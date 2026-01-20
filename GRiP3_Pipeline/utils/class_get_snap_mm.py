import pyrealsense2 as rs
import numpy as np
import cv2
import os

class RealSenseCapture:
    def __init__(
        self,
        save_directory="/GRiP3_Pipeline/sample_data/real_world/XY",
        rgb_filename="rgn.png",
        depth_filename="depth.npy",
        resolution_width=640,
        resolution_height=480,
        fps=30
    ):
        """
        Inizializza i parametri per la cattura da RealSense.
        """
        self.save_directory = save_directory
        self.rgb_filename = rgb_filename
        self.depth_filename = depth_filename
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.fps = fps

        # Creazione della cartella di salvataggio se non esiste
        os.makedirs(self.save_directory, exist_ok=True)

        # Configura il pipeline e le opzioni di streaming
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.depth,
            self.resolution_width,
            self.resolution_height,
            rs.format.z16,
            self.fps
        )
        self.config.enable_stream(
            rs.stream.color,
            self.resolution_width,
            self.resolution_height,
            rs.format.bgr8,
            self.fps
        )

        # Creiamo un oggetto per allineare il frame di depth al frame di colore.
        self.align = rs.align(rs.stream.color)

    def run_capture(self):
        """
        Avvia la cattura da RealSense. Mostra l'immagine a schermo e
        attende la pressione di 's' per salvare (e poi terminare) o ESC
        per uscire senza salvare.
        """
        # Avvia lo streaming
        try:
            profile = self.pipeline.start(self.config)
            print("Streaming started. Press 's' to save the image and depth data, or 'ESC' to exit without saving.")
        except Exception as e:
            print(f"Failed to start the RealSense pipeline: {e}")
            return

        # Ottieni il depth scale per convertire i valori in metri
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()  # ad es. 0.001 se i valori sono in millimetri
        print(f"Depth Scale ottenuto: {depth_scale}")

        try:
            while True:
                # Attendi un set di frame (depth e color)
                frames = self.pipeline.wait_for_frames()
                # Allinea il frame di depth al frame color
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Converti i frame in array NumPy
                raw_depth = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Converti il depth in metri
                depth_image = raw_depth.astype(np.float32) * depth_scale

                # (Opzionale) Crea una mappa di colori per la depth per visualizzare meglio i dettagli
                # Qui moltiplichiamo per 255 se vogliamo visualizzare come immagine (questa parte non altera i dati salvati)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=255/depth_image.max()),
                    cv2.COLORMAP_JET
                )

                # Mostra a schermo le immagini color e depth
                cv2.imshow('RealSense - Color', color_image)
                cv2.imshow('RealSense - Depth', depth_colormap)

                # Gestione input da tastiera
                key = cv2.waitKey(1) & 0xFF

                # Se l'utente preme 's', salva e termina
                if key == ord('s'):
                    rgb_path = os.path.join(self.save_directory, self.rgb_filename)
                    cv2.imwrite(rgb_path, color_image)
                    print(f"Saved RGB image to {rgb_path}")

                    depth_path = os.path.join(self.save_directory, self.depth_filename)
                    np.save(depth_path, depth_image)
                    print(f"Saved depth data to {depth_path}")

                    break  # Esci dopo aver salvato

                # Se l'utente preme ESC (27), esci senza salvare
                elif key == 27:  # ESC
                    print("Exiting without saving.")
                    break

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            # Ferma lo streaming e chiudi le finestre
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("Stream stopped and windows closed.")

# Per eseguire la cattura
if __name__ == "__main__":
    capture = RealSenseCapture()
    capture.run_capture()