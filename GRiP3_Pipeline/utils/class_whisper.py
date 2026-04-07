import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import whisper
import os


class WhisperTranscriber:
    def __init__(self, output_dir="/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/whisper"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.wav_path = os.path.join(self.output_dir, "prompt.wav")
        # You can later add additional configuration parameters as needed.

    def record_and_transcribe(self, duration=5, samplerate=16000, channels=1):
        # --- Step 1: List devices
        print("\n🎧 Available audio devices:\n")
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"[{idx}] {dev['name']}")

        # --- Step 2: Select input device
        try:
            device_id = int(input("\n🔧 Enter the device ID for your microphone: "))
            sd.default.device = device_id
        except Exception as e:
            print(f"Error selecting device: {e}")
            return ""

        # --- Step 3: Record from the selected device
        print("\n🎙️ Recording... Speak now!")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                       channels=channels, dtype='int16')
        sd.wait()
        print("✅ Recording done.\n")

        # --- Step 4: Save WAV file
        write(self.wav_path, samplerate, audio)
        print(f"💾 Audio saved at: {self.wav_path}")

        # --- Step 5: Transcribe using Whisper
        print("🔍 Loading Whisper model...")
        model = whisper.load_model("turbo")  # Options: "tiny", "small", "medium", "large", "turbo"
        result = model.transcribe(self.wav_path, language="en", temperature=0.0, task="transcribe")
        transcription = result["text"].strip()
        print("\n📝 Transcription:")
        print(transcription)
        return transcription
