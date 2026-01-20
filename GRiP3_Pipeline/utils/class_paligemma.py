import sys
import os
import re
import functools
from typing import List, Tuple

# ---- Libraries for VAE & annotation ----
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import PIL.Image
import cv2
import supervision as sv
import json

from huggingface_hub import hf_hub_download

# ---- Libraries for PyTorch / Transformers / PEFT ----
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForPreTraining, PaliGemmaProcessor


class PaliGemmaInference:
    """
    This class encapsulates:
      - Loading the PEFT model and base model.
      - Decoding and reconstructing masks via VQ-VAE.
      - Annotating with Supervision.
      - Automatically creating and saving the label mask.

    Modes available via infer(...):
      - "command": returns only the decoded text.
      - "detect": returns bounding box detections only.
         **Now also computes and saves the target 3D position in the robot base frame as a JSON file.**
      - "segmentation": returns annotated images and optionally saves the label mask.
    """
    # Regex for segmentation mode: expects 4 <loc####> plus 16 <seg###> tokens.
    _SEGMENT_DETECT_RE = re.compile(
        r'(.*?)' +
        r'<loc(\d{4})>' * 4 + r'\s*' +
        r'(?:' + (r'<seg(\d{3})>' * 16) + ')' +
        r'\s*([^;<>]+)? ?(?:; )?'
    )

    def __init__(self,
                 peft_model_path: str,
                 base_model_id: str,
                 local_vae_path: str = "./vae-oid.npz",
                 device: str = None):
        """
        :param peft_model_path: Trained PEFT checkpoint.
        :param base_model_id:   ID of the base model (e.g., "google/paligemma-3b-mix-448").
        :param local_vae_path:  Path to the vae-oid.npz file.
        :param device:          "cuda" or "cpu". If None, autodetect.
        """
        self.peft_model_path = peft_model_path
        self.base_model_id = base_model_id
        self.local_vae_path = local_vae_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Download vae-oid.npz if needed.
        self._download_vae_npz_if_needed()

        # 2) Load the PEFT model and processor.
        self._load_model()

        # 3) Initialize the decoder.
        self._init_decoder()

    def _download_vae_npz_if_needed(self):
        if not os.path.exists(self.local_vae_path):
            print("Downloading vae-oid.npz from Hugging Face...")
            try:
                hf_hub_download(
                    repo_id="google/paligemma-hf",
                    filename="vae-oid.npz",
                    repo_type="space",
                    local_dir="."
                )
                print("Downloaded vae-oid.npz successfully.")
            except Exception as e:
                print(f"Failed to download vae-oid.npz: {e}")
                sys.exit(1)
        else:
            print("vae-oid.npz already exists. Skipping download.")

    def _load_model(self):
        print(f"Loading PEFT model from: {self.peft_model_path}")
        config = PeftConfig.from_pretrained(self.peft_model_path)
        base_model = AutoModelForPreTraining.from_pretrained(self.base_model_id)
        self.model = PeftModel.from_pretrained(base_model, self.peft_model_path)
        self.processor = PaliGemmaProcessor.from_pretrained(self.base_model_id)
        self.model.to(self.device)

    def _get_params(self, checkpoint):
        """Converts PyTorch weights (npz) to Flax parameters."""

        def transp(kernel):
            return np.transpose(kernel, (2, 3, 1, 0))

        def conv(name):
            return {
                'bias': checkpoint[name + '.bias'],
                'kernel': transp(checkpoint[name + '.weight']),
            }

        def resblock(name):
            return {
                'Conv_0': conv(name + '.0'),
                'Conv_1': conv(name + '.2'),
                'Conv_2': conv(name + '.4'),
            }

        return {
            '_embeddings': checkpoint['_vq_vae._embedding'],
            'Conv_0': conv('decoder.0'),
            'ResBlock_0': resblock('decoder.2.net'),
            'ResBlock_1': resblock('decoder.3.net'),
            'ConvTranspose_0': conv('decoder.4'),
            'ConvTranspose_1': conv('decoder.6'),
            'ConvTranspose_2': conv('decoder.8'),
            'ConvTranspose_3': conv('decoder.10'),
            'Conv_1': conv('decoder.12'),
        }

    def _init_decoder(self):
        with open(self.local_vae_path, 'rb') as f:
            ckpt = dict(np.load(f))
        params = self._get_params(ckpt)

        class ResBlock(nn.Module):
            features: int

            @nn.compact
            def __call__(self, x):
                original_x = x
                x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
                x = nn.relu(x)
                x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
                x = nn.relu(x)
                x = nn.Conv(features=self.features, kernel_size=(1, 1), padding=0)(x)
                return x + original_x

        class Decoder(nn.Module):
            @nn.compact
            def __call__(self, x):
                num_res_blocks = 2
                dim = 128
                num_upsample_layers = 4
                x = nn.Conv(features=dim, kernel_size=(1, 1), padding=0)(x)
                x = nn.relu(x)
                for _ in range(num_res_blocks):
                    x = ResBlock(features=dim)(x)
                for _ in range(num_upsample_layers):
                    x = nn.ConvTranspose(
                        features=dim,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding=2,
                        transpose_kernel=True
                    )(x)
                    x = nn.relu(x)
                    dim //= 2
                x = nn.Conv(features=1, kernel_size=(1, 1), padding=0)(x)
                return x

        def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
            batch_size, num_tokens = codebook_indices.shape
            assert num_tokens == 16, f"Expected 16 seg tokens, got {num_tokens}"
            _, embedding_dim = embeddings.shape
            encodings = jnp.take(embeddings, codebook_indices.reshape((-1)), axis=0)
            encodings = encodings.reshape((batch_size, 4, 4, embedding_dim))
            return encodings

        def reconstruct_masks(codebook_indices):
            quantized = _quantized_values_from_codebook_indices(codebook_indices, params['_embeddings'])
            return Decoder().apply({'params': params}, quantized)

        self._reconstruct_fn = jax.jit(reconstruct_masks, backend='cpu')

    # --------------------- Inference Method ---------------------
    def infer(self,
              prompt: str,
              image_path: str,
              output_path: str = None,
              classes: List[str] = None,
              save_label_mask: bool = True,
              label_mask_path: str = None,
              mode: str = "segmentation"
              ) -> any:
        """
        Performs inference in one of three modes:
          - "command": returns only the decoded text.
          - "detect": returns bounding box detections only.
                     In addition, computes and saves the 3D target position in the robot base frame.
          - "segmentation": produces segmentation overlays and optionally saves the label mask.
        """
        if not os.path.exists(image_path):
            print(f"ERROR: image not found at {image_path}")
            sys.exit(1)

        # Load image
        pil_image = PIL.Image.open(image_path).convert("RGB")
        w, h = pil_image.size

        # Preprocess and generate output
        inputs = self.processor(
            images=pil_image,
            text=prompt,
            padding="longest",
            do_convert_rgb=True,
            return_tensors="pt"
        ).to(self.device)
        inputs = inputs.to(dtype=self.model.dtype)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=2048, do_sample=False)
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)
        print("\nDecoded Output:")
        print(decoded_output)

        # For "command" mode, return the text only.
        if mode == "command":
            return decoded_output.strip()

        # Process output tokens: remove initial "segment" or "detect" if present.
        lines = decoded_output.split('\n', 1)
        if len(lines) > 1:
            line0 = lines[0].lower()
            if "segment " in line0 or "detect " in line0:
                cleaned_text = lines[1].strip()
            else:
                cleaned_text = decoded_output.strip()
        else:
            cleaned_text = decoded_output.strip()

        # --------------------- DETECT MODE ---------------------
        if mode == "detect":
            # Parse detections (expecting 4 <loc####> tokens and an optional class).
            detections = self._convert_to_detections_detect(cleaned_text, (w, h), classes or [])
            if len(detections) == 0:
                print("No bounding boxes found for detect mode.")
                return detections

            annotated_image = self._annotate_detection(np.array(pil_image), detections)
            if output_path:
                PIL.Image.fromarray(annotated_image).save(output_path)
                print(f"Bounding-box-only image saved at: {output_path}")

            # ---------------- New utils: Compute & Save 3D Target Poses ----------------
            # Load depth image (assumed in meters)
            depth_path = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/sample_data/real_world/XY/depth.npy"
            depth_image = np.load(depth_path)

            # Provided extrinsics (camera_pose) and intrinsic parameters.
            camera_pose = np.array([[7.7483457e-01, -4.2044988e-01, 4.7207338e-01, -1.5715596e-01],
                                    [-6.3216394e-01, -5.1492232e-01, 5.7898515e-01, -5.9072340e-01],
                                    [-3.5311221e-04, -7.4704546e-01, -6.6477287e-01, 2.2176746e-01],
                                    [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            intrinsics = np.array([[606.0859375, 0, 310.45910645],
                                   [0, 605.17205811, 252.21199036],
                                   [0, 0, 1]], dtype=np.float32)
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

            # Create an empty list to store the target poses.
            target_poses = []

            for i in range(len(detections)):
                # Get bounding box coordinates [x1, y1, x2, y2]
                bbox = detections.xyxy[i]
                center_u = int(round((bbox[0] + bbox[2]) / 2.0))
                center_v = int(round((bbox[1] + bbox[3]) / 2.0))

                # Get depth value at the center pixel.
                z = depth_image[center_v, center_u]
                # Convert pixel coordinates (with depth) to 3D camera coordinates.
                X_cam = (center_u - cx) * z / fx
                Y_cam = (center_v - cy) * z / fy
                Z_cam = z
                cam_coords_hom = np.array([X_cam, Y_cam, Z_cam, 1]).reshape(4, 1)
                # Transform the camera coordinates to robot base frame.
                robot_coords = camera_pose @ cam_coords_hom
                robot_coords = robot_coords.flatten()[:3]
                print(f"Detection {i + 1} center in robot base coordinates: {robot_coords}")
                target_poses.append(robot_coords.tolist())

            # Save the target poses to a JSON file.
            json_save_path = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/sample_data/real_world/XY/target_pose.json"
            with open(json_save_path, "w") as json_file:
                json.dump(target_poses, json_file, indent=2)
            print(f"Target poses saved to: {json_save_path}")
            # -----------------------------------------------------------------------------

            return detections

        # --------------------- SEGMENTATION MODE ---------------------
        classes = classes or []
        detections = self._convert_to_detections(cleaned_text, (w, h), classes)
        if output_path:
            annotated_image = self._annotate_image(np.array(pil_image), detections)
            PIL.Image.fromarray(annotated_image).save(output_path)
            print(f"Annotated image (segmentation) saved at: {output_path}")
        if save_label_mask:
            self._create_and_save_label_mask_auto(
                detections=detections,
                pil_image=pil_image,
                image_path=image_path,
                label_mask_path=label_mask_path
            )
        return detections

    # --------------------- DETECTION PARSING METHODS ---------------------
    def _convert_to_detections_detect(self,
                                      text: str,
                                      resolution_wh: Tuple[int, int],
                                      classes: List[str]) -> sv.Detections:
        w, h = resolution_wh
        detect_re = re.compile(
            r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([^;<>]+)?'
        )
        match = detect_re.search(text)
        if not match:
            return sv.Detections.empty()
        y1f = int(match.group(1)) / 1024.0
        x1f = int(match.group(2)) / 1024.0
        y2f = int(match.group(3)) / 1024.0
        x2f = int(match.group(4)) / 1024.0
        y1, x1 = round(y1f * h), round(x1f * w)
        y2, x2 = round(y2f * h), round(x2f * w)
        name = match.group(5)
        if name is None:
            name = ""
        name = name.strip()
        xyxy = np.array([[x1, y1, x2, y2]], dtype=int)
        mask = np.zeros((1, h, w), dtype=bool)
        if name and name not in classes:
            classes.append(name)
        cls_id = classes.index(name) if name else -1
        detections = sv.Detections(
            xyxy=xyxy,
            mask=mask,
            class_id=np.array([cls_id], dtype=int)
        )
        detections["class_name"] = [name]
        return detections

    # --------------------- SEGMENTATION PARSING METHODS ---------------------
    def _convert_to_detections(self, text: str, resolution_wh: Tuple[int, int], classes: List[str]) -> sv.Detections:
        w, h = resolution_wh
        parsed = self._extract_objs(text, w, h)
        if not parsed:
            return sv.Detections.empty()
        xyxy_list = []
        mask_list = []
        class_id_list = []
        class_name_list = []
        for obj in parsed:
            if 'xyxy' not in obj or 'name' not in obj:
                continue
            (x1, y1, x2, y2) = obj['xyxy']
            xyxy_list.append([x1, y1, x2, y2])
            m = obj['mask']
            if m is None:
                bin_m = np.zeros((h, w), dtype=bool)
            else:
                _, thres_m = cv2.threshold(m, 0.5, 1.0, cv2.THRESH_BINARY)
                bin_m = thres_m.astype(bool)
            mask_list.append(bin_m)
            name = obj['name'].strip()
            if name not in classes:
                classes.append(name)
            cls_id = classes.index(name)
            class_id_list.append(cls_id)
            class_name_list.append(name)
        detections = sv.Detections(
            xyxy=np.array(xyxy_list, dtype=int),
            mask=np.array(mask_list, dtype=bool),
            class_id=np.array(class_id_list, dtype=int)
        )
        detections["class_name"] = class_name_list
        return detections

    def _extract_objs(self, text: str, width: int, height: int):
        objs = []
        while text:
            m = self._SEGMENT_DETECT_RE.match(text)
            if not m:
                break
            groups = list(m.groups())
            before = groups.pop(0)
            class_name = groups.pop()
            if class_name is None:
                class_name = ""
            y1, x1, y2, x2 = [int(v) / 1024 for v in groups[:4]]
            y1, x1, y2, x2 = map(round, (y1 * height, x1 * width, y2 * height, x2 * width))
            seg_indices = groups[4:20]
            if seg_indices[0] is None:
                mask = None
            else:
                seg_indices = [int(s) for s in seg_indices]
                mask64 = self._reconstruct_fn(np.array(seg_indices, dtype=np.int32)[None])[..., 0]
                mask64 = np.array(mask64)[0]
                mask64 = np.clip(mask64 * 0.5 + 0.5, 0, 1)
                mask = np.zeros((height, width), dtype=np.float32)
                if x2 > x1 and y2 > y1:
                    mask64_u8 = (mask64 * 255).astype(np.uint8)
                    resized = cv2.resize(mask64_u8, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                    mask[y1:y2, x1:x2] = resized.astype(np.float32) / 255.0
            content = m.group()
            if before:
                objs.append(dict(content=before))
                content = content[len(before):]
            objs.append({
                'content': content,
                'xyxy': (x1, y1, x2, y2),
                'mask': mask,
                'name': class_name.strip()
            })
            text = text[len(before) + len(content):]
        if text:
            objs.append(dict(content=text))
        return objs

    # --------------------- ANNOTATION METHODS ---------------------
    def _annotate_image(self, image_np: np.ndarray, detections: sv.Detections) -> np.ndarray:
        mask_annotator = sv.MaskAnnotator(color=sv.Color.GREEN)
        annotated_image = mask_annotator.annotate(scene=image_np.copy(), detections=detections)
        box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        return annotated_image

    def _annotate_detection(self, image_np: np.ndarray, detections: sv.Detections) -> np.ndarray:
        box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
        annotated_image = box_annotator.annotate(scene=image_np.copy(), detections=detections)
        for i, cid in enumerate(detections.class_id):
            if cid < 0:
                detections.class_id[i] = 0
                detections["class_name"][i] = ""
        label_annotator = sv.LabelAnnotator()
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        return annotated_image

    def _create_and_save_label_mask_auto(self,
                                         detections: sv.Detections,
                                         pil_image: PIL.Image.Image,
                                         image_path: str,
                                         label_mask_path: str = None
                                         ) -> None:
        w, h = pil_image.size
        label_mask = np.full((h, w), 0, dtype=np.uint8)
        print(f"Sfondo => label_value=0")
        if len(detections) == 0:
            print("Nessun detection, maschera = 0 ovunque (sfondo).")
            if label_mask_path is None:
                base, ext = os.path.splitext(image_path)
                label_mask_path = base + "_mask.png"
            cv2.imwrite(label_mask_path, label_mask)
            print(f"Mask salvata in: {label_mask_path}")
            return
        for i in range(len(detections)):
            detection_mask = detections.mask[i]
            label_value = 101 + i
            class_name = detections["class_name"][i]
            print(f"Oggetto {i + 1}: '{class_name}' => label_value={label_value}")
            label_mask[detection_mask] = label_value
        if label_mask_path is None:
            label_mask_path = os.path.join(os.path.dirname(image_path), "seg.png")
        cv2.imwrite(label_mask_path, label_mask)
        print(f"Mask salvata in: {label_mask_path}")
