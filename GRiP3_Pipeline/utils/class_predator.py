#!/usr/bin/env python3
import os
import sys
import subprocess
import numpy as np
import open3d as o3d
from PIL import Image
import torch

class PredatorPipeline:
    def __init__(self,
                 sample_dir: str,
                 predator_script: str,
                 predator_cfg: str,
                 label: int = 101):
        """
        :param sample_dir: directory containing rgb.png, depth.npy, seg.png.
        :param predator_script: absolute path to your demo_save.py
        :param predator_cfg: absolute path to the config .yaml for Predator
        :param label: segmentation label to keep and assign
        """
        self.sample_dir   = sample_dir
        self.pth_path     = os.path.join(sample_dir, "scene.pth")
        self.ply_path     = os.path.join(sample_dir, "merged.ply")
        self.npz_path     = os.path.join(sample_dir, "full_object_pointcloud.npz")
        self.rgb_file     = os.path.join(sample_dir, "rgb.png")
        self.depth_file   = os.path.join(sample_dir, "depth.npy")
        self.mask_file    = os.path.join(sample_dir, "seg.png")
        self.pred_script  = predator_script
        self.pred_cfg     = predator_cfg
        self.label        = label

        # camera intrinsics
        self.fx, self.fy, self.cx, self.cy = (
            606.0859375, 605.17205811, 310.45910645, 252.21199036
        )
        self.depth_scale = 1.0

    def build_scene_pth(self):
        # 1) load inputs
        rgb   = np.array(Image.open(self.rgb_file))      # only used for masking
        depth = np.load(self.depth_file)
        mask  = np.array(Image.open(self.mask_file))

        # 2) backproject to 3D
        H, W = depth.shape
        Z = (depth.flatten() * self.depth_scale)
        valid = Z > 0
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.flatten()[valid]
        v = v.flatten()[valid]
        Z = Z[valid]
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        xyz = np.stack([X, Y, Z], axis=1)

        # 3) apply mask to keep only your object label
        mask_flat = mask.flatten()[valid] == self.label
        obj_xyz   = xyz[mask_flat]

        # 4) MAD filter outliers in Z
        Zv = obj_xyz[:, 2]
        med, mad = np.median(Zv), np.median(np.abs(Zv - np.median(Zv)))
        keep = np.abs(Zv - med) < (3 * mad)
        obj_xyz = obj_xyz[keep]

        # 5) scale ×10 so units match Predator’s training
        obj_xyz *= 10.0

        # 6) save **only** the XYZ coords (N×3) into scene.pth
        arr = obj_xyz.astype(np.float32)
        torch.save(arr, self.pth_path)
        print(f"[1] scene.pth saved: {self.pth_path} (shape={arr.shape})")

    def run_predator(self):
        # 1) avoid MKL/GOMP clash
        env = os.environ.copy()
        env["MKL_SERVICE_FORCE_INTEL"] = "1"

        # 2) build command
        cmd = [
            sys.executable,
            self.pred_script,
            self.pred_cfg
        ]

        # 3) run inside OverlapPredator so Results/ is correct
        demo_root = os.path.abspath(os.path.join(
            os.path.dirname(self.pred_script), ".."
        ))
        subprocess.check_call(cmd, env=env, cwd=demo_root)

        # 4) move merged.ply into sample_dir
        src = os.path.join(demo_root, "Results", "merged.ply")
        if os.path.isfile(src):
            os.replace(src, self.ply_path)
        print(f"[2] Predator registration done → {self.ply_path}")

    def convert_npz(self):
        # 1) load merged.ply
        pcd = o3d.io.read_point_cloud(self.ply_path)
        pts = np.asarray(pcd.points, dtype=np.float64)

        # 2) downscale ÷10 back to meters
        pts /= 10.0

        # 3) dummy RGB zeros
        rgb0 = np.zeros_like(pts, dtype=np.float64)

        # 4) build and save NPZ (6 dims: xyz + rgb0)
        full_inputs = np.concatenate([pts, rgb0], axis=1)
        full_seg    = np.full((full_inputs.shape[0],), self.label, dtype=np.uint32)
        os.makedirs(os.path.dirname(self.npz_path), exist_ok=True)
        np.savez(self.npz_path,
                 full_inputs=full_inputs,
                 full_seg=full_seg)
        print(f"[3] NPZ saved: {self.npz_path} (shape={full_inputs.shape})")

    def run(self):
        self.build_scene_pth()
        self.run_predator()
        self.convert_npz()
