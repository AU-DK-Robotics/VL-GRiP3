#!/usr/bin/env python3
"""
Rigenera sempre (sovrascrive) i file di output per tac1, tac2 e tac3:
  • training/CAD/tacX_ok.pth
  • training/tacX/tacX_*.pth
  • configs/indoor/train_info.pkl   (tutti gli oggetti)
  • configs/indoor/val_info.pkl     (tutti gli oggetti)

Per tac1: 15 coppie train
Per tac2: 10 coppie train
Per tac3: 18 coppie train
Il resto in validation.
"""
import os, glob, pickle
import numpy as np
import open3d as o3d
import torch

# CONFIG -----------------------------------------------
ROOT = "/home/au-robotics/MircoProjects/Predator/OverlapPredator/training"
CAD_DIR = os.path.join(ROOT, "CAD")
OBJECTS = [("tac1", 64), ("tac2", 68), ("tac3", 54)]  # (object_name, n_train)
N_POINTS = 55000  # punti da campionare se mesh


# -------------------------------------------------------

# helper ------------------------------------------------
def read_as_pcd(path, n_pts):
    geo = o3d.io.read_triangle_mesh(path)
    if isinstance(geo, o3d.geometry.TriangleMesh) and len(geo.triangles) > 0:
        return geo.sample_points_uniformly(number_of_points=n_pts)
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty geometry: {path}")
    return pcd


def save_pth(pcd, out_path):
    xyz = np.asarray(pcd.points, dtype=np.float32)
    torch.save(xyz, out_path)
    print(f"✓ wrote {os.path.relpath(out_path, ROOT)} ({xyz.shape[0]} pts)")


def load_mat(txt_path):
    M = np.loadtxt(txt_path, dtype=np.float32).reshape(4, 4)
    R = M[:3, :3]
    t = M[:3, 3].reshape(3, 1)  # ensure shape (3,1)
    return R, t


rel = lambda p: os.path.relpath(p, ROOT).replace('\\', '/')


# -------------------------------------------------------


def main():
    info = {'src': [], 'tgt': [], 'rot': [], 'trans': []}
    train_ids = []
    val_ids = []
    offset = 0

    # convert each object
    for obj_name, n_train in OBJECTS:
        # CAD
        cad_stl = os.path.join(CAD_DIR, f"{obj_name}_ok.stl")
        cad_pth = cad_stl.replace('.stl', '.pth')
        print(f"Converting CAD {obj_name} (overwrite)...")
        save_pth(read_as_pcd(cad_stl, N_POINTS), cad_pth)

        # scenes
        scene_dir = os.path.join(ROOT, obj_name)
        pattern = os.path.join(scene_dir, f"{obj_name}_*.ply")
        pairs = sorted(glob.glob(pattern))
        total = 0
        for ply in pairs:
            idx = os.path.basename(ply).split('_')[-1].split('.')[0]
            txt = os.path.join(scene_dir, f"{idx}.txt")
            if not os.path.isfile(txt):
                print(f"⛔ missing {txt}, skipping {ply}")
                continue
            # convert scene pcd
            pth = ply.replace('.ply', '.pth')
            print(f"Converting scene {obj_name}_{idx} (overwrite)...")
            save_pth(read_as_pcd(ply, N_POINTS), pth)
            # load GT
            R, t = load_mat(txt)
            # append
            info['src'].append(rel(cad_pth))
            info['tgt'].append(rel(pth))
            info['rot'].append(R)
            info['trans'].append(t)
            total += 1
        # assign train/val indices for this object
        ids = list(range(offset, offset + total))
        train_ids += ids[:n_train]
        val_ids += ids[n_train:]
        offset += total
        print(f"  → {obj_name}: total {total}, train {n_train}, val {total - n_train}")

    # dump combined pickle
    os.makedirs('configs/indoor', exist_ok=True)
    with open('configs/indoor/training_vlam_info.pkl', 'wb') as f:
        pickle.dump({k: [info[k][i] for i in train_ids] for k in info}, f)
    with open('configs/indoor/validation_vlam_info.pkl', 'wb') as f:
        pickle.dump({k: [info[k][i] for i in val_ids] for k in info}, f)

    print(f"\n✓ train_info.pkl   ({len(train_ids)} pairs)")
    print(f"✓ val_info.pkl     ({len(val_ids)} pairs)")


if __name__ == '__main__':
    main()

