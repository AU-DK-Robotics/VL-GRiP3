#!/usr/bin/env python3
import os
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

def show_depth():
    depth = np.load("depth.npy")
    plt.figure(figsize=(8, 6))
    plt.title("Depth Image")
    plt.imshow(depth, cmap="gray")
    plt.axis("off")
    plt.show()

def pc_from_numpy(xyz, rgb=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def main():
    # 1) depth
    # if os.path.exists("depth.npy"):
    #     print("[1] Showing depth.npy…")
    #     show_depth()
    # else:
    #     print("→ depth.npy not found, skipping.")

    # 2) scene.pth
    if os.path.exists("scene.pth"):
        print("[2] Loading scene.pth…")
        scene_data = torch.load("scene.pth", weights_only=False)
        if isinstance(scene_data, np.ndarray):
            scene_xyz = scene_data
        else:
            scene_xyz = scene_data.numpy()
        scene_pcd = pc_from_numpy(scene_xyz)
        scene_pcd.paint_uniform_color([0.216, 0.494, 0.722])
    else:
        print("→ scene.pth not found.")
        scene_pcd = None

    # # 3) full_object_pointcloud.npz
    # if os.path.exists("full_object_pointcloud.npz"):
    #     print("[3] Loading full_object_pointcloud.npz…")
    #     data    = np.load("full_object_pointcloud.npz")
    #     full_in = data["full_inputs"]
    #     obj_xyz = full_in[:, :3]
    #     obj_rgb = full_in[:, 3:6] / 255.0
    #     if np.all(obj_rgb == 0):
    #         obj_rgb = np.tile([0.0,1.0,0.0], (obj_xyz.shape[0],1))
    #     obj_pcd = pc_from_numpy(obj_xyz, obj_rgb)
    # else:
    #     print("→ full_object_pointcloud.npz not found.")
    #     obj_pcd = None

    # 4) merged.ply
    if os.path.exists("merged.ply"):
        print("[4] Loading merged.ply…")
        merged_pcd = o3d.io.read_point_cloud("merged.ply")
        merged_pcd.paint_uniform_color([0.267, 0.596, 0.329])
    else:
        print("→ merged.ply not found.")
        merged_pcd = None

    # 5) visualize all together
    to_draw = [pc for pc in (merged_pcd, scene_pcd) if pc is not None]
    if not to_draw:
        print("Nothing to visualize.")
        return

    print("[5] Opening Open3D viewer with all point clouds…")
    o3d.visualization.draw_geometries(
        to_draw,
        window_name="Merged(Green)|Scene(Blue)",
        width=1024, height=768
    )

if __name__ == "__main__":
    main()
