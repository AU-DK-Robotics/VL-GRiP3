"""
Scripts for pairwise registration demo

Author: Shengyu Huang
Last modified: 26.05.2025 – merged PLY saving added / dtype‑safe
"""
import os, torch, time, shutil, json, glob, sys, copy, argparse
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch import optim, nn
import open3d as o3d

cwd = os.getcwd()
sys.path.append(cwd)
from datasets.indoor import IndoorDataset
from datasets.dataloader_limits import get_dataloader
from models.architectures import KPFCNN
from lib.utils import load_obj, setup_seed, natural_key, load_config
from lib.benchmark_utils import ransac_pose_estimation, to_o3d_pcd, get_blue, get_yellow, to_tensor
from lib.trainer import Trainer
from lib.loss import MetricLoss

setup_seed(0)


class ThreeDMatchDemo(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """

    def __init__(self, config, src_path, tgt_path):
        super().__init__()
        self.config = config
        self.src_path = src_path
        self.tgt_path = tgt_path

    def __len__(self):
        return 1

    def __getitem__(self, item):
        # load pointcloud tensors (N,3) float32
        src_pcd = torch.load(self.src_path, weights_only=False).astype(np.float32)
        tgt_pcd = torch.load(self.tgt_path, weights_only=False).astype(np.float32)

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        # fake ground‑truth info
        rot = np.eye(3, dtype=np.float32)
        trans = np.ones((3, 1), dtype=np.float32)
        correspondences = torch.ones(1, 2).long()

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)


def lighter(color, percent):
    """assumes color is rgb between (0,0,0) and (1,1,1)"""
    color = np.asarray(color)
    white = np.array([1, 1, 1])
    return color + (white - color) * percent


def draw_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, src_saliency, tgt_saliency, tsfm):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. overlap colors
    rot, trans = to_tensor(tsfm[:3, :3]), to_tensor(tsfm[:3, 3][:, None])
    src_overlap = src_overlap[:, None].repeat(1, 3).numpy()
    tgt_overlap = tgt_overlap[:, None].repeat(1, 3).numpy()
    src_overlap_color = lighter(get_yellow(), 1 - src_overlap)
    tgt_overlap_color = lighter(get_blue(), 1 - tgt_overlap)
    src_pcd_overlap = copy.deepcopy(src_pcd_before)
    src_pcd_overlap.transform(tsfm)
    tgt_pcd_overlap = copy.deepcopy(tgt_pcd_before)
    src_pcd_overlap.colors = o3d.utility.Vector3dVector(src_overlap_color)
    tgt_pcd_overlap.colors = o3d.utility.Vector3dVector(tgt_overlap_color)

    ########################################
    # 3. draw registrations
    src_pcd_after = copy.deepcopy(src_pcd_before)
    src_pcd_after.transform(tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input', width=960, height=540, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='Inferred overlap region', width=960, height=540, left=0, top=600)
    vis2.add_geometry(src_pcd_overlap)
    vis2.add_geometry(tgt_pcd_overlap)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name='Our registration', width=960, height=540, left=960, top=0)
    vis3.add_geometry(src_pcd_after)
    vis3.add_geometry(tgt_pcd_before)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis3.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_overlap)
        vis2.update_geometry(tgt_pcd_overlap)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        vis3.update_geometry(src_pcd_after)
        vis3.update_geometry(tgt_pcd_before)
        if not vis3.poll_events():
            break
        vis3.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()


def main(config, demo_loader):
    config.model.eval()
    c_loader_iter = iter(demo_loader)
    with torch.no_grad():
        try:
            inputs = next(c_loader_iter)
        except StopIteration:
            c_loader_iter = iter(demo_loader)
            inputs = next(c_loader_iter)

        # --------------------------------------------------
        # move inputs to device
        # --------------------------------------------------
        for k, v in inputs.items():
            if isinstance(v, list):
                inputs[k] = [item.to(config.device) for item in v]
            else:
                inputs[k] = v.to(config.device)

        # --------------------------------------------------
        # forward pass
        # --------------------------------------------------
        feats, scores_overlap, scores_saliency = config.model(inputs)
        pcd = inputs['points'][0]
        len_src = inputs['stack_lengths'][0][0]

        src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
        src_raw = src_pcd.clone()
        tgt_raw = tgt_pcd.clone()

        src_feats, tgt_feats = feats[:len_src].cpu(), feats[len_src:].cpu()
        src_overlap, src_saliency = scores_overlap[:len_src].cpu(), scores_saliency[:len_src].cpu()
        tgt_overlap, tgt_saliency = scores_overlap[len_src:].cpu(), scores_saliency[len_src:].cpu()

        # probabilistic sampling
        src_scores = (src_overlap * src_saliency).flatten()
        tgt_scores = (tgt_overlap * tgt_saliency).flatten()

        if src_pcd.size(0) > config.n_points:
            probs = (src_scores / src_scores.sum()).numpy()
            idx = np.random.choice(src_pcd.size(0), config.n_points, replace=False, p=probs)
            src_pcd, src_feats = src_pcd[idx], src_feats[idx]
        if tgt_pcd.size(0) > config.n_points:
            probs = (tgt_scores / tgt_scores.sum()).numpy()
            idx = np.random.choice(tgt_pcd.size(0), config.n_points, replace=False, p=probs)
            tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

        # --------------------------------------------------
        # estimate pose with RANSAC
        # --------------------------------------------------
        tsfm = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)

        # --------------------------------------------------
        # SAVE merged cloud (.ply)
        # --------------------------------------------------
        out_dir = "/home/au-robotics/MircoProjects/VL_GRiP3/OverlapPredator/Results"
        os.makedirs(out_dir, exist_ok=True)

        # dtype‑safe conversion (float32)
        rot_mat = torch.from_numpy(tsfm[:3, :3].astype(np.float32)).to(src_pcd.device)
        trans_vec = torch.from_numpy(tsfm[:3, 3].astype(np.float32)).to(src_pcd.device)

        src_pcd_after = src_pcd @ rot_mat.T + trans_vec  # (N,3)
        merged_xyz = torch.cat([src_pcd_after, tgt_pcd], dim=0).cpu().numpy()

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_xyz)
        out_path = os.path.join(out_dir, "merged.ply")
        o3d.io.write_point_cloud(out_path, merged_pcd, write_ascii=True)
        print(f"[INFO] Point‑cloud merged saved to: {out_path}")

        # show results (optional)
        draw_registration_result(
            src_raw,
            tgt_raw,
            src_overlap,
            tgt_overlap,
            src_saliency,
            tgt_saliency,
            tsfm,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file.')
    args = parser.parse_args()

    config = edict(load_config(args.config))
    config.device = torch.device('cuda' if config.get('gpu_mode', False) else 'cpu')

    # build architecture list
    config.architecture = ['simple', 'resnetb']
    for _ in range(config.num_layers - 1):
        config.architecture += ['resnetb_strided', 'resnetb', 'resnetb']
    for _ in range(config.num_layers - 2):
        config.architecture += ['nearest_upsample', 'unary']
    config.architecture += ['nearest_upsample', 'last_unary']

    config.model = KPFCNN(config).to(config.device)

    # datasets & loaders
    info_train = load_obj(config.train_info)
    train_set = IndoorDataset(info_train, config, data_augmentation=True)
    demo_set = ThreeDMatchDemo(config, config.src_pcd, config.tgt_pcd)

    _, neighborhood_limits = get_dataloader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    demo_loader, _ = get_dataloader(
        dataset=demo_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,
        neighborhood_limits=neighborhood_limits,
    )

    # load pretrained weights
    assert config.pretrain, "Pretrained weights not specified in config."
    state = torch.load(config.pretrain, weights_only=False)
    config.model.load_state_dict(state['state_dict'])

    # run demo
    main(config, demo_loader)
