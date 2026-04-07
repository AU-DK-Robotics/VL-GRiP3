import hydra
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .m2t2.dataset import load_rgb_xyz, collate
from .m2t2.dataset_utils import denormalize_rgb, sample_points
from .m2t2.meshcat_utils import (
    create_visualizer, make_frame, visualize_grasp, visualize_pointcloud
)
from .m2t2.m2t2 import M2T2
from .m2t2.plot_utils import get_set_colors
from .m2t2.train_utils import to_cpu, to_gpu
from pathlib import Path
from omegaconf import OmegaConf

class M2T2Inference:
    def __init__(self, cfg=None):
        here = Path(__file__).resolve().parent
        # If no configuration is passed, load the base config.yaml using OmegaConf.
        if cfg is None:
            cfg_path = here / "config.yaml"
            cfg = OmegaConf.load(str(cfg_path))
        self.cfg = cfg

        # Hard-utils the 4 parameters (overriding any CLI overrides)
        self.cfg.eval.checkpoint = str(here / "m2t2.pth")
        self.cfg.eval.data_dir   = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/sample_data/real_world/MX"
        self.cfg.eval.mask_thresh = 0.55
        self.cfg.eval.num_runs    = 5

    def parse_transformation(self, matrix):
        rotation = [row[:3] for row in matrix[:3]]
        translation = [matrix[i][3] for i in range(3)]
        return {"rotation_matrix": rotation, "translation_vector": translation}

    def generate_sorted_tcp_poses(self, input_json, output_json):
        with open(input_json, "r") as f:
            data = json.load(f)
        grasps = data.get("grasps", [])
        pose_list = []
        for pose in grasps:
            translation = np.array(pose["translation_vector"])
            rotation_matrix = np.array(pose["rotation_matrix"])
            rotvec = R.from_matrix(rotation_matrix).as_rotvec()
            tcp_pose = np.concatenate([translation, rotvec])
            confidence = pose.get("confidence", 0)
            pose_list.append({"confidence": confidence, "tcp_pose": tcp_pose.tolist()})
        pose_list.sort(key=lambda x: x["confidence"], reverse=True)
        output_dict = {}
        for i, entry in enumerate(pose_list, start=1):
            output_dict[str(i)] = entry["tcp_pose"]
        with open(output_json, "w") as f:
            json.dump(output_dict, f, indent=2)

    def load_and_predict(self, data_dir):
        data, meta_data = load_rgb_xyz(
            data_dir,
            self.cfg.data.robot_prob,
            self.cfg.data.world_coord,
            self.cfg.data.jitter_scale,
            self.cfg.data.grid_resolution,
            self.cfg.eval.surface_range
        )
        obj_label = 101
        scene_points = data['points'].numpy()
        scene_inputs = data['inputs'].numpy()
        scene_seg = data['seg'].numpy()

        # Load the non-occluded object point cloud.
        obj_data = np.load(
            "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/sample_data/real_world/MX/full_object_pointcloud.npz"
        )
        obj_inputs = obj_data['full_inputs']
        obj_seg = obj_data['full_seg']

        data['object_inputs'] = torch.from_numpy(obj_inputs).float()

        obj_mask_original = (scene_seg == obj_label)
        scene_points_filtered = scene_points[~obj_mask_original]
        scene_inputs_filtered = scene_inputs[~obj_mask_original]
        scene_seg_filtered = scene_seg[~obj_mask_original]

        if self.cfg.eval.world_coord and 'camera_pose' in meta_data:
            cam_pose = meta_data['camera_pose'].astype(np.float32)
            obj_xyz = obj_inputs[:, :3]
            ones = np.ones((obj_xyz.shape[0], 1), dtype=np.float32)
            obj_xyz_hom = np.hstack([obj_xyz, ones])
            obj_xyz_world = (cam_pose @ obj_xyz_hom.T).T[:, :3]
        else:
            obj_xyz_world = obj_inputs[:, :3]

        full_xyz = np.concatenate([scene_points_filtered, obj_xyz_world], axis=0)
        full_rgb = np.concatenate([scene_inputs_filtered[:, 3:6], obj_inputs[:, 3:6]], axis=0)
        full_labels = np.concatenate([scene_seg_filtered, obj_seg], axis=0)
        full_inputs = np.concatenate([full_xyz, full_rgb], axis=1)

        inputs_torch = torch.from_numpy(full_inputs).float()
        points_torch = inputs_torch[:, :3]
        seg_torch = torch.from_numpy(full_labels).long()

        data['inputs'] = inputs_torch
        data['points'] = points_torch
        data['seg'] = seg_torch

        if 'object_label' in meta_data:
            data['task'] = 'place'
        else:
            data['task'] = 'pick'

        model = M2T2.from_config(self.cfg.m2t2)
        ckpt = torch.load(self.cfg.eval.checkpoint)
        model.load_state_dict(ckpt['model'])
        model = model.cuda().eval()

        inputs, xyz, seg = data['inputs'], data['points'], data['seg']
        obj_inputs = data['object_inputs']

        outputs = {
            'grasps': [],
            'grasp_confidence': [],
            'grasp_contacts': [],
            'placements': [],
            'placement_confidence': [],
            'placement_contacts': []
        }

        for _ in range(self.cfg.eval.num_runs):
            pt_idx = sample_points(xyz, self.cfg.data.num_points)
            data['inputs'] = inputs[pt_idx]
            data['points'] = xyz[pt_idx]
            data['seg'] = seg[pt_idx]

            pt_idx = sample_points(obj_inputs, self.cfg.data.num_object_points)
            data['object_inputs'] = obj_inputs[pt_idx]

            data_batch = collate([data])
            to_gpu(data_batch)
            with torch.no_grad():
                model_outputs = model.infer(data_batch, self.cfg.eval)
            to_cpu(model_outputs)
            for key in outputs:
                if 'place' in key and len(outputs[key]) > 0:
                    outputs[key] = [
                        torch.cat([prev, cur])
                        for prev, cur in zip(outputs[key], model_outputs[key][0])
                    ]
                else:
                    outputs[key].extend(model_outputs[key][0])
        data['inputs'], data['points'], data['seg'] = inputs, xyz, seg
        return data, outputs

    def run(self):
        data, outputs = self.load_and_predict(self.cfg.eval.data_dir)

        vis = create_visualizer()
        rgb = denormalize_rgb(data['inputs'][:, 3:].T.unsqueeze(2)).squeeze(2).T
        rgb = (rgb.numpy() * 255).astype('uint8')
        xyz = data['points'].numpy()

        # --- ORIGINAL LOGIC ---
        # Always show the camera frame. Then, if the data are not already in world coordinates,
        # transform the points from camera to world coordinates.
        cam_pose = data.get('cam_pose', None)
        if cam_pose is not None:
            cam_pose = cam_pose.double().numpy()
            make_frame(vis, 'camera', T=cam_pose)
            if not self.cfg.eval.world_coord:
                xyz = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]
        # ------------------------

        visualize_pointcloud(vis, 'scene', xyz, rgb, size=0.005)
        best_poses = {}

        if data['task'] == 'pick':
            best_poses["task"] = "pick"
            best_grasps_list = []
            best_conf_list = []
            best_color_list = []
            colors = get_set_colors()
            num_runs = len(outputs['grasps'])
            for i in range(num_runs):
                run_grasps = outputs['grasps'][i]
                run_conf = outputs['grasp_confidence'][i]
                if run_grasps.shape[0] == 0:
                    continue
                run_conf_np = run_conf.numpy()
                best_idx = np.argmax(run_conf_np)
                best_grasp = run_grasps[best_idx].cpu().numpy()
                best_conf = run_conf_np[best_idx]
                run_color = colors[i % len(colors)]
                best_grasps_list.append(best_grasp)
                best_conf_list.append(best_conf)
                best_color_list.append(run_color)
            overall_conf = np.array(best_conf_list)
            sorted_overall_idx = np.argsort(-overall_conf)
            top3_overall_idx = sorted_overall_idx[:3]
            final_best_grasps = [best_grasps_list[i] for i in top3_overall_idx]
            final_best_conf = [best_conf_list[i] for i in top3_overall_idx]
            final_best_colors = [best_color_list[i] for i in top3_overall_idx]
            if cam_pose is not None and not self.cfg.eval.world_coord:
                final_best_grasps = np.array([cam_pose @ pose for pose in final_best_grasps])
            for j, grasp in enumerate(final_best_grasps):
                visualize_grasp(
                    vis,
                    f"object_00/grasps/{j:03d}",
                    grasp,
                    final_best_colors[j],
                    linewidth=4.0
                )
            best_poses["grasps"] = [
                {
                    **self.parse_transformation(grasp.tolist()),
                    "confidence": float(conf),
                    "color": final_best_colors[j]
                }
                for j, (grasp, conf) in enumerate(zip(final_best_grasps, final_best_conf))
            ]
        else:
            best_poses["task"] = "place"
            ee_pose = data.get('ee_pose', None)
            if ee_pose is not None:
                ee_pose = ee_pose.double().numpy()
                make_frame(vis, 'ee', T=ee_pose)
            obj_xyz_ee, obj_rgb = data['object_inputs'].split([3, 3], dim=1)
            obj_xyz_ee = (obj_xyz_ee + data['object_center']).numpy()
            obj_rgb = denormalize_rgb(obj_rgb.T.unsqueeze(2)).squeeze(2).T
            obj_rgb = (obj_rgb.numpy() * 255).astype('uint8')
            visualize_pointcloud(vis, 'object', obj_xyz_ee, obj_rgb, size=0.005)
            merged_placements = np.concatenate([p for p in outputs['placements']], axis=0)
            merged_conf = np.concatenate([c.numpy() for c in outputs['placement_confidence']], axis=0)
            sorted_idx = np.argsort(-merged_conf)
            best_idx = sorted_idx[:3]
            best_placements = merged_placements[best_idx]
            best_placement_conf = merged_conf[best_idx]
            if cam_pose is not None and not self.cfg.eval.world_coord:
                best_placements = np.array([cam_pose @ pose for pose in best_placements])
            for j, placement in enumerate(best_placements):
                visualize_grasp(
                    vis,
                    f"orientation_00/placements/{j:02d}/gripper",
                    placement,
                    [0, 255, 0],
                    linewidth=0.2
                )
                obj_xyz_placed = obj_xyz_ee @ placement[:3, :3].T + placement[:3, 3]
                visualize_pointcloud(
                    vis,
                    f"orientation_00/placements/{j:02d}/object",
                    obj_xyz_placed,
                    obj_rgb,
                    size=0.01
                )
            best_poses["placements"] = [
                {
                    **self.parse_transformation(placement.tolist()),
                    "confidence": float(conf)
                }
                for placement, conf in zip(best_placements, best_placement_conf)
            ]

        output_filename_alt = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/sample_data/real_world/MX/best_poses.json"
        with open(output_filename_alt, "w") as f:
            json.dump(best_poses, f, indent=2)
        print(f"Salvato '{output_filename_alt}'.")
        sorted_output_file = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/sample_data/real_world/MX/sorted_tcp_poses.json"
        self.generate_sorted_tcp_poses(output_filename_alt, sorted_output_file)
        print(f"Generato '{sorted_output_file}'.")
        if best_poses["task"] == "pick":
            for i, grasp in enumerate(best_poses["grasps"]):
                print(f"Pose {i + 1}: confidence = {grasp['confidence']}")
        else:
            for i, placement in enumerate(best_poses["placements"]):
                print(f"Placement {i + 1}: confidence = {placement['confidence']}")
        print("Inferenza M2T2 completata.")


@hydra.main(config_path='.', config_name='config', version_base='1.3')
def main(cfg):
    """
    Hydra entry point: load config.yaml and instantiate M2T2Inference.
    The 4 parameters (checkpoint, data_dir, mask_thresh, num_runs) are forced
    within the class.
    """
    inference = M2T2Inference(cfg)
    inference.run()


if __name__ == '__main__':
    main()

