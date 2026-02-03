# VL-GRiP3: A Hierarchical Pipeline Leveraging Vision-Language Models for Autonomous Robotic 3D Grasping

This repository represents the official implementation of the paper:
Paper: [Title of my paper](https://YOUR-LINK-HERE)


VL_GRiP3 is an end-to-end pipeline for **vision–language-driven grasping** on a UR3 robot with a Robotiq gripper.

![VL-GRiP3](assets/banner.png)

This framework provides a transparent, modular pipeline that decomposes language understanding, perception, and action planning for robotic manipulation. A single vision–language model (VLM) backbone interprets natural-language commands, localizes the target, and produces high-level action intent. To handle occlusions from a single RGB-D view, CAD-augmented point cloud registration reconstructs a more complete 3D representation at low hardware cost. An M2T2-based grasp planner then predicts geometry-aware 3D grasp poses from the augmented point cloud, enabling reliable manipulation of irregular industrial parts in SME manufacturing settings.

High-level flow:

1. Capture RGB-D with **RealSense**
2. Segment the commanded object + detect the target area with **PaliGemma**
3. Register the CAD model to the scene with **OverlapPredator**
4. Generate grasp candidates with **M2T2**
5. Decode a high-level action script from language + image (PaliGemma action head)
6. Execute the sequence on the UR3 via **RTDE**

## Citation
If you find this code useful for your work or use it in your project, please consider citing:

```bash
tba
```

## Installation
This code has been tested on

- Python 3.10.19, PyTorch 2.3.1+cu121, CUDA 12.1, NVIDIA RTX 4090 (24GB VRAM)


## Requirements

To clone the repo and install dependencies, run:

```bash
git clone https://github.com/AU-DK-Robotics/VL-GRiP3.git
cd VL-GRiP3
python -m pip install -r requirements.txt
```

## External Dependencies (Required)

This repository includes (as submodules/third-party code) parts of the official implementations below.  
After cloning this repo, you **must** install/build them by following the instructions in their respective pages:

- [OverlapPredator — requirements & setup](https://github.com/prs-eth/OverlapPredator/blob/main/requirements.txt)
- [M2T2 — installation instructions](https://github.com/NVlabs/M2T2?tab=readme-ov-file)


`
Download [model weights](https://huggingface.co/polonara/paligemma-action-module/tree/main) using:
```bash
cd GRiP3_Pipeline && git clone https://huggingface.co/polonara/checkpoints
```

## Run
After creating the virtual environment, created your own dataset, and trained Predator, VL-GRiP3 can be run using:

```bash

python main.py
```
If you want to use OpenAI Whisper module, please run:
```bash
python main_whisper.py
```

Example command and repository directory tree:

```text
move tac1 in red target
move tac2 in green target
move tac3 in yellow target


GRiP3_Pipeline/
├─ checkpoints/
│  ├─ paligemma-segm-module/       # Segmentation head weights (PEFT)
│  └─ paligemma-action-module/     # Action command head weights (PEFT)
│  
│
├─ sample_data/
│  └─ real_world/
│     └─ MX/
│        ├─ rgb.png                # Captured RGB (RealSense)
│        ├─ depth.npy              # Captured depth (RealSense)
│        ├─ meta_data.pkl          # Camera intrinsics, extrinsics, etc. for M2T2
│        ├─ seg.png                # Segmentation mask (PaliGemma)
│        ├─ segmentation_overlay.png
│        ├─ detection_target_overlay.png
│        ├─ target_pose.json       # Target MX in robot frame
│        ├─ scene.pth              # Scene point cloud for Predator
│        ├─ full_object_pointcloud.npz
│        │                         # Registered object+scene cloud for M2T2
│        ├─ best_poses.json        # Best grasps/placements (M2T2)
│        └─ sorted_tcp_poses.json  # Sorted TCP poses for UR3Commands
│
└─ utils/
   ├─ class_get_snap_mm.py         # RealSenseCapture (RGB + depth grabber)
   ├─ class_paligemma.py           # PaliGemmaInference (seg/detect/command)
   ├─ class_predator.py            # PredatorPipeline (scene.pth + OverlapPredator + NPZ)
   ├─ class_m2t2.py                # M2T2Inference (loads config.yaml + runs grasping)
   ├─ class_robot_library.py       # UR3Commands + RobotiqGripper (RTDE control)
   ├─ class_whisper                # Whisper class
   ├─ ft_action_module.py          # Script to fine-tune the action (command) PaliGemma head
   └─ ft_segm_module.py            # Script to fine-tune the segmentation PaliGemma head

```

## Acknowledgments

In this project we use (parts of) the official implementations of the following works:

- [OverlapPredator](https://github.com/prs-eth/OverlapPredator)
- [M2T2](https://github.com/NVlabs/M2T2)

We thank the respective authors for open-sourcing their methods. We would also like to thank reviewers for their valuable inputs.



