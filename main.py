import os
import re
import time
import sys
from GRiP3_Pipeline.utils.class_m2t2 import M2T2Inference
from GRiP3_Pipeline.utils.class_robot_library import UR3Commands
from GRiP3_Pipeline.utils.class_get_snap_mm import RealSenseCapture
from GRiP3_Pipeline.utils.class_predator import PredatorPipeline
import torch  # Per svuotare la cache GPU (se usi PyTorch)
from GRiP3_Pipeline.utils.class_paligemma import PaliGemmaInference

def process_user_prompt(user_prompt: str):
    # 1) Extract the command verb and object for segmentation.
    object_match = re.search(
        r"(?P<verb>move|place|put|position|shift|locate)\s+(?:the\s+)?(?P<object>.+?)(?=\s+(?:to|in|on|into|at|positive|negative)\b|$)",
        user_prompt,
        re.IGNORECASE
    )
    if object_match:
        verb = object_match.group("verb").lower()
        object_phrase = object_match.group("object").strip()
    else:
        raise ValueError("Could not extract a valid object for segmentation.")

    object_seg_prompt = "<image> segment " + object_phrase.lower()

    # 2) Check for shift commands.
    direction_match = re.search(
        r"\b(positive|negative)\s+([xyz])(?:\s+by\s+([-+]?[0-9]*\.?[0-9]+))?\b",
        user_prompt,
        re.IGNORECASE
    )
    if direction_match:
        label = direction_match.group(1).lower()
        direction = direction_match.group(2).lower()
        offset = direction_match.group(3)
        if offset:
            command_prompt = f"<image> shift {label} {direction} by {offset}"
        else:
            command_prompt = f"<image> shift {label} {direction}"
        return object_seg_prompt, None, command_prompt

    # 3) Look for a target location pattern.
    target_match = re.search(
        r"\b(?:to|in|on|into|at|inside)\s+(?:the\s+)?(?P<color>\w+)\s+(?P<target>\w+)",
        user_prompt,
        re.IGNORECASE
    )
    if target_match:
        color = target_match.group("color").lower()
        target_type = target_match.group("target").lower()
        target_seg_prompt = f"<image> detect {color} {target_type}"
        command_prompt = f"<image> move in {color} target"
        return object_seg_prompt, target_seg_prompt, command_prompt

    # 4) Fallback.
    command_remainder = user_prompt[object_match.end():].strip()
    if not command_remainder:
        raise ValueError("The command part after the object could not be extracted.")
    if not command_remainder.lower().startswith(verb):
        command_prompt = f"<image> {verb} " + command_remainder
    else:
        command_prompt = "<image> " + command_remainder

    return object_seg_prompt, None, command_prompt

if __name__ == "__main__":
    # 1. Capture the scene using RealSense.
    capture_dir = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/sample_data/real_world/XY"
    rgb_filename = "rgb.png"
    depth_filename = "depth.npy"

    capture = RealSenseCapture(
        save_directory=capture_dir,
        rgb_filename=rgb_filename,
        depth_filename=depth_filename,
        resolution_width=640,
        resolution_height=480,
        fps=30
    )
    print("Starting scene capture...")
    capture.run_capture()

    # 2. Process user prompt.
    user_prompt = input("Enter your prompt (es. 'move the black cube to the location defined as ...'): ")
    seg_prompt, seg_target_prompt, cmd_prompt = process_user_prompt(user_prompt)
    print("Segmentation prompt:", seg_prompt)
    print("Segmentation Target prompt:", seg_target_prompt)
    print("Command prompt:", cmd_prompt)

    # --- Decide which TAC object to use for registration (tac1/tac2/tac3) ---
    m = re.search(r"\b(tac1|tac2|tac3)\b", user_prompt, re.IGNORECASE)
    if m:
        cad_name = m.group(1).lower()  # "tac1" / "tac2" / "tac3"
        print("Detected object for registration:", cad_name)
    else:
        print("\n[ERROR] No 'tac1', 'tac2' or 'tac3' found in the prompt.")
        print("Please specify which tac to move, e.g.: 'move tac2 in red target'.")
        sys.exit(1)

    image_path = os.path.join(capture_dir, rgb_filename)
    image_path_action_inf = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/utils/dataset/train_basic_cmd/1.jpg"
    seg_output_path = os.path.join(capture_dir, "segmentation_overlay.png")
    seg_output_path_target = os.path.join(capture_dir, "detection_target_overlay.png")
    classes = ["cylinder", "cube", "prism", "triangle"]

    # 3. Run segmentation inference.
    seg_model_path = "GRiP3_Pipeline/checkpoints/paligemma-segm-module"
    inferencer = PaliGemmaInference(
        peft_model_path=seg_model_path,
        base_model_id="google/paligemma-3b-mix-448"
    )

    print("Running inference for segmentation...")
    seg_detections = inferencer.infer(
        prompt=seg_prompt,
        image_path=image_path,
        output_path=seg_output_path,
        classes=classes,
        mode="segmentation"
    )

    if seg_target_prompt:
        seg_detections_target = inferencer.infer(
            prompt=seg_target_prompt,
            image_path=image_path,
            output_path=seg_output_path_target,
            classes=classes,
            mode="detect"
        )

    del inferencer
    torch.cuda.empty_cache()
    print("Segmentation model unloaded. GPU cache cleared.")

    # 4. Registration via PredatorPipeline
    input("Press Enter to proceed with registration...")

    cfg_base = "/home/au-robotics/MircoProjects/VL_GRiP3/OverlapPredator/configs/test"

    cad_cfg_map = {
        "tac1": os.path.join(cfg_base, "vl_grip3_tac1.yaml"),
        "tac2": os.path.join(cfg_base, "vl_grip3_tac2.yaml"),
        "tac3": os.path.join(cfg_base, "vl_grip3_tac3.yaml"),
    }

    predator_cfg_path = cad_cfg_map.get(cad_name)
    if predator_cfg_path is None:
        print(f"[ERROR] No Predator config found for cad_name='{cad_name}'.")
        sys.exit(1)

    print("Using Predator config:", predator_cfg_path)

    while True:
        pipeline = PredatorPipeline(
            sample_dir=capture_dir,
            predator_script="/home/au-robotics/MircoProjects/VL_GRiP3/OverlapPredator/scripts/demo_save.py",
            predator_cfg=predator_cfg_path,
            label=101,
        )
        pipeline.run()
        print(f"Registration completed. NPZ available in: {pipeline.npz_path}")
        repeat = input("Type 'R' to repeat or press Enter to continue: ")
        if repeat.lower() != 'r':
            break

    # 5. Grasping.
    input("Press Enter to proceed with M2T2 inference:")
    while True:
        m2t2_inference = M2T2Inference()
        m2t2_inference.run()
        repeat = input("If the visualization is satisfactory, press Enter; otherwise type 'R' to repeat the inference: ")
        if repeat.lower() != 'r':
            break

    # 6. Run command inference.
    cmd_model_path = "GRiP3_Pipeline/checkpoints/paligemma-action-module"
    inferencer_cmd = PaliGemmaInference(
        peft_model_path=cmd_model_path,
        base_model_id="google/paligemma-3b-mix-448"
    )

    print("Running inference for command...")
    cmd_detections = inferencer_cmd.infer(
        prompt=cmd_prompt,
        image_path=image_path_action_inf,
        classes=classes,
        mode="command"
    )
    # Optionally: print("Comando decodificato:", cmd_detections)

    input("Press Enter to proceed with grasping...")

    # 6. Execute the command on the robot.
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    ur3 = UR3Commands()

    # Example command string:
    # "connect ; approach ; gripper open ; grasp ; gripper close ; approach ; target(red) ; gripper open ; home pose ; disconnect"
    command_line = cmd_detections.split('\n')[-1].strip()
    print("Sequence of commands to execute:", command_line)

    command_mapping = {
        "gripper open": ur3.gripper_open,
        "gripper close": ur3.gripper_close,
        "grasp": ur3.move_to_grasping,
        "approach": ur3.approach,
        "home pose": ur3.move_to_home,
    }

    commands = [cmd.strip() for cmd in command_line.split(';') if cmd.strip()]

    for cmd in commands:
        normalized_cmd = cmd.lower().strip()

        if normalized_cmd == "connect":
            print("Running the command: connect")
            ur3.connect()

        elif normalized_cmd == "disconnect":
            print("Executing the command: disconnect")
            ur3.disconnect()

        elif normalized_cmd.startswith("positive(") or normalized_cmd.startswith("negative("):
            pattern_shift = r"^(positive|negative)\(\s*([xyz])\s*,\s*([\d\.]+)\s*\)$"
            match_shift = re.match(pattern_shift, normalized_cmd, re.IGNORECASE)
            if match_shift:
                shift_type, axis, offset_str = match_shift.groups()
                try:
                    offset = float(offset_str)
                except ValueError:
                    print(f"Invalid offset value in: {cmd}")
                    continue
                print(f"Executing the command: {cmd}")
                if shift_type.lower() == "positive":
                    ur3.positive_shift(axis, offset)
                else:
                    ur3.negative_shift(axis, offset)
            else:
                print(f"Invalid shift command format: {cmd}")

        elif normalized_cmd.startswith("target("):
            # Ignore any parameter. Just call move_to_target()
            print("Executing the target command (ignoring parameter)")
            ur3.move_to_target()

        elif normalized_cmd == "target":
            print("Executing the target command")
            ur3.move_to_target()

        elif normalized_cmd in command_mapping:
            print(f"Executing the command: {cmd}")
            command_mapping[normalized_cmd]()

        else:
            print(f"Command '{cmd}' not recognized.")

        time.sleep(0.1)

#WITH JUST KEYBOARD