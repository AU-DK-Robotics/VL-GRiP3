import json
import time
import sys
import re
import rtde_receive
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

class UR3Commands:
    """
    UR3 Robotics Command Library via RTDE.

    Methods:
        - connect(): Establish connection to the robot using a fixed IP.
        - disconnect(): Close RTDE connections.
        - move_to_pose(): Move the robot to a specified pose.
        - move_to_grasping(): Read the pose from a JSON file and move the robot to that pose.
        - move_to_target(): Move the robot to the specified target.
            (Ignores the color parameter; uses the x,y position saved in the JSON file.)
        - move_to_home(): Move the robot to the home configuration.
        - gripper_open() / gripper_close(): Controls the opening/closing of the gripper.
        - approach(): Performs a deapproach move (raising the z by 0.05 from the grasping pose).
        - get_current_tcp_pose(), positive_shift(), negative_shift(): Operations on the current pose.
    """

    def __init__(self, rtde_control_interface=None, rtde_receive_interface=None,
                 move_delay=4.0,      # Waiting time after a movement
                 gripper_delay=1.0):  # Waiting time after gripper operations
        """
        :param rtde_control_interface: Instance of RTDEControlInterface (optional).
        :param rtde_receive_interface: Instance of RTDEReceiveInterface (optional).
        :param move_delay: Time to wait after each move command.
        :param gripper_delay: Time to wait after each gripper command.
        """
        self.rtde_c = rtde_control_interface
        self.rtde_r = rtde_receive_interface
        self.move_delay = move_delay
        self.gripper_delay = gripper_delay

        # Velocity and acceleration parameters for Cartesian motions
        self.speed = 0.08
        self.acceleration = 0.1

        # Fixed poses for targets
        self._yellow_target_pose = [0.3506093066335152, 0.026065163598190763, 0.08630217204639501,
                                     1.85011627085003, -2.538905414868407, -3.633083091122777e-05]
        self._green_target_pose = [0.3092810912107801, -0.16723593843130086, 0.08633799189949548,
                                    1.0619274431058556, -2.956625921081819, 0.00014242509976082317]
        self._red_target_pose   = [-0.09621386458169016, -0.33818094057402187, 0.08633208721832436,
                                    -1.0218855726535594, -2.9706414975151993, 1.7414612154461717e-05]
        self._home_pose         = [0.19463757949913976, -0.23026603411194024, 0.27530714604309037,
                                    2.267162875376877, -2.174720802168727, 0.0001074923254140672]

        # Default key for grasping pose
        self.default_grasp_pose_key = "1"

        # Variable for gripper (initialized after connection)
        self.gripper = None

    def _init_gripper(self):
        try:
            self.gripper = RobotiqGripper(self.rtde_c)
            print("[UR3Commands] Activating the gripper...")
            self.gripper.activate()
            self.gripper.set_force(0)
            self.gripper.set_speed(30)
        except Exception as e:
            print(f"[UR3Commands] Error initializing gripper: {e}")
            self.gripper = None

    def connect(self):
        """
        Establishes connection with the robot using a static IP.
        If the connection fails, the whole program is terminated.
        """
        robot_ip = "192.168.1.254"  # IP address
        try:
            self.rtde_c = RTDEControlInterface(robot_ip)
            self.rtde_r = RTDEReceiveInterface(robot_ip)

            # Extra safety: check both interfaces
            if self.rtde_c is None or self.rtde_r is None:
                print(f"[connect] RTDE interfaces not initialized correctly for {robot_ip}.")
                print("[connect] Aborting pipeline because robot is not reachable.")
                sys.exit(1)

            print(f"[connect] Connection established with the robot {robot_ip}.")
            self._init_gripper()
        except Exception as e:
            print(f"[connect] Error connecting to robot {robot_ip}: {e}")
            print("[connect] Aborting pipeline because robot is not reachable.")
            sys.exit(1)

    def disconnect(self):
        """
        Closes RTDE connections.
        """
        if self.rtde_c is not None:
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
            print("[disconnect] RTDEControlInterface disconnected.")
        if self.rtde_r is not None:
            self.rtde_r.disconnect()
            print("[disconnect] RTDEReceiveInterface disconnected.")

    def _read_pose_from_json(self, file_path, pose_key):
        """
        Reads the pose associated with the given key from the JSON file.
        :param file_path: Path to the JSON file.
        :param pose_key: Key of the pose to extract.
        :return: Pose as a list of 6 values or None.
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            pose = data.get(str(pose_key), None)
            if pose is None:
                print(f"Error: The pose with key '{pose_key}' is not defined in the JSON file.")
            return pose
        except Exception as e:
            print("Error reading JSON file:", e)
            return None

    def move_to_pose(self, pose, speed=None, acceleration=None):
        """
        Moves the robot to the specified pose.
        :param pose: List of 6 values [x, y, z, rx, ry, rz].
        :param speed: Speed (default=self.speed).
        :param acceleration: Acceleration (default=self.acceleration).
        """
        actual_speed = speed if speed is not None else self.speed
        actual_acceleration = acceleration if acceleration is not None else self.acceleration
        print(f"[move_to_pose] Move to {pose} (speed={actual_speed}, accel={actual_acceleration})")
        if self.rtde_c is not None:
            self.rtde_c.moveL(pose, actual_speed, actual_acceleration)
        else:
            print("[move_to_pose] Error: Invalid RTDE connection!")
        time.sleep(self.move_delay)

    def move_to_grasping(self):
        """
        Reads the grasping pose from a JSON file and moves the robot towards it.
        """
        pose_file = "/home/au-robotics/MircoProjects/Finale/pali/sample_data/real_world/XY/sorted_tcp_poses.json"
        grasp_pose = self._read_pose_from_json(pose_file, self.default_grasp_pose_key)
        if grasp_pose is None:
            print("Unable to perform move_to_grasping: Pose not available.")
            return
        self.move_to_pose(grasp_pose)

    def move_to_target(self, _ignored_color=None):
        """
        Moves the robot to the target based on the target JSON file.
        Reads the JSON file (/home/au-robotics/MircoProjects/VLAM/GRiP3_Pipeline/sample_data/real_world/99/target_pose.json)
        extracting the x and y values. These values are then integrated with the current
        pose obtained with get_current_tcp_pose() to keep z, rx, ry, rz unchanged.
        """
        target_json_path = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/sample_data/real_world/XY/target_pose.json"
        try:
            with open(target_json_path, "r") as f:
                data = json.load(f)
            # I take the first bearing and only x,y
            target_xy = data[0][:2]
        except Exception as e:
            print(f"[move_to_target] Error reading target JSON file: {e}")
            return

        current_pose = self.get_current_tcp_pose()
        if current_pose is None:
            print("[move_to_target] Unable to get current TCP pose.")
            return

        # 1) build and execute the usual move (swap in X,Y)
        new_pose = current_pose.copy()
        new_pose[0] = target_xy[0]
        new_pose[1] = target_xy[1]
        print(f"[move_to_target] Move to the target(x,y): {new_pose}")
        self.move_to_pose(new_pose)

        # 2) now lower Z by 1 cm and execute that move
        lowered_pose = new_pose.copy()
        lowered_pose[2] -= 0.015  # 1 cm down
        print(f"[move_to_target] Lowering of 1 cm along Z: {lowered_pose}")
        self.move_to_pose(lowered_pose)

    def move_to_home(self):
        """
        Moves the robot to the home position.
        """
        print("[move_to_home] Moving to home configuration.")
        self.move_to_pose(self._home_pose)

    def gripper_open(self):
        """
        Open the gripper.
        """
        if self.gripper is not None:
            print("[gripper_open] Opening the gripper.")
            self.gripper.open()
        else:
            print("[gripper_open] Error: gripper not initialized.")
        time.sleep(self.gripper_delay)

    def gripper_close(self):
        """
        Close the gripper.
        """
        if self.gripper is not None:
            print("[gripper_close] Closing the gripper.")
            self.gripper.close()
        else:
            print("[gripper_close] Error: gripper not initialized.")
        time.sleep(self.gripper_delay)

    def approach(self):
        """
        Performs a deapproach motion: takes the grasping pose from the JSON and adds +0.05 to the z-coordinate.
        """
        pose_file = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/sample_data/real_world/XY/sorted_tcp_poses.json"
        grasp_pose = self._read_pose_from_json(pose_file, self.default_grasp_pose_key)
        if grasp_pose is None:
            print("Unable to perform approach: Grasping pose not available.")
            return
        deapproach_pose = grasp_pose.copy()
        deapproach_pose[2] += 0.05
        print(f"[approach] Deapproach movement towards the pose {deapproach_pose}")
        self.move_to_pose(deapproach_pose)

    def get_current_tcp_pose(self):
        """
        Returns the current TCP pose of the robot.
        """
        if self.rtde_r is not None:
            current_pose = self.rtde_r.getActualTCPPose()
            print(f"[get_current_tcp_pose] The current pose is: {current_pose}")
            return current_pose
        else:
            print("[get_current_tcp_pose] Error: Invalid RTDE connection!")
            return None

    def positive_shift(self, axis, offset):
        """
        Performs a positive shift on the current pose along the specified axis.
        :param axis: 'x', 'y' or 'z'
        :param offset: Offset (in meters, positive value)
        """
        current_pose = self.get_current_tcp_pose()
        if current_pose is None:
            print("Unable to get current pose for positive_shift.")
            return
        new_pose = current_pose.copy()
        if axis.lower() == "x":
            new_pose[0] += offset
        elif axis.lower() == "y":
            new_pose[1] += offset
        elif axis.lower() == "z":
            new_pose[2] += offset
        else:
            print("Axis not recognized for positive_shift.")
            return
        print(f"[positive_shift] New pose after long positive shift {axis}: {new_pose}")
        self.move_to_pose(new_pose)

    def negative_shift(self, axis, offset):
        """
        Performs a negative shift on the current pose along the specified axis.
        :param axis: 'x', 'y' or 'z'
        :param offset: Offset (in meters, positive value)
        """
        current_pose = self.get_current_tcp_pose()
        if current_pose is None:
            print("Unable to get current pose for negative_shift.")
            return
        new_pose = current_pose.copy()
        if axis.lower() == "x":
            new_pose[0] -= offset
        elif axis.lower() == "y":
            new_pose[1] -= offset
        elif axis.lower() == "z":
            new_pose[2] -= offset
        else:
            print("Axis not recognized for negative_shift.")
            return
        print(f"[negative_shift] New pose after long negative shift {axis}: {new_pose}")
        self.move_to_pose(new_pose)


#ADDED DE-APPOACH BEFORE OPENING GRIPPER