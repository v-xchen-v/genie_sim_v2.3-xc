import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from PIL import Image
import os,time, pickle
from typing import Dict, Any
from pathlib import Path

from kinematics.urdf_coordinate_transformer import URDFCoordinateTransformer
from kinematics.g1_relax_ik import G1RelaxSolver
from scipy.spatial.transform import Rotation as R


class VLAInputProcessor:
    # def __init__(self, image_list, task_instruction, robot_state, task_name: str, 
    #              log_obs=True, additional_info=None):
    #     self.image_list = image_list
    #     self.task_instruction = task_instruction
    #     self.robot_state = robot_state
    #     self.additional_info = additional_info if additional_info is not None else {}
    #     self.task_name = task_name
    #     self.log_obs = log_obs

    #     # extra
    #     self._log_dir_registry = {}
    def __init__(self, log_obs=False):        
        
        self.log_obs = log_obs
        if self.log_obs:
            # Initialize log directory registry if logging is enabled
            self.task_name = "iros_pack_in_the_supermarket"  # Placeholder, can be set later
            self._log_dir_registry = {}
            
        self.fk_urdf_path = Path(__file__).parent / "kinematics/configs/g1/G1_omnipicker.urdf"
        if not self.fk_urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {self.fk_urdf_path}")
        self.fk_urdf_path = str(self.fk_urdf_path.resolve())
        self.coord_transformer = URDFCoordinateTransformer(self.fk_urdf_path)
        
        self.ik_urdf_path = Path(__file__).parent / "kinematics/configs/g1/G1_NO_GRIPPER.urdf"
        if not self.ik_urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {self.ik_urdf_path}")
        self.ik_urdf_path = str(self.ik_urdf_path.resolve())
        self.ik_config_path = Path(__file__).parent / "kinematics/configs/g1/g1_solver.yaml"
        if not self.ik_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.ik_config_path}")
        self.ik_config_path = str(self.ik_config_path.resolve())
        
        # Initialize the IK solvers for both arms
        self.left_arm_ik_solver = G1RelaxSolver(
            urdf_path=self.ik_urdf_path,
            config_path=self.ik_config_path,
            arm="left"
        )
        self.right_arm_ik_solver = G1RelaxSolver(
            urdf_path=self.ik_urdf_path,
            config_path=self.ik_config_path,
            arm="right"
        )
        self.last_left_arm_joint_angles = None
        self.last_right_arm_joint_angles = None

    def process(self, img_h, img_l, img_r, lang, state, task_substep_index, head_joint_cfg=None):
        """
        Process the input images, task description, and robot state.
        
        Args:
            img_h: Head image
            img_l: Left wrist image
            img_r: Right wrist image
            lang: Task description in natural language
            state: Robot state information
            
        Returns:
            A dictionary containing processed images, task description, and robot state.
        """
        self.image_list = [img_h, img_l, img_r]
        self.task_instruction = lang
        self.robot_state = state
        self.curr_task_substep_index = task_substep_index
        
        return self.prepare_input(head_joint_cfg)
        
    
    def prepare_input(self, head_joint_cfg=None):
        """
        Prepare the input for the VLA model.
        
        Returns:
            A dictionary containing the processed images, task description, and robot state.
        """
        obs_dict = self.get_observations(head_joint_cfg)
        if self.log_obs:
            # Log observations if required
            self._log_observations(obs_dict, log_dir="./obs_logs")
       
        input_data = {
            "image_list" : [
                Image.fromarray(obs_dict["images"]["cam_top"]),
                Image.fromarray(obs_dict["images"]["head_left"]),
                Image.fromarray(obs_dict["images"]["head_right"]),
            ],
            "task_description": obs_dict["task_description"],
            "robot_state": obs_dict["robot_state"],
        }
        
        return input_data
        
    def get_observations(self, head_joint_cfg):
        """
        Get observations for the policy step.
        
        Returns:
            A dictionary containing the observations.
        """
        # Preprocess images
        processed_images = self.preprocess_images()
        cam_top_img = processed_images[0]
        head_left_img = processed_images[1]
        head_right_img = processed_images[2]
        
        # Preprocess task instruction
        processed_task_instruction = self.preprocess_instruction(self.task_instruction, self.curr_task_substep_index)
        
        if self.robot_state is not None:
            # Preprocess robot state
            processed_robot_state = self.preprocess_single_robot_state(self.robot_state, head_joint_cfg)
        else:
            processed_robot_state = None
            
        # Construct the observations dictionary
        obs_dict = {
            "task_description": processed_task_instruction,
            "images": {
                "cam_top": cam_top_img,
                "head_left": head_left_img,
                "head_right": head_right_img,
            },
            "robot_state": processed_robot_state,
        }
        return obs_dict
    
    def preprocess_robot_state(self, robot_joints_list, head_joint_cfg):
        """
        Preprocess the robot state which is joint angle to ee pose.
        Ref:
        "robot_state": {
                "ROBOT_LEFT_TRANS": robot_ee_left_translation_in_head_cam.tolist(),
                "ROBOT_LEFT_ROT_EULER": robot_ee_left_rotation_euler_xyz_in_head_cam.tolist(),
                "ROBOT_LEFT_GRIPPER": np.zeros((1,), dtype=np.float32).tolist(),
                "ROBOT_RIGHT_TRANS": robot_ee_right_translation_in_head_cam.tolist(),
                "ROBOT_RIGHT_ROT_EULER": robot_ee_right_rotation_euler_xyz_in_head_cam.tolist(),
                "ROBOT_RIGHT_GRIPPER": np.zeros((1,), dtype=np.float32).tolist(),
            },
            
        Args:
            robot_joints: a numpy array of joint angles in order consistent with ROS:
            "idx21_arm_l_joint1",
            "idx22_arm_l_joint2",
            "idx23_arm_l_joint3",
            "idx24_arm_l_joint4",
            "idx25_arm_l_joint5",
            "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "idx41_gripper_l_outer_joint1",
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
            "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
            "idx81_gripper_r_outer_joint1",
        """
        # Iterate through the joint angles and compute the end-effector poses
        robot_states = []
        for robot_joints in robot_joints_list:
            single_robot_state = self.preprocess_single_robot_state(robot_joints, head_joint_cfg)
            robot_states.append(single_robot_state)

        # Combine robot states in list with key to one-dict format
        combined_robot_state = {    
            "ROBOT_LEFT_TRANS": [],
            "ROBOT_LEFT_ROT_EULER": [],
            "ROBOT_LEFT_GRIPPER": [],
            "ROBOT_RIGHT_TRANS": [],
            "ROBOT_RIGHT_ROT_EULER": [],
            "ROBOT_RIGHT_GRIPPER": [],
        }
        
        for state in robot_states:
            combined_robot_state["ROBOT_LEFT_TRANS"].append(state["ROBOT_LEFT_TRANS"])
            combined_robot_state["ROBOT_LEFT_ROT_EULER"].append(state["ROBOT_LEFT_ROT_EULER"])
            combined_robot_state["ROBOT_LEFT_GRIPPER"].append(state["ROBOT_LEFT_GRIPPER"])
            combined_robot_state["ROBOT_RIGHT_TRANS"].append(state["ROBOT_RIGHT_TRANS"])
            combined_robot_state["ROBOT_RIGHT_ROT_EULER"].append(state["ROBOT_RIGHT_ROT_EULER"])
            combined_robot_state["ROBOT_RIGHT_GRIPPER"].append(state["ROBOT_RIGHT_GRIPPER"])
        # Convert lists to numpy arrays
        for key in combined_robot_state:
            combined_robot_state[key] = np.array(combined_robot_state[key], dtype=np.float32)
        return combined_robot_state

    def preprocess_single_robot_state(self, robot_joints, head_joint_cfg):
        """
        Preprocess a single robot state, which is joint angle to ee pose.
        Ref:
        "robot_state": {
                "ROBOT_LEFT_TRANS": robot_ee_left_translation_in_head_cam.tolist(),
                "ROBOT_LEFT_ROT_EULER": robot_ee_left_rotation_euler_xyz_in_head_cam.tolist(),
                "ROBOT_LEFT_GRIPPER": np.zeros((1,), dtype=np.float32).tolist(),
                "ROBOT_RIGHT_TRANS": robot_ee_right_translation_in_head_cam.tolist(),
                "ROBOT_RIGHT_ROT_EULER": robot_ee_right_rotation_euler_xyz_in_head_cam.tolist(),
                "ROBOT_RIGHT_GRIPPER": np.zeros((1,), dtype=np.float32).tolist(),
            },
        Args:
            robot_joints: a numpy array of joint angles in order consistent with ROS:
            "idx21_arm_l_joint1",
            "idx22_arm_l_joint2",
            "idx23_arm_l_joint3",
            "idx24_arm_l_joint4",
            "idx25_arm_l_joint5",
            "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "idx41_gripper_l_outer_joint1",
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
            "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
            "idx81_gripper_r_outer_joint1",
        """
        left_arm_joints = robot_joints[:7]
        right_arm_joints = robot_joints[8:15]
        left_gripper_joint = robot_joints[7]
        right_gripper_joint = robot_joints[15]
        
        # Transform left arm joints to end-effector pose
        T_left_ee_pose_in_armbase_coord = self.left_arm_ik_solver.compute_fk(left_arm_joints)
        T_right_ee_pose_in_armbase_coord = self.right_arm_ik_solver.compute_fk(right_arm_joints)
        
        # transform end-effector pose to arm base coord to head camera coord
        T_armbase_to_headlink2 = self.coord_transformer.relative_transform("arm_base_link", "head_link2", joint_values=head_joint_cfg)
        # T_left_ee_pose_in_headlink2_coord_wa = self.coord_transformer.transform_pose(
        #         T_left_ee_pose_in_armbase_coord, "arm_base_link", "head_link2", joint_values=head_joint_cfg
        #     )[0]
        # T_right_ee_pose_in_headlink2_coord_wa = self.coord_transformer.transform_pose(
        #         T_right_ee_pose_in_armbase_coord, "arm_base_link", "head_link2", joint_values=head_joint_cfg
        #     )[0]
        T_left_ee_pose_in_headlink2_coord = np.linalg.inv(T_armbase_to_headlink2) @ T_left_ee_pose_in_armbase_coord[0]
        T_right_ee_pose_in_headlink2_coord = np.linalg.inv(T_armbase_to_headlink2) @ T_right_ee_pose_in_armbase_coord[0]
        
        """Head_Came in head_link2 coord

        tx, ty, tz: [0.0858, -0.04119, 0.0]

        rx, ry, rz(degree): [-180.0, -90.0, 0.0]
        
        rw, rx, ry, rz: [0.0, -0.70711, 0, 0.70711]"""
        T_head_link2_to_head_cam = np.eye(4)
        T_head_link2_to_head_cam[:3, 3] = np.array([0.0858, -0.04119, 0.0])  # Translation
        # DO NOT USE EULER!!!
        T_head_link2_to_head_cam[:3, :3] = R.from_euler(
            'xyz', [-180.0, -90.0, 0.0], degrees=True
        ).as_matrix()  # Rotation in XYZ order
        T_head_link2_to_head_cam[:3, :3] = R.from_quat(
            [0.0, -0.70711, 0, 0.70711], scalar_first=True
        ).as_matrix()  # Rotation from quaternion

        T_left_ee_pose_in_headcam_coord = np.linalg.inv(T_head_link2_to_head_cam) @ T_left_ee_pose_in_headlink2_coord
        T_right_ee_pose_in_headcam_coord = np.linalg.inv(T_head_link2_to_head_cam) @ T_right_ee_pose_in_headlink2_coord


        # # convert from sim cam coord to real cam coord, since urdf consistent with real cam coord
        T_left_ee_pose_in_real_headcam_coord= self.T_obj_in_simcam_to_T_obj_in_realcam(T_left_ee_pose_in_headcam_coord)
        T_right_ee_pose_in_real_headcam_coord= self.T_obj_in_simcam_to_T_obj_in_realcam(T_right_ee_pose_in_headcam_coord)
        
        # Decompose the transformation matrices to get translation and rotation
        left_ee_translation, left_ee_rotation = self.coord_transformer.decompose_transform(T_left_ee_pose_in_real_headcam_coord)
        right_ee_translation, right_ee_rotation = self.coord_transformer.decompose_transform(T_right_ee_pose_in_real_headcam_coord)
        
        # Convert rotation to Euler angles in XYZ order
        left_ee_rotation_euler_xyz = np.array(left_ee_rotation)
        right_ee_rotation_euler_xyz = np.array(right_ee_rotation)
        
        # Prepare the robot state dictionary
        robot_state = {
            "ROBOT_LEFT_TRANS": left_ee_translation.tolist(),
            "ROBOT_LEFT_ROT_EULER": left_ee_rotation_euler_xyz.tolist(),
            "ROBOT_LEFT_GRIPPER": np.array([left_gripper_joint]).tolist(),
            "ROBOT_RIGHT_TRANS": right_ee_translation.tolist(),
            "ROBOT_RIGHT_ROT_EULER": right_ee_rotation_euler_xyz.tolist(),
            "ROBOT_RIGHT_GRIPPER": np.array([right_gripper_joint]).tolist(),
        }
        return robot_state
    
    
    def T_obj_in_simcam_to_T_obj_in_realcam(
        self,
        # T_simcam_in_world: np.ndarray,
        T_obj_in_simcam: np.ndarray,
    ) -> np.ndarray:
        """
        Convert object pose from simcam frame to realcam frame.

        Inputs:
            T_simcam_in_world: 4x4 pose of simcam in world coordinates
            T_obj_in_simcam: 4x4 pose of object in simcam coordinates

        Output:
            T_obj_in_realcam: 4x4 pose of object in realcam coordinates
        """
        # The vla model output action is in sim camera coordinate, but we should executa in sim camera then sim world cam
        R_real2sim = np.array(
            [
                [1, 0, 0],  # realcam +X → simcam +X
                [0, -1, 0],  # realcam +Y → simcam -Y
                [0, 0, -1],  # realcam +Z → simcam -Z
            ]
        )
        R_sim2real = R_real2sim.T
        # np.array(
        #     [
        #         [1, 0,  0],
        #         [0, 0,  1],
        #         [0, -1, 0],
        #     ]
        # )

        T_sim2real = np.eye(4)
        T_sim2real[:3, :3] = R_sim2real

        T_real2sim = np.linalg.inv(T_sim2real)  # T_real2sim
        T_obj_in_realcam = T_real2sim @ T_obj_in_simcam
        return T_obj_in_realcam

    def preprocess_instruction(self, lang, substep_index=0):
        """
        Preprocess the task instruction.
        
        Args:
            lang: Task description in natural language
            substep_index: Index of the current substep
            
        Returns:
            The processed task instruction.
        """
        instruction = self._obs_instruction(lang, substep_index)
        return instruction
    
    def preprocess_images(self):
        """
        Preprocess the images in the image_list.
        This could include resizing, normalization, etc.
        """
        processed_images = []
        for img in self.image_list:
            # Perform preprocessing on each image
            processed_img = self._preprocess_single_image(img)
            processed_images.append(processed_img)
        return processed_images
    
            
    def _preprocess_single_image(self, img):
        """
        Preprocess a single image.
        
        Args:
            img: The image to preprocess.
        
        Returns:
            The preprocessed image.
        """
        # Resize the image to a fixed size (e.g., 224x224)
        img = self._resize_image(img, target_size=(224, 224))
        return img        
    
    def _obs_instruction(self, lang,  substep_index=0):
        instruction_splits = self._split_instruction(lang)
        
        # Strategy 1: Keep the last instruction if the index exceeds the length
        # if substep_index >= len(instruction_splits):
        #     return instruction_splits[-1]  # Return the last instruction if index exceeds

        # Strategy 2: Use modulo to wrap around so that if time is enough, try more times of whole loop of instructions
        if substep_index < 0:
            raise ValueError("Substep index cannot be negative.")
        substep_index = substep_index % len(instruction_splits)  # Wrap around

        instruction = instruction_splits[
            substep_index
        ]  
        return instruction
    
    def _split_instruction(self, instruction: str):
        """
        Split the instruction into individual actions.
        """
        # Split by semicolon and strip whitespace
        subinstruction = [
            action.strip() for action in instruction.split(";") if action.strip()
        ]
        if len(subinstruction) == 0:
            raise ValueError("Instruction is empty or only contains semicolons.")
        return subinstruction
        
        
        
    def _resize_image(self, img, target_size=(224, 224)):
        """
        Resize the image to the target size while maintaining aspect ratio.
        """
        h, w = img.shape[:2]
        target_aspect = 4 / 3
        current_aspect = w / h

        # Allow small floating point tolerance
        if abs(current_aspect - target_aspect) < 1e-3:
            padded = img  # No padding needed
        elif current_aspect > target_aspect:
            # Image is too wide → pad height
            new_h = int(w / target_aspect)
            pad_total = new_h - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            padded = np.pad(
                img,
                ((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            # Image is too tall → pad width
            new_w = int(h * target_aspect)
            pad_total = new_w - w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded = np.pad(
                img,
                ((0, 0), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        img = Image.fromarray(padded)
        img = img.resize(target_size, Image.LANCZOS)
        return np.array(img)
    
    def _get_unique_log_dir(self, base_dir, task_name):
        """
        base_dir/
        └── task_name/
            ├── iter_1/
            ├── iter_2/
            └── ...
            """
        task_base_dir = os.path.join(base_dir, task_name)
        os.makedirs(task_base_dir, exist_ok=True)

        i = 1
        while True:
            iter_log_dir = os.path.join(task_base_dir, f"iter_{i}")
            if not os.path.exists(iter_log_dir):
                os.makedirs(iter_log_dir)
                return iter_log_dir
            i += 1
            
    def _create_and_register_log_dir(self, log_dir: str):
        """Make sure log directory of single run is same and unique."""
        if (log_dir, self.task_name) not in self._log_dir_registry:
            task_log_dir = self._get_unique_log_dir(log_dir, self.task_name)
            self._log_dir_registry[(log_dir, self.task_name)] = task_log_dir
        return self._log_dir_registry[(log_dir, self.task_name)]            

    def _log_observations(self, obs_dict: Dict[str, Any], log_dir: str):
        """Log observations to a specified directory."""
        task_log_dir = self._create_and_register_log_dir(log_dir)
            
        # Use timestamp or step_id as filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # also save the observations
        obs_filename = os.path.join(
            task_log_dir, f"observations_{timestamp}.pkl"
        )
        with open(obs_filename, "wb") as f:
            # pickle.dump(observations, f)
            pickle.dump(obs_dict, f)
        
if __name__ == "__main__":
    # Example usage
    image_list = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
    task_instruction = "Pick up the red ball and place it in the blue box."
    robot_state = {"joint_positions": [0.0] * 16, "gripper_state": "open"}
    
    processor = VLAInputProcessor(image_list, task_instruction, robot_state)
    input_data = processor.prepare_input()
    
    print("Prepared Input Data:")
    print(input_data)
    # This will print the processed images, task description, and robot state.
