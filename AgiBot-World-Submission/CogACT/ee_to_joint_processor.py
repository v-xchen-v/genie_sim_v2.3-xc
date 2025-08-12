import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional
from pathlib import Path

from kinematics.g1_relax_ik import G1RelaxSolver
from kinematics.urdf_coordinate_transformer import URDFCoordinateTransformer


class EEtoJointProcessor:
    """
    Convert VLA ee pose (camera frame) -> arm base frame -> 7-DoF joint angles.
    Uses urdfpy FK + relax IK.
    """
    ### ------------Public API------------ ###
    def __init__(self):
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
        
    
    def get_joint_cmd(self, vla_act_dict, 
                      head_joint_cfg: Optional[Dict[str, float]],
                      curr_arm_joint_angles: np.ndarray) -> np.ndarray:
        """
        Get joint command from VLA action.
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
        vla_act_dict = self._process(vla_act_dict, head_joint_cfg=head_joint_cfg, curr_arm_joint_angles=curr_arm_joint_angles)
        left_arm_joint_angles = vla_act_dict.get("ROBOT_LEFT_JOINTS")
        right_arm_joint_angles = vla_act_dict.get("ROBOT_RIGHT_JOINTS")
        
        if left_arm_joint_angles is None or right_arm_joint_angles is None:
            raise ValueError("VLA action does not contain joint angles for both arms.")
        
        # get gripper joint angles
        left_gripper_joint = self._cmd_gripper("left", vla_act_dict)
        right_gripper_joint = self._cmd_gripper("right", vla_act_dict)
        
        # Combine left and right arm joint angles and gripper commands
        joint_cmd = np.concatenate([
            left_arm_joint_angles,
            left_gripper_joint.reshape(-1, 1),
            right_arm_joint_angles,
            right_gripper_joint.reshape(-1, 1)
        ], axis=1)  # [num_steps, 16]
        
        return joint_cmd
    
    ### ------------Private API------------ ###
    def _process(self, vla_act_dict: dict, head_joint_cfg: Optional[Dict[str, float]], curr_arm_joint_angles) -> np.ndarray:
        """
        Process VLA action to joint angles.
        vla_act_dict: dict with keys 'rotation' and 'translation' for the end-effector pose.
        arm: "left" or "right" to specify which arm to use.
        head_joint_cfg: Optional dictionary for head joint configuration.
        """

        # Convert VLA action to joint angles
        left_arm_joint_angles = self.vla_pose_to_joints(
            vla_act_dict,
            arm="left",
            head_joint_cfg=head_joint_cfg,
            curr_arm_joint_angles=curr_arm_joint_angles
        ) # [num_steps, 7] joint angles for left arm
        right_arm_joint_angles = self.vla_pose_to_joints(
            vla_act_dict,
            arm="right",
            head_joint_cfg=head_joint_cfg,
            curr_arm_joint_angles=curr_arm_joint_angles
        ) # [num_steps, 7] joint angles for right arm
        
        # Add the joint angles into the action dict for controller
        vla_act_dict["ROBOT_LEFT_JOINTS"] = left_arm_joint_angles
        vla_act_dict["ROBOT_RIGHT_JOINTS"] = right_arm_joint_angles
        
        return vla_act_dict
    
    def _act_gripper(self, arm: str, vla_act_dict: dict) -> np.ndarray:
        """
        Get gripper joint command from VLA action.
        """
        if arm not in ["left", "right"]:
            raise ValueError("Arm must be 'left' or 'right'.")
        
        if arm == "left":
            left_gripper_joint = np.array(vla_act_dict.get("ROBOT_LEFT_GRIPPER")).reshape(-1, 1)  # [num_steps, 1]
            if left_gripper_joint is None:
                raise ValueError("VLA action does not contain gripper joint angles for the left arm.")
            return left_gripper_joint
        
        # arm == "right"
        right_gripper_joint = np.array(vla_act_dict.get("ROBOT_RIGHT_GRIPPER")).reshape(-1, 1)  # [num_steps, 1]
        if right_gripper_joint is None:
            raise ValueError("VLA action does not contain gripper joint angles for the right arm.")
        return right_gripper_joint
        
    def _cmd_gripper(self, arm: str, vla_act_dict: dict) -> np.ndarray:
        """
        calculate gripper command from vla action.
        the gripper value is in [0, 1], where 0 is fully open and 1 is fully closed.
        we need to *70 for the gripper joint command.
        """
        if arm not in ["left", "right"]:
            raise ValueError("Arm must be 'left' or 'right'.")
        
        gripper_joint = self._act_gripper(arm, vla_act_dict)
        print(f"gripper value shape: {gripper_joint.shape}, gripper value: {gripper_joint}")
        
        # convert to joint command
        gripper_cmd = np.clip(gripper_joint * 120/70.0, 0, 1)  # [num_steps, 1]
        print(f"gripper command shape: {gripper_cmd.shape}, gripper command: {gripper_cmd}")
        
        return gripper_cmd
    
    def vla_pose_to_joints(
        self,
        vla_act_dict: dict,
        # rot_key: str,
        # trans_key: str,
        arm: str,
        head_joint_cfg: Dict[str, float],
        curr_arm_joint_angles,
    ) -> np.ndarray:
        """
        Convert VLA action to joint angles.
        vla_act_dict: dict with keys 'rotation' and 'translation' for the end-effector pose.
        """
        if arm not in ["left", "right"]:
            raise ValueError("Arm must be 'left' or 'right'.")
        
        if arm == "left":
            rot_key = "ROBOT_LEFT_ROT_EULER"
            trans_key = "ROBOT_LEFT_TRANS"
        else:  # arm == "right"
            rot_key = "ROBOT_RIGHT_ROT_EULER"
            trans_key = "ROBOT_RIGHT_TRANS"



        # Extract rotation and translation from the VLA action dictionary
        rotation_delta = vla_act_dict.get(rot_key) # [num_steps, 3] euler angles
        translation_delta = vla_act_dict.get(trans_key) # [num_steps, 3]
                
        # for each step, convert to 4x4 pose matrix
        if isinstance(rotation_delta, list) and isinstance(translation_delta, list):
            if len(rotation_delta) != len(translation_delta):
                raise ValueError("Rotation and translation lists must have the same length.")
        
        """
        Handle delta
        Refer:
        rotation_sum = R.from_euler("xyz", ee_left_rot_eular_xyz_s, degrees=False)
        rotaion_list = []
        for delta in ee_left_rot_euler_xyz_delta:
            rotation_sum = R.from_euler("xyz", delta, degrees=False) * rotation_sum
            rotaion_list.append(rotation_sum.as_euler("xyz", degrees=False))
        return np.array(rotaion_list)
        """
        
        curr_left_arm_joint_angles = curr_arm_joint_angles[:7]
        curr_right_arm_joint_angles = curr_arm_joint_angles[8:15]
        self.last_left_arm_joint_angles = curr_left_arm_joint_angles if arm == "left" else self.last_left_arm_joint_angles
        self.last_right_arm_joint_angles = curr_right_arm_joint_angles if arm == "right" else self.last_right_arm_joint_angles
        T_left_ee_pose_in_armbase_coord = self.left_arm_ik_solver.compute_fk(curr_left_arm_joint_angles)[0]
        T_right_ee_pose_in_armbase_coord = self.right_arm_ik_solver.compute_fk(curr_right_arm_joint_angles)[0]
        

        
        # convert into head cam coord frame
        if arm == "left":
            # since URDF only have head_link2 but not cam link so that we first transform to head_link2
            # then transform to head cam link
            # T_ee_pose_in_headlink2_coord_wa = self.coord_transformer.transform_pose(
            #     T_left_ee_pose_in_armbase_coord, "arm_base_link", "head_link2", joint_values=head_joint_cfg
            # )

            T_armbase_to_headlink2 = self.coord_transformer.relative_transform("arm_base_link", "head_link2", head_joint_cfg)
            T_ee_pose_in_headlink2_coord = np.linalg.inv(T_armbase_to_headlink2) @ T_left_ee_pose_in_armbase_coord
            # print("T_ee_pose_in_headlink2_coord:", T_ee_pose_in_headlink2_coord)
            
        else:  # arm == "right"
            # T_ee_pose_in_headlink2_coord_wa = self.coord_transformer.transform_pose(
            #     T_right_ee_pose_in_armbase_coord, "arm_base_link", "head_link2", joint_values=head_joint_cfg
            # )
            T_armbase_to_headlink2 = self.coord_transformer.relative_transform("arm_base_link", "head_link2", head_joint_cfg)
            T_ee_pose_in_headlink2_coord = np.linalg.inv(T_armbase_to_headlink2) @ T_right_ee_pose_in_armbase_coord
            # print("T_ee_pose_in_headlink2_coord:", T_ee_pose_in_headlink2_coord)
            
        """Head_Came in head_link2 coord

        tx, ty, tz: [0.0858, -0.04119, 0.0]

        rx, ry, rz(degree): [-180.0, -90.0, 0.0]
        
        rw, rx, ry, rz: [0.0, -0.70711, 0, 0.70711]"""
        T_head_link2_to_head_cam = np.eye(4)
        T_head_link2_to_head_cam[:3, 3] = np.array([0.0858, -0.04119, 0.0])  # Translation
        # T_head_link2_to_head_cam[:3, :3] = R.from_euler(
        #     'xyz', [-180.0, -90.0, 0.0], degrees=True
        # ).as_matrix()  # Rotation in XYZ order
        T_head_link2_to_head_cam[:3, :3] = R.from_quat(
            [0.0, -0.70711, 0, 0.70711], scalar_first=True
        ).as_matrix()

        T_ee_pose_in_headcam_coord = np.linalg.inv(T_head_link2_to_head_cam) @ T_ee_pose_in_headlink2_coord

        # URDF cam coordinate is consistent with real camera coordinate, so that we could skip the conversion
        # # the readed joint angles is from sim but delta predicted is based on real camera coordinate
        # T_left_ee_pose_in_armbase_coord = self.realcam_to_simcam(T_obj_in_simcam=T_left_ee_pose_in_armbase_coord)
        # T_right_ee_pose_in_armbase_coord = self.realcam_to_simcam(T_obj_in_simcam=T_right_ee_pose_in_armbase_coord)

        curr_ee_rot = T_ee_pose_in_headcam_coord[:3, :3] # [3x3] rotation matrix
        curr_ee_trans = T_ee_pose_in_headcam_coord[:3, 3] # [tx, ty, tz]


        rotation_list = []
        rotation_sum = curr_ee_rot
        translation_list = []
        for rot_delta in rotation_delta:
            rotation_sum = R.from_euler("xyz", rot_delta, degrees=False).as_matrix() * rotation_sum
            rotation_list.append(rotation_sum)

        translation_list = []
        translation_sum = curr_ee_trans
        for trans_delta in translation_delta:
            translation_sum += trans_delta
            translation_list.append(translation_sum.copy())
        # We got the real cam coord based end-effector poses in 16 steps, now we need to convert them to arm base frame    
        
        poses = []
        T_ee_pose_arm_base_frame_list = []
        for rot_matrix, trans_vec in zip(rotation_list, translation_list):
            if rot_matrix.shape != (3, 3) or len(trans_vec) != 3:
                raise ValueError("Rotation and translation vectors must be of length 3.")
            # rot_matrix = R.from_euler('xyz', rot_vec, degrees=False).as_matrix()
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :3] = rot_matrix
            pose_4x4[:3, 3] = trans_vec

            # # # here we need to convert the pose from real cam to sim cam
            # pose_4x4 = self.realcam_to_simcam(T_obj_in_realcam=pose_4x4)

            poses.append(pose_4x4) # based on real cam coord

            """Head_Came in head_link2 coord

            tx, ty, tz: [0.0858, -0.04119, 0.0]

            rx, ry, rz(degree): [-180.0, -90.0, 0.0]
            
            rw, rx, ry, rz: [0.0, -0.70711, 0, 0.70711]
            """
            T_head_link2_to_head_cam = np.eye(4)
            T_head_link2_to_head_cam[:3, 3] = np.array([0.0858, -0.04119, 0.0])  # Translation
            # T_head_link2_to_head_cam[:3, :3] = R.from_euler(
            #     'xyz', [-180.0, -90.0, 0.0], degrees=True
            # ).as_matrix()  # Rotation in XYZ order
            T_head_link2_to_head_cam[:3, :3] = R.from_quat(
                [0.0, -0.70711, 0, 0.70711], scalar_first=True
            ).as_matrix()


            # convert pose from real cam coord to head link2 coord
            T_ee_pose_in_headlink2_coord = T_head_link2_to_head_cam @ pose_4x4
    

            # Get the end-effector pose in the arm base frame
            # T_ee_pose_arm_base_frame_wa = self.coord_transformer.transform_pose(
            #     T_ee_pose_in_headlink2_coord, "head_link2", "arm_base_link", joint_values=head_joint_cfg
            # )
            # head_angles = [head_joint_cfg.get("idx11_head_joint1", 0.0), head_joint_cfg.get("idx12_head_joint2", 0.0)]
            # T_arm_base_link_to_head_link2 = self.head_ik_solver.compute_fk(head_angles)
            # T_ee_pose_arm_base_frame2 = T_arm_base_link_to_head_link2 @ T_ee_pose_in_headlink2_coord
            T_headlink2_to_armbase = self.coord_transformer.relative_transform("head_link2", "arm_base_link", head_joint_cfg)
            T_ee_pose_arm_base_frame = np.linalg.inv(T_headlink2_to_armbase) @ T_ee_pose_in_headlink2_coord

            # """IN USD for iros_pack_in_the_supermarket:
            # arm_base_link:
            # tx, ty, tz: [0.2835, 0.0, 1.21263]
            # rx, ry, rz(degree): [0.0, 29.999, 0.0]

            # head_link2:
            # tx, ty, tz: [0.42504, 0.0, 1.35731]
            # rx, ry, rz(degree): [-90.0, 0.0, 54.99]
            # """
            # T_arm_base_link_in_usd = np.eye(4)
            # T_arm_base_link_in_usd[:3, 3] = np.array([0.2835, 0.0, 1.21263])  # Translation
            # T_arm_base_link_in_usd[:3, :3] = R.from_euler(
            #     'xyz', [0.0, 29.999, 0.0], degrees=True
            # ).as_matrix()  # Rotation in XYZ order

            # T_head_link2_in_usd = np.eye(4)
            # T_head_link2_in_usd[:3, 3] = np.array([0.42504, 0.0, 1.35731])  # Translation
            # T_head_link2_in_usd[:3, :3] = R.from_euler(
            #     'xyz', [-90.0, 0.0, 54.99], degrees=True
            # ).as_matrix()  # Rotation in XYZ order

            # # Convert the pose from head link2 coord to arm base frame
            # T_arm_base_to_head_link2_in_usd = np.linalg.inv(T_arm_base_link_in_usd) @ T_head_link2_in_usd

            # T_ee_pose_arm_base_frame = T_arm_base_to_head_link2_in_usd @ T_ee_pose_in_headlink2_coord

            T_ee_pose_arm_base_frame_list.append(T_ee_pose_arm_base_frame2)
        
        # Got ee poses in arm base frame, now solve IK for each pose
        if arm == "left":
            ik_solver = self.left_arm_ik_solver
        else:  # arm == "right"
            ik_solver = self.right_arm_ik_solver
        
        joint_angles_list = []
        for T_ee in T_ee_pose_arm_base_frame_list:
            #  set last joint state for the solver
            if arm == "left" and self.last_left_arm_joint_angles is not None:
                ik_solver.set_current_state(self.last_left_arm_joint_angles)
                ik_solver.compute_fk(self.last_left_arm_joint_angles)  # update the FK cache
            elif arm == "right" and self.last_right_arm_joint_angles is not None:
                ik_solver.set_current_state(self.last_right_arm_joint_angles)
                
            joint_angles = ik_solver.solve_from_pose(T_ee)

            # optional: compute the ee pose error
            computed_ee_pose = ik_solver.compute_fk(joint_angles)  # update the FK cache
            # compute T_ee and computed_ee_pose
            T_ee_computed = computed_ee_pose[0]
            T_ee_error = T_ee_computed @ np.linalg.inv(T_ee)  # Error in pose
            print(f"End-effector pose error for {arm} arm: {T_ee_error[:3, 3]}")  # Translation error
            print(f"End-effector rotation error for {arm} arm: {R.from_matrix(T_ee_error[:3, :3]).as_euler('xyz', degrees=True)}")  # Rotation error

            joint_angles_list.append(joint_angles)
            
            # update the last joint angles for the solver
            if arm == "left":
                self.last_left_arm_joint_angles = joint_angles
            else:  # arm == "right"
                self.last_right_arm_joint_angles = joint_angles
        
        # Convert list of joint angles to a numpy array
        joint_angles_array = np.array(joint_angles_list)
        
        # if joint_angles_array.ndim == 1:
        #     return joint_angles_array
        
        # # TODO: Implement the conversion from VLA pose to joint angles
        # joint_angles = np.zeros(7)  # Placeholder for the actual joint angles

        return joint_angles_array

    def realcam_to_simcam(
        self,
        # T_realcam_in_world: np.ndarray,
        T_obj_in_realcam: np.ndarray,
    ) -> np.ndarray:
        """
        Convert object pose from realcam frame to simcam frame.

        Inputs:
            T_realcam_in_world: 4x4 pose of realcam in world coordinates
            T_obj_in_realcam: 4x4 pose of object in realcam coordinates

        Output:
            T_obj_in_simcam: 4x4 pose of object in simcam coordinates
        """
        # the vla model input obs-robot state should be in real camera coordinate
        R_real2sim = np.array(
            [
                [1, 0, 0],  # realcam +X → simcam +X
                [0, -1, 0],  # realcam +Y → simcam -Y
                [0, 0, -1],  # realcam +Z → simcam -Z
            ]
        )

        T_real2sim = np.eye(4)
        T_real2sim[:3, :3] = R_real2sim

        # T_obj_in_simcam = T_real2sim @ T_obj_in_realcam
        T_sim2real = np.linalg.inv(T_real2sim)  # T_sim2real
        T_obj_in_simcam = T_sim2real @ T_obj_in_realcam
        return T_obj_in_simcam

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
