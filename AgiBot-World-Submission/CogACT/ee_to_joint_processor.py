import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from scipy.spatial.transform import Rotation
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
        
        self.transformer = URDFCoordinateTransformer(self.fk_urdf_path)
        
        self.ik_urdf_path = Path(__file__).parent / "kinematics/configs/g1/G1_NO_GRIPPER.urdf"
        if not self.ik_urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {self.ik_urdf_path}")
        
        # Initialize the IK solvers for both arms
        self.left_arm_ik_solver = G1RelaxSolver(
            urdf_path=self.ik_urdf_path,
            config_path=Path(__file__).parent / "kinematics/configs/g1/g1_solver.yaml",
            arm="left"
        )
        self.right_arm_ik_solver = G1RelaxSolver(
            urdf_path=self.ik_urdf_path,
            config_path=Path(__file__).parent / "kinematics/configs/g1/g1_solver.yaml",
            arm="right"
        )
        
    def vla_pose_to_joints(
        self,
        vla_act_dict: dict,
        rot_key: str,
        trans_key: str,
    ) -> np.ndarray:
        """
        Convert VLA action to joint angles.
        vla_act_dict: dict with keys 'rotation' and 'translation' for the end-effector pose.
        """
        # Extract rotation and translation from the VLA action dictionary
        rotation = vla_act_dict.get(rot_key)
        translation = vla_act_dict.get(trans_key)

        # TODO: Implement the conversion from VLA pose to joint angles
        joint_angles = np.zeros(7)  # Placeholder for the actual joint angles

        return joint_angles
