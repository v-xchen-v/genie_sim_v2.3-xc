import numpy as np
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R
# --- Compatibility patches for older libraries like urdfpy ---
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.Set = collections.abc.Set
collections.Iterable = collections.abc.Iterable

import math
import fractions
fractions.gcd = math.gcd

import numpy as np
np.int = int
np.float_ = float
np.float = float

from pathlib import Path

class URDFCoordinateTransformer:
    def __init__(self, urdf_path):
        self.robot = URDF.load(urdf_path)

    def relative_transform(self, link_from, link_to, joint_values=None):
        """
        Compute the homogeneous transform from link_from -> link_to.

        joint_values: dict of joint_name -> angle (radians) for movable joints.
                      Unspecified joints default to 0.0.
        """
        if joint_values is None:
            joint_values = {}

        fk = self.robot.link_fk(joint_values)
        T_base_from = fk[self.robot.link_map[link_from]]
        T_base_to = fk[self.robot.link_map[link_to]]
        return np.linalg.inv(T_base_from) @ T_base_to

    def reverse_transform(self, link_from, link_to, joint_values=None):
        """
        Compute the transform from link_to -> link_from (reverse).
        """
        return np.linalg.inv(self.relative_transform(link_from, link_to, joint_values))

    def decompose_transform(self, T):
        """
        Break transform into translation vector and Euler XYZ rotation.
        """
        R_mat = T[:3, :3]
        t_vec = T[:3, 3]
        euler_xyz = R.from_matrix(R_mat).as_euler('xyz')
        return t_vec, euler_xyz

    def transform_point(self, point, link_from, link_to, joint_values=None):
        """
        Transform a 3D point from link_from's coordinate frame to link_to's frame.
        """
        T = self.relative_transform(link_from, link_to, joint_values)
        point_h = np.append(np.array(point), 1.0)  # to homogeneous
        return (T @ point_h)[:3]


# === Example Usage ===
if __name__ == "__main__":
    # change the path of URDF file as relative to this script
    urdf_path = Path(__file__).parent / "configs/g1/G1_omnipicker.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # Initialize the transformer with the URDF path
    transformer = URDFCoordinateTransformer(urdf_path)

    joint_cfg = {
        "idx11_head_joint1": 0.10,
        "idx12_head_joint2": -0.20
    }

    # arm_r_base_link -> head_link2
    T_armr_to_head = transformer.relative_transform("arm_r_base_link", "head_link2", joint_cfg)
    T_head_to_armr = transformer.reverse_transform("arm_r_base_link", "head_link2", joint_cfg)

    # arm_l_base_link -> head_link2
    T_arml_to_head = transformer.relative_transform("arm_l_base_link", "head_link2", joint_cfg)
    T_head_to_arml = transformer.reverse_transform("arm_l_base_link", "head_link2", joint_cfg)

    print("arm_r_base_link -> head_link2:\n", T_armr_to_head)
    print("head_link2 -> arm_r_base_link:\n", T_head_to_armr)
    print("arm_l_base_link -> head_link2:\n", T_arml_to_head)
    print("head_link2 -> arm_l_base_link:\n", T_head_to_arml)

    # Example point transformation
    point_in_head = [0.1, 0.0, 0.0]
    point_in_armr = transformer.transform_point(point_in_head, "head_link2", "arm_r_base_link", joint_cfg)
    print("Point in head_link2:", point_in_head, "-> in arm_r_base_link:", point_in_armr)
