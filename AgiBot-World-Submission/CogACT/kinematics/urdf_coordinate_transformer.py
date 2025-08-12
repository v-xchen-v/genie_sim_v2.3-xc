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

    # def transform_pose(
    #     self,
    #     T_obj, # pose
    #     link_from,
    #     link_to,
    #     joint_values=None,
    #     output="matrix",          # "matrix" | "tuple_quat" | "tuple_euler"
    #     euler_order="xyz"         # used when parsing/returning eulers
    # ):
    #     """
    #     Transform a pose expressed in link_from's frame into link_to's frame.

    #     pose:
    #       - 4x4 homogeneous matrix

    #     Returns:
    #       - if output=="matrix": 4x4 homogeneous matrix
    #       - if output=="tuple_quat": (t, quat[x,y,z,w])
    #       - if output=="tuple_euler": (t, euler in euler_order)
    #     """
    #     # transform between frames
    #     T_rel = self.relative_transform(link_from, link_to, joint_values)

    #     # apply transform
    #     T_out = T_rel @ T_obj

    #     if output == "matrix":
    #         return T_out
    #     elif output == "tuple_quat":
    #         t = T_out[:3, 3]
    #         q = R.from_matrix(T_out[:3, :3]).as_quat()  # [x,y,z,w]
    #         return t, q
    #     elif output == "tuple_euler":
    #         t = T_out[:3, 3]
    #         e = R.from_matrix(T_out[:3, :3]).as_euler(euler_order)
    #         return t, e
    #     else:
    #         raise ValueError("output must be 'matrix', 'tuple_quat', or 'tuple_euler'.")

    # helper: coerce various pose formats to 4x4 matrix
    def _pose_to_matrix(self, pose, euler_order="xyz"):
        if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
            return pose

        # dict input
        if isinstance(pose, dict):
            t = np.asarray(pose.get("t", [0, 0, 0]), dtype=float)
            if "quat" in pose:
                Rm = R.from_quat(pose["quat"]).as_matrix()
            elif "euler" in pose:
                order = pose.get("order", euler_order)
                Rm = R.from_euler(order, pose["euler"]).as_matrix()
            elif "R" in pose:
                Rm = np.asarray(pose["R"], dtype=float)
            else:
                Rm = np.eye(3)
            return self._make_T(Rm, t)

        # tuple/list input
        if isinstance(pose, (list, tuple)) and len(pose) == 2:
            t = np.asarray(pose[0], dtype=float)
            rot = np.asarray(pose[1], dtype=float)

            if rot.shape == (4,):  # quat
                Rm = R.from_quat(rot).as_matrix()
            elif rot.shape == (3,):  # euler
                Rm = R.from_euler(euler_order, rot).as_matrix()
            elif rot.shape == (3, 3):  # rotation matrix
                Rm = rot
            else:
                raise ValueError("Unrecognized rotation format in pose tuple.")
            return self._make_T(Rm, t)

        raise ValueError("Unsupported pose format. Provide 4x4, (t, rot), or dict.")

    @staticmethod
    def _make_T(Rm, t):
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = np.asarray(t, dtype=float)
        return T

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
