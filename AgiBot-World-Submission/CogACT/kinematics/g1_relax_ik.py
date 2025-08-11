import numpy as np
from ik_solver import Solver, RobotPart
from scipy.spatial.transform import Rotation
from typing import Literal


class G1RelaxSolver:
    def __init__(
        self,
        urdf_path: str,
        config_path: str,
        arm: Literal["left", "right"] = "right",
        debug: bool = False,
    ):
        self.arm = arm
        self.robot_part = RobotPart.LEFT_ARM if arm == "left" else RobotPart.RIGHT_ARM
        self._solver = Solver(part=self.robot_part, urdf_path=urdf_path, config_path=config_path)
        self._solver.set_debug_mode(debug)
        self.last_pose = None  # (pos, rot) tuple

    def set_current_state(self, joint_angles: np.ndarray):
        """Sets initial joint state, syncing target with FK"""
        self._solver.sync_target_with_joints(joint_angles)

    def update_target(
        self,
        pos: np.ndarray,
        rot: np.ndarray = None,
        quat: np.ndarray = None,
    ):
        """
        Update IK target from either rotation matrix or quaternion.
        Stores internally for reuse.
        """
        if quat is None:
            assert rot is not None
            self._solver.update_target_mat(pos, rot)
        else:
            self._solver.update_target_quat(pos, quat)

        self.last_pose = (pos, rot if rot is not None else Rotation.from_quat(quat).as_matrix())

    def solve(self) -> np.ndarray:
        """Return joint solution for current target"""
        return self._solver.solve()

    def solve_from_pose(self, pose_4x4: np.ndarray) -> np.ndarray:
        """High-level: update target from SE(3) matrix and solve"""
        pos = pose_4x4[:3, 3]
        rot = pose_4x4[:3, :3]
        self.update_target(pos, rot)
        return self.solve()

    def solve_from_pos_quat(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """High-level: update target from pos + quat and solve"""
        self.update_target(pos, quat=quat)
        return self.solve()

    def get_current_pose(self) -> np.ndarray:
        """Return 4x4 matrix of current target"""
        if self.last_pose is None:
            mat = self._solver.get_current_target()[0]
            return np.array(mat)
        pos, rot = self.last_pose
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    def compute_fk(self, joint_angles: np.ndarray) -> list[np.ndarray]:
        """Compute FK from joint angles. Returns list of 4x4 poses."""
        return self._solver.compute_fk(joint_angles)
