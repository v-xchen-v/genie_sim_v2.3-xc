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

if __name__ == "__main__":
    import os
    from pathlib import Path
    # Example usage
    # solver = G1RelaxSolver(urdf_path="path/to/urdf", config_path="path/to/config", arm="right")
    # Get absolute path relative to this script
    URDF_PATH = Path(__file__).parent / "configs/g1/G1_NO_GRIPPER.urdf"
    CONFIG_PATH = Path(__file__).parent / "configs/g1/g1_solver.yaml"
    
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF file not found: {URDF_PATH}")
    
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    
    # Ensure the path is absolute
    URDF_PATH = str(URDF_PATH.resolve())
    CONFIG_PATH = str(CONFIG_PATH.resolve())

    # Initialize the solver
    solver = G1RelaxSolver(
        urdf_path=URDF_PATH,
        config_path=CONFIG_PATH,
        arm="right"
    )

    # Optional: Sync target with initial joint configuration
    initial_joint_angles = np.zeros(7)
    solver.set_current_state(initial_joint_angles)

    # Example 1: Solve from 4x4 SE(3) pose
    pose_matrix = np.eye(4)
    pose_matrix[:3, 3] = [0.3, 0.2, 0.5]  # Set translation only
    joint_solution = solver.solve_from_pose(pose_matrix)
    print("Joint solution from SE(3) pose:\n", joint_solution)

    # Example 2: Solve from position and quaternion
    position = np.array([0.4, 0.1, 0.3])
    quaternion_xyzw = np.array([0, 0, 0, 1])  # Identity quaternion
    joint_solution = solver.solve_from_pos_quat(position, quaternion_xyzw)
    print("Joint solution from pos + quat:\n", joint_solution)

    # Example 3: Forward Kinematics
    print("\n=== Forward Kinematics ===")
    test_joint_angles = np.array([0.1, -0.2, 0.3, -0.1, 0.5, -0.4, 0.2])
    ee_pose = solver.compute_fk(test_joint_angles)
    print("End-effector pose from FK:\n", ee_pose)
