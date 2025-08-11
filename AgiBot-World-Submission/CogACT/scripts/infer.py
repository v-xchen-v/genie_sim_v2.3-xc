# Reference: AgiBot-World/UniVLA/scripts/infer.py
import os
import sys
from pathlib import Path
import cv2

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

import rclpy
import os
import threading
from genie_sim_ros import SimROSNode
from cv_bridge import CvBridge
import numpy as np
from PIL import Image
from cogact_policy import CogActAPIPolicy
from vlainputprocessor import VLAInputProcessor
from kinematics.urdf_coordinate_transformer import URDFCoordinateTransformer
from kinematics.g1_relax_ik import G1RelaxSolver

def get_instruction(task_name):

    if task_name == "iros_clear_the_countertop_waste":
        lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "iros_restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
    elif task_name == "iros_clear_table_in_the_restaurant":
        lang = "Pick up the bowl on the table near the right arm with the right arm.;Place the bowl on the plate on the table with the right arm."
    elif task_name == "iros_stamp_the_seal":
        lang = "Pick up the stamp from the ink pad on the table with the right arm.;Stamp the document on the table with the stamp in the right arm.;Place the stamp into the ink pad on the table with the right arm."
    elif task_name == "iros_pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm.;Put the grape juice into the felt bag on the table with the right arm."
    elif task_name == "iros_heat_the_food_in_the_microwave":
        lang = "Open the door of the microwave oven with the right arm.;Pick up the plate with bread on the table with the right arm.;Put the plate containing bread into the microwave oven with the right arm.;Push the plate that was not placed properly into the microwave oven the right arm.;Close the door of the microwave oven with the left arm.;Press the start button on the right side of the microwave oven with the right arm."
    elif task_name == "iros_open_drawer_and_store_items":
        lang = "Pull the top drawer of the drawer cabinet with the right arm.;Pick up the Rubik's Cube on the drawer cabinet with the right arm.;Place the Rubik's Cube into the drawer with the right arm.;Push the top drawer of the drawer cabinet with the right arm."
    elif task_name == "iros_pack_moving_objects_from_conveyor":
        lang = "Pick up the hand cream from the conveyor belt with the right arm;Place the hand cream held in the right arm into the box on the table"
    elif task_name == "iros_pickup_items_from_the_freezer":
        lang = "Open the freezer door with the right arm;Pick up the caviar from the freezer with the right arm;Place the caviar held in the right arm into the shopping cart;Close the freezer door with both arms"
    elif task_name == "iros_make_a_sandwich":
        lang = "Pick up the bread slice from the toaster on the table with the right arm;Place the picked bread slice into the plate on the table with the right arm;Pick up the ham slice from the box on the table with the left arm;Place the picked ham slice onto the bread slice in the plate on the table with the left arm;Pick up the lettuce slice from the box on the table with the right arm;Place the picked lettuce slice onto the ham slice in the plate on the table with the right arm;Pick up the bread slice from the toaster on the table with the right arm;Place the bread slice onto the lettuce slice in the plate on the table with the right arm"
    else:
        raise ValueError("task does not exist")

    return lang

def get_head_joint_cfg(task_name):
    # Define your joint configurations per task
    task_joint_cfgs = {
        "iros_clear_the_countertop_waste": {
            "idx11_head_joint1": 0.,
            "idx12_head_joint2": 0.4593
        },
        "iros_restock_supermarket_items": {
            "idx11_head_joint1": 0.,
            "idx12_head_joint2": 0.3839745594177246
        },
        "iros_clear_table_in_the_restaurant": {
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.43633231
        },
        "iros_stamp_the_seal": {
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.384
        },
        "iros_pack_in_the_supermarket": {
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.43633231
        },
        "iros_heat_the_food_in_the_microwave": {
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.43633231
        },
        "iros_open_drawer_and_store_items": {
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.2617993878
        },
        "iros_pack_moving_objects_from_conveyor": {
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.4362
        },
        "iros_pickup_items_from_the_freezer": {
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.4362
        },
        "iros_make_a_sandwich": {
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.43634
        }
    }
    
    if task_name in task_joint_cfgs:
        return task_joint_cfgs[task_name]
    else:
        raise ValueError(f"Joint configuration for task '{task_name}' not defined.")
    
def get_sim_time(sim_ros_node):
    sim_time = sim_ros_node.get_clock().now().nanoseconds * 1e-9
    return sim_time

def infer(policy):
    
    
    rclpy.init()
    current_path = os.getcwd()
    sim_ros_node = SimROSNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
    spin_thread.start()
    
    bridge = CvBridge()
    count = 0
    SIM_INIT_TIME = 10

    lang = get_instruction(task_name="iros_pack_in_the_supermarket")
    
    transformer = URDFCoordinateTransformer("kinematics/configs/g1/G1_omnipicker.urdf")
    g1_ik_solver = G1RelaxSolver(
        urdf_path="kinematics/configs/g1/G1_NO_GRIPPER.urdf",
        config_path="kinematics/configs/g1/g1_solver.yaml",
        arm="right",
        debug=False,
    )
    # Optional: Sync target with initial joint configuration
    initial_joint_angles = np.zeros(7)
    g1_ik_solver.set_current_state(initial_joint_angles)

    # Example 1: Solve from 4x4 SE(3) pose
    pose_matrix = np.eye(4)
    pose_matrix[:3, 3] = [0.3, 0.2, 0.5]  # Set translation only
    joint_solution = g1_ik_solver.solve_from_pose(pose_matrix)
    print("Joint solution from SE(3) pose:\n", joint_solution)

    # Example 2: Solve from position and quaternion
    position = np.array([0.4, 0.1, 0.3])
    quaternion_xyzw = np.array([0, 0, 0, 1])  # Identity quaternion
    joint_solution = g1_ik_solver.solve_from_pos_quat(position, quaternion_xyzw)
    print("Joint solution from pos + quat:\n", joint_solution)

    # Example 3: Forward Kinematics
    print("\n=== Forward Kinematics ===")
    test_joint_angles = np.array([0.1, -0.2, 0.3, -0.1, 0.5, -0.4, 0.2])
    ee_pose = g1_ik_solver.compute_fk(test_joint_angles)
    print("End-effector pose from FK:\n", ee_pose)

    head_joint_cfg = get_head_joint_cfg(task_name="iros_pack_in_the_supermarket")

    # arm_r_base_link -> head_link2
    T_armr_to_head = transformer.relative_transform("arm_r_base_link", "head_link2", head_joint_cfg)
    T_head_to_armr = transformer.reverse_transform("arm_r_base_link", "head_link2", head_joint_cfg)

    # arm_l_base_link -> head_link2
    T_arml_to_head = transformer.relative_transform("arm_l_base_link", "head_link2", head_joint_cfg)
    T_head_to_arml = transformer.reverse_transform("arm_l_base_link", "head_link2", head_joint_cfg)

    print("arm_r_base_link -> head_link2:\n", T_armr_to_head)
    print("head_link2 -> arm_r_base_link:\n", T_head_to_armr)
    print("arm_l_base_link -> head_link2:\n", T_arml_to_head)
    print("head_link2 -> arm_l_base_link:\n", T_head_to_arml)

    # # Example point transformation
    # point_in_head = [0.1, 0.0, 0.0]
    # point_in_armr = transformer.transform_point(point_in_head, "head_link2", "arm_r_base_link", joint_cfg)
    # print("Point in head_link2:", point_in_head, "-> in arm_r_base_link:", point_in_armr)


    while rclpy.ok():
        img_h_raw = sim_ros_node.get_img_head()
        img_l_raw = sim_ros_node.get_img_left_wrist()
        img_r_raw = sim_ros_node.get_img_right_wrist()
        act_raw = sim_ros_node.get_joint_state()
        
        if (
            img_h_raw
            and img_l_raw
            and img_r_raw
            and act_raw
            and img_h_raw.header.stamp
            == img_l_raw.header.stamp
            == img_r_raw.header.stamp
        ):
            sim_time = get_sim_time(sim_ros_node)
            if sim_time > SIM_INIT_TIME:
                count += 1
                
                img_h = bridge.compressed_imgmsg_to_cv2(
                    img_h_raw, desired_encoding="rgb8"
                )
                img_l = bridge.compressed_imgmsg_to_cv2(
                    img_l_raw, desired_encoding="rgb8"
                )
                img_r = bridge.compressed_imgmsg_to_cv2(
                    img_r_raw, desired_encoding="rgb8"
                )

                # # save images if needed for debugging
                # cv2.imwrite(f"{current_path}/img_h_{count}.jpg", img_h)
                # cv2.imwrite(f"{current_path}/img_l_{count}.jpg", img_l)
                # cv2.imwrite(f"{current_path}/img_r_{count}.jpg", img_r)
                
                # state = np.array(act_raw.position)
                state = None

                input_processor = VLAInputProcessor()
                curr_task_substep_index=0
                input = input_processor.process(
                    img_h, img_l, img_r, lang, state, curr_task_substep_index
                )
                # obs = get_observations(img_h, img_l, img_r, lang, state)
                # if cfg.with_proprio:
                #     action = policy.step(img_h, img_l, img_r, lang, state)
                # else:
                # print(f"instruction: {input["task_description"]}")
                action = policy.step(input["image_list"], input["task_description"], input["robot_state"], verbose=True )
                
                if action:
                    task_substep_progress = _action_task_substep_progress(action)
                    if task_substep_progress[0][0] > 0.95:
                        curr_task_substep_index += 1
                        input_processor.curr_task_substep_index = curr_task_substep_index
                        print(f"Task substep index updated to: {curr_task_substep_index}")
                        
                # send command from model to sim
                # sim_ros_node.publish_joint_command(action)
                sim_ros_node.loop_rate.sleep()

def _action_task_substep_progress(action_raw):
    """
    Get the task substep progress from the action raw.
    """
    return action_raw["PROGRESS"] # shape: []
    
def get_observations(img_h, img_l, img_r, lang, joint_positions):
    """
    Prepare observations for the policy step.
    
    Args:
        img_h: Head image
        img_l: Left wrist image
        img_r: Right wrist image
        lang: Language instruction
        joint_positions: Joint positions of the robot
        
    Returns:
        A dictionary containing the observations.
    """
    instruction = "Pick up the red can."
    image_path = "franka_arm.jpeg"
    image = Image.open(image_path)
    translation = np.zeros((3,), dtype=np.float32)
    rotation = np.zeros((3,), dtype=np.float32)
    gripper = np.zeros((1,), dtype=np.float32)
    robot_state = {
                "ROBOT_LEFT_TRANS": translation.tolist(),
                "ROBOT_LEFT_ROT_EULER": rotation.tolist(),
                "ROBOT_LEFT_GRIPPER": gripper.tolist(),
                "ROBOT_RIGHT_TRANS": translation.tolist(),
                "ROBOT_RIGHT_ROT_EULER": rotation.tolist(),
                "ROBOT_RIGHT_GRIPPER": gripper.tolist(),
    }

    dummy_obs = {
        "img_list": [image],
        "task_description": instruction,
        "robot_state": robot_state,
        "image_format": "JPEG",
    }
    
    return dummy_obs
    
def get_policy():
    PORT=15020 # 40k steps
    ip = "10.190.172.212"
    policy = CogActAPIPolicy(ip_address=ip, port=PORT)  # Adjust IP and port as needed
    return policy  # Placeholder for actual policy loading logic

if __name__ == "__main__":

    policy = get_policy()
    infer(policy)
