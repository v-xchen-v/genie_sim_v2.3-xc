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
from ee_pose_to_joint_processor import EEtoJointProcessor
from pathlib import Path

import time
import argparse

# Initialize ee_to_joint_processor at module level
ee_to_joint_processor = EEtoJointProcessor()
input_processor = VLAInputProcessor(log_obs=False)

def get_instruction(task_name):

    if task_name == "iros_clear_the_countertop_waste":
        lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "iros_restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
        # lang = "Pick up the grape juice on the table with the right arm.;Place the grape juice on the shelf where the grape juice is located with the right arm."
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

# def get_instruction_splites(task_name, substep_index=0):
#     full_instruction = get_instruction(task_name)
#     instructions = full_instruction.split(";")
#     if substep_index < len(instructions):
#         return instructions[substep_index]
#     else:
#         # the last substep is the final step, so return the last instruction
#         return instructions[-1]
    
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
            # "idx12_head_joint2": 0.43633231, # not consiste with GUI joint state and iros but it works, find it by seting task to pack_in_the supermarket but work well
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
            "idx12_head_joint2": 0.43633231,
            # "idx12_head_joint2": 0.3839745594177246, 
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

def infer(policy, task_name):
    
    
    rclpy.init()
    current_path = os.getcwd()
    sim_ros_node = SimROSNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
    spin_thread.start()
    
    bridge = CvBridge()
    count = 0
    SIM_INIT_TIME = 10

    # Use the passed task_name parameter instead of hardcoded value
    print(f"Running task: {task_name}")
    
    lang = get_instruction(task_name=task_name)
    curr_task_substep_index = 0
    head_joint_cfg = get_head_joint_cfg(task_name=task_name)
    
    # Counter for inference steps in current substep
    substep_inference_counter = 0

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
                
                
                state = np.array(act_raw.position)
                # state = None # if use model without state

                if act_raw.position is not None and len(act_raw.position) == 0:
                    print("No joint state received, skipping iteration.")
                    continue

                
                model_input = input_processor.process(
                    img_h, img_l, img_r, lang, state, curr_task_substep_index, head_joint_cfg=head_joint_cfg
                )
                # obs = get_observations(img_h, img_l, img_r, lang, state)
                # if cfg.with_proprio:
                #     action = policy.step(img_h, img_l, img_r, lang, state)
                # else:
                # print(f"instruction: {input["task_description"]}")
                action = policy.step(model_input["image_list"], model_input["task_description"], model_input["robot_state"], verbose=False)
                
                if action:
                    substep_inference_counter += 1
                    task_substep_progress = _action_task_substep_progress(action)
                    print(f"------------Task substep progress: {task_substep_progress[0][0]}------------")
                    print(f"Substep: {curr_task_substep_index}, Inference Counter: {substep_inference_counter}, Instruction: {model_input['task_description']}")
                    
                    # New strategy: Combined progress threshold and inference counter
                    # Define minimum progress threshold per task
                    progress_threshold_dict = {
                        "iros_pack_in_the_supermarket": 0.3,
                        "iros_restock_supermarket_items": 0.9,
                        "iros_stamp_the_seal": 0.99,
                        "iros_clear_the_countertop_waste": 0.4,
                        "iros_clear_table_in_the_restaurant": 0.6,
                        "iros_heat_the_food_in_the_microwave": 0.6,
                        "iros_open_drawer_and_store_items": 0.4,
                        "iros_pack_moving_objects_from_conveyor": 0.4,
                        "iros_pickup_items_from_the_freezer": 0.4,
                        "iros_make_a_sandwich": 0.9,
                    }
                    
                    # Define minimum inference counter per task
                    min_inference_counter_dict = {
                        # "iros_pack_in_the_supermarket": 20,
                        "iros_pack_in_the_supermarket": 12, # 1-8 steps
                        "iros_restock_supermarket_items": 5, # 1-16 steps
                        "iros_stamp_the_seal": 10,
                        "iros_clear_the_countertop_waste": 6,
                        "iros_clear_table_in_the_restaurant": 10,
                        "iros_heat_the_food_in_the_microwave": 40,
                        "iros_open_drawer_and_store_items": 32,
                        "iros_pack_moving_objects_from_conveyor": 6,
                        "iros_pickup_items_from_the_freezer": 24,
                        "iros_make_a_sandwich": 12,
                    }
                    
                    # Get thresholds for current task
                    progress_threshold = progress_threshold_dict.get(task_name, 0.4)
                    min_inference_counter = min_inference_counter_dict.get(task_name, 6)
                    
                    # Check both conditions: progress above threshold AND counter above minimum
                    if ((task_substep_progress[0][0] > progress_threshold and 
                        substep_inference_counter >= min_inference_counter)):
                        # or (task_substep_progress[0][0] > progress_threshold and progress_threshold > 0.95):
                        curr_task_substep_index += 1
                        substep_inference_counter = 0  # Reset counter for new substep
                        print(f"✅ ADVANCING: Progress ({task_substep_progress[0][0]:.3f} > {progress_threshold}) AND Counter ({substep_inference_counter} >= {min_inference_counter})")
                        print(f"Task substep index updated to: {curr_task_substep_index}")
                    else:
                        progress_ok = "✅" if task_substep_progress[0][0] > progress_threshold else "❌"
                        counter_ok = "✅" if substep_inference_counter >= min_inference_counter else "❌"
                        print(f"STAYING: Progress {progress_ok} ({task_substep_progress[0][0]:.3f} > {progress_threshold}) AND Counter {counter_ok} ({substep_inference_counter} >= {min_inference_counter})")
                    
                    # option 2: switch task substep by user input (manual control), temporary approach when progress is not reliable
                    # print("Do you want to advance to the next substep? (yes/no): ", end="", flush=True)
                    # user_input = input().strip().lower()
                    # if user_input == "yes" or user_input == "y":
                    #     curr_task_substep_index += 1
                    #     print(f"Task substep index updated to: {curr_task_substep_index}")
                    # else:
                    #     print(f"Continuing with current substep: {curr_task_substep_index}")
                    joint_cmd = ee_to_joint_processor.get_joint_cmd(action, head_joint_cfg, curr_arm_joint_angles=act_raw.position)
                # print(f"Joint command shape: {joint_cmd.shape}, Joint command: {joint_cmd}")


                # send command from model to sim
                execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # default
                # execution_steps = [0, 1, 2, 3]
                # execution_steps = [0]
                # execution_steps = [0, 1]
                if task_name == "iros_stamp_the_seal":
                    # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                elif task_name == "iros_pack_in_the_supermarket":
                    # execution_steps = [0, 1, 2, 3]
                    # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7]
                elif task_name == "iros_make_a_sandwich":
                    # execution_steps = [0, 1, 2, 3]
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                elif task_name == "iros_clear_the_countertop_waste":
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                elif task_name == "iros_heat_the_food_in_the_microwave":
                    # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    execution_steps = [0, 1, 2, 3]
                elif task_name == "iros_open_drawer_and_store_items":
                    # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    execution_steps = [0, 1, 2, 3]
                elif task_name == "iros_pack_moving_objects_from_conveyor":
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    execution_steps = execution_steps[::4]  # Take every 4th step for execution
                elif task_name == "iros_pickup_items_from_the_freezer":
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    # execution_steps = [0, 1, 2, 3]
                elif task_name == "iros_restock_supermarket_items":
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                elif task_name == "iros_clear_table_in_the_restaurant":
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                else:
                    print(f"Task {task_name} not recognized, using default execution steps.")

                for step_index in execution_steps:
                    num_ik_iterations = 1
                    delta_joint_angles = joint_cmd[(step_index+1)*num_ik_iterations-1] - act_raw.position
                    # print(f"Delta joint angles for step {step_index}: \n")
                    # print(f"Delta left arm joint angles: {delta_joint_angles[:7]}\n")
                    # print(f"Delta right arm joint angles: {delta_joint_angles[8:15]}\n")
                    # print(f"Delta left gripper joint angles: {delta_joint_angles[7]}\n")
                    # print(f"Delta right gripper joint angles: {delta_joint_angles[15]}\n")
                    
                    # print gripper joint angles in degrees
                    print(f"Step {step_index} - Left gripper joint angle: {np.rad2deg(delta_joint_angles[7])}, Right gripper joint angle: {np.rad2deg(delta_joint_angles[15])}")

                    # Convert delta joint angles to joint state message
                    for i in range(num_ik_iterations):
                        joint_arr = joint_cmd[step_index * num_ik_iterations + i].tolist()
                        if task_name == "iros_pack_moving_objects_from_conveyor":
                            # drop during lifting, more tight grasp is need
                            joint_arr[7] *= 1.5
                            joint_arr[15] *= 1.5
                        sim_ros_node.publish_joint_command(
                            joint_arr
                        )
                        # sim_ros_node.publish_joint_command(joint_cmd[step_index])
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
    
def get_policy_wo_state():
    PORT=15020 
    ip = "10.190.172.212"
    policy = CogActAPIPolicy(ip_address=ip, port=PORT)  # Adjust IP and port as needed
    return policy  # Placeholder for actual policy loading logic

def get_policy():
    # PORT=14020 
    PORT=14030
    ip = "10.190.172.212"
    policy = CogActAPIPolicy(ip_address=ip, port=PORT)  # Adjust IP and port as needed
    return policy  # Placeholder for actual policy loading logic

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run robot inference with specified task')
    parser.add_argument('--task_name', type=str, required=True,
                        choices=[
                            "iros_pack_in_the_supermarket",
                            "iros_restock_supermarket_items", 
                            "iros_stamp_the_seal",
                            "iros_make_a_sandwich",
                            "iros_clear_the_countertop_waste",
                            "iros_clear_table_in_the_restaurant",
                            "iros_heat_the_food_in_the_microwave",
                            "iros_open_drawer_and_store_items",
                            "iros_pack_moving_objects_from_conveyor",
                            "iros_pickup_items_from_the_freezer"
                        ],
                        help='Name of the task to run')
    
    args = parser.parse_args()

    policy = get_policy()
    # policy = get_policy_wo_state()
    
    infer(policy, args.task_name)
