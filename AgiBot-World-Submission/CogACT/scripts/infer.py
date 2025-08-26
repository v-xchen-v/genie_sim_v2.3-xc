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
import signal
import atexit
import logging
from datetime import datetime

# Global variables for cleanup
video_writer_global = None
video_segment_counter_global = 0
task_log_dir_global = None
task_name_global = None

def setup_logging(log_dir, task_name, enable_file_logging=True):
    """
    Set up logging to output to both console and file.
    
    Args:
        log_dir: Directory to save log files
        task_name: Name of the task (used in log filename)
        enable_file_logging: Whether to enable file logging (default: True)
    
    Returns:
        str: Path to the log file if file logging is enabled, None otherwise
    """
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    log_file_path = None
    if enable_file_logging:
        # File handler (optional)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"{task_name}_inference_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"📝 Logging to file: {log_file_path}")
    else:
        logger.info("📝 File logging disabled - console output only")
    
    return log_file_path

class LogCapture:
    """Capture print statements and redirect them to logging"""
    
    def __init__(self, logger):
        self.logger = logger
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def write(self, message):
        # Filter out empty messages and newlines
        message = message.strip()
        if message:
            self.logger.info(message)
        # Also write to original stdout for immediate console output
        self.original_stdout.write(message + '\n' if message else '\n')
        self.original_stdout.flush()
    
    def flush(self):
        self.original_stdout.flush()

def redirect_print_to_logging(logger):
    """Redirect print statements to logging while keeping console output"""
    log_capture = LogCapture(logger)
    return log_capture

# Initialize ee_to_joint_processor at module level
ee_to_joint_processor = EEtoJointProcessor()
input_processor = VLAInputProcessor(log_obs=False, resize_mode="1x1")  # "4x3_pad_resize" or "1x1", if is a aug model use "1x1", else use "4x3_pad_resize"

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

def get_task_progression_config():
    """Get task-specific configuration for substep progression."""
    return {
        "progress_thresholds": {
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
        },
        "min_inference_counters": {
            "iros_pack_in_the_supermarket": 12,  # 1-8 steps
            "iros_restock_supermarket_items": 5,  # 1-16 steps
            "iros_stamp_the_seal": 10,
            "iros_clear_the_countertop_waste": 6,
            "iros_clear_table_in_the_restaurant": 10,
            "iros_heat_the_food_in_the_microwave": 40,
            "iros_open_drawer_and_store_items": 32,
            "iros_pack_moving_objects_from_conveyor": 10, # steps: 4, 8, 12, 16, 6 is enough for pickup directly but not enough for failed and retry
            "iros_pickup_items_from_the_freezer": 24,
            "iros_make_a_sandwich": 12,
        },
        "max_inference_counters": {
            "iros_pack_in_the_supermarket": 48,  # 1-8 steps
            "iros_heat_the_food_in_the_microwave": 40,  # 1-8 steps
        }
    }

def check_progress_based_advancement(task_substep_progress, task_name, substep_inference_counter, config):
    """Strategy 3: Check if substep should advance based on progress threshold OR max counter."""
    progress_threshold = 0.95  # High threshold for reliable progress signal
    max_inference_counter = config["max_inference_counters"].get(task_name, 20)
    
    progress_list = np.array(task_substep_progress[0])
    current_progress = task_substep_progress[0][0]
    
    # Check advancement conditions
    counter_exceeded = substep_inference_counter >= max_inference_counter
    progress_exceeded = np.any(progress_list > progress_threshold)
    
    should_advance = counter_exceeded or progress_exceeded
    
    if should_advance:
        reason = "counter exceeded" if counter_exceeded else "progress exceeded"
        print(f"✅ ADVANCING (Strategy 3): {reason} - Progress ({current_progress:.3f}), Counter ({substep_inference_counter})")
    else:
        print(f"❌ NOT ADVANCING (Strategy 3): Progress ({current_progress:.3f} <= {progress_threshold}), Counter ({substep_inference_counter} < {max_inference_counter})")
    
    return should_advance

def check_restrict_progress_advancement(task_substep_progress, task_name, substep_inference_counter, config):
    """Strategy 2: Strictly follow progress signal only - advance when progress > threshold."""
    progress_threshold = 0.95  # High threshold for reliable progress signal
    
    progress_list = np.array(task_substep_progress[0])
    current_progress = task_substep_progress[0][0]
    
    # Only advance based on progress, ignore counter
    progress_exceeded = np.any(progress_list > progress_threshold)
    should_advance = progress_exceeded
    
    if should_advance:
        print(f"✅ ADVANCING (Strategy 2): Progress exceeded - Progress ({current_progress:.3f} > {progress_threshold})")
    else:
        print(f"❌ NOT ADVANCING (Strategy 2): Progress not met - Progress ({current_progress:.3f} <= {progress_threshold})")
    
    return should_advance

def check_legacy_advancement(task_substep_progress, task_name, substep_inference_counter, config):
    """Strategy 1: Check if substep should advance based on legacy logic (progress AND counter thresholds)."""
    progress_threshold = config["progress_thresholds"].get(task_name, 0.4)
    min_inference_counter = config["min_inference_counters"].get(task_name, 6)
    
    current_progress = task_substep_progress[0][0]
    progress_met = current_progress > progress_threshold
    counter_met = substep_inference_counter >= min_inference_counter
    
    should_advance = progress_met and counter_met
    
    if should_advance:
        print(f"✅ ADVANCING (Strategy 1): Progress ({current_progress:.3f} > {progress_threshold}) AND Counter ({substep_inference_counter} >= {min_inference_counter})")
    else:
        progress_ok = "✅" if progress_met else "❌"
        counter_ok = "✅" if counter_met else "❌"
        print(f"STAYING (Strategy 1): Progress {progress_ok} ({current_progress:.3f} > {progress_threshold}) AND Counter {counter_ok} ({substep_inference_counter} >= {min_inference_counter})")
    
    return should_advance

def check_restrict_inference_count_advancement(task_substep_progress, task_name, substep_inference_counter, config):
    """Strategy 4: Advance only when inference counter exceeds minimum threshold (ignore progress)."""
    min_inference_counter = config["min_inference_counters"].get(task_name, 6)
    current_progress = task_substep_progress[0][0]
    
    # Only advance based on counter, ignore progress completely
    counter_exceeded = substep_inference_counter >= min_inference_counter
    should_advance = counter_exceeded
    
    if should_advance:
        print(f"✅ ADVANCING (Strategy 4): Counter exceeded - Counter ({substep_inference_counter} >= {min_inference_counter}), Progress ({current_progress:.3f})")
    else:
        print(f"❌ NOT ADVANCING (Strategy 4): Counter not met - Counter ({substep_inference_counter} < {min_inference_counter}), Progress ({current_progress:.3f})")
    
    return should_advance

def handle_substep_progression(action, task_name, curr_task_substep_index, substep_inference_counter, model_input, mode="by_progress"):
    """
    Handle substep progression logic based on task progress and inference counter.
    
    Args:
        action: Action containing progress information
        task_name: Name of the current task
        curr_task_substep_index: Current substep index
        substep_inference_counter: Current inference counter for this substep
        model_input: Model input containing task description
        mode: Strategy selection:
            - "legacy": Strategy 1 - Advance when progress > low_threshold AND counter >= min_counter
            - "restrict_progress": Strategy 2 - Advance only when progress > high_threshold (strict progress following)
            - "by_progress": Strategy 3 - Advance when progress > high_threshold OR counter >= max_counter
            - "restrict_inference_count": Strategy 4 - Advance only when counter >= min_counter (ignore progress)
    
    Returns:
        tuple: (new_curr_task_substep_index, new_substep_inference_counter)
    """
    substep_inference_counter += 1
    task_substep_progress = _action_task_substep_progress(action)
    
    # Log current state
    print(f"------------Task substep progress: {task_substep_progress[0][0]}------------")
    print(f"Substep: {curr_task_substep_index}, Inference Counter: {substep_inference_counter}, Mode: {mode}")
    print(f"Instruction: {model_input['task_description']}")
    
    # Get task configuration
    config = get_task_progression_config()
    
    # Determine if we should advance based on the selected strategy
    if mode == "legacy":
        should_advance = check_legacy_advancement(task_substep_progress, task_name, substep_inference_counter, config)
    elif mode == "restrict_progress":
        should_advance = check_restrict_progress_advancement(task_substep_progress, task_name, substep_inference_counter, config)
    elif mode == "by_progress":
        should_advance = check_progress_based_advancement(task_substep_progress, task_name, substep_inference_counter, config)
    elif mode == "restrict_inference_count":
        should_advance = check_restrict_inference_count_advancement(task_substep_progress, task_name, substep_inference_counter, config)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'legacy', 'restrict_progress', 'by_progress', or 'restrict_inference_count'")
    
    # hard-code seting for pack_moving_objects_from_conveyor, since current the model is not very good at progress prediction, when the model is ready,
    # remove this hard-code
    if task_name == "iros_pack_moving_objects_from_conveyor":
        mode = "restrict_inference_count"
        should_advance = check_restrict_inference_count_advancement(task_substep_progress, task_name, substep_inference_counter, config)

    # Update indices if advancing
    if should_advance:
        curr_task_substep_index += 1
        substep_inference_counter = 0  # Reset counter for new substep
    
    return curr_task_substep_index, substep_inference_counter

def _get_unique_log_dir(base_dir, task_name):
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


def infer(policy, task_name, enable_video_recording=False, enable_file_logging=True):
    """
    Main inference loop for robot task execution.
    
    Features:
    - Records initial joint angles at startup
    - Executes task substeps with configurable progression strategies
    - Automatically returns to initial pose when task sequence completes and loops back
    - Supports continuous task repetition
    - Saves video in multiple segments for robustness against unexpected exits (optional)
    - Logs output to both console and file (optional)
    
    Args:
        policy: The policy object for inference
        task_name: Name of the task to execute
        enable_video_recording: Whether to record video during inference (default: False)
        enable_file_logging: Whether to save logs to file (default: True)
    """
    global video_writer_global, video_segment_counter_global, task_log_dir_global, task_name_global
    
    
    rclpy.init()
    current_path = os.getcwd()
    sim_ros_node = SimROSNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
    spin_thread.start()
    
    bridge = CvBridge()
    count = 0
    SIM_INIT_TIME = 8

    # Set up logging directory (shared with video if enabled)
    log_dir = "./inference_logs" if not enable_video_recording else "./video_recordings"
    if enable_video_recording:
        task_log_dir = _get_unique_log_dir(log_dir, task_name)
        log_file_path = setup_logging(task_log_dir, task_name, enable_file_logging)
    else:
        os.makedirs(log_dir, exist_ok=True)
        task_log_dir = _get_unique_log_dir(log_dir, task_name)
        log_file_path = setup_logging(task_log_dir, task_name, enable_file_logging)
    
    # Get logger instance
    logger = logging.getLogger()

    # Use the passed task_name parameter instead of hardcoded value
    logger.info(f"🚀 Starting inference for task: {task_name}")
    if enable_file_logging:
        logger.info(f"📁 Log directory: {task_log_dir}")
        logger.info(f"📝 Log file: {log_file_path}")
    
    lang = get_instruction(task_name=task_name)
    curr_task_substep_index = 0
    head_joint_cfg = get_head_joint_cfg(task_name=task_name)
    
    # Counter for inference steps in current substep
    substep_inference_counter = 0
    
    # Variables to store initial joint angles and track initialization
    initial_joint_angles = None
    is_initialized = False
    
    # Get total number of substeps for loop-back detection
    total_substeps = get_num_substeps(task_name)
    logger.info(f"📊 Task '{task_name}' has {total_substeps} substeps")
    
    # Track whether we've returned to initial pose for this cycle
    returned_to_initial_this_cycle = False

    # Initialize video recording variables (only if enabled)
    video_writer = None
    video_segment_counter = 0
    
    if enable_video_recording:
        # Set global variables for cleanup
        task_log_dir_global = task_log_dir
        task_name_global = task_name

        import imageio
        video_segment_counter_global = video_segment_counter
        VIDEO_FPS = 2 
        video_writer = imageio.get_writer(os.path.join(task_log_dir, f"{task_name}_inference_segment_{video_segment_counter:03d}.mp4"), fps=VIDEO_FPS)
        video_writer_global = video_writer
        logger.info(f"🎥 Recording video to: {os.path.join(task_log_dir, f'{task_name}_inference_segment_{video_segment_counter:03d}.mp4')}")
    else:
        logger.info("📵 Video recording disabled")

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
                # img_h_pil = Image.fromarray(img_h)
                # img_l_pil = Image.fromarray(img_l)
                # img_r_pil = Image.fromarray(img_r)

                # Process and record video if enabled
                if enable_video_recording and video_writer:
                    # Process image to same height before combining
                    target_height = 224
                    img_h_render = cv2.resize(img_h, (int(img_h.shape[1] * target_height / img_h.shape[0]), target_height))
                    img_l_render = cv2.resize(img_l, (int(img_l.shape[1] * target_height / img_l.shape[0]), target_height))
                    img_r_render = cv2.resize(img_r, (int(img_r.shape[1] * target_height / img_r.shape[0]), target_height))

                    # Combine images side by side for video
                    combined_img = np.hstack((img_l_render, img_h_render, img_r_render))
                    # rgb to bgr for opencv
                    combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
                    video_writer.append_data(combined_img)

                    # save the video before exit
                    SAVE_VIDEO_EVERY_N_INFERENCE = 10
                    if count % SAVE_VIDEO_EVERY_N_INFERENCE == 0 and count > 0:
                        video_writer.close()
                        video_segment_counter += 1
                        video_segment_counter_global = video_segment_counter
                        VIDEO_FPS = 2  # Define VIDEO_FPS here if not already defined
                        video_writer = imageio.get_writer(os.path.join(task_log_dir, f"{task_name}_inference_segment_{video_segment_counter:03d}.mp4"), fps=VIDEO_FPS)
                        video_writer_global = video_writer
                        logger.info(f"🎥 Saved intermediate video segment {video_segment_counter-1:03d} at count {count}, starting segment {video_segment_counter:03d}")
                
                state = np.array(act_raw.position)
                # state = None # if use model without state

                if act_raw.position is not None and len(act_raw.position) == 0:
                    logger.warning("No joint state received, skipping iteration.")
                    continue

                # Record initial joint angles on first valid iteration
                if not is_initialized and act_raw.position is not None and len(act_raw.position) > 0:
                    initial_joint_angles = np.array(act_raw.position).copy()
                    is_initialized = True
                    logger.info(f"📝 Recorded initial joint angles: {initial_joint_angles}")

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
                    # Handle substep progression logic
                    curr_task_substep_index, substep_inference_counter = handle_substep_progression(
                        action, task_name, curr_task_substep_index, substep_inference_counter, model_input, mode="by_progress"
                    )
                    
                    # Check if we've completed all substeps and looped back to first instruction
                    # This happens when we advance from the last substep (total_substeps - 1) to substep 0
                    logger.debug(f"curr_task_substep_index: {curr_task_substep_index}, total_substeps: {total_substeps}")
                    if (curr_task_substep_index > 0 and curr_task_substep_index % total_substeps == 0 and 
                        is_initialized and initial_joint_angles is not None and not returned_to_initial_this_cycle):
                        logger.info("🎉 Task sequence completed! Moving back to initial pose...")
                        
                        # Move to initial pose using interpolation
                        current_joints = np.array(act_raw.position)
                        target_joints = initial_joint_angles
                        
                        # Use more steps for returning to initial pose to ensure smooth movement
                        num_steps = 10
                        interpolated_steps = interpolate_joints(current_joints, target_joints, num_steps=num_steps)
                        
                        # Send interpolated joint commands to return to initial pose
                        for step_idx, interp_joints in enumerate(interpolated_steps):
                            sim_ros_node.publish_joint_command(interp_joints)
                            logger.debug(f"🏠 Moving to initial pose - Step {step_idx + 1}/{num_steps}")
                            sim_ros_node.loop_rate.sleep()
                        
                        logger.info("✅ Returned to initial pose! Starting new task cycle...")
                        returned_to_initial_this_cycle = True
                        
                        # Update act_raw to reflect the new position
                        act_raw = sim_ros_node.get_joint_state()
                    
                    # Reset the flag when we start a new cycle (advance from substep 0)
                    if curr_task_substep_index % total_substeps == 1:
                        returned_to_initial_this_cycle = False
                    
                joint_cmd = ee_to_joint_processor.get_joint_cmd(action, head_joint_cfg, curr_arm_joint_angles=act_raw.position, task_name=task_name)
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
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                elif task_name == "iros_make_a_sandwich":
                    # execution_steps = [0, 1, 2, 3]
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                elif task_name == "iros_clear_the_countertop_waste":
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                elif task_name == "iros_heat_the_food_in_the_microwave":
                    # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    # execution_steps = [0, 1, 2, 3]
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7]
                elif task_name == "iros_open_drawer_and_store_items":
                    # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    execution_steps = [0, 1, 2, 3]
                elif task_name == "iros_pack_moving_objects_from_conveyor":
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

                    if curr_task_substep_index % total_substeps == 0: # must be fast to pick up object from conveyor, so that use less steps
                        execution_steps = execution_steps[::4]  # Take every 4th step for execution
                    else:
                        execution_steps = execution_steps[:8]  # Use first 8 steps for placing into box

                elif task_name == "iros_pickup_items_from_the_freezer":
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    # execution_steps = [0, 1, 2, 3]
                elif task_name == "iros_restock_supermarket_items":
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                elif task_name == "iros_clear_table_in_the_restaurant":
                    execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                else:
                    logger.warning(f"Task {task_name} not recognized, using default execution steps.")

                for step_index in execution_steps:
                    num_ik_iterations = 1
                    delta_joint_angles = joint_cmd[(step_index+1)*num_ik_iterations-1] - act_raw.position
                    # print(f"Delta joint angles for step {step_index}: \n")
                    # print(f"Delta left arm joint angles: {delta_joint_angles[:7]}\n")
                    # print(f"Delta right arm joint angles: {delta_joint_angles[8:15]}\n")
                    # print(f"Delta left gripper joint angles: {delta_joint_angles[7]}\n")
                    # print(f"Delta right gripper joint angles: {delta_joint_angles[15]}\n")
                    

                    # Convert delta joint angles to joint state message
                    for i in range(num_ik_iterations):
                        joint_arr = joint_cmd[step_index * num_ik_iterations + i]
                        if task_name == "iros_pack_moving_objects_from_conveyor" or task_name == "iros_restock_supermarket_items" \
                            or task_name == "iros_make_a_sandwich":
                            # drop during lifting, more tight grasp is need
                            joint_arr[7] *= 1.5
                            joint_arr[15] *= 1.5
                        
                        # Interpolate between current joint positions and target joint positions
                        act_raw = sim_ros_node.get_joint_state()
                        current_joints = act_raw.position  # [16,]
                        target_joints = joint_arr          # [16,]
                        if task_name == "iros_clear_countertop_waste" or \
                            (task_name == "iros_pack_moving_objects_from_conveyor" and curr_task_substep_index%total_substeps==0) \
                            or task_name == "iros_make_a_sandwich" :
                            # or task_name =="iros_clear_table_in_the_restaurant":
                            num_steps = 1
                        else:
                            num_steps = 2
                        interpolated_steps = interpolate_joints(current_joints, target_joints, num_steps=num_steps)
                        # Send interpolated joint commands
                        for interp_joints in interpolated_steps:
                            sim_ros_node.publish_joint_command(interp_joints)
                            # print gripper joint angles in degrees for the final target
                            if interp_joints == interpolated_steps[-1]:
                                logger.debug(f"Step {step_index} - Left gripper joint angle: {np.rad2deg(interp_joints[7]):.1f}°, Right gripper joint angle: {np.rad2deg(interp_joints[15]):.1f}°")
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
    # # PORT=14020 
    # PORT=14030 # 08/15/2025 tested
    # PORT=15030 # predict 1 step
    # PORT=16030 # predict 3 step

    # # Split data by ADC timepoint and object z > threshold for pick
    # PORT=17020 # no aug, step 20k
    PORT=17030 # no aug, step 30k
    # PORT=17040 # no aug, step 40k
    # PORT=18020 # aug, step~20k


    ip = "10.190.172.212"
    policy = CogActAPIPolicy(ip_address=ip, port=PORT)  # Adjust IP and port as needed
    return policy  # Placeholder for actual policy loading logic

def interpolate_joints(current_joints, target_joints, num_steps=5):
    """
    Interpolate between current joint positions and target joint positions.
    
    Args:
        current_joints: Current joint positions [16,] or [16, 1]
        target_joints: Target joint positions [16,] or [16, 1] 
        num_steps: Number of interpolation steps (default: 5)
        
    Returns:
        List of interpolated joint arrays, each of shape [16,]
    """
    # Ensure inputs are numpy arrays and flatten to 1D
    current = np.array(current_joints).flatten()
    target = np.array(target_joints).flatten()
    
    if current.shape != target.shape:
        raise ValueError(f"Shape mismatch: current_joints {current.shape} vs target_joints {target.shape}")
    
    # Create interpolation steps (excluding start point, including end point)
    interpolated_joints = []
    for i in range(1, num_steps + 1):
        alpha = i / num_steps  # interpolation factor from 0 to 1
        interpolated = current + alpha * (target - current)

        # exclude the gripper joints from interpolation
        interpolated[7] = target[7]  # left gripper joint
        interpolated[15] = target[15]  # right gripper joint

        # Append the interpolated joint positions to the list
        interpolated_joints.append(interpolated.tolist())
    
    return interpolated_joints

def get_num_substeps(task_name):
    """Get the number of substeps for a given task."""
    lang = get_instruction(task_name)
    substeps = [action.strip() for action in lang.split(";") if action.strip()]
    return len(substeps)

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
    parser.add_argument('--enable_video_recording', action='store_true', default=False,
                        help='Enable video recording of inference images. Videos are saved as segments and merged on normal exit. (default: False)')
    parser.add_argument('--enable_file_logging', action='store_true', default=True,
                        help='Enable logging to file in addition to console output. (default: True)')
    parser.add_argument('--disable_file_logging', action='store_true', default=False,
                        help='Disable file logging - console output only. Overrides --enable_file_logging. (default: False)')
    
    args = parser.parse_args()

    # Handle logging flags (disable takes precedence)
    enable_file_logging = args.enable_file_logging and not args.disable_file_logging

    policy = get_policy()
    # policy = get_policy_wo_state()
    
    infer(policy, args.task_name, enable_video_recording=args.enable_video_recording, enable_file_logging=enable_file_logging)
