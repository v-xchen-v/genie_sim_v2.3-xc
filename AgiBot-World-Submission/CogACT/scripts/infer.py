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
from cogact_policy import CogActPolicy
from config_loader import InferenceConfig
from vlainputprocessor import VLAInputProcessor
from kinematics.urdf_coordinate_transformer import URDFCoordinateTransformer
from kinematics.g1_relax_ik import G1RelaxSolver
from ee_pose_to_joint_processor import EEtoJointProcessor
from config_loader import get_config
from video_utils import save_inference_images, save_joint_step_images, VideoRecordingManager
from task_substep_processor import (
    get_full_instruction, 
    get_num_substeps, 
    handle_substep_progression
)
from sensor_msgs.msg import (
    CompressedImage,
)

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

# Load configuration
config = get_config()

# def get_instruction_splites(task_name, substep_index=0):
#     full_instruction = get_instruction(task_name)
#     instructions = full_instruction.split(";")
#     if substep_index < len(instructions):
#         return instructions[substep_index]
#     else:
#         # the last substep is the final step, so return the last instruction
#         return instructions[-1]
    
def get_sim_time(sim_ros_node):
    sim_time = sim_ros_node.get_clock().now().nanoseconds * 1e-9
    return sim_time

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
        
        
def decode_depth_image(msg: CompressedImage) -> np.ndarray:
    # Convert CompressedImage -> OpenCV image
    # cv2.imdecode returns a uint8/uint16/float32 array depending on encoding
    np_arr = np.frombuffer(msg.data, np.uint8)
    depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    # Now depth_img is a numpy array (H x W)
    # - If Isaac Sim compressed it as PNG16 → dtype=uint16
    # - If it used PNG32F → dtype=float32
    print(depth_img.shape, depth_img.dtype)

    # # Convert to meters if needed
    # if depth_img.dtype == np.uint16:
    #     # often stored in millimeters
    #     depth_m = depth_img.astype(np.float32)
    # elif depth_img.dtype == np.float32:
    #     depth_m = depth_img
    # else:
    #     raise ValueError(f"Unexpected depth dtype: {depth_img.dtype}")
    return depth_img # depth_m is uint16 in meters
            
def depth_to_rgb_uint8(depth_img: np.ndarray, min_val: int, max_val: int, cmap_name: str = "turbo") -> np.ndarray:
    import matplotlib
    """
    将16位深度图转化为伪彩色RGB图 (uint8, 每通道0~255)
    Args:
        depth_img (np.ndarray): 输入深度图 (uint16)
        min_val (int): 深度最小值 (归一化下界)
        max_val (int): 深度最大值 (归一化上界)
        cmap_name (str): 颜色映射 (matplotlib colormap)，如 "turbo", "viridis"
    Returns:
        rgb_img (np.ndarray): 伪彩色RGB图 (uint8, HxWx3)
    """
    # 转 float32 方便归一化
    depth = depth_img.astype(np.float32)
 
    # 归一化到 [0,1]
    depth_norm = (depth - min_val) / (max_val - min_val)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
 
    # colormap 映射 → float32 [0,1]
    # cmap = cm.get_cmap(cmap_name)
    cmap = matplotlib.colormaps[cmap_name]
    rgb_float = cmap(depth_norm)[..., :3]  # RGBA → RGB
 
    # 转成 uint8 (0~255)
    rgb_img = (rgb_float * 255).astype(np.uint8)
 
    return rgb_img

def infer(policy, task_name, enable_video_recording=False, enable_file_logging=True):
    """
    Main inference loop for robot task execution.
    
    Features:
    - Records initial joint angles at startup
    - Executes task substeps with configurable progression strategies
    - Automatically returns to initial pose when task sequence completes and loops back
    - Supports continuous task repetition
    - Saves individual images during inference for debugging and analysis (optional)
    - Logs output to both console and file (optional)
    
    Args:
        policy: The policy object for inference
        task_name: Name of the task to execute
        enable_video_recording: Whether to save individual images during inference (default: False)
        enable_file_logging: Whether to save logs to file (default: True)
    """
    global video_writer_global, video_segment_counter_global, task_log_dir_global, task_name_global
    

        # Get logger instance
    logger = logging.getLogger()

    # Get coordinate mode from policy configuration
    coord_mode = config.get_coordinate_mode()
    image_strategy = config.get_image_strategy()
    if task_name == "iros_pack_moving_objects_from_conveyor":
        image_strategy = "rgb_only" # for fast moving task, rgb only is better
    pose_strategy = config.get_pose_strategy(task_name)
    logger.info(f"🔧 Using coordinate mode: {coord_mode}")
    logger.info(f"🖼️ Using image strategy: {image_strategy}")
    logger.info(f"📐 Using pose strategy: {pose_strategy}")

    # Initialize ee_to_joint_processor at module level with coordinate mode and pose strategy
    ee_to_joint_processor = EEtoJointProcessor(logger=logger, coord_mode=coord_mode, pose_strategy=pose_strategy)
    input_processor = VLAInputProcessor(logger=logger, log_obs=False, resize_mode=config.resize_mode, coord_mode=coord_mode, image_strategy=image_strategy)  # "4x3_pad_resize" or "1x1", if is a aug model use "1x1", else use "4x3_pad_resize"
    
    rclpy.init()
    current_path = os.getcwd()
    
    # Get ROS configuration from config
    ros_loop_rate = config.get_task_ros_loop_rate(task_name)
    
    # Initialize SimROSNode with configuration
    sim_ros_node = SimROSNode(
        loop_rate=ros_loop_rate,
    )
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
    spin_thread.start()
    
    bridge = CvBridge()
    count = 0
    SIM_INIT_TIME = 8
        
    # Set up logging directory (shared with image saving if enabled)
    log_dir = "./inference_logs" if not enable_video_recording else "./video_recordings"
    if enable_video_recording:
        task_log_dir = _get_unique_log_dir(log_dir, task_name)
        log_file_path = setup_logging(task_log_dir, task_name, enable_file_logging)
    else:
        os.makedirs(log_dir, exist_ok=True)
        task_log_dir = _get_unique_log_dir(log_dir, task_name)
        log_file_path = setup_logging(task_log_dir, task_name, enable_file_logging)
    

    
    
    # Copy configuration file to task directory for reproducibility
    import shutil
    config_dest_path = os.path.join(task_log_dir, 'inference_config.yaml')
    try:
        config.save_config_copy(config_dest_path)
        logger.info(f"📋 Configuration copied to: {config_dest_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to copy configuration file: {e}")
    
    # Log ROS configuration
    logger.info(f"🤖 ROS Configuration:")
    logger.info(f"   └─ Loop rate: {ros_loop_rate} Hz")
    if ros_loop_rate != config.ros_loop_rate:
        logger.info(f"   └─ Using task-specific loop rate (default: {config.ros_loop_rate} Hz)")
    
    # Save configuration summary as JSON
    config_summary_path = os.path.join(task_log_dir, 'inference_config_summary.json')
    try:
        config.save_config_summary(config_summary_path)
        logger.info(f"📊 Configuration summary saved to: {config_summary_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save configuration summary: {e}")
    
    # Update SIM_INIT_TIME from config
    SIM_INIT_TIME = 8
    
    lang = get_full_instruction(task_name=task_name)
    curr_task_substep_index = 0
    head_joint_cfg = config.get_head_joint_cfg(task_name=task_name)
    
    # Counter for inference steps in current substep
    substep_inference_counter = 0
    
    # Variables to store initial joint angles and track initialization
    initial_joint_angles = None
    is_initialized = False
    
    # Get total number of substeps for loop-back detection
    total_substeps = get_num_substeps(task_name)
    logger.info(f"📊 Task '{task_name}' has {total_substeps} substeps")
    
    # Log image strategy configuration
    if image_strategy == "rgb_depth":
        depth_cameras = config.get_depth_cameras_config()
        enabled_cameras = [cam for cam, enabled in depth_cameras.items() if enabled]
        logger.info(f"📷 Depth images ENABLED for cameras: {enabled_cameras}")
        if config.depth_save_debug_images:
            logger.info(f"🎨 Depth debug images will be saved with colormap: {config.depth_colormap}")
    else:
        logger.info("📷 Depth images DISABLED - using RGB-only strategy")
    
    # Track whether we've returned to initial pose for this cycle
    returned_to_initial_this_cycle = False

    # Initialize video recording manager
    video_manager = VideoRecordingManager(task_log_dir, task_name, logger)
    if enable_video_recording:  # Note: despite the name, this now saves individual images
        # Set global variables for cleanup
        task_log_dir_global = task_log_dir
        task_name_global = task_name
        
        video_manager.enable_recording()
    else:
        video_manager.disable_recording()

    while rclpy.ok():
        img_h_raw = sim_ros_node.get_img_head()
        img_l_raw = sim_ros_node.get_img_left_wrist()
        img_r_raw = sim_ros_node.get_img_right_wrist()
        
        # ADD depth images if needed - controlled by image strategy
        depth_img_h_raw = None
        depth_img_l_raw = None
        depth_img_r_raw = None
        
        if image_strategy == "rgb_depth":
            depth_cameras = config.get_depth_cameras_config()
            if depth_cameras.get('head', True):
                depth_img_h_raw = sim_ros_node.get_depth_img_head()
            if depth_cameras.get('left_wrist', True):
                depth_img_l_raw = sim_ros_node.get_depth_img_left_wrist()
            if depth_cameras.get('right_wrist', True):
                depth_img_r_raw = sim_ros_node.get_depth_img_right_wrist()
        
        act_raw = sim_ros_node.get_joint_state()
        
        # Check if we have valid images and joint state
        depth_check_passed = True
        if image_strategy == "rgb_depth":
            depth_cameras = config.get_depth_cameras_config()
            if depth_cameras.get('head', True) and depth_img_h_raw is None:
                depth_check_passed = False
            if depth_cameras.get('left_wrist', True) and depth_img_l_raw is None:
                depth_check_passed = False
            if depth_cameras.get('right_wrist', True) and depth_img_r_raw is None:
                depth_check_passed = False
        
        if (
            img_h_raw
            and img_l_raw
            and img_r_raw
            and act_raw
            and img_h_raw.header.stamp
            == img_l_raw.header.stamp
            == img_r_raw.header.stamp
            and depth_check_passed
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

                
                # save images if needed for debugging
                # cv2.imwrite(f"{current_path}/img_h_{count}.jpg", img_h)
                # cv2.imwrite(f"{current_path}/img_l_{count}.jpg", img_l)
                # cv2.imwrite(f"{current_path}/img_r_{count}.jpg", img_r)
                # img_h_pil = Image.fromarray(img_h)
                # img_l_pil = Image.fromarray(img_l)
                # img_r_pil = Image.fromarray(img_r)

                # Process depth images if enabled
                depth_imgs_processed = {}
                if image_strategy == "rgb_depth":
                    depth_cameras = config.get_depth_cameras_config()
                    
                    if depth_cameras.get('head', True) and depth_img_h_raw is not None:
                        depth_img_h = decode_depth_image(depth_img_h_raw)
                        # logger.info(f"Type of depth_img_h: {type(depth_img_h)}, shape: {depth_img_h.shape}, dtype: {depth_img_h.dtype}")
                        depth_imgs_processed['head'] = depth_img_h
                        
                        # Save debug images if enabled
                        if config.depth_save_debug_images:
                            depth_img_h_normalized = depth_to_rgb_uint8(
                                depth_img_h, 
                                min_val=config.depth_range_min, 
                                max_val=config.depth_range_max, 
                                cmap_name=config.depth_colormap
                            )
                            cv2.imwrite(f"{current_path}/depth_img_h_{count}.png", depth_img_h_normalized)
                    
                    if depth_cameras.get('left_wrist', True) and depth_img_l_raw is not None:
                        depth_img_l = decode_depth_image(depth_img_l_raw)
                        depth_imgs_processed['left_wrist'] = depth_img_l
                        
                        if config.depth_save_debug_images:
                            depth_img_l_normalized = depth_to_rgb_uint8(
                                depth_img_l,
                                min_val=config.depth_range_min,
                                max_val=config.depth_range_max,
                                cmap_name=config.depth_colormap
                            )
                            cv2.imwrite(f"{current_path}/depth_img_l_{count}.png", depth_img_l_normalized)
                    
                    if depth_cameras.get('right_wrist', True) and depth_img_r_raw is not None:
                        depth_img_r = decode_depth_image(depth_img_r_raw)
                        depth_imgs_processed['right_wrist'] = depth_img_r
                        
                        if config.depth_save_debug_images:
                            depth_img_r_normalized = depth_to_rgb_uint8(
                                depth_img_r,
                                min_val=config.depth_range_min,
                                max_val=config.depth_range_max,
                                cmap_name=config.depth_colormap
                            )
                            cv2.imwrite(f"{current_path}/depth_img_r_{count}.png", depth_img_r_normalized)

                # Save individual images if video recording is enabled
                if enable_video_recording:
                    timestamp = f"{count:06d}"
                    video_manager.save_inference_step_images(img_h, img_l, img_r, timestamp, save_individual=False)
                
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

                if image_strategy == "rgb_depth":
                    # Use depth images
                    model_input = input_processor.process(
                        img_h, img_l, img_r, lang, state, curr_task_substep_index, head_joint_cfg=head_joint_cfg,
                        img_depth_h=depth_imgs_processed.get('head'),
                        img_depth_l=depth_imgs_processed.get('left_wrist'),
                        img_depth_r=depth_imgs_processed.get('right_wrist')
                    )
                else:
                    # RGB only
                    model_input = input_processor.process(
                        img_h, img_l, img_r, lang, state, curr_task_substep_index, head_joint_cfg=head_joint_cfg
                    )
                # obs = get_observations(img_h, img_l, img_r, lang, state)
                # if cfg.with_proprio:
                #     action = policy.step(img_h, img_l, img_r, lang, state)
                # else:
                # print(f"instruction: {input["task_description"]}")
                action = policy.step(model_input["image_list"], model_input["task_description"], model_input["robot_state"], verbose=False)
                logger.debug(f"Step {count} | Substep {curr_task_substep_index+1}/{total_substeps} | Action: {action}")
                if action:                    
                    # Handle substep progression logic
                    task_advance_strategy = config.get_progression_strategy(task_name)
                    curr_task_substep_index, substep_inference_counter = handle_substep_progression(
                        action, task_name, curr_task_substep_index, substep_inference_counter, model_input, mode=task_advance_strategy, logger=logger
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


                # # send command from model to sim
                # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # default
                # if task_name == "iros_stamp_the_seal":
                #     # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                #     execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                # elif task_name == "iros_pack_in_the_supermarket":
                #     # execution_steps = [0, 1, 2, 3]
                #     execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                # elif task_name == "iros_make_a_sandwich":
                #     # execution_steps = [0, 1, 2, 3]
                #     execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                # elif task_name == "iros_clear_the_countertop_waste":
                #     execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                # elif task_name == "iros_heat_the_food_in_the_microwave":
                #     # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                #     # execution_steps = [0, 1, 2, 3]
                #     execution_steps = [0, 1, 2, 3, 4, 5, 6, 7]
                # elif task_name == "iros_open_drawer_and_store_items":
                #     # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                #     execution_steps = [0, 1, 2, 3]
                # elif task_name == "iros_pack_moving_objects_from_conveyor":
                #     execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

                #     if curr_task_substep_index % total_substeps == 0: # must be fast to pick up object from conveyor, so that use less steps
                #         execution_steps = execution_steps[::4]  # Take every 4th step for execution
                #     else:
                #         execution_steps = execution_steps[:8]  # Use first 8 steps for placing into box

                # elif task_name == "iros_pickup_items_from_the_fr'ee'zer":
                #     # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                #     # execution_steps = [0, 1, 2, 3]
                #     # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7]
                #     execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                # elif task_name == "iros_restock_supermarket_items":
                #     execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                #     # execution_steps = [0, 1, 2, 3, 4, 5, 6, 7] # bad for 17030
                #     # execution_steps = [0, 1, 2, 3] # bad for 17030
                # elif task_name == "iros_clear_table_in_the_restaurant":
                #     execution_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                # else:
                #     logger.warning(f"Task {task_name} not recognized, using default execution steps.")
                execution_steps = config.get_execution_steps(task_name, curr_task_substep_index, total_substeps)
                logger.info(f"🔄 Executing steps: {execution_steps} for substep {curr_task_substep_index} of task '{task_name}'")

                for step_index in execution_steps:
                    num_ik_iterations = config.get_ik_iterations(task_name)
                    # delta_joint_angles = joint_cmd[(step_index+1)*num_ik_iterations-1] - act_raw.position
                    # print(f"Delta joint angles for step {step_index}: \n")
                    # print(f"Delta left arm joint angles: {delta_joint_angles[:7]}\n")
                    # print(f"Delta right arm joint angles: {delta_joint_angles[8:15]}\n")
                    # print(f"Delta left gripper joint angles: {delta_joint_angles[7]}\n")
                    # print(f"Delta right gripper joint angles: {delta_joint_angles[15]}\n")
                    

                    # Convert delta joint angles to joint state message
                    for i in range(num_ik_iterations):
                        joint_arr = joint_cmd[step_index * num_ik_iterations + i]
                        # if  task_name == "iros_make_a_sandwich": # task_name == "iros_pack_moving_objects_from_conveyor" or
                        #   or task_name == "iros_restock_supermarket_items" \
                            
                            # drop during lifting, more tight grasp is need
                            # joint_arr[7] *= 1.5
                            # joint_arr[15] *= 1.5
                        # if task_name == "iros_restock_supermarket_items":
                        #     joint_arr[7] *= 1.7
                        #     joint_arr[15] *= 1.7

                        # Interpolate between current joint positions and target joint positions
                        act_raw = sim_ros_node.get_joint_state()
                        if enable_video_recording and config.save_per_joint_step_images and i == 0:
                            # Save images for each joint step if enabled
                            video_manager.save_joint_step_images(sim_ros_node, bridge, count, step_index, save_individual=False)

                        current_joints = act_raw.position  # [16,]
                        target_joints = joint_arr          # [16,]
                        # if task_name == "iros_clear_countertop_waste" or \
                        #     (task_name == "iros_pack_moving_objects_from_conveyor" and curr_task_substep_index%total_substeps==0) \
                        #     or task_name == "iros_make_a_sandwich" \
                        #     or task_name == "iros_restock_supermarket_items" :
                        #     # or task_name =="iros_clear_table_in_the_restaurant":
                        #     num_steps = 1
                        # else:
                        #     num_steps = 2
                        num_interpolation_steps = config.get_interpolation_steps(task_name, curr_task_substep_index, total_substeps)
                        logger.info(f"Interpolating to target joints over {num_interpolation_steps} steps")
                        interpolated_steps = interpolate_joints(current_joints, target_joints, num_steps=num_interpolation_steps)
                        # Send interpolated joint commands
                        for interp_joints in interpolated_steps:
                            sim_ros_node.publish_joint_command(interp_joints)
                            # print gripper joint angles in degrees for the final target
                            if interp_joints == interpolated_steps[-1]:
                                logger.info(f"Step {step_index} - Left gripper joint angle: {np.rad2deg(interp_joints[7]):.1f}°, Right gripper joint angle: {np.rad2deg(interp_joints[15]):.1f}°")
                            sim_ros_node.loop_rate.sleep()

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
    
def create_policy(config: InferenceConfig):
    """
    Create and return a policy instance based on configuration.
    
    Returns:
        CogActPolicy: Configured policy instance
    """
    from cogact_policy import CogActPolicy
    
    inference_mode = config.inference_mode
    if inference_mode == "api":
        logging.info(f"Connecting to policy API at {config.policy_ip}:{config.policy_port}")
        return CogActPolicy(
            inference_mode="api",
            ip_address=config.policy_ip,
            port=config.policy_port
        )
    elif inference_mode == "local":
        logging.info(f"Loading local policy from checkpoint: {config.local_checkpoint_path} with norm key: {config.unnorm_key}")
        return CogActPolicy(
            inference_mode="local",
            saved_model_path=config.local_checkpoint_path,
            unnorm_key=config.unnorm_key
        )
    else:
        raise ValueError(f"Unsupported inference mode: {inference_mode}")

def get_policy():
    """Get policy instance using configuration settings"""
    return create_policy(config)

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
    # parser.add_argument('--enable_video_recording', action='store_true', default=True,
    #                     help='Enable saving of individual images during inference. Images are saved in the log directory. (default: False)')
    parser.add_argument('--enable_file_logging', action='store_true', default=True,
                        help='Enable logging to file in addition to console output. (default: True), otherwise, console output only.')
    args = parser.parse_args()

    policy = get_policy()
    # policy = get_policy_wo_state()

    enable_video_recording = config.enable_video_recording
    if args.task_name == "iros_pack_moving_objects_from_conveyor":
        enable_video_recording = False # always disable video recording for this task, disable video recording since it's sensitive to timing and video recording may cause lag

    infer(policy, args.task_name, enable_video_recording=enable_video_recording, enable_file_logging=args.enable_file_logging)
