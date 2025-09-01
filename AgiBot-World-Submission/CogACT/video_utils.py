"""
Video and Image Recording Utilities for Robot Inference

This module provides utilities for recording and saving images during robot inference.
It includes functions for resizing images, saving individual images, and handling
both inference step images and joint step images.

Usage:
    from video_utils import save_inference_images, save_joint_step_images, resize_images_to_target_height
"""

import os
import cv2
import numpy as np
from cv_bridge import CvBridge


def resize_images_to_target_height(img_h, img_l, img_r, target_height=224):
    """
    Resize images to the same target height while maintaining aspect ratio.
    
    Args:
        img_h: Head image (numpy array)
        img_l: Left wrist image (numpy array)
        img_r: Right wrist image (numpy array)
        target_height: Target height for all images (default: 224)
    
    Returns:
        tuple: (resized_head, resized_left, resized_right) images
    """
    img_h_render = cv2.resize(img_h, (int(img_h.shape[1] * target_height / img_h.shape[0]), target_height))
    img_l_render = cv2.resize(img_l, (int(img_l.shape[1] * target_height / img_l.shape[0]), target_height))
    img_r_render = cv2.resize(img_r, (int(img_r.shape[1] * target_height / img_r.shape[0]), target_height))
    return img_h_render, img_l_render, img_r_render


def save_inference_images(img_h, img_l, img_r, task_log_dir, timestamp, logger=None, save_individual=False):
    """
    Save images from inference step.
    
    Args:
        img_h: Head image (numpy array)
        img_l: Left wrist image (numpy array) 
        img_r: Right wrist image (numpy array)
        task_log_dir: Directory to save images
        timestamp: Timestamp string for filename
        logger: Logger instance for error reporting (optional)
        save_individual: Whether to save individual camera images (default: False)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create images directory if it doesn't exist
        images_dir = os.path.join(task_log_dir)
        os.makedirs(images_dir, exist_ok=True)
        
        # Resize images to target height
        img_h_render, img_l_render, img_r_render = resize_images_to_target_height(img_h, img_l, img_r)

        # Save individual camera images if requested
        if save_individual:
            cv2.imwrite(os.path.join(images_dir, f"{timestamp}_head.jpg"), cv2.cvtColor(img_h_render, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(images_dir, f"{timestamp}_left_wrist.jpg"), cv2.cvtColor(img_l_render, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(images_dir, f"{timestamp}_right_wrist.jpg"), cv2.cvtColor(img_r_render, cv2.COLOR_RGB2BGR))
        
        # Save combined image
        combined_img = np.hstack((img_l_render, img_h_render, img_r_render))
        cv2.imwrite(os.path.join(images_dir, f"{timestamp}_combined.jpg"), combined_img)
        
        return True
        
    except Exception as e:
        if logger:
            logger.warning(f"Error saving images: {e}")
        else:
            print(f"Error saving images: {e}")
        return False


def save_joint_step_images(sim_ros_node, bridge, task_log_dir, count, step_index, logger=None, save_individual=False):
    """
    Save images from joint step execution.
    
    Args:
        sim_ros_node: ROS node for getting images
        bridge: CV bridge for image conversion
        task_log_dir: Directory to save images
        count: Current inference count
        step_index: Current step index
        logger: Logger instance for error reporting (optional)
        save_individual: Whether to save individual camera images (default: False)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get current images from ROS node
        img_h_step = bridge.compressed_imgmsg_to_cv2(
            sim_ros_node.get_img_head(), desired_encoding="rgb8"
        )
        img_l_step = bridge.compressed_imgmsg_to_cv2(
            sim_ros_node.get_img_left_wrist(), desired_encoding="rgb8"
        )
        img_r_step = bridge.compressed_imgmsg_to_cv2(
            sim_ros_node.get_img_right_wrist(), desired_encoding="rgb8"
        )
        
        # Create step timestamp
        step_timestamp = f"{count:06d}_{step_index:06d}"
        
        # Save images using the common function
        return save_inference_images(img_h_step, img_l_step, img_r_step, task_log_dir, step_timestamp, logger, save_individual)
        
    except Exception as e:
        if logger:
            logger.warning(f"Error saving step images: {e}")
        else:
            print(f"Error saving step images: {e}")
        return False


def setup_video_recording_directory(task_log_dir, task_name, logger=None):
    """
    Set up directory structure for video recording.
    
    Args:
        task_log_dir: Base directory for task logs
        task_name: Name of the task
        logger: Logger instance for info logging (optional)
    
    Returns:
        str: Path to the images directory
    """
    images_dir = task_log_dir
    os.makedirs(images_dir, exist_ok=True)
    
    if logger:
        logger.info(f"📷 Image saving enabled - images will be saved to: {images_dir}")
        logger.info(f"📁 Image naming format: XXXXXX_[head|left_wrist|right_wrist|combined].jpg")
    else:
        print(f"📷 Image saving enabled - images will be saved to: {images_dir}")
        print(f"📁 Image naming format: XXXXXX_[head|left_wrist|right_wrist|combined].jpg")
    
    return images_dir


class VideoRecordingManager:
    """
    Manager class for handling video recording operations.
    
    This class provides a unified interface for managing video recording
    during robot inference, including setup, image saving, and cleanup.
    """
    
    def __init__(self, task_log_dir, task_name, logger=None):
        """
        Initialize the video recording manager.
        
        Args:
            task_log_dir: Directory to save images
            task_name: Name of the task
            logger: Logger instance (optional)
        """
        self.task_log_dir = task_log_dir
        self.task_name = task_name
        self.logger = logger
        self.images_dir = None
        self.is_enabled = False
    
    def enable_recording(self):
        """Enable video recording and set up directory structure."""
        self.images_dir = setup_video_recording_directory(self.task_log_dir, self.task_name, self.logger)
        self.is_enabled = True
    
    def disable_recording(self):
        """Disable video recording."""
        self.is_enabled = False
        if self.logger:
            self.logger.info("📵 Image saving disabled")
        else:
            print("📵 Image saving disabled")
    
    def save_inference_step_images(self, img_h, img_l, img_r, timestamp, save_individual=False):
        """
        Save images from an inference step.
        
        Args:
            img_h: Head image
            img_l: Left wrist image
            img_r: Right wrist image
            timestamp: Timestamp string for filename
            save_individual: Whether to save individual camera images
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_enabled:
            return False
        
        return save_inference_images(img_h, img_l, img_r, self.task_log_dir, timestamp, self.logger, save_individual)
    
    def save_joint_step_images(self, sim_ros_node, bridge, count, step_index, save_individual=False):
        """
        Save images from a joint step.
        
        Args:
            sim_ros_node: ROS node for getting images
            bridge: CV bridge for image conversion
            count: Current inference count
            step_index: Current step index
            save_individual: Whether to save individual camera images
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_enabled:
            return False
        
        return save_joint_step_images(sim_ros_node, bridge, self.task_log_dir, count, step_index, self.logger, save_individual)
