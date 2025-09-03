#!/usr/bin/env python3
"""
Configuration loader for robot inference using YAML configuration file.
"""

import yaml
import os
from typing import Dict, List, Any, Union


class InferenceConfig:
    """Configuration loader and manager for robot inference."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration.
        
        Args:
            config_path: Path to the YAML config file. If None, uses default location.
        """
        if config_path is None:
            # Default to config file in the same directory as this script
            config_path = os.path.join(os.path.dirname(__file__), 'inference_config.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()
    
    # =========================================================================
    # IMAGE PROCESSING CONFIGURATION
    # =========================================================================
    
    @property
    def resize_mode(self) -> str:
        """Get image resize mode."""
        return self.config['image_processing']['resize_mode']
    
    @property
    def depth_images_enabled(self) -> bool:
        """Get whether depth images are enabled."""
        return self.config['image_processing'].get('depth_images', {}).get('enabled', False)
    
    @property 
    def depth_save_debug_images(self) -> bool:
        """Get whether to save depth debug images."""
        return self.config['image_processing'].get('depth_images', {}).get('save_debug_images', False)
    
    @property
    def depth_colormap(self) -> str:
        """Get depth image colormap."""
        return self.config['image_processing'].get('depth_images', {}).get('colormap', 'turbo')
    
    @property
    def depth_range_min(self) -> float:
        """Get minimum depth value for normalization."""
        return self.config['image_processing'].get('depth_images', {}).get('depth_range', {}).get('min_val', 0.1)
    
    @property
    def depth_range_max(self) -> float:
        """Get maximum depth value for normalization."""
        return self.config['image_processing'].get('depth_images', {}).get('depth_range', {}).get('max_val', 3.0)
    
    def get_depth_cameras_config(self) -> Dict[str, bool]:
        """Get depth camera configuration."""
        default_cameras = {'head': True, 'left_wrist': True, 'right_wrist': True}
        return self.config['image_processing'].get('depth_images', {}).get('cameras', default_cameras)
    
    # @property
    # def log_observations(self) -> bool:
    #     """Get whether to log observations."""
    #     return self.config['image_processing']['log_observations']
    
    # @property
    # def target_image_height(self) -> int:
    #     """Get target image height for resizing."""
    #     return self.config['image_processing']['target_image_height']
    
    # @property
    # def save_individual_camera_images(self) -> bool:
    #     """Get whether to save individual camera images."""
    #     return self.config['image_processing']['save_individual_camera_images']
    
    # @property
    # def save_combined_image(self) -> bool:
    #     """Get whether to save combined image."""
    #     return self.config['image_processing']['save_combined_image']
    
    # @property
    # def image_save_format(self) -> str:
    #     """Get image save format."""
    #     return self.config['image_processing']['image_save_format']
    
    @property
    def save_per_joint_step_images(self) -> bool:
        """Get whether to save images per joint execution step."""
        return self.config['image_processing']['save_per_joint_step_images']
    
    # @property
    # def joint_step_image_prefix(self) -> str:
    #     """Get prefix for joint step images."""
    #     return self.config['image_processing']['joint_step_image_prefix']
    
    # @property
    # def inference_step_image_prefix(self) -> str:
    #     """Get prefix for inference step images."""
    #     return self.config['image_processing']['inference_step_image_prefix']
    
    # =========================================================================
    # TASK EXECUTION CONFIGURATION
    # =========================================================================
    
    def get_execution_steps(self, task_name: str, substep_index: int = None, total_substeps: int = None) -> List[int]:
        """
        Get execution steps for a specific task and substep.
        
        Args:
            task_name: Name of the task
            substep_index: Current substep index (for special cases)
            total_substeps: Total number of substeps (for special cases)
            
        Returns:
            List of execution step indices
        """
        task_steps = self.config['task_execution']['task_execution_steps']
        
        if task_name not in task_steps:
            return self.config['task_execution']['default_execution_steps']
        
        steps_config = task_steps[task_name]
        
        # Handle special case for conveyor task
        if task_name == "iros_pack_moving_objects_from_conveyor" and isinstance(steps_config, dict):
            if substep_index is not None and total_substeps is not None:
                if substep_index % total_substeps == 0:  # Pickup substep
                    return steps_config["pickup_substep"]
                else:  # Place substep
                    return steps_config["place_substep"]
            return self.config['task_execution']['default_execution_steps']
        
        return steps_config
    
    def get_gripper_config(self) -> Dict[str, Any]:
        """Get complete gripper configuration."""
        return self.config['task_execution']['gripper_config']
    
    def get_gripper_strategy(self, task_name: str) -> str:
        """Get gripper strategy for a specific task."""
        gripper_config = self.get_gripper_config()
        strategy_config = gripper_config['strategy_per_task']
        return strategy_config.get(task_name, strategy_config['default'])
    
    def get_gripper_ratio(self, task_name: str, strategy: str = None) -> float:
        """Get gripper ratio for a specific task and strategy."""
        gripper_config = self.get_gripper_config()
        
        # Use provided strategy or get from task configuration
        if strategy is None:
            strategy = self.get_gripper_strategy(task_name)
        
        # Check for task-specific ratio first
        task_ratios = gripper_config['ratios_per_task'].get(task_name, {})
        if strategy in task_ratios:
            return task_ratios[strategy]
        
        # Fall back to default ratio
        return gripper_config['default_ratios'][strategy]
    
    def get_gripper_timing_adjustment(self, task_name: str, sequence_length: int) -> int:
        """Get gripper timing adjustment (frames to shift forward) for a task."""
        gripper_config = self.get_gripper_config()
        timing_config = gripper_config['timing_adjustment']
        
        # Check task-specific timing first
        task_timing = timing_config.get('task_specific', {}).get(task_name, {})
        
        # Map sequence length to frame key
        frame_key = f"frames_{sequence_length}"
        
        # Get task-specific timing or fall back to default
        if frame_key in task_timing:
            return task_timing[frame_key]
        elif frame_key in timing_config:
            return timing_config[frame_key]
        else:
            return timing_config['default']
    
    # def get_gripper_signal_filter_params(self) -> Dict[str, Any]:
    #     """Get gripper signal filter parameters."""
    #     gripper_config = self.get_gripper_config()
    #     return gripper_config['signal_filter']

    # def get_gripper_multipliers(self, task_name: str) -> Dict[str, float]:
    #     """Get gripper force multipliers for a task."""
    #     multipliers = self.config['task_execution']['gripper_force_multipliers']
    #     return multipliers.get(task_name, {"left": 1.0, "right": 1.0})
    
    def get_interpolation_steps(self, task_name: str, substep_index: int = None, total_substeps: int = None) -> int:
        """Get number of interpolation steps for a task."""
        interpolation_config = self.config['task_execution']['interpolation_steps']
        if task_name not in interpolation_config:
            return interpolation_config["default"]

        interpolation_steps_config = interpolation_config[task_name]

        # Special case for conveyor task pickup
        # Handle special case for conveyor task
        if task_name == "iros_pack_moving_objects_from_conveyor" and isinstance(interpolation_steps_config, dict):
            if substep_index is not None and total_substeps is not None:
                if substep_index % total_substeps == 0:  # Pickup substep
                    return interpolation_steps_config["pickup_substep"]
                else:  # Place substep
                    return interpolation_steps_config["place_substep"]
            return self.config['task_execution']['default_execution_steps']
        
        return interpolation_config.get(task_name, interpolation_config["default"])
    
    def get_ik_iterations(self, task_name: str = None) -> int:
        """Get number of IK iterations for solving end-effector poses."""
        ik_config = self.config['task_execution']['ik_config']['iterations']
        if task_name and task_name in ik_config:
            return ik_config[task_name]
        return ik_config['default']
    
    def get_ik_error_logging_enabled(self) -> bool:
        """Get whether IK error logging is enabled."""
        return self.config['task_execution']['ik_config'].get('enable_error_logging', True)
    
    # =========================================================================
    # TASK PROGRESSION CONFIGURATION
    # =========================================================================
    
    def get_task_progression_config(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """Get task-specific configuration for substep progression."""
        return self.config['task_progression']
    
    def get_progression_strategy(self, task_name: str) -> str:
        """Get progression strategy for a task."""
        strategies = self.config['task_progression']['task_progression_strategies']
        return strategies.get(task_name, strategies["default"])
    
    # =========================================================================
    # HEAD JOINT CONFIGURATIONS
    # =========================================================================
    
    def get_head_joint_cfg(self, task_name: str) -> Dict[str, float]:
        """Get head joint configuration for a specific task."""
        head_configs = self.config['head_joint_configurations']
        if task_name in head_configs:
            return head_configs[task_name]
        else:
            raise ValueError(f"Joint configuration for task '{task_name}' not defined.")
    
    # =========================================================================
    # POLICY CONFIGURATION
    # =========================================================================
    
    @property
    def policy_ip(self) -> str:
        """Get policy API IP address."""
        return self.config['policy']['ip']
    
    @property
    def policy_port(self) -> int:
        """Get policy API port."""
        return self.config['policy']['port']
    
    # # =========================================================================
    # # SIMULATION CONFIGURATION
    # # =========================================================================
    
    # @property
    # def sim_init_time(self) -> int:
    #     """Get simulation initialization time."""
    #     return self.config['simulation']['init_time']
    
    # @property
    # def return_to_initial_steps(self) -> int:
    #     """Get number of steps for returning to initial pose."""
    #     return self.config['simulation']['return_to_initial_steps']
    
    # =========================================================================
    # LOGGING CONFIGURATION
    # =========================================================================
    @property
    def enable_video_recording(self) -> bool:
        """Get whether to enable video recording."""
        return self.config['logging'].get('enable_video_recording', False)

    # @property
    # def default_log_dir(self) -> str:
    #     """Get default log directory."""
    #     return self.config['logging']['default_log_dir']
    
    # @property
    # def default_video_recording_dir(self) -> str:
    #     """Get default video recording directory."""
    #     return self.config['logging']['default_video_recording_dir']
    
    # @property
    # def log_format(self) -> str:
    #     """Get log format string."""
    #     return self.config['logging']['log_format']
    
    # @property
    # def log_date_format(self) -> str:
    #     """Get log date format string."""
    #     return self.config['logging']['log_date_format']
    
    # @property
    # def image_save_log_frequency(self) -> int:
    #     """Get image save log frequency."""
    #     return self.config['logging']['image_save_log_frequency']
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for saving."""
        return self.config.copy()
    
    def save_config_copy(self, destination_path: str):
        """Save a copy of the configuration file to the specified path."""
        import shutil
        shutil.copy2(self.config_path, destination_path)
    
    def save_config_summary(self, destination_path: str):
        """Save configuration summary as JSON."""
        import json
        with open(destination_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# Global configuration instance
_config_instance = None

def get_config(config_path: str = None) -> InferenceConfig:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Path to config file. Only used on first call.
        
    Returns:
        InferenceConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = InferenceConfig(config_path)
    return _config_instance

def reload_config():
    """Reload the global configuration from file."""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload()
