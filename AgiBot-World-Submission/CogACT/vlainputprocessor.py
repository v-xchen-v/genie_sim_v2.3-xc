import numpy as np
from PIL import Image
import os,time, pickle
from typing import Dict, Any


class VLAInputProcessor:
    # def __init__(self, image_list, task_instruction, robot_state, task_name: str, 
    #              log_obs=True, additional_info=None):
    #     self.image_list = image_list
    #     self.task_instruction = task_instruction
    #     self.robot_state = robot_state
    #     self.additional_info = additional_info if additional_info is not None else {}
    #     self.task_name = task_name
    #     self.log_obs = log_obs

    #     # extra
    #     self._log_dir_registry = {}
    def __init__(self, log_obs=True):
        
        self.log_obs = log_obs
        if self.log_obs:
            # Initialize log directory registry if logging is enabled
            self.task_name = "default_task"  # Placeholder, can be set later
            self._log_dir_registry = {}
        
    def process(self, img_h, img_l, img_r, lang, state):
        """
        Process the input images, task description, and robot state.
        
        Args:
            img_h: Head image
            img_l: Left wrist image
            img_r: Right wrist image
            lang: Task description in natural language
            state: Robot state information
            
        Returns:
            A dictionary containing processed images, task description, and robot state.
        """
        self.image_list = [img_h, img_l, img_r]
        self.task_instruction = lang
        self.robot_state = state
        
        return self.prepare_input()
        
    
    def prepare_input(self):
        """
        Prepare the input for the VLA model.
        
        Returns:
            A dictionary containing the processed images, task description, and robot state.
        """
        obs_dict = self.get_observations()
        if self.log_obs:
            # Log observations if required
            self._log_observations(obs_dict, log_dir="./obs_logs")
       
        input_data = {
            "image_list" : [
                Image.fromarray(obs_dict["images"]["cam_top"]),
                Image.fromarray(obs_dict["images"]["head_left"]),
                Image.fromarray(obs_dict["images"]["head_right"]),
            ],
            "task_description": obs_dict["task_description"],
            "robot_state": obs_dict["robot_state"],
        }
        
        return input_data
        
    def get_observations(self):
        """
        Get observations for the policy step.
        
        Returns:
            A dictionary containing the observations.
        """
        # Preprocess images
        processed_images = self.preprocess_images()
        cam_top_img = processed_images[0]
        head_left_img = processed_images[1]
        head_right_img = processed_images[2]
        
        
        # Construct the observations dictionary
        obs_dict = {
            "task_description": self.task_instruction,
            "images": {
                "cam_top": cam_top_img,
                "head_left": head_left_img,
                "head_right": head_right_img,
            },
            "robot_state": self.robot_state,
        }
        return obs_dict
    
    def preprocess_images(self):
        """
        Preprocess the images in the image_list.
        This could include resizing, normalization, etc.
        """
        processed_images = []
        for img in self.image_list:
            # Perform preprocessing on each image
            processed_img = self._preprocess_single_image(img)
            processed_images.append(processed_img)
        return processed_images
    
            
    def _preprocess_single_image(self, img):
        """
        Preprocess a single image.
        
        Args:
            img: The image to preprocess.
        
        Returns:
            The preprocessed image.
        """
        # Resize the image to a fixed size (e.g., 224x224)
        img = self._resize_image(img, target_size=(224, 224))
        return img        
    
    def _resize_image(self, img, target_size=(224, 224)):
        """
        Resize the image to the target size while maintaining aspect ratio.
        """
        h, w = img.shape[:2]
        target_aspect = 4 / 3
        current_aspect = w / h

        # Allow small floating point tolerance
        if abs(current_aspect - target_aspect) < 1e-3:
            padded = img  # No padding needed
        elif current_aspect > target_aspect:
            # Image is too wide → pad height
            new_h = int(w / target_aspect)
            pad_total = new_h - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            padded = np.pad(
                img,
                ((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            # Image is too tall → pad width
            new_w = int(h * target_aspect)
            pad_total = new_w - w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded = np.pad(
                img,
                ((0, 0), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        img = Image.fromarray(padded)
        img = img.resize(target_size, Image.LANCZOS)
        return np.array(img)
    
    def _get_unique_log_dir(self, base_dir, task_name):
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
            
    def _create_and_register_log_dir(self, log_dir: str):
        """Make sure log directory of single run is same and unique."""
        if (log_dir, self.task_name) not in self._log_dir_registry:
            task_log_dir = self._get_unique_log_dir(log_dir, self.task_name)
            self._log_dir_registry[(log_dir, self.task_name)] = task_log_dir
        return self._log_dir_registry[(log_dir, self.task_name)]            

    def _log_observations(self, obs_dict: Dict[str, Any], log_dir: str):
        """Log observations to a specified directory."""
        task_log_dir = self._create_and_register_log_dir(log_dir)
            
        # Use timestamp or step_id as filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # also save the observations
        obs_filename = os.path.join(
            task_log_dir, f"observations_{timestamp}.pkl"
        )
        with open(obs_filename, "wb") as f:
            # pickle.dump(observations, f)
            pickle.dump(obs_dict, f)
        
if __name__ == "__main__":
    # Example usage
    image_list = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
    task_instruction = "Pick up the red ball and place it in the blue box."
    robot_state = {"joint_positions": [0.0] * 16, "gripper_state": "open"}
    
    processor = VLAInputProcessor(image_list, task_instruction, robot_state)
    input_data = processor.prepare_input()
    
    print("Prepared Input Data:")
    print(input_data)
    # This will print the processed images, task description, and robot state.
