# Reference: AgiBot-World/UniVLA/experiments/robot/geniesim/genie_model.py

class CogActAPIPolicy:
    def __init__(self, ip_address, port):
        self.url = f"http://{ip_address}:{port}/api/inference"
        pass
        
    # def reset(self):
    #     pass
    
    def step(self, img_list, task_description: str, robot_state: dict, image_format: str="JPEG", verbose: bool=False):
        """
        Args:
            img_list: List of images (head, left wrist, right wrist)
            task_description: Task description in natural language
            robot_state: Dictionary containing robot state information
            image_format: Format of the images (default is "JPEG")
            verbose: If True, print additional information

        Returns:
            action: Action to be performed by the robot or None.
        """
        # Placeholder for actual implementation
        action = None
        
        # Process images and task description to generate an action
        # This is where the actual policy logic would go
        
        if verbose:
            print(f"Processed images: {len(img_list)} images")
            print(f"Task description: {task_description}")
            print(f"Robot state: {robot_state}")
        
        return action
