# Reference: AgiBot-World/UniVLA/experiments/robot/geniesim/genie_model.py
from PIL import Image
import io
import json
import requests
import time
import numpy as np
import os
from abc import ABC, abstractmethod

class BaseCogActPolicy(ABC):
    """Base class for CogAct policy implementations"""
    
    @abstractmethod
    def step(self, img_list, task_description: str, robot_state: dict, image_format: str="JPEG", verbose: bool=False):
        """Main inference method"""
        pass

class CogActAPIPolicy(BaseCogActPolicy):
    def __init__(self, ip_address, port):
        self.url = f"http://{ip_address}:{port}/api/inference"
        self.inference_mode = "api"
        
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
        if robot_state is not None:
            response = self._call_infer(
                img_list, task_description, robot_state, self.url, image_format, verbose
            )
        else:
            response = self._call_infer_wo_state(
                img_list, task_description, self.url, image_format, verbose
            )
        
        # Process images and task description to generate an action
        # This is where the actual policy logic would go
        
        if verbose:
            print(f"Processed images: {len(img_list)} images")
            print(f"Task description: {task_description}")
            print(f"Robot state: {robot_state}")
        
        return response

    def _call_infer(
        self,
        image_list: Image.Image,
        task_description: str,
        robot_status: dict,
        url: str,
        image_format: str = "JPEG",
        verbose: bool = False,
    ):
        # TODO(XI): Change and integrate calling CogAct Infer here.
        """
        Get action from the image and task description.
        :param image: PIL Image to be processed.
        :param task_description: Task description.
        :return: Action or None.
        """

        start_time = time.time()
        files = []
        for i, img in enumerate(image_list):
            # print(f"Image {i} type: {type(img)}")
            if img is not None:
                if isinstance(img, Image.Image):
                    img_bytes = io.BytesIO()
                    if image_format == "JPEG":
                        img.save(img_bytes, format="JPEG")
                    elif image_format == "PNG":
                        img.save(img_bytes, format="PNG")
                    else:
                        raise ValueError("Unsupported image format. Use 'JPEG' or 'PNG'.")
                    img_bytes.seek(0)
                    files.append(
                        (
                            f"image_{i}",
                            (
                                (f"image_{i}.png", img_bytes, "image/png")
                                if image_format == "PNG"
                                else (f"image_{i}.jpg", img_bytes, "image/jpeg")
                            ),
                        )
                    )
                elif isinstance(img, np.ndarray):
                    img_bytes = io.BytesIO()
                    np.save(img_bytes, img)
                    img_bytes.seek(0)
                    files.append(
                        (
                            f"image_{i}",
                            (f"image_{i}.npy", img_bytes, "application/octet-stream"),
                        )
                    )
                else:
                    raise ValueError("Unsupported image type. Use PIL Image or numpy ndarray.")

        json_bytes = io.BytesIO(
            json.dumps(
                {
                    "task_description": task_description,
                    "state": robot_status,
                }
            ).encode("utf-8")
        )
        files.append(("json", ("data.json", json_bytes, "application/json")))
        if verbose:
            print(f"Save time: {time.time() - start_time}")
        start_request_time = time.time()
        response = requests.post(url, files=files)
        end_request_time = time.time()

        if verbose:
            print("Request time: ", end_request_time - start_request_time)
            print("Total time: ", end_request_time - start_time)

        if response.status_code == 200:
            return response.json()
        else:
            print("Failed to get a response from the API")
            print(response.text)

    def _call_infer_wo_state(
        self,
        image_list: Image.Image,
        task_description: str,
        url: str,
        image_format: str = "JPEG",
        verbose: bool = False,
    ):
        # TODO(XI): Change and integrate calling CogAct Infer here.
        """
        Get action from the image and task description.
        :param image: PIL Image to be processed.
        :param task_description: Task description.
        :return: Action or None.
        """

        start_time = time.time()
        files = []
        for i, img in enumerate(image_list):
            if img is not None:
                img_bytes = io.BytesIO()
                if image_format == "JPEG":
                    img.save(img_bytes, format="JPEG")
                elif image_format == "PNG":
                    img.save(img_bytes, format="PNG")
                else:
                    raise ValueError("Unsupported image format. Use 'JPEG' or 'PNG'.")
                img_bytes.seek(0)
                files.append(
                    (
                        f"image_{i}",
                        (
                            (f"image_{i}.png", img_bytes, "image/png")
                            if image_format == "PNG"
                            else (f"image_{i}.jpg", img_bytes, "image/jpeg")
                        ),
                    )
                )

        json_bytes = io.BytesIO(
            json.dumps(
                {
                    "task_description": task_description,
                }
            ).encode("utf-8")
        )
        files.append(("json", ("data.json", json_bytes, "application/json")))
        if verbose:
            print(f"Save time: {time.time() - start_time}")
        start_request_time = time.time()
        response = requests.post(url, files=files)
        end_request_time = time.time()

        if verbose:
            print("Request time: ", end_request_time - start_request_time)
            print("Total time: ", end_request_time - start_time)

        if response.status_code == 200:
            return response.json()
        else:
            print("Failed to get a response from the API")
            print(response.text)

class CogActPolicy:
    """CogAct policy wrapper to choose between API and local inference modes"""
    def __init__(self, inference_mode="api", ip_address="localhost", port=8000):
        if inference_mode == "api":
            self.policy = CogActAPIPolicy(ip_address, port)
        # elif inference_mode == "local":
        #     self.policy = CogActLocalPolicy(
        #         checkpoint_path="/path/to/your/model/checkpoint.pth",  # Update this path
        #         model_config={
        #             "model_type": "your_model_type",
        #             "input_size": [3, 224, 224],
        #             "output_dim": 7
        #         }
        #     )
        else:
            raise ValueError("Unsupported inference mode. Use 'api'.")

    def step(self, img_list, task_description: str, robot_state: dict=None, image_format: str="JPEG", verbose: bool=False):
        return self.policy.step(img_list, task_description, robot_state, image_format, verbose)
