# Reference: AgiBot-World/UniVLA/experiments/robot/geniesim/genie_model.py
from PIL import Image
import io
import json
import requests
import time
import numpy as np
import os
import sys
import pathlib
from abc import ABC, abstractmethod

# Import for local inference
try:
    # Add the inference scripts path to sys.path
    current_dir = pathlib.Path(__file__).parent.resolve()
    inference_dir = current_dir / "inference" / "scripts_vla" / "serving"
    if inference_dir.exists() and str(inference_dir) not in sys.path:
        sys.path.append(str(inference_dir))
    
    from direct_inference import CogACTInferencer
    DIRECT_INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Direct inference not available: {e}")
    DIRECT_INFERENCE_AVAILABLE = False

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

class CogACTLocalPolicy(BaseCogActPolicy):
    """Local CogACT policy implementation using direct inference"""
    
    def __init__(
        self,
        saved_model_path: str,
        unnorm_key: str = None,
        image_size: list = [224, 224],
        cfg_scale: float = 1.5,
        num_ddim_steps: int = 10,
        use_ddim: bool = True,
        use_bf16: bool = True,
        action_ensemble: bool = True,
        adaptive_ensemble_alpha: float = 0.1,
        action_ensemble_horizon: int = 2,
        action_chunking: bool = False,
        action_chunking_window: int = None,
        device: str = "cuda",
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the local CogACT policy.
        
        Args:
            saved_model_path: Path to the saved model directory
            unnorm_key: Key for action normalization
            image_size: Input image size [height, width]
            cfg_scale: Classifier-free guidance scale
            num_ddim_steps: Number of DDIM sampling steps
            use_ddim: Whether to use DDIM sampling
            use_bf16: Whether to use bfloat16 precision
            action_ensemble: Whether to use action ensemble
            adaptive_ensemble_alpha: Alpha parameter for adaptive ensemble
            action_ensemble_horizon: Horizon for action ensemble
            action_chunking: Whether to use action chunking
            action_chunking_window: Window size for action chunking
            device: Device to run inference on
            verbose: Whether to print verbose information
            **kwargs: Additional arguments passed to CogACTInferencer
        """
        if not DIRECT_INFERENCE_AVAILABLE:
            raise ImportError("Direct inference is not available. Please check if the direct_inference module is properly installed.")
        
        self.inference_mode = "local"
        self.verbose = verbose
        
        # Initialize the CogACT inferencer
        self.inferencer = CogACTInferencer(
            saved_model_path=saved_model_path,
            unnorm_key=unnorm_key,
            image_size=image_size,
            cfg_scale=cfg_scale,
            num_ddim_steps=num_ddim_steps,
            use_ddim=use_ddim,
            use_bf16=use_bf16,
            action_ensemble=action_ensemble,
            adaptive_ensemble_alpha=adaptive_ensemble_alpha,
            action_ensemble_horizon=action_ensemble_horizon,
            action_chunking=action_chunking,
            action_chunking_window=action_chunking_window,
            device=device,
            verbose=verbose,
            **kwargs
        )
        
        if self.verbose:
            print(f"*** CogACT Local Policy initialized successfully ***")
    
    def step(self, img_list, task_description: str, robot_state: dict, image_format: str="JPEG", verbose: bool=False):
        """
        Perform local inference using the CogACT model.
        
        Args:
            img_list: List of images (head, left wrist, right wrist)
            task_description: Task description in natural language
            robot_state: Dictionary containing robot state information
            image_format: Format of the images (default is "JPEG") - not used in local inference
            verbose: If True, print additional information

        Returns:
            action: Action dictionary to be performed by the robot or None.
        """
        start_time = time.time()
        
        try:
            # Use the predict method from CogACTInferencer which handles preprocessing
            action = self.inferencer.predict(
                images=img_list,
                task_description=task_description,
                state=robot_state,
                save_logs=False  # Set to True if you want to save logs
            )
            
            end_time = time.time()
            
            if verbose or self.verbose:
                print(f"Local inference completed in {(end_time - start_time) * 1000:.2f} ms")
                print(f"Processed images: {len(img_list)} images")
                print(f"Task description: {task_description}")
                print(f"Robot state: {robot_state}")
                print(f"Predicted action: {action}")
            
            return action
            
        except Exception as e:
            print(f"Local inference failed: {str(e)}")
            return None
    
    def reset(self):
        """Reset the action ensemble state"""
        if hasattr(self.inferencer, 'reset'):
            self.inferencer.reset()

class CogActPolicy:
    """CogAct policy wrapper to choose between API and local inference modes"""
    def __init__(self, inference_mode="api", **kwargs):
        """
        Initialize CogAct policy.
        
        Args:
            inference_mode: "api" or "local"
            **kwargs: Mode-specific arguments:
                For API mode:
                    - ip_address: IP address for API mode (default: "localhost")
                    - port: Port for API mode (default: 8000)
                For local mode:
                    - saved_model_path: Path to the saved model directory (required for local mode)
                    - unnorm_key: Key for action normalization
                    - image_size: Input image size [height, width] (default: [224, 224])
                    - cfg_scale: Classifier-free guidance scale (default: 1.5)
                    - num_ddim_steps: Number of DDIM sampling steps (default: 10)
                    - use_ddim: Whether to use DDIM sampling (default: True)
                    - use_bf16: Whether to use bfloat16 precision (default: True)
                    - action_ensemble: Whether to use action ensemble (default: True)
                    - adaptive_ensemble_alpha: Alpha parameter for adaptive ensemble (default: 0.1)
                    - action_ensemble_horizon: Horizon for action ensemble (default: 2)
                    - action_chunking: Whether to use action chunking (default: False)
                    - action_chunking_window: Window size for action chunking (default: None)
                    - device: Device to run inference on (default: "cuda")
                    - verbose: Whether to print verbose information (default: False)
        """
        if inference_mode == "api":
            # Extract API-specific parameters
            ip_address = kwargs.get('ip_address', 'localhost')
            port = kwargs.get('port', 8000)
            self.policy = CogActAPIPolicy(ip_address, port)
        elif inference_mode == "local":
            if not DIRECT_INFERENCE_AVAILABLE:
                raise ValueError("Direct inference is not available. Cannot use local inference mode.")
            
            # Extract required parameters for local inference
            saved_model_path = kwargs.get('saved_model_path')
            if saved_model_path is None:
                raise ValueError("saved_model_path is required for local inference mode")
            unnorm_key = kwargs.get('unnorm_key')
            if unnorm_key is None:
                raise ValueError("unnorm_key is required for local inference mode")
            
            self.policy = CogACTLocalPolicy(
                saved_model_path=saved_model_path,
                unnorm_key=unnorm_key,
                image_size=kwargs.get('image_size', [224, 224]),
                cfg_scale=kwargs.get('cfg_scale', 1.5),
                num_ddim_steps=kwargs.get('num_ddim_steps', 10),
                use_ddim=kwargs.get('use_ddim', True),
                use_bf16=kwargs.get('use_bf16', True),
                action_ensemble=kwargs.get('action_ensemble', True),
                adaptive_ensemble_alpha=kwargs.get('adaptive_ensemble_alpha', 0.1),
                action_ensemble_horizon=kwargs.get('action_ensemble_horizon', 2),
                action_chunking=kwargs.get('action_chunking', False),
                action_chunking_window=kwargs.get('action_chunking_window', None),
                device=kwargs.get('device', "cuda"),
                verbose=kwargs.get('verbose', False),
                **{k: v for k, v in kwargs.items() if k not in [
                    'saved_model_path', 'unnorm_key', 'image_size', 'cfg_scale', 
                    'num_ddim_steps', 'use_ddim', 'use_bf16', 'action_ensemble',
                    'adaptive_ensemble_alpha', 'action_ensemble_horizon', 'action_chunking',
                    'action_chunking_window', 'device', 'verbose'
                ]}
            )
        else:
            raise ValueError("Unsupported inference mode. Use 'api' or 'local'.")

    def step(self, img_list, task_description: str, robot_state: dict=None, image_format: str="JPEG", verbose: bool=False):
        return self.policy.step(img_list, task_description, robot_state, image_format, verbose)
    
    def reset(self):
        """Reset the policy state (useful for action ensemble in local mode)"""
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
