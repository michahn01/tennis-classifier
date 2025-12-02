import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import os
from tennis.common.model_utils import get_model_class, requires_pose
from tennis.common.extract_features import extract_features_from_video
from tennis.common.extract_pose import extract_pose_from_video
from tennis.dataset import TennisDataset

class InferenceEngine:
    def __init__(self, model_type, model_weights_path, device=None, num_frames=30):
        """Initializes the InferenceEngine class.

        Args:
            model_type (str): Type of model (baseline, attention, skeleton, skeleton_attention).
            model_weights_path (str): Path to the trained model weights.
            device (str, optional): Device to run on (e.g., 'cuda', 'cpu'). If None, auto-detects.
            num_frames (int): Number of frames to sample from the video.
        """
        self.model_type = model_type
        self.model_weights_path = model_weights_path
        self.num_frames = num_frames
        self.landmarker = None # Initialize early to avoid AttributeError in __del__
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"Inference initialized on device: {self.device}")

        # Load Model Architecture
        model_class = get_model_class(model_type)

        hidden_size = 256
        lstm_layers = 1
        dropout = 0.1

        import re
        match = re.search(r"h(\d+)_l(\d+)_d([\d.]+)", model_weights_path)
        if match:
            hidden_size = int(match.group(1))
            lstm_layers = int(match.group(2))
            dropout = float(match.group(3))
            print(f"Inferred hyperparameters: hidden_size={hidden_size}, lstm_layers={lstm_layers}, dropout={dropout}")
        else:
            print("Could not infer hyperparameters from filename, using defaults.")

        self.model = model_class(
            num_classes=6,
            input_dim=2048,
            lstm_hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout
        ).to(self.device)

        # Load Weights
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

        # Fix for PyTorch 2.6+ weights_only default
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device, weights_only=False))
        self.model.eval()

        # Initialize Feature Extractor (ResNet50)
        # We'll load it once and keep it in memory
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(self.resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules).to(self.device)
        self.feature_extractor.eval()
        
        self.transform = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        # Initialize Pose Landmarker if needed
        self.requires_pose = requires_pose(model_type)
        self.landmarker = None
        if self.requires_pose:
            import mediapipe as mp
            from tennis.common.extract_pose import download_model
            
            model_path = "pose_landmarker_full.task"
            download_model(model_path)
            
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU
                ),
                running_mode=VisionRunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = PoseLandmarker.create_from_options(options)

    def predict(self, video_path):
        """Runs inference on a single video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            predicted_class (str): The predicted class name.
            probabilities (dict): Dictionary mapping class names to probabilities.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found at {video_path}")

        # Extract RGB Features
        # extract_features_from_video returns (T, 2048)
        rgb_features = extract_features_from_video(
            video_path, 
            self.num_frames, 
            self.device, 
            model=self.feature_extractor, 
            transform=self.transform
        )
        
        # Add batch dimension: (1, T, 2048)
        rgb_input = rgb_features.unsqueeze(0)

        # Extract Pose Features (if needed)
        if self.requires_pose:
            # extract_pose_from_video returns (T, 99)
            pose_features = extract_pose_from_video(
                video_path, 
                self.num_frames, 
                landmarker=self.landmarker
            )
            # Add batch dimension: (1, T, 99)
            pose_input = pose_features.unsqueeze(0).to(self.device)
            
            # Model input is a tuple
            model_input = (rgb_input, pose_input)
        else:
            model_input = rgb_input

        # Inference
        with torch.no_grad():
            # output is (1, num_classes)
            probs = self.model(model_input)
            
        # Process Results
        probs = probs.squeeze(0).cpu().numpy() # (num_classes,)
        class_names = list(TennisDataset.CLASS_MAPPING.keys())
        
        result = {class_name: float(prob) for class_name, prob in zip(class_names, probs)}
        predicted_idx = probs.argmax()
        predicted_class = class_names[predicted_idx]
        
        return predicted_class, result

    def __del__(self):
        if self.landmarker:
            self.landmarker.close()
