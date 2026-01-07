"""
Video Deepfake Detection Model

This module implements a deep learning model for detecting video deepfakes
including face swaps, facial reenactment, and synthetic videos.

Architecture:
- EfficientNet-B0 for spatial feature extraction
- 3D CNN for temporal analysis
- Attention mechanism for frame importance
- Binary classification head
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple, Optional
import numpy as np


class VideoDeepfakeDetector(nn.Module):
    """
    Video deepfake detection model using EfficientNet + 3D CNN + Attention.
    
    Args:
        num_frames: Number of frames to process
        pretrained: Use pretrained EfficientNet weights
        dropout: Dropout probability
        freeze_backbone: Whether to freeze EfficientNet
    """
    
    def __init__(
        self,
        num_frames: int = 30,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.num_frames = num_frames
        
        # EfficientNet-B0 for spatial features
        efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Remove classification head
        self.spatial_features = nn.Sequential(
            *list(efficientnet.children())[:-1]
        )
        
        if freeze_backbone:
            for param in self.spatial_features.parameters():
                param.requires_grad = False
        
        # Feature dimension from EfficientNet-B0
        self.feature_dim = 1280
        
        # 3D Convolutional layers for temporal analysis
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Frame-level attention
        self.frame_attention = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim + 128, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            frames: Video frames [batch, num_frames, channels, height, width]
            
        Returns:
            predictions: Authenticity scores [batch, 1]
            attention_weights: Frame attention weights [batch, num_frames]
        """
        batch_size, num_frames, c, h, w = frames.shape
        
        # Extract spatial features for each frame
        frames_flat = frames.view(batch_size * num_frames, c, h, w)
        spatial_features = self.spatial_features(frames_flat)  # [batch*frames, feature_dim, 1, 1]
        spatial_features = spatial_features.view(batch_size, num_frames, self.feature_dim)
        
        # Frame-level attention
        attention_scores = self.frame_attention(spatial_features)  # [batch, frames, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted spatial features
        weighted_spatial = torch.sum(
            spatial_features * attention_weights,
            dim=1
        )  # [batch, feature_dim]
        
        # Temporal features with 3D CNN
        # Reshape for 3D conv: [batch, 1, frames, h, w]
        frames_3d = frames.mean(dim=2, keepdim=True)  # Convert to grayscale
        frames_3d = frames_3d.permute(0, 2, 1, 3, 4)  # [batch, 1, frames, h, w]
        
        # Resize to fixed size for 3D conv
        frames_3d = nn.functional.interpolate(
            frames_3d,
            size=(num_frames, 64, 64),
            mode='trilinear',
            align_corners=False
        )
        
        temporal_features = self.temporal_conv(frames_3d)  # [batch, 128, 1, 1, 1]
        temporal_features = temporal_features.view(batch_size, 128)
        
        # Fuse spatial and temporal features
        fused = torch.cat([weighted_spatial, temporal_features], dim=1)
        fused = self.fusion(fused)
        
        # Classification
        predictions = self.classifier(fused)
        
        return predictions, attention_weights.squeeze(-1)
    
    def predict(
        self,
        frames: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Predict if video is deepfake.
        
        Args:
            frames: Video frames numpy array [num_frames, height, width, channels]
            threshold: Classification threshold
            
        Returns:
            Dictionary with prediction results
        """
        self.eval()
        
        # Convert to tensor and normalize
        frames_tensor = torch.tensor(frames).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [frames, C, H, W]
        frames_tensor = frames_tensor / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        
        # Add batch dimension
        frames_tensor = frames_tensor.unsqueeze(0)
        
        with torch.no_grad():
            predictions, attention_weights = self.forward(frames_tensor)
            score = predictions.item()
            
        return {
            "authenticity_score": score,
            "is_authentic": score >= threshold,
            "is_deepfake": score < threshold,
            "confidence": abs(score - 0.5) * 2,
            "frame_attention": attention_weights[0].cpu().numpy().tolist()
        }


class LightweightVideoDetector(nn.Module):
    """
    Lightweight video detector for edge deployment.
    Uses MobileNetV3 for efficient spatial feature extraction.
    """
    
    def __init__(
        self,
        num_frames: int = 15,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.num_frames = num_frames
        
        # MobileNetV3-Small for spatial features
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        self.spatial_features = nn.Sequential(
            *list(mobilenet.children())[:-1]
        )
        
        self.feature_dim = 576  # MobileNetV3-Small output
        
        # Lightweight temporal modeling
        self.temporal_lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            frames: Video frames [batch, num_frames, channels, height, width]
            
        Returns:
            predictions: Authenticity scores [batch, 1]
        """
        batch_size, num_frames, c, h, w = frames.shape
        
        # Extract spatial features
        frames_flat = frames.view(batch_size * num_frames, c, h, w)
        spatial_features = self.spatial_features(frames_flat)
        spatial_features = spatial_features.view(batch_size, num_frames, self.feature_dim)
        
        # Temporal modeling
        lstm_out, _ = self.temporal_lstm(spatial_features)
        
        # Use last LSTM output
        final_features = lstm_out[:, -1, :]
        
        # Classification
        predictions = self.classifier(final_features)
        
        return predictions


class FaceManipulationDetector(nn.Module):
    """
    Specialized detector for face manipulation detection.
    Focuses on facial regions and inconsistencies.
    """
    
    def __init__(self):
        super().__init__()
        
        # Face-specific feature extractor
        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Inconsistency detector
        self.inconsistency_detector = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, face_crops: torch.Tensor) -> torch.Tensor:
        """
        Detect face manipulation.
        
        Args:
            face_crops: Cropped face regions [batch, 3, height, width]
            
        Returns:
            predictions: Manipulation scores [batch, 1]
        """
        features = self.face_encoder(face_crops)
        features = features.view(features.size(0), -1)
        predictions = self.inconsistency_detector(features)
        return predictions


def load_pretrained_video_model(
    model_path: str,
    device: str = "cpu"
) -> VideoDeepfakeDetector:
    """
    Load pretrained video detection model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = VideoDeepfakeDetector()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Test model
    model = VideoDeepfakeDetector(num_frames=30)
    
    # Dummy input (30 frames, 224x224)
    dummy_frames = torch.randn(2, 30, 3, 224, 224)
    
    predictions, attention = model(dummy_frames)
    print(f"Model output shape: {predictions.shape}")
    print(f"Attention weights shape: {attention.shape}")
    
    # Test lightweight model
    lightweight = LightweightVideoDetector(num_frames=15)
    dummy_frames_light = torch.randn(2, 15, 3, 224, 224)
    predictions = lightweight(dummy_frames_light)
    print(f"Lightweight model output: {predictions.shape}")
