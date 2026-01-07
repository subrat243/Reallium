"""
Multimodal Fusion Model

This module combines audio and video predictions for improved deepfake detection.
Uses late fusion with learned weights.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np

from .audio_detector import AudioDeepfakeDetector
from .video_detector import VideoDeepfakeDetector


class MultimodalFusionDetector(nn.Module):
    """
    Multimodal fusion model combining audio and video predictions.
    
    Args:
        audio_model: Pretrained audio detection model
        video_model: Pretrained video detection model
        fusion_method: Fusion method ('concat', 'attention', 'weighted')
        freeze_modalities: Whether to freeze audio/video models
    """
    
    def __init__(
        self,
        audio_model: Optional[AudioDeepfakeDetector] = None,
        video_model: Optional[VideoDeepfakeDetector] = None,
        fusion_method: str = 'attention',
        freeze_modalities: bool = True
    ):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # Initialize models if not provided
        self.audio_model = audio_model or AudioDeepfakeDetector()
        self.video_model = video_model or VideoDeepfakeDetector()
        
        if freeze_modalities:
            for param in self.audio_model.parameters():
                param.requires_grad = False
            for param in self.video_model.parameters():
                param.requires_grad = False
        
        # Fusion layers based on method
        if fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif fusion_method == 'attention':
            self.audio_attention = nn.Linear(1, 1)
            self.video_attention = nn.Linear(1, 1)
            self.fusion = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif fusion_method == 'weighted':
            # Learnable weights
            self.audio_weight = nn.Parameter(torch.tensor(0.5))
            self.video_weight = nn.Parameter(torch.tensor(0.5))
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(
        self,
        audio_input: Optional[torch.Tensor] = None,
        video_input: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with multimodal fusion.
        
        Args:
            audio_input: Audio waveform [batch, time]
            video_input: Video frames [batch, num_frames, C, H, W]
            audio_mask: Audio attention mask
            
        Returns:
            predictions: Fused authenticity scores [batch, 1]
            modality_outputs: Individual modality predictions
        """
        modality_outputs = {}
        
        # Audio prediction
        if audio_input is not None:
            audio_pred, audio_attn = self.audio_model(audio_input, audio_mask)
            modality_outputs['audio'] = audio_pred
            modality_outputs['audio_attention'] = audio_attn
        else:
            audio_pred = None
        
        # Video prediction
        if video_input is not None:
            video_pred, video_attn = self.video_model(video_input)
            modality_outputs['video'] = video_pred
            modality_outputs['video_attention'] = video_attn
        else:
            video_pred = None
        
        # Fusion
        if audio_pred is not None and video_pred is not None:
            # Both modalities available
            if self.fusion_method == 'concat':
                combined = torch.cat([audio_pred, video_pred], dim=1)
                fused_pred = self.fusion(combined)
                
            elif self.fusion_method == 'attention':
                # Compute attention weights for each modality
                audio_attn_weight = torch.sigmoid(self.audio_attention(audio_pred))
                video_attn_weight = torch.sigmoid(self.video_attention(video_pred))
                
                # Normalize weights
                total_weight = audio_attn_weight + video_attn_weight
                audio_attn_weight = audio_attn_weight / total_weight
                video_attn_weight = video_attn_weight / total_weight
                
                # Weighted combination
                weighted = torch.cat([
                    audio_pred * audio_attn_weight,
                    video_pred * video_attn_weight
                ], dim=1)
                fused_pred = self.fusion(weighted)
                
                modality_outputs['audio_fusion_weight'] = audio_attn_weight
                modality_outputs['video_fusion_weight'] = video_attn_weight
                
            elif self.fusion_method == 'weighted':
                # Learnable weighted average
                weights = torch.softmax(
                    torch.stack([self.audio_weight, self.video_weight]),
                    dim=0
                )
                fused_pred = (
                    audio_pred * weights[0] +
                    video_pred * weights[1]
                )
                
                modality_outputs['audio_weight'] = weights[0]
                modality_outputs['video_weight'] = weights[1]
        
        elif audio_pred is not None:
            # Only audio available
            fused_pred = audio_pred
        elif video_pred is not None:
            # Only video available
            fused_pred = video_pred
        else:
            raise ValueError("At least one modality must be provided")
        
        return fused_pred, modality_outputs
    
    def predict(
        self,
        audio: Optional[np.ndarray] = None,
        frames: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Predict if media is deepfake using multimodal fusion.
        
        Args:
            audio: Audio waveform numpy array
            frames: Video frames numpy array
            threshold: Classification threshold
            
        Returns:
            Dictionary with prediction results
        """
        self.eval()
        
        # Prepare inputs
        audio_tensor = None
        if audio is not None:
            audio_tensor = torch.tensor(audio).unsqueeze(0).float()
        
        video_tensor = None
        if frames is not None:
            frames_tensor = torch.tensor(frames).float()
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [frames, C, H, W]
            frames_tensor = frames_tensor / 255.0
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames_tensor = (frames_tensor - mean) / std
            video_tensor = frames_tensor.unsqueeze(0)
        
        with torch.no_grad():
            predictions, modality_outputs = self.forward(
                audio_input=audio_tensor,
                video_input=video_tensor
            )
            score = predictions.item()
        
        result = {
            "authenticity_score": score,
            "is_authentic": score >= threshold,
            "is_deepfake": score < threshold,
            "confidence": abs(score - 0.5) * 2,
            "modality_scores": {}
        }
        
        # Add individual modality scores
        if 'audio' in modality_outputs:
            result['modality_scores']['audio'] = modality_outputs['audio'].item()
        if 'video' in modality_outputs:
            result['modality_scores']['video'] = modality_outputs['video'].item()
        
        # Add fusion weights if available
        if 'audio_fusion_weight' in modality_outputs:
            result['fusion_weights'] = {
                'audio': modality_outputs['audio_fusion_weight'].item(),
                'video': modality_outputs['video_fusion_weight'].item()
            }
        elif 'audio_weight' in modality_outputs:
            result['fusion_weights'] = {
                'audio': modality_outputs['audio_weight'].item(),
                'video': modality_outputs['video_weight'].item()
            }
        
        return result


class EnsembleDetector(nn.Module):
    """
    Ensemble of multiple detection models for improved robustness.
    """
    
    def __init__(
        self,
        models: list,
        voting_method: str = 'soft'  # 'soft' or 'hard'
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.voting_method = voting_method
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Returns:
            predictions: Ensemble predictions [batch, 1]
        """
        predictions = []
        
        for model in self.models:
            if isinstance(model, MultimodalFusionDetector):
                pred, _ = model(*args, **kwargs)
            else:
                pred = model(*args, **kwargs)
                if isinstance(pred, tuple):
                    pred = pred[0]
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        if self.voting_method == 'soft':
            # Average probabilities
            ensemble_pred = torch.mean(predictions, dim=0)
        else:
            # Majority voting
            binary_preds = (predictions > 0.5).float()
            ensemble_pred = (torch.mean(binary_preds, dim=0) > 0.5).float()
        
        return ensemble_pred


def create_multimodal_detector(
    audio_checkpoint: Optional[str] = None,
    video_checkpoint: Optional[str] = None,
    fusion_method: str = 'attention',
    device: str = 'cpu'
) -> MultimodalFusionDetector:
    """
    Create multimodal detector with pretrained models.
    
    Args:
        audio_checkpoint: Path to audio model checkpoint
        video_checkpoint: Path to video model checkpoint
        fusion_method: Fusion method to use
        device: Device to load models on
        
    Returns:
        Multimodal fusion detector
    """
    # Load audio model
    audio_model = None
    if audio_checkpoint:
        audio_model = AudioDeepfakeDetector()
        checkpoint = torch.load(audio_checkpoint, map_location=device)
        audio_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load video model
    video_model = None
    if video_checkpoint:
        video_model = VideoDeepfakeDetector()
        checkpoint = torch.load(video_checkpoint, map_location=device)
        video_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create fusion model
    model = MultimodalFusionDetector(
        audio_model=audio_model,
        video_model=video_model,
        fusion_method=fusion_method,
        freeze_modalities=True
    )
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test multimodal fusion
    model = MultimodalFusionDetector(fusion_method='attention')
    
    # Dummy inputs
    dummy_audio = torch.randn(2, 16000 * 10)
    dummy_video = torch.randn(2, 30, 3, 224, 224)
    
    predictions, outputs = model(
        audio_input=dummy_audio,
        video_input=dummy_video
    )
    
    print(f"Fused predictions shape: {predictions.shape}")
    print(f"Modality outputs: {outputs.keys()}")
    
    # Test with only audio
    audio_only, _ = model(audio_input=dummy_audio)
    print(f"Audio-only prediction: {audio_only.shape}")
    
    # Test with only video
    video_only, _ = model(video_input=dummy_video)
    print(f"Video-only prediction: {video_only.shape}")
