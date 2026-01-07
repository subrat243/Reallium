"""
Audio Deepfake Detection Model

This module implements a deep learning model for detecting audio deepfakes
including voice cloning, synthetic speech, and audio manipulation.

Architecture:
- Wav2Vec2 for feature extraction
- BiLSTM for temporal modeling
- Attention mechanism for focus on manipulated regions
- Binary classification head
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Dict, Tuple, Optional
import numpy as np


class AudioDeepfakeDetector(nn.Module):
    """
    Audio deepfake detection model using Wav2Vec2 + BiLSTM + Attention.
    
    Args:
        pretrained_model: Wav2Vec2 model name or path
        hidden_size: Hidden size for LSTM layers
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        freeze_encoder: Whether to freeze Wav2Vec2 encoder
    """
    
    def __init__(
        self,
        pretrained_model: str = "facebook/wav2vec2-base",
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        # Wav2Vec2 feature extractor
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model)
        
        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Get feature dimension from Wav2Vec2
        self.feature_dim = self.wav2vec2.config.hidden_size
        
        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_values: Audio waveform tensor [batch, time]
            attention_mask: Attention mask [batch, time]
            
        Returns:
            predictions: Authenticity scores [batch, 1]
            attention_weights: Attention weights [batch, time]
        """
        # Extract features with Wav2Vec2
        wav2vec_output = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        features = wav2vec_output.last_hidden_state  # [batch, time, feature_dim]
        
        # Temporal modeling with BiLSTM
        lstm_output, _ = self.lstm(features)  # [batch, time, hidden_size*2]
        
        # Self-attention
        attn_output, attn_weights = self.attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # Global average pooling
        pooled = torch.mean(attn_output, dim=1)  # [batch, hidden_size*2]
        
        # Classification
        predictions = self.classifier(pooled)  # [batch, 1]
        
        return predictions, attn_weights
    
    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Predict if audio is deepfake.
        
        Args:
            audio: Audio waveform numpy array
            sample_rate: Sample rate (must be 16kHz for Wav2Vec2)
            threshold: Classification threshold
            
        Returns:
            Dictionary with prediction results
        """
        self.eval()
        
        # Convert to tensor
        input_values = torch.tensor(audio).unsqueeze(0).float()
        
        with torch.no_grad():
            predictions, attention_weights = self.forward(input_values)
            score = predictions.item()
            
        return {
            "authenticity_score": score,
            "is_authentic": score >= threshold,
            "is_deepfake": score < threshold,
            "confidence": abs(score - 0.5) * 2  # 0-1 scale
        }


class LightweightAudioDetector(nn.Module):
    """
    Lightweight audio detector for edge deployment.
    Uses smaller architecture optimized for mobile devices.
    """
    
    def __init__(
        self,
        input_features: int = 128,  # Mel spectrogram features
        hidden_size: int = 128,
        num_layers: int = 1
    ):
        super().__init__()
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Mel spectrogram features [batch, features, time]
            
        Returns:
            predictions: Authenticity scores [batch, 1]
        """
        # Convolutional features
        conv_out = self.conv_layers(x)  # [batch, 256, time/8]
        
        # Reshape for LSTM
        conv_out = conv_out.permute(0, 2, 1)  # [batch, time/8, 256]
        
        # LSTM
        lstm_out, _ = self.lstm(conv_out)
        
        # Global average pooling
        pooled = torch.mean(lstm_out, dim=1)
        
        # Classification
        predictions = self.classifier(pooled)
        
        return predictions


def load_pretrained_audio_model(
    model_path: str,
    device: str = "cpu"
) -> AudioDeepfakeDetector:
    """
    Load pretrained audio detection model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = AudioDeepfakeDetector()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Test model
    model = AudioDeepfakeDetector()
    
    # Dummy input (10 seconds at 16kHz)
    dummy_audio = torch.randn(2, 16000 * 10)
    
    predictions, attention = model(dummy_audio)
    print(f"Model output shape: {predictions.shape}")
    print(f"Attention weights shape: {attention.shape}")
    
    # Test lightweight model
    lightweight = LightweightAudioDetector()
    dummy_mel = torch.randn(2, 128, 1000)  # Mel spectrogram
    predictions = lightweight(dummy_mel)
    print(f"Lightweight model output: {predictions.shape}")
