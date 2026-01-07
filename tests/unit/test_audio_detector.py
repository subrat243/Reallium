"""
Unit tests for audio detection model.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.audio_detector import (
    AudioDeepfakeDetector,
    LightweightAudioDetector,
    load_pretrained_audio_model
)


class TestAudioDeepfakeDetector:
    """Test suite for audio detection model."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return AudioDeepfakeDetector(
            pretrained_model="facebook/wav2vec2-base",
            hidden_size=128,
            num_layers=1,
            freeze_encoder=True
        )
    
    def test_model_initialization(self, model):
        """Test model can be initialized."""
        assert model is not None
        assert model.wav2vec2 is not None
        assert model.lstm is not None
        assert model.attention is not None
        assert model.classifier is not None
    
    def test_forward_pass(self, model):
        """Test forward pass with dummy data."""
        # 10 seconds at 16kHz
        batch_size = 2
        audio_length = 16000 * 10
        
        dummy_audio = torch.randn(batch_size, audio_length)
        
        model.eval()
        with torch.no_grad():
            predictions, attention_weights = model(dummy_audio)
        
        assert predictions.shape == (batch_size, 1)
        assert torch.all((predictions >= 0) & (predictions <= 1))
        assert attention_weights is not None
    
    def test_predict_method(self, model):
        """Test predict method."""
        # Generate dummy audio
        audio = np.random.randn(16000 * 5).astype(np.float32)
        
        result = model.predict(audio, threshold=0.5)
        
        assert 'authenticity_score' in result
        assert 'is_authentic' in result
        assert 'is_deepfake' in result
        assert 'confidence' in result
        
        assert 0 <= result['authenticity_score'] <= 1
        assert 0 <= result['confidence'] <= 1
        assert result['is_authentic'] != result['is_deepfake']
    
    def test_different_audio_lengths(self, model):
        """Test with different audio lengths."""
        lengths = [16000, 16000 * 5, 16000 * 10]
        
        for length in lengths:
            audio = np.random.randn(length).astype(np.float32)
            result = model.predict(audio)
            assert result is not None


class TestLightweightAudioDetector:
    """Test suite for lightweight audio detector."""
    
    @pytest.fixture
    def model(self):
        """Create lightweight model instance."""
        return LightweightAudioDetector(
            input_features=128,
            hidden_size=64,
            num_layers=1
        )
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.conv_layers is not None
        assert model.lstm is not None
        assert model.classifier is not None
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        batch_size = 2
        features = 128
        time_steps = 1000
        
        dummy_input = torch.randn(batch_size, features, time_steps)
        
        model.eval()
        with torch.no_grad():
            predictions = model(dummy_input)
        
        assert predictions.shape == (batch_size, 1)
        assert torch.all((predictions >= 0) & (predictions <= 1))
    
    def test_model_size(self, model):
        """Test model is lightweight."""
        num_params = sum(p.numel() for p in model.parameters())
        
        # Should be less than 1M parameters
        assert num_params < 1_000_000


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_threshold_variations(threshold):
    """Test different classification thresholds."""
    model = LightweightAudioDetector()
    dummy_input = torch.randn(1, 128, 1000)
    
    model.eval()
    with torch.no_grad():
        predictions = model(dummy_input)
    
    score = predictions.item()
    is_authentic = score >= threshold
    is_deepfake = score < threshold
    
    assert is_authentic != is_deepfake


def test_batch_processing():
    """Test batch processing capability."""
    model = LightweightAudioDetector()
    
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 128, 1000)
        
        model.eval()
        with torch.no_grad():
            predictions = model(dummy_input)
        
        assert predictions.shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
