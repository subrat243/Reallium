"""
Unit tests for video detection model.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.video_detector import (
    VideoDeepfakeDetector,
    LightweightVideoDetector,
    FaceManipulationDetector
)


class TestVideoDeepfakeDetector:
    """Test suite for video detection model."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return VideoDeepfakeDetector(
            num_frames=15,
            pretrained=False,  # Faster for testing
            freeze_backbone=True
        )
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.spatial_features is not None
        assert model.temporal_conv is not None
        assert model.frame_attention is not None
        assert model.classifier is not None
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        batch_size = 2
        num_frames = 15
        channels = 3
        height = 224
        width = 224
        
        dummy_frames = torch.randn(batch_size, num_frames, channels, height, width)
        
        model.eval()
        with torch.no_grad():
            predictions, attention_weights = model(dummy_frames)
        
        assert predictions.shape == (batch_size, 1)
        assert torch.all((predictions >= 0) & (predictions <= 1))
        assert attention_weights.shape == (batch_size, num_frames)
    
    def test_predict_method(self, model):
        """Test predict method."""
        num_frames = 15
        height = 224
        width = 224
        
        # Generate dummy frames
        frames = np.random.randint(0, 255, (num_frames, height, width, 3), dtype=np.uint8)
        
        result = model.predict(frames, threshold=0.5)
        
        assert 'authenticity_score' in result
        assert 'is_authentic' in result
        assert 'is_deepfake' in result
        assert 'confidence' in result
        assert 'frame_attention' in result
        
        assert 0 <= result['authenticity_score'] <= 1
        assert len(result['frame_attention']) == num_frames
    
    def test_frame_attention_weights(self, model):
        """Test frame attention weights sum to 1."""
        dummy_frames = torch.randn(1, 15, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            _, attention_weights = model(dummy_frames)
        
        # Attention weights should sum to approximately 1
        attention_sum = torch.sum(attention_weights, dim=1)
        assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=0.1)


class TestLightweightVideoDetector:
    """Test suite for lightweight video detector."""
    
    @pytest.fixture
    def model(self):
        """Create lightweight model."""
        return LightweightVideoDetector(
            num_frames=10,
            pretrained=False
        )
    
    def test_model_initialization(self, model):
        """Test initialization."""
        assert model is not None
        assert model.spatial_features is not None
        assert model.temporal_lstm is not None
        assert model.classifier is not None
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        batch_size = 2
        num_frames = 10
        
        dummy_frames = torch.randn(batch_size, num_frames, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            predictions = model(dummy_frames)
        
        assert predictions.shape == (batch_size, 1)
        assert torch.all((predictions >= 0) & (predictions <= 1))
    
    def test_model_size(self, model):
        """Test model is lightweight."""
        num_params = sum(p.numel() for p in model.parameters())
        
        # Should be less than 5M parameters
        assert num_params < 5_000_000


class TestFaceManipulationDetector:
    """Test suite for face manipulation detector."""
    
    @pytest.fixture
    def model(self):
        """Create model."""
        return FaceManipulationDetector()
    
    def test_model_initialization(self, model):
        """Test initialization."""
        assert model is not None
        assert model.face_encoder is not None
        assert model.inconsistency_detector is not None
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        batch_size = 4
        face_crops = torch.randn(batch_size, 3, 128, 128)
        
        model.eval()
        with torch.no_grad():
            predictions = model(face_crops)
        
        assert predictions.shape == (batch_size, 1)
        assert torch.all((predictions >= 0) & (predictions <= 1))


@pytest.mark.parametrize("num_frames", [5, 10, 15, 30])
def test_variable_frame_counts(num_frames):
    """Test with different frame counts."""
    model = LightweightVideoDetector(num_frames=num_frames)
    dummy_frames = torch.randn(1, num_frames, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        predictions = model(dummy_frames)
    
    assert predictions.shape == (1, 1)


def test_inference_speed():
    """Test inference speed is reasonable."""
    import time
    
    model = LightweightVideoDetector(num_frames=15)
    model.eval()
    
    dummy_frames = torch.randn(1, 15, 3, 224, 224)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_frames)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(20):
            start = time.time()
            _ = model(dummy_frames)
            times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    
    # Should be less than 500ms on CPU
    assert avg_time < 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
