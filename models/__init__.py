"""Models package for deepfake detection."""

from .audio_detector import (
    AudioDeepfakeDetector,
    LightweightAudioDetector,
    load_pretrained_audio_model
)

from .video_detector import (
    VideoDeepfakeDetector,
    LightweightVideoDetector,
    FaceManipulationDetector,
    load_pretrained_video_model
)

from .multimodal_fusion import (
    MultimodalFusionDetector,
    EnsembleDetector,
    create_multimodal_detector
)

__all__ = [
    'AudioDeepfakeDetector',
    'LightweightAudioDetector',
    'VideoDeepfakeDetector',
    'LightweightVideoDetector',
    'FaceManipulationDetector',
    'MultimodalFusionDetector',
    'EnsembleDetector',
    'load_pretrained_audio_model',
    'load_pretrained_video_model',
    'create_multimodal_detector'
]
