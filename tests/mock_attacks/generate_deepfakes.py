"""
Mock Deepfake Generation

Generate synthetic deepfakes for testing purposes.
Uses open-source tools to create test datasets.
"""

import numpy as np
import torch
import torchaudio
import cv2
from pathlib import Path
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockAudioDeepfakeGenerator:
    """
    Generate mock audio deepfakes for testing.
    """
    
    def __init__(self, output_dir: str = "tests/fixtures/audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_synthetic_voice(
        self,
        duration_seconds: float = 10.0,
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, str]:
        """
        Generate synthetic voice using simple signal processing.
        
        Args:
            duration_seconds: Duration of audio
            sample_rate: Sample rate
            
        Returns:
            Audio array and file path
        """
        logger.info(f"Generating synthetic voice ({duration_seconds}s)")
        
        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, num_samples)
        
        # Generate synthetic voice with multiple harmonics
        fundamental_freq = 150  # Hz (typical male voice)
        audio = np.zeros(num_samples)
        
        for harmonic in range(1, 6):
            freq = fundamental_freq * harmonic
            amplitude = 1.0 / harmonic
            audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add formants (resonances)
        formant_freqs = [800, 1200, 2500]  # Hz
        for formant_freq in formant_freqs:
            audio += 0.3 * np.sin(2 * np.pi * formant_freq * t)
        
        # Add noise
        noise = np.random.normal(0, 0.05, num_samples)
        audio += noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Save
        output_path = self.output_dir / f"synthetic_voice_{duration_seconds}s.wav"
        torchaudio.save(
            str(output_path),
            torch.tensor(audio).unsqueeze(0),
            sample_rate
        )
        
        logger.info(f"Saved to {output_path}")
        return audio, str(output_path)
    
    def generate_voice_cloning_artifact(
        self,
        duration_seconds: float = 10.0,
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, str]:
        """
        Generate audio with voice cloning artifacts.
        
        Args:
            duration_seconds: Duration
            sample_rate: Sample rate
            
        Returns:
            Audio array and file path
        """
        logger.info("Generating voice cloning artifact")
        
        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, num_samples)
        
        # Base signal
        audio = np.sin(2 * np.pi * 200 * t)
        
        # Add spectral artifacts (common in TTS)
        for i in range(0, num_samples, sample_rate // 10):
            audio[i:i+100] *= 0.5  # Periodic drops
        
        # Add phase discontinuities
        for i in range(0, num_samples, sample_rate):
            audio[i:i+50] = -audio[i:i+50]
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Save
        output_path = self.output_dir / f"cloning_artifact_{duration_seconds}s.wav"
        torchaudio.save(
            str(output_path),
            torch.tensor(audio).unsqueeze(0),
            sample_rate
        )
        
        return audio, str(output_path)


class MockVideoDeepfakeGenerator:
    """
    Generate mock video deepfakes for testing.
    """
    
    def __init__(self, output_dir: str = "tests/fixtures/video"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_synthetic_face(
        self,
        num_frames: int = 30,
        width: int = 224,
        height: int = 224
    ) -> Tuple[np.ndarray, str]:
        """
        Generate synthetic face video.
        
        Args:
            num_frames: Number of frames
            width: Frame width
            height: Frame height
            
        Returns:
            Video frames and file path
        """
        logger.info(f"Generating synthetic face video ({num_frames} frames)")
        
        frames = []
        
        for i in range(num_frames):
            # Create frame with moving circle (simulating face)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Moving circle
            center_x = int(width / 2 + 50 * np.sin(2 * np.pi * i / num_frames))
            center_y = int(height / 2)
            radius = 50
            
            cv2.circle(frame, (center_x, center_y), radius, (255, 200, 150), -1)
            
            # Add eyes
            cv2.circle(frame, (center_x - 15, center_y - 10), 5, (0, 0, 0), -1)
            cv2.circle(frame, (center_x + 15, center_y - 10), 5, (0, 0, 0), -1)
            
            # Add noise
            noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            frames.append(frame)
        
        frames = np.array(frames)
        
        # Save video
        output_path = self.output_dir / f"synthetic_face_{num_frames}frames.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        logger.info(f"Saved to {output_path}")
        return frames, str(output_path)
    
    def generate_face_swap_artifact(
        self,
        num_frames: int = 30,
        width: int = 224,
        height: int = 224
    ) -> Tuple[np.ndarray, str]:
        """
        Generate video with face swap artifacts.
        
        Args:
            num_frames: Number of frames
            width: Frame width
            height: Frame height
            
        Returns:
            Video frames and file path
        """
        logger.info("Generating face swap artifact")
        
        frames = []
        
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Face region
            center_x, center_y = width // 2, height // 2
            cv2.circle(frame, (center_x, center_y), 60, (255, 200, 150), -1)
            
            # Add blending artifacts (sharp boundaries)
            if i % 5 == 0:  # Periodic artifacts
                cv2.circle(frame, (center_x, center_y), 61, (200, 150, 100), 2)
            
            # Color mismatch
            if i % 3 == 0:
                frame[:, :, 0] = frame[:, :, 0] * 0.8  # Reduce blue channel
            
            frames.append(frame)
        
        frames = np.array(frames)
        
        # Save
        output_path = self.output_dir / f"face_swap_artifact_{num_frames}frames.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        return frames, str(output_path)


def generate_test_dataset(
    num_audio_samples: int = 10,
    num_video_samples: int = 10
):
    """
    Generate complete test dataset.
    
    Args:
        num_audio_samples: Number of audio samples
        num_video_samples: Number of video samples
    """
    logger.info("Generating test dataset...")
    
    # Audio
    audio_gen = MockAudioDeepfakeGenerator()
    
    for i in range(num_audio_samples // 2):
        audio_gen.generate_synthetic_voice(duration_seconds=5.0 + i)
        audio_gen.generate_voice_cloning_artifact(duration_seconds=5.0 + i)
    
    # Video
    video_gen = MockVideoDeepfakeGenerator()
    
    for i in range(num_video_samples // 2):
        video_gen.generate_synthetic_face(num_frames=30 + i * 10)
        video_gen.generate_face_swap_artifact(num_frames=30 + i * 10)
    
    logger.info("Test dataset generation complete!")


if __name__ == "__main__":
    generate_test_dataset(num_audio_samples=10, num_video_samples=10)
