"""
Detection service for deepfake analysis.
"""

import torch
import numpy as np
import cv2
import librosa
from pathlib import Path
from typing import Dict, List, Optional
import time
import hashlib
from sqlalchemy.orm import Session
import logging

from ..database.database import get_db
from ..database.models import Detection, AuditLog
from ..api.config import settings

logger = logging.getLogger(__name__)


class DetectionService:
    """Service for deepfake detection."""
    
    def __init__(self):
        self.device = settings.ML_DEVICE
        self.audio_model = None
        self.video_model = None
        self.fusion_model = None
        self._load_models()
    
    def _load_models(self):
        """Load ML models."""
        try:
            # TODO: Load actual models when trained
            logger.info("Loading detection models...")
            
            # Placeholder - would load actual models here
            # from models import load_pretrained_audio_model, load_pretrained_video_model
            # self.audio_model = load_pretrained_audio_model("path/to/model")
            # self.video_model = load_pretrained_video_model("path/to/model")
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    async def detect(
        self,
        file_path: str,
        media_type: str,
        threshold: float = 0.5,
        return_explainability: bool = False,
        user_id: str = None,
        db: Session = None
    ) -> Dict:
        """
        Detect deepfake in media file.
        
        Args:
            file_path: Path to media file
            media_type: Type of media ('audio' or 'video')
            threshold: Classification threshold
            return_explainability: Return explainability data
            user_id: User ID
            db: Database session
            
        Returns:
            Detection results
        """
        if db is None:
            db = next(get_db())
        
        start_time = time.time()
        
        try:
            # Compute file hash
            file_hash = self._compute_file_hash(file_path)
            
            # Check if already analyzed
            existing = db.query(Detection).filter(
                Detection.media_hash == file_hash,
                Detection.user_id == user_id
            ).first()
            
            if existing:
                logger.info(f"Using cached detection for hash {file_hash}")
                return self._detection_to_dict(existing)
            
            # Perform detection based on media type
            if media_type == "audio":
                result = await self._detect_audio(file_path, threshold)
            elif media_type == "video":
                result = await self._detect_video(file_path, threshold)
            else:
                raise ValueError(f"Unsupported media type: {media_type}")
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Save detection result
            detection = Detection(
                user_id=user_id,
                media_type=media_type,
                media_hash=file_hash,
                authenticity_score=result['authenticity_score'],
                is_authentic=result['is_authentic'],
                is_deepfake=result['is_deepfake'],
                confidence=result['confidence'],
                audio_score=result.get('audio_score'),
                video_score=result.get('video_score'),
                model_version=settings.AUDIO_MODEL_VERSION if media_type == "audio" else settings.VIDEO_MODEL_VERSION,
                explainability_data=result.get('explainability') if return_explainability else None,
                processing_time_ms=processing_time
            )
            
            db.add(detection)
            db.commit()
            db.refresh(detection)
            
            # Audit log
            audit = AuditLog(
                user_id=user_id,
                event_type="detection",
                event_description=f"Detection performed on {media_type}",
                metadata={
                    "detection_id": detection.id,
                    "is_deepfake": result['is_deepfake']
                }
            )
            db.add(audit)
            db.commit()
            
            return self._detection_to_dict(detection)
            
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            raise
    
    async def _detect_audio(self, file_path: str, threshold: float) -> Dict:
        """Detect audio deepfake."""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=16000)
            
            # TODO: Use actual model when available
            # For now, return mock result
            score = np.random.uniform(0.3, 0.9)
            
            return {
                "authenticity_score": float(score),
                "is_authentic": score >= threshold,
                "is_deepfake": score < threshold,
                "confidence": float(abs(score - 0.5) * 2),
                "audio_score": float(score)
            }
            
        except Exception as e:
            logger.error(f"Audio detection error: {e}")
            raise
    
    async def _detect_video(self, file_path: str, threshold: float) -> Dict:
        """Detect video deepfake."""
        try:
            # Load video
            cap = cv2.VideoCapture(file_path)
            
            frames = []
            frame_count = 0
            max_frames = 30
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            # TODO: Use actual model when available
            # For now, return mock result
            score = np.random.uniform(0.3, 0.9)
            
            return {
                "authenticity_score": float(score),
                "is_authentic": score >= threshold,
                "is_deepfake": score < threshold,
                "confidence": float(abs(score - 0.5) * 2),
                "video_score": float(score)
            }
            
        except Exception as e:
            logger.error(f"Video detection error: {e}")
            raise
    
    def _detection_to_dict(self, detection: Detection) -> Dict:
        """Convert detection model to dictionary."""
        return {
            "detection_id": detection.id,
            "authenticity_score": detection.authenticity_score,
            "is_authentic": detection.is_authentic,
            "is_deepfake": detection.is_deepfake,
            "confidence": detection.confidence,
            "modality_scores": {
                "audio": detection.audio_score,
                "video": detection.video_score
            } if detection.audio_score or detection.video_score else None,
            "explainability": detection.explainability_data,
            "processing_time_ms": detection.processing_time_ms
        }
    
    async def get_user_history(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
        db: Session = None
    ) -> List[Dict]:
        """Get user detection history."""
        if db is None:
            db = next(get_db())
        
        detections = db.query(Detection).filter(
            Detection.user_id == user_id
        ).order_by(
            Detection.created_at.desc()
        ).limit(limit).offset(offset).all()
        
        return [self._detection_to_dict(d) for d in detections]
    
    async def get_detection(
        self,
        detection_id: str,
        user_id: str,
        db: Session = None
    ) -> Optional[Dict]:
        """Get specific detection."""
        if db is None:
            db = next(get_db())
        
        detection = db.query(Detection).filter(
            Detection.id == detection_id,
            Detection.user_id == user_id
        ).first()
        
        if detection:
            return self._detection_to_dict(detection)
        return None
    
    async def delete_detection(
        self,
        detection_id: str,
        user_id: str,
        db: Session = None
    ) -> bool:
        """Delete detection."""
        if db is None:
            db = next(get_db())
        
        detection = db.query(Detection).filter(
            Detection.id == detection_id,
            Detection.user_id == user_id
        ).first()
        
        if detection:
            db.delete(detection)
            db.commit()
            return True
        return False
    
    async def submit_feedback(
        self,
        detection_id: str,
        user_id: str,
        is_correct: bool,
        notes: Optional[str] = None,
        db: Session = None
    ):
        """Submit feedback on detection."""
        if db is None:
            db = next(get_db())
        
        from ..database.models import Feedback
        
        feedback = Feedback(
            detection_id=detection_id,
            user_id=user_id,
            is_correct=is_correct,
            notes=notes
        )
        
        db.add(feedback)
        db.commit()
        
        logger.info(f"Feedback submitted for detection {detection_id}")
