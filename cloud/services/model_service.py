"""
Model management service.
"""

from typing import List, Dict, Optional
from pathlib import Path
from sqlalchemy.orm import Session
import logging

from ..database.database import get_db
from ..database.models import ModelVersion
from ..api.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """Service for model management."""
    
    def __init__(self):
        self.storage_path = Path(settings.MODEL_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def list_models(
        self,
        model_type: Optional[str] = None,
        active_only: bool = True,
        db: Session = None
    ) -> List[Dict]:
        """
        List available models.
        
        Args:
            model_type: Filter by model type
            active_only: Only return active models
            db: Database session
            
        Returns:
            List of models
        """
        if db is None:
            db = next(get_db())
        
        query = db.query(ModelVersion)
        
        if model_type:
            query = query.filter(ModelVersion.model_type == model_type)
        
        if active_only:
            query = query.filter(ModelVersion.is_active == True)
        
        models = query.order_by(ModelVersion.created_at.desc()).all()
        
        return [self._model_to_dict(m) for m in models]
    
    async def get_model(self, model_id: str, db: Session = None) -> Optional[Dict]:
        """Get model by ID."""
        if db is None:
            db = next(get_db())
        
        model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
        
        if model:
            return self._model_to_dict(model)
        return None
    
    async def upload_model(
        self,
        file,
        name: str,
        version: str,
        model_type: str,
        uploaded_by: str,
        db: Session = None
    ) -> Dict:
        """
        Upload new model.
        
        Args:
            file: Uploaded file
            name: Model name
            version: Model version
            model_type: Model type
            uploaded_by: User ID
            db: Database session
            
        Returns:
            Created model info
        """
        if db is None:
            db = next(get_db())
        
        # Save file
        file_path = self.storage_path / f"{name}_{version}_{model_type}.tflite"
        
        contents = await file.read()
        file_path.write_bytes(contents)
        
        file_size_mb = len(contents) / (1024 * 1024)
        
        # Create model version
        model = ModelVersion(
            name=name,
            version=version,
            model_type=model_type,
            file_path=str(file_path),
            file_size_mb=file_size_mb,
            uploaded_by=uploaded_by
        )
        
        db.add(model)
        db.commit()
        db.refresh(model)
        
        logger.info(f"Model uploaded: {name} v{version}")
        
        return self._model_to_dict(model)
    
    async def activate_model(self, model_id: str, db: Session = None) -> bool:
        """Activate a model version."""
        if db is None:
            db = next(get_db())
        
        model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
        
        if not model:
            return False
        
        # Deactivate other models of same type
        db.query(ModelVersion).filter(
            ModelVersion.model_type == model.model_type,
            ModelVersion.is_active == True
        ).update({"is_active": False})
        
        # Activate this model
        model.is_active = True
        db.commit()
        
        logger.info(f"Model activated: {model.name} v{model.version}")
        
        return True
    
    async def delete_model(self, model_id: str, db: Session = None) -> bool:
        """Delete a model."""
        if db is None:
            db = next(get_db())
        
        model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
        
        if not model:
            return False
        
        # Delete file
        file_path = Path(model.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete from database
        db.delete(model)
        db.commit()
        
        logger.info(f"Model deleted: {model.name} v{model.version}")
        
        return True
    
    async def get_latest_edge_models(self, db: Session = None) -> Dict:
        """Get latest models for edge deployment."""
        if db is None:
            db = next(get_db())
        
        audio_model = db.query(ModelVersion).filter(
            ModelVersion.model_type == "audio",
            ModelVersion.is_active == True
        ).first()
        
        video_model = db.query(ModelVersion).filter(
            ModelVersion.model_type == "video",
            ModelVersion.is_active == True
        ).first()
        
        fusion_model = db.query(ModelVersion).filter(
            ModelVersion.model_type == "fusion",
            ModelVersion.is_active == True
        ).first()
        
        return {
            "audio": self._model_to_dict(audio_model) if audio_model else None,
            "video": self._model_to_dict(video_model) if video_model else None,
            "fusion": self._model_to_dict(fusion_model) if fusion_model else None
        }
    
    async def check_updates(
        self,
        current_versions: Dict[str, str],
        db: Session = None
    ) -> List[Dict]:
        """
        Check for model updates.
        
        Args:
            current_versions: Current model versions on edge device
            db: Database session
            
        Returns:
            List of available updates
        """
        if db is None:
            db = next(get_db())
        
        updates = []
        
        for model_type, current_version in current_versions.items():
            latest = db.query(ModelVersion).filter(
                ModelVersion.model_type == model_type,
                ModelVersion.is_active == True
            ).first()
            
            if latest and latest.version != current_version:
                updates.append(self._model_to_dict(latest))
        
        return updates
    
    def _model_to_dict(self, model: ModelVersion) -> Dict:
        """Convert model to dictionary."""
        return {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "model_type": model.model_type,
            "size_mb": model.file_size_mb,
            "accuracy": model.accuracy,
            "created_at": model.created_at.isoformat(),
            "is_active": model.is_active
        }
