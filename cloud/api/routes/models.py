"""
Model Management API Routes

Endpoints for model versioning and distribution.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import logging

from ..config import settings
from ..database.models import User, ModelVersion
from ..services.model_service import ModelService
from ..middleware.auth import get_current_user, require_role

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize model service
model_service = ModelService()


class ModelInfo(BaseModel):
    """Model information model."""
    id: str
    name: str
    version: str
    model_type: str  # 'audio', 'video', 'fusion'
    size_mb: float
    accuracy: Optional[float] = None
    created_at: datetime
    is_active: bool


class ModelListResponse(BaseModel):
    """Model list response."""
    models: List[ModelInfo]
    total: int


@router.get("/", response_model=ModelListResponse)
async def list_models(
    model_type: Optional[str] = None,
    active_only: bool = True,
    current_user: User = Depends(get_current_user)
):
    """
    List available models.
    
    Args:
        model_type: Filter by model type
        active_only: Only return active models
        current_user: Authenticated user
        
    Returns:
        List of models
    """
    models = await model_service.list_models(
        model_type=model_type,
        active_only=active_only
    )
    
    return ModelListResponse(
        models=[ModelInfo(**m) for m in models],
        total=len(models)
    )


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get model information.
    
    Args:
        model_id: Model ID
        current_user: Authenticated user
        
    Returns:
        Model information
    """
    model = await model_service.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelInfo(**model)


@router.post("/upload", response_model=ModelInfo)
async def upload_model(
    file: UploadFile = File(...),
    name: str = None,
    version: str = None,
    model_type: str = None,
    current_user: User = Depends(require_role("admin"))
):
    """
    Upload a new model.
    
    Args:
        file: Model file
        name: Model name
        version: Model version
        model_type: Model type
        current_user: Authenticated admin user
        
    Returns:
        Uploaded model information
    """
    try:
        model = await model_service.upload_model(
            file=file,
            name=name,
            version=version,
            model_type=model_type,
            uploaded_by=current_user.id
        )
        
        return ModelInfo(**model)
        
    except Exception as e:
        logger.error(f"Model upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/activate")
async def activate_model(
    model_id: str,
    current_user: User = Depends(require_role("admin"))
):
    """
    Activate a model version.
    
    Args:
        model_id: Model ID
        current_user: Authenticated admin user
        
    Returns:
        Success message
    """
    success = await model_service.activate_model(model_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"message": "Model activated successfully"}


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    current_user: User = Depends(require_role("admin"))
):
    """
    Delete a model.
    
    Args:
        model_id: Model ID
        current_user: Authenticated admin user
        
    Returns:
        Success message
    """
    success = await model_service.delete_model(model_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"message": "Model deleted successfully"}


@router.get("/edge/latest")
async def get_latest_edge_models(
    current_user: User = Depends(get_current_user)
):
    """
    Get latest models for edge deployment.
    
    Args:
        current_user: Authenticated user
        
    Returns:
        Latest model versions and download URLs
    """
    models = await model_service.get_latest_edge_models()
    
    return {
        "audio_model": models.get("audio"),
        "video_model": models.get("video"),
        "fusion_model": models.get("fusion")
    }


@router.post("/edge/check-updates")
async def check_model_updates(
    current_versions: dict,
    current_user: User = Depends(get_current_user)
):
    """
    Check for model updates.
    
    Args:
        current_versions: Current model versions on edge device
        current_user: Authenticated user
        
    Returns:
        Available updates
    """
    updates = await model_service.check_updates(current_versions)
    
    return {
        "updates_available": len(updates) > 0,
        "updates": updates
    }
