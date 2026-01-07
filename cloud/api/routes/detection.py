"""
Detection API Routes

Endpoints for deepfake detection.
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List
from pydantic import BaseModel
import tempfile
import os
from pathlib import Path
import logging

from ..config import settings
from ..database.models import Detection, User
from ..services.detection_service import DetectionService
from ..middleware.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize detection service
detection_service = DetectionService()


class DetectionRequest(BaseModel):
    """Detection request model."""
    media_url: Optional[str] = None
    threshold: float = 0.5
    return_explainability: bool = False


class DetectionResponse(BaseModel):
    """Detection response model."""
    detection_id: str
    authenticity_score: float
    is_authentic: bool
    is_deepfake: bool
    confidence: float
    modality_scores: Optional[Dict[str, float]] = None
    explainability: Optional[Dict] = None
    processing_time_ms: float


@router.post("/analyze", response_model=DetectionResponse)
async def analyze_media(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    return_explainability: bool = False,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze uploaded media for deepfake detection.
    
    Args:
        file: Uploaded media file (audio/video)
        threshold: Classification threshold
        return_explainability: Return explainability data
        current_user: Authenticated user
        
    Returns:
        Detection results
    """
    # Validate file size
    file_size_mb = 0
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    
    if file_size_mb > settings.MEDIA_MAX_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MEDIA_MAX_SIZE_MB}MB"
        )
    
    # Save to temporary file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name
    
    try:
        # Detect media type
        media_type = "video" if suffix.lower() in ['.mp4', '.avi', '.mov'] else "audio"
        
        # Run detection
        result = await detection_service.detect(
            file_path=temp_path,
            media_type=media_type,
            threshold=threshold,
            return_explainability=return_explainability,
            user_id=current_user.id
        )
        
        return DetectionResponse(**result)
        
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post("/batch", response_model=List[DetectionResponse])
async def batch_analyze(
    files: List[UploadFile] = File(...),
    threshold: float = 0.5,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user)
):
    """
    Batch analyze multiple media files.
    
    Args:
        files: List of uploaded files
        threshold: Classification threshold
        background_tasks: Background task handler
        current_user: Authenticated user
        
    Returns:
        List of detection results
    """
    if not settings.FEATURE_BATCH_PROCESSING:
        raise HTTPException(status_code=403, detail="Batch processing not enabled")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        try:
            # Process each file
            contents = await file.read()
            suffix = Path(file.filename).suffix
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name
            
            media_type = "video" if suffix.lower() in ['.mp4', '.avi', '.mov'] else "audio"
            
            result = await detection_service.detect(
                file_path=temp_path,
                media_type=media_type,
                threshold=threshold,
                return_explainability=False,
                user_id=current_user.id
            )
            
            results.append(DetectionResponse(**result))
            
            # Cleanup
            os.unlink(temp_path)
            
        except Exception as e:
            logger.error(f"Batch detection error for {file.filename}: {e}")
            # Continue with other files
            continue
    
    return results


@router.get("/history", response_model=List[DetectionResponse])
async def get_detection_history(
    limit: int = 10,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """
    Get detection history for current user.
    
    Args:
        limit: Number of results to return
        offset: Offset for pagination
        current_user: Authenticated user
        
    Returns:
        List of past detections
    """
    history = await detection_service.get_user_history(
        user_id=current_user.id,
        limit=limit,
        offset=offset
    )
    
    return history


@router.get("/{detection_id}", response_model=DetectionResponse)
async def get_detection(
    detection_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get specific detection result.
    
    Args:
        detection_id: Detection ID
        current_user: Authenticated user
        
    Returns:
        Detection result
    """
    result = await detection_service.get_detection(
        detection_id=detection_id,
        user_id=current_user.id
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    return DetectionResponse(**result)


@router.delete("/{detection_id}")
async def delete_detection(
    detection_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete a detection record.
    
    Args:
        detection_id: Detection ID
        current_user: Authenticated user
        
    Returns:
        Success message
    """
    success = await detection_service.delete_detection(
        detection_id=detection_id,
        user_id=current_user.id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    return {"message": "Detection deleted successfully"}


@router.post("/feedback")
async def submit_feedback(
    detection_id: str,
    is_correct: bool,
    notes: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Submit feedback on detection result.
    
    Args:
        detection_id: Detection ID
        is_correct: Whether the detection was correct
        notes: Optional feedback notes
        current_user: Authenticated user
        
    Returns:
        Success message
    """
    await detection_service.submit_feedback(
        detection_id=detection_id,
        user_id=current_user.id,
        is_correct=is_correct,
        notes=notes
    )
    
    return {"message": "Feedback submitted successfully"}
