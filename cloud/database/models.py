"""
Database models for deepfake detection system.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import enum

from .database import Base


class UserRole(str, enum.Enum):
    """User roles."""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # MFA
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    detections = relationship("Detection", back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")


class Detection(Base):
    """Detection result model."""
    __tablename__ = "detections"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Media info
    media_type = Column(String, nullable=False)  # 'audio', 'video', 'multimodal'
    media_url = Column(String, nullable=True)
    media_hash = Column(String, nullable=True, index=True)
    
    # Detection results
    authenticity_score = Column(Float, nullable=False)
    is_authentic = Column(Boolean, nullable=False)
    is_deepfake = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Modality scores
    audio_score = Column(Float, nullable=True)
    video_score = Column(Float, nullable=True)
    
    # Model info
    model_version = Column(String, nullable=False)
    
    # Explainability
    explainability_data = Column(JSON, nullable=True)
    
    # Performance
    processing_time_ms = Column(Float, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="detections")
    feedbacks = relationship("Feedback", back_populates="detection", cascade="all, delete-orphan")


class Feedback(Base):
    """User feedback on detection results."""
    __tablename__ = "feedbacks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    detection_id = Column(String, ForeignKey("detections.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    is_correct = Column(Boolean, nullable=False)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    detection = relationship("Detection", back_populates="feedbacks")
    user = relationship("User", back_populates="feedbacks")


class ModelVersion(Base):
    """Model version tracking."""
    __tablename__ = "model_versions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    version = Column(String, nullable=False, index=True)
    model_type = Column(String, nullable=False)  # 'audio', 'video', 'fusion'
    
    # Model info
    file_path = Column(String, nullable=False)
    file_size_mb = Column(Float, nullable=False)
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    training_dataset = Column(String, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    uploaded_by = Column(String, ForeignKey("users.id"), nullable=True)


class AuditLog(Base):
    """Audit log for security events."""
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    
    # Event info
    event_type = Column(String, nullable=False, index=True)  # 'login', 'detection', 'model_upload', etc.
    event_description = Column(Text, nullable=False)
    
    # Request info
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)


class ThreatIntelligence(Base):
    """Threat intelligence data."""
    __tablename__ = "threat_intelligence"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Threat info
    threat_type = Column(String, nullable=False, index=True)  # 'deepfake_campaign', 'new_technique', etc.
    severity = Column(String, nullable=False)  # 'low', 'medium', 'high', 'critical'
    
    # Details
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    indicators = Column(JSON, nullable=True)  # IOCs, patterns, etc.
    
    # Source
    source = Column(String, nullable=False)  # 'osint', 'internal', 'partner'
    source_url = Column(String, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
