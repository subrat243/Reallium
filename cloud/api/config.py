"""
Configuration settings for the API.
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/deepfake_db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: str = ""
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = False
    API_SECRET_KEY: str = "change-this-secret-key"
    API_ALGORITHM: str = "HS256"
    API_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # Model Storage
    MODEL_STORAGE_TYPE: str = "local"
    MODEL_STORAGE_PATH: str = "/models"
    MODEL_STORAGE_BUCKET: str = "deepfake-models"
    
    # Media Storage
    MEDIA_STORAGE_TYPE: str = "local"
    MEDIA_STORAGE_PATH: str = "/media"
    MEDIA_MAX_SIZE_MB: int = 100
    
    # ML Configuration
    ML_DEVICE: str = "cpu"
    ML_BATCH_SIZE: int = 32
    ML_NUM_WORKERS: int = 4
    
    # Security
    JWT_SECRET_KEY: str = "change-this-jwt-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 30
    MFA_ENABLED: bool = True
    
    # RBAC
    RBAC_ENABLED: bool = True
    DEFAULT_ROLE: str = "viewer"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Feature Flags
    FEATURE_BATCH_PROCESSING: bool = True
    FEATURE_EXPLAINABILITY: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
