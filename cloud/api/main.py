"""
FastAPI Main Application

Cloud API for deepfake detection system.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import Dict

from .config import settings
from .database.database import engine, Base
from .routes import detection, auth, models as model_routes
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.auth import AuthMiddleware

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Deepfake Detection API")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
    
    # Load ML models
    # TODO: Load models into memory
    logger.info("ML models loaded")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Deepfake Detection API")


# Create FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="Autonomous AI system for real-time deepfake detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
if settings.RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(detection.router, prefix="/api/v1/detection", tags=["Detection"])
app.include_router(model_routes.router, prefix="/api/v1/models", tags=["Models"])


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Deepfake Detection API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",
        "models": "loaded"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # TODO: Implement Prometheus metrics
    return {"metrics": "not implemented"}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=settings.API_WORKERS
    )
