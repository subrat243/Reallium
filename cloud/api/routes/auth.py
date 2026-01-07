"""
Authentication API Routes

Endpoints for user authentication and authorization.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
import pyotp
import logging

from ..config import settings
from ..database.models import User
from ..services.auth_service import AuthService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize auth service
auth_service = AuthService()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


class UserCreate(BaseModel):
    """User registration model."""
    email: EmailStr
    password: str
    full_name: str
    role: Optional[str] = "viewer"


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    mfa_enabled: bool
    created_at: datetime


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class MFASetupResponse(BaseModel):
    """MFA setup response."""
    secret: str
    qr_code_url: str


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """
    Register a new user.
    
    Args:
        user_data: User registration data
        
    Returns:
        Created user
    """
    try:
        user = await auth_service.create_user(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            role=user_data.role
        )
        
        return UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            mfa_enabled=user.mfa_enabled,
            created_at=user.created_at
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    User login.
    
    Args:
        form_data: Login credentials
        
    Returns:
        Access and refresh tokens
    """
    user = await auth_service.authenticate_user(
        email=form_data.username,
        password=form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    # Generate tokens
    access_token = auth_service.create_access_token(user.id)
    refresh_token = auth_service.create_refresh_token(user.id)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60
    )


@router.post("/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(current_user: User = Depends(auth_service.get_current_user)):
    """
    Setup MFA for user.
    
    Args:
        current_user: Authenticated user
        
    Returns:
        MFA secret and QR code URL
    """
    if not settings.MFA_ENABLED:
        raise HTTPException(status_code=403, detail="MFA not enabled")
    
    # Generate secret
    secret = pyotp.random_base32()
    
    # Save secret to user
    await auth_service.update_user_mfa_secret(current_user.id, secret)
    
    # Generate QR code URL
    totp = pyotp.TOTP(secret)
    qr_url = totp.provisioning_uri(
        name=current_user.email,
        issuer_name="Deepfake Detection"
    )
    
    return MFASetupResponse(
        secret=secret,
        qr_code_url=qr_url
    )


@router.post("/mfa/verify")
async def verify_mfa(
    token: str,
    current_user: User = Depends(auth_service.get_current_user)
):
    """
    Verify MFA token.
    
    Args:
        token: MFA token
        current_user: Authenticated user
        
    Returns:
        Success message
    """
    if not current_user.mfa_secret:
        raise HTTPException(status_code=400, detail="MFA not setup")
    
    totp = pyotp.TOTP(current_user.mfa_secret)
    
    if not totp.verify(token):
        raise HTTPException(status_code=401, detail="Invalid MFA token")
    
    # Enable MFA for user
    await auth_service.enable_user_mfa(current_user.id)
    
    return {"message": "MFA verified and enabled"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """
    Refresh access token.
    
    Args:
        refresh_token: Refresh token
        
    Returns:
        New access and refresh tokens
    """
    try:
        user_id = auth_service.verify_refresh_token(refresh_token)
        
        access_token = auth_service.create_access_token(user_id)
        new_refresh_token = auth_service.create_refresh_token(user_id)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.JWT_EXPIRE_MINUTES * 60
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user(current_user: User = Depends(auth_service.get_current_user)):
    """
    Get current user information.
    
    Args:
        current_user: Authenticated user
        
    Returns:
        User information
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        mfa_enabled=current_user.mfa_enabled,
        created_at=current_user.created_at
    )


@router.post("/logout")
async def logout(current_user: User = Depends(auth_service.get_current_user)):
    """
    Logout user (invalidate tokens).
    
    Args:
        current_user: Authenticated user
        
    Returns:
        Success message
    """
    # TODO: Implement token blacklisting
    return {"message": "Logged out successfully"}
