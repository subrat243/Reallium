"""
Authentication service.
"""

from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ..api.config import settings
from ..database.database import get_db
from ..database.models import User, UserRole, AuditLog
import logging

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


class AuthService:
    """Authentication service."""
    
    def __init__(self):
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire = settings.JWT_EXPIRE_MINUTES
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password."""
        return pwd_context.hash(password)
    
    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token.
        
        Args:
            user_id: User ID
            expires_delta: Token expiration time
            
        Returns:
            JWT token
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str) -> str:
        """
        Create refresh token.
        
        Args:
            user_id: User ID
            
        Returns:
            Refresh token
        """
        expire = datetime.utcnow() + timedelta(days=7)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> str:
        """
        Verify JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            User ID
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id: str = payload.get("sub")
            
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            return user_id
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def verify_refresh_token(self, token: str) -> str:
        """Verify refresh token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            user_id: str = payload.get("sub")
            return user_id
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    
    async def create_user(
        self,
        email: str,
        password: str,
        full_name: str,
        role: str = "viewer",
        db: Session = None
    ) -> User:
        """
        Create new user.
        
        Args:
            email: User email
            password: Plain password
            full_name: Full name
            role: User role
            db: Database session
            
        Returns:
            Created user
        """
        if db is None:
            db = next(get_db())
        
        # Check if user exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            raise ValueError("User already exists")
        
        # Create user
        user = User(
            email=email,
            hashed_password=self.get_password_hash(password),
            full_name=full_name,
            role=UserRole(role)
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Audit log
        audit = AuditLog(
            user_id=user.id,
            event_type="user_created",
            event_description=f"User {email} created"
        )
        db.add(audit)
        db.commit()
        
        logger.info(f"User created: {email}")
        return user
    
    async def authenticate_user(
        self,
        email: str,
        password: str,
        db: Session = None
    ) -> Optional[User]:
        """
        Authenticate user.
        
        Args:
            email: User email
            password: Plain password
            db: Database session
            
        Returns:
            User if authenticated, None otherwise
        """
        if db is None:
            db = next(get_db())
        
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            return None
        
        if not self.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Audit log
        audit = AuditLog(
            user_id=user.id,
            event_type="login",
            event_description=f"User {email} logged in"
        )
        db.add(audit)
        db.commit()
        
        return user
    
    async def get_current_user(
        self,
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db)
    ) -> User:
        """
        Get current authenticated user.
        
        Args:
            token: JWT token
            db: Database session
            
        Returns:
            Current user
        """
        user_id = self.verify_token(token)
        
        user = db.query(User).filter(User.id == user_id).first()
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )
        
        return user
    
    async def update_user_mfa_secret(self, user_id: str, secret: str, db: Session = None):
        """Update user MFA secret."""
        if db is None:
            db = next(get_db())
        
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.mfa_secret = secret
            db.commit()
    
    async def enable_user_mfa(self, user_id: str, db: Session = None):
        """Enable MFA for user."""
        if db is None:
            db = next(get_db())
        
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.mfa_enabled = True
            db.commit()
