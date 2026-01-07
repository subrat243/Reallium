"""
Authentication middleware.
"""

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..services.auth_service import AuthService, oauth2_scheme
from ..database.database import get_db
from ..database.models import User, UserRole

auth_service = AuthService()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user."""
    return await auth_service.get_current_user(token, db)


def require_role(required_role: str):
    """
    Dependency to require specific role.
    
    Args:
        required_role: Required user role
        
    Returns:
        Dependency function
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        role_hierarchy = {
            "viewer": 0,
            "operator": 1,
            "admin": 2
        }
        
        user_level = role_hierarchy.get(current_user.role.value, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return current_user
    
    return role_checker
