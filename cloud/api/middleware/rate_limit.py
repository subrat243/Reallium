"""
Rate limiting middleware.
"""

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.requests = defaultdict(list)
        self.per_minute = settings.RATE_LIMIT_PER_MINUTE
        self.per_hour = settings.RATE_LIMIT_PER_HOUR
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        
        # Get client IP
        client_ip = request.client.host
        
        # Clean old requests
        now = datetime.now()
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < timedelta(hours=1)
        ]
        
        # Check rate limits
        recent_requests = self.requests[client_ip]
        
        # Per minute check
        minute_ago = now - timedelta(minutes=1)
        requests_last_minute = sum(1 for req_time in recent_requests if req_time > minute_ago)
        
        if requests_last_minute >= self.per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip} (per minute)")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Per hour check
        if len(recent_requests) >= self.per_hour:
            logger.warning(f"Rate limit exceeded for {client_ip} (per hour)")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Hourly rate limit exceeded. Please try again later."
            )
        
        # Record request
        self.requests[client_ip].append(now)
        
        # Process request
        response = await call_next(request)
        
        return response
