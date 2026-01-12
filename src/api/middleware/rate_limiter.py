"""
Rate Limiting Middleware for I.R.I.S.
Prevents API abuse and ensures fair resource usage
"""
import time
from collections import defaultdict
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    requests_per_hour: int = 500
    generation_per_minute: int = 10
    generation_per_hour: int = 100
    burst_limit: int = 5  # Max requests in 1 second


@dataclass
class ClientStats:
    """Track request statistics for a client"""
    minute_requests: list = field(default_factory=list)
    hour_requests: list = field(default_factory=list)
    generation_minute: list = field(default_factory=list)
    generation_hour: list = field(default_factory=list)
    last_request: float = 0
    burst_count: int = 0


class RateLimiter:
    """
    In-memory rate limiter with multiple time windows
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.clients: Dict[str, ClientStats] = defaultdict(ClientStats)
        self._cleanup_interval = 300  # Cleanup every 5 minutes
        self._last_cleanup = time.time()
    
    def _get_client_key(self, request: Request) -> str:
        """Extract client identifier from request"""
        # Try X-Forwarded-For for proxied requests
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _cleanup_old_entries(self, stats: ClientStats, current_time: float):
        """Remove expired entries from tracking lists"""
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        
        stats.minute_requests = [t for t in stats.minute_requests if t > minute_ago]
        stats.hour_requests = [t for t in stats.hour_requests if t > hour_ago]
        stats.generation_minute = [t for t in stats.generation_minute if t > minute_ago]
        stats.generation_hour = [t for t in stats.generation_hour if t > hour_ago]
    
    def _global_cleanup(self):
        """Periodically clean up inactive clients"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = current_time
        hour_ago = current_time - 3600
        
        # Remove clients with no recent activity
        inactive_clients = [
            key for key, stats in self.clients.items()
            if stats.last_request < hour_ago
        ]
        for key in inactive_clients:
            del self.clients[key]
    
    def check_rate_limit(
        self, 
        request: Request, 
        is_generation: bool = False
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Check if request is within rate limits
        
        Returns:
            Tuple of (allowed, error_message, retry_after_seconds)
        """
        current_time = time.time()
        client_key = self._get_client_key(request)
        stats = self.clients[client_key]
        
        # Cleanup old entries
        self._cleanup_old_entries(stats, current_time)
        self._global_cleanup()
        
        # Check burst limit (requests within 1 second)
        if current_time - stats.last_request < 1:
            stats.burst_count += 1
            if stats.burst_count > self.config.burst_limit:
                return False, "Too many requests. Please slow down.", 1
        else:
            stats.burst_count = 1
        
        stats.last_request = current_time
        
        # Check minute limit
        if len(stats.minute_requests) >= self.config.requests_per_minute:
            oldest = min(stats.minute_requests)
            retry_after = int(60 - (current_time - oldest)) + 1
            return False, f"Rate limit exceeded. Try again in {retry_after}s.", retry_after
        
        # Check hour limit
        if len(stats.hour_requests) >= self.config.requests_per_hour:
            oldest = min(stats.hour_requests)
            retry_after = int(3600 - (current_time - oldest)) + 1
            return False, f"Hourly limit exceeded. Try again in {retry_after // 60}m.", retry_after
        
        # Additional checks for generation endpoints
        if is_generation:
            if len(stats.generation_minute) >= self.config.generation_per_minute:
                oldest = min(stats.generation_minute)
                retry_after = int(60 - (current_time - oldest)) + 1
                return False, f"Generation limit exceeded. Try again in {retry_after}s.", retry_after
            
            if len(stats.generation_hour) >= self.config.generation_per_hour:
                oldest = min(stats.generation_hour)
                retry_after = int(3600 - (current_time - oldest)) + 1
                return False, f"Hourly generation limit exceeded. Try again in {retry_after // 60}m.", retry_after
            
            stats.generation_minute.append(current_time)
            stats.generation_hour.append(current_time)
        
        # Record request
        stats.minute_requests.append(current_time)
        stats.hour_requests.append(current_time)
        
        return True, None, None
    
    def get_client_stats(self, request: Request) -> Dict:
        """Get rate limit stats for a client"""
        client_key = self._get_client_key(request)
        stats = self.clients[client_key]
        current_time = time.time()
        
        self._cleanup_old_entries(stats, current_time)
        
        return {
            "requests_this_minute": len(stats.minute_requests),
            "requests_this_hour": len(stats.hour_requests),
            "generations_this_minute": len(stats.generation_minute),
            "generations_this_hour": len(stats.generation_hour),
            "limits": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "generation_per_minute": self.config.generation_per_minute,
                "generation_per_hour": self.config.generation_per_hour
            }
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        
        # Paths that count as generation requests
        self.generation_paths = {
            "/api/generate",
            "/ws/generate",
            "/api/upscale",
            "/api/variation"
        }
        
        # Paths exempt from rate limiting
        self.exempt_paths = {
            "/health",
            "/api/health",
            "/docs",
            "/openapi.json",
            "/favicon.ico"
        }
    
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        
        # Skip rate limiting for exempt paths
        if path in self.exempt_paths or path.startswith("/static"):
            return await call_next(request)
        
        # Check if this is a generation request
        is_generation = path in self.generation_paths
        
        # Check rate limit
        allowed, error_message, retry_after = self.rate_limiter.check_rate_limit(
            request, 
            is_generation=is_generation
        )
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": error_message,
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        return await call_next(request)


# Global rate limiter instance
rate_limiter = RateLimiter()
