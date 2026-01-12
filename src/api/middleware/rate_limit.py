"""
Rate Limiting Middleware for I.R.I.S.
Prevents API abuse and ensures fair resource usage
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse


def get_client_ip(request: Request) -> str:
    """Get client IP, considering proxy headers"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(key_func=get_client_ip)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for rate limit exceeded"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": f"Too many requests. {exc.detail}",
            "retry_after": getattr(exc, 'retry_after', 60)
        }
    )


# Rate limit configurations
RATE_LIMITS = {
    "generate": "10/minute",      # 10 generations per minute
    "generate_burst": "3/second", # Max 3 concurrent
    "api_default": "60/minute",   # General API calls
    "gallery": "30/minute",       # Gallery requests
    "model_switch": "5/minute"    # Model switching
}
