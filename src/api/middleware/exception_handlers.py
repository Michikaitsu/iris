"""
Global Exception Handlers for I.R.I.S.
Provides consistent error responses across the API
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback

from src.core.exceptions import (
    IRISException,
    ModelLoadError,
    ModelNotFoundError,
    VRAMExhaustedError,
    NSFWContentError,
    GenerationError,
    QueueFullError,
    InvalidParameterError,
    UpscaleError,
    WebSocketError,
    FileOperationError
)
from src.utils.logger import create_logger

logger = create_logger("ExceptionHandler")


def setup_exception_handlers(app: FastAPI):
    """Register all exception handlers with the FastAPI app"""
    
    @app.exception_handler(IRISException)
    async def iris_exception_handler(request: Request, exc: IRISException):
        """Handle all custom I.R.I.S. exceptions"""
        logger.warning(f"{exc.error_code}: {exc.message}")
        
        # Map error codes to HTTP status codes
        status_codes = {
            "MODEL_LOAD_ERROR": 503,
            "MODEL_NOT_FOUND": 404,
            "VRAM_EXHAUSTED": 507,
            "NSFW_CONTENT": 400,
            "GENERATION_ERROR": 500,
            "QUEUE_FULL": 503,
            "INVALID_PARAMETER": 400,
            "UPSCALE_ERROR": 500,
            "WEBSOCKET_ERROR": 500,
            "FILE_OPERATION_ERROR": 500
        }
        
        status_code = status_codes.get(exc.error_code, 500)
        
        return JSONResponse(
            status_code=status_code,
            content=exc.to_dict()
        )
    
    @app.exception_handler(VRAMExhaustedError)
    async def vram_exception_handler(request: Request, exc: VRAMExhaustedError):
        """Special handler for VRAM errors with helpful suggestions"""
        logger.error(f"VRAM Exhausted: {exc.message}")
        
        return JSONResponse(
            status_code=507,
            content={
                **exc.to_dict(),
                "suggestions": [
                    "Reduce image resolution (try 512x512)",
                    "Lower the number of steps",
                    "Enable DRAM extension in settings",
                    "Close other GPU-intensive applications",
                    "Try a lighter model (e.g., anything_v5)"
                ]
            }
        )
    
    @app.exception_handler(NSFWContentError)
    async def nsfw_exception_handler(request: Request, exc: NSFWContentError):
        """Handler for NSFW content detection"""
        logger.warning(f"NSFW Content Blocked: {exc.details.get('category', 'unknown')}")
        
        return JSONResponse(
            status_code=400,
            content={
                **exc.to_dict(),
                "allowed_content": [
                    "Artistic body proportions",
                    "Stylized anime/manga characters",
                    "Fashion and elegant poses",
                    "Non-explicit artistic content"
                ]
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle standard HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Catch-all handler for unexpected exceptions"""
        # Log the full traceback for debugging
        logger.error(f"Unexpected error: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Check for common PyTorch/CUDA errors
        error_str = str(exc).lower()
        
        if "cuda out of memory" in error_str or "out of memory" in error_str:
            return JSONResponse(
                status_code=507,
                content={
                    "error": "VRAM_EXHAUSTED",
                    "message": "GPU memory exhausted during operation",
                    "suggestions": [
                        "Reduce image resolution",
                        "Lower the number of steps",
                        "Enable DRAM extension"
                    ]
                }
            )
        
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "MODEL_NOT_FOUND",
                    "message": "The requested model could not be found",
                    "details": {"original_error": str(exc)}
                }
            )
        
        # Generic error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {
                    "type": type(exc).__name__,
                    "message": str(exc)
                }
            }
        )
    
    logger.info("Exception handlers registered")
