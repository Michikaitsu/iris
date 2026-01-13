"""
I.R.I.S. Server - Main FastAPI Application
Refactored modular architecture
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import torch
import os
import gc
import json
import time
import io
import base64
import asyncio
import subprocess
import numpy as np
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Services
from src.api.services.pipeline import pipeline_service, MODEL_CONFIGS
from src.api.services.nsfw_filter import check_nsfw_prompt
from src.api.services.history import generation_history
from src.api.services.queue import generation_queue

# Routes
from src.api.routes import system, gallery, queue, history, templates

# Utils
from src.core.config import Config
from src.utils.logger import create_logger
from src.utils.file_manager import FileManager

# Exceptions
from src.core.exceptions import (
    IRISException,
    ModelLoadError,
    VRAMExhaustedError,
    NSFWContentError,
    GenerationError,
    QueueFullError,
    InvalidParameterError,
    ModelNotLoadedError
)

logger = create_logger("IRISServer")

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = BASE_DIR / "assets"
STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIR = Config.BASE_DIR / "frontend"

# WebSocket clients
connected_clients = []
gallery_clients = []

# Stats
generation_stats = {
    "total_images": 0,
    "total_time": 0
}

# Discord Bot Process (global for shutdown handling)
_discord_bot_process = None

# Bot Status File for communication
BOT_STATUS_FILE = BASE_DIR / "static" / "data" / "bot_status.json"

def update_bot_status_file(status: str, details: str = None):
    """Update bot status file for Discord bot to read"""
    try:
        BOT_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        status_data = {
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "total_generated": generation_stats["total_images"]
        }
        with open(BOT_STATUS_FILE, 'w') as f:
            json.dump(status_data, f)
    except Exception as e:
        logger.error(f"Failed to update bot status file: {e}")


def _patch_torchvision_compat():
    """Fix torchvision >= 0.18 compatibility with basicsr/realesrgan"""
    try:
        import torchvision.transforms.functional_tensor
    except ImportError:
        import sys
        import types
        import torchvision.transforms.functional as F
        
        # Create dummy module with required functions
        functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')
        functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor
        logger.info("Applied torchvision compatibility patch")


def _free_vram_for_upscaling():
    """Free VRAM by temporarily offloading the diffusion model
    
    This is crucial for low VRAM GPUs (4-6GB) where the diffusion model
    takes up most of the available memory.
    """
    import gc
    
    freed_mb = 0
    
    if not torch.cuda.is_available():
        return freed_mb
    
    try:
        # Get initial VRAM usage
        initial_allocated = torch.cuda.memory_allocated(0) / (1024**2)
        
        # Move diffusion pipeline to CPU if loaded
        if pipeline_service.pipe is not None:
            try:
                pipeline_service.pipe.to("cpu")
                logger.info("Moved diffusion pipeline to CPU for upscaling")
            except Exception as e:
                logger.warning(f"Could not move pipeline to CPU: {e}")
        
        if pipeline_service.img2img_pipe is not None:
            try:
                pipeline_service.img2img_pipe.to("cpu")
            except Exception:
                pass
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Calculate freed memory
        final_allocated = torch.cuda.memory_allocated(0) / (1024**2)
        freed_mb = initial_allocated - final_allocated
        
        if freed_mb > 100:
            logger.info(f"Freed {freed_mb:.0f}MB VRAM for upscaling")
        
    except Exception as e:
        logger.warning(f"VRAM cleanup failed: {e}")
    
    return freed_mb


def _restore_vram_after_upscaling():
    """Restore diffusion model to GPU after upscaling"""
    if not torch.cuda.is_available() or pipeline_service.device != "cuda":
        return
    
    try:
        if pipeline_service.pipe is not None:
            pipeline_service.pipe.to(pipeline_service.device)
            logger.info("Restored diffusion pipeline to GPU")
        
        if pipeline_service.img2img_pipe is not None:
            pipeline_service.img2img_pipe.to(pipeline_service.device)
            
    except Exception as e:
        logger.warning(f"Could not restore pipeline to GPU: {e}")


async def _load_upscaler(use_cpu: bool = False, tile_size: int = 256):
    """Load Real-ESRGAN upscaler
    
    Args:
        use_cpu: Force CPU mode (slower but no VRAM needed)
        tile_size: Tile size for processing (smaller = less VRAM, 0 = no tiling)
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        device = "cpu" if use_cpu else pipeline_service.device
        use_half = False if use_cpu else (pipeline_service.dtype == torch.float16)
        
        # Use the general x4plus model (more reliable URL)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        pipeline_service.upscaler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=tile_size,  # Tiling reduces VRAM usage significantly
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            device=device
        )
        mode_str = "CPU" if use_cpu else f"CUDA (tile={tile_size})"
        logger.success(f"Real-ESRGAN upscaler loaded ({mode_str})")
    except Exception as e:
        logger.warning(f"Real-ESRGAN not available: {e}")
        pipeline_service.upscaler = None


async def _load_upscaler_anime_v3(use_cpu: bool = False, tile_size: int = 256):
    """Load Real-ESRGAN Anime v3 upscaler (faster, optimized for anime)
    
    Args:
        use_cpu: Force CPU mode (slower but no VRAM needed)
        tile_size: Tile size for processing (smaller = less VRAM, 0 = no tiling)
    """
    if hasattr(pipeline_service, 'upscaler_anime') and pipeline_service.upscaler_anime:
        return  # Already loaded
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        device = "cpu" if use_cpu else pipeline_service.device
        use_half = False if use_cpu else (pipeline_service.dtype == torch.float16)
        
        # Anime v3 uses smaller model (6 blocks instead of 23)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        pipeline_service.upscaler_anime = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
            model=model,
            tile=tile_size,  # Tiling reduces VRAM usage
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            device=device
        )
        mode_str = "CPU" if use_cpu else f"CUDA (tile={tile_size})"
        logger.success(f"Real-ESRGAN Anime v3 upscaler loaded ({mode_str})")
    except Exception as e:
        logger.warning(f"Anime v3 upscaler not available: {e}")
        pipeline_service.upscaler_anime = None


async def _load_upscaler_bsrgan(use_cpu: bool = False, tile_size: int = 256):
    """Load BSRGAN-style upscaler (optimized for compressed/degraded images)
    
    Uses Real-ESRGAN x4plus with tile-based processing for better handling
    of compression artifacts and degraded images.
    
    Args:
        use_cpu: Force CPU mode (slower but no VRAM needed)
        tile_size: Tile size for processing (smaller = less VRAM)
    """
    if hasattr(pipeline_service, 'upscaler_bsrgan') and pipeline_service.upscaler_bsrgan:
        return  # Already loaded
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        device = "cpu" if use_cpu else pipeline_service.device
        use_half = False if use_cpu else (pipeline_service.dtype == torch.float16)
        
        # Use Real-ESRGAN x4plus with tile processing for degraded images
        # Tiling helps with compression artifacts by processing smaller regions
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        pipeline_service.upscaler_bsrgan = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=tile_size if tile_size > 0 else 256,  # Smaller tiles = better for degraded images
            tile_pad=32,  # More padding for smoother transitions
            pre_pad=10,
            half=use_half,
            device=device
        )
        mode_str = "CPU" if use_cpu else f"CUDA (tile={tile_size})"
        logger.success(f"BSRGAN-style upscaler loaded ({mode_str})")
    except Exception as e:
        # Fallback: use same as Real-ESRGAN
        logger.warning(f"BSRGAN upscaler failed, using Real-ESRGAN as fallback: {e}")
        pipeline_service.upscaler_bsrgan = pipeline_service.upscaler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI startup and shutdown"""
    logger.info("Starting I.R.I.S. Server...")
    logger.info("=" * 70)
    
    try:
        # Apply torchvision compatibility patch for Real-ESRGAN
        _patch_torchvision_compat()
        
        # Detect device
        pipeline_service.detect_device()
        
        # Load default model
        await pipeline_service.load_model("anime_kawai")
        
        # Pre-load Real-ESRGAN upscaler
        await _load_upscaler()
        
        # Setup queue callback
        generation_queue.set_process_callback(process_queue_item)
        
        logger.info("=" * 70)
        logger.info("Server ready at http://localhost:8000")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown: Stop Discord bot if running
    global _discord_bot_process
    if _discord_bot_process is not None and _discord_bot_process.poll() is None:
        logger.info("Stopping Discord bot on shutdown...")
        try:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(_discord_bot_process.pid)], 
                              capture_output=True, timeout=5)
            else:
                _discord_bot_process.kill()
                _discord_bot_process.wait(timeout=2)
            logger.success("Discord bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Discord bot: {e}")
        finally:
            _discord_bot_process = None
    
    logger.info("Shutting down I.R.I.S. Server...")
    pipeline_service.cleanup()


# Create FastAPI app
app = FastAPI(
    title="I.R.I.S. API",
    version="1.0.0",
    description="Intelligent Rendering & Image Synthesis",
    lifespan=lifespan
)

# Rate Limiting
try:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from src.api.middleware.rate_limit import limiter, rate_limit_exceeded_handler
    
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    logger.success("Rate limiting enabled")
except ImportError:
    logger.warning("slowapi not installed - rate limiting disabled")

# Mount static directories
try:
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
    logger.success(f"Assets mounted: {ASSETS_DIR}")
except Exception as e:
    logger.warning(f"Failed to mount assets: {e}")

try:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.success(f"Static mounted: {STATIC_DIR}")
except Exception as e:
    logger.warning(f"Failed to mount static: {e}")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(system.router)
app.include_router(gallery.router)
app.include_router(queue.router)
app.include_router(history.router)
app.include_router(templates.router)


# ============ Exception Handlers ============

@app.exception_handler(IRISException)
async def iris_exception_handler(request: Request, exc: IRISException):
    """Handle all I.R.I.S. custom exceptions"""
    status_codes = {
        "vram_exhausted": 507,
        "nsfw_blocked": 422,
        "model_load_error": 503,
        "model_not_loaded": 503,
        "queue_full": 429,
        "invalid_parameter": 400,
        "generation_error": 500
    }
    status_code = status_codes.get(exc.code, 500)
    return JSONResponse(status_code=status_code, content=exc.to_dict())


@app.exception_handler(VRAMExhaustedError)
async def vram_exception_handler(request: Request, exc: VRAMExhaustedError):
    """Handle VRAM exhaustion with cleanup"""
    if pipeline_service.device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    return JSONResponse(
        status_code=507,
        content={
            **exc.to_dict(),
            "suggestion": "Try reducing resolution or steps, or enable DRAM extension"
        }
    )


@app.exception_handler(NSFWContentError)
async def nsfw_exception_handler(request: Request, exc: NSFWContentError):
    """Handle NSFW content detection"""
    return JSONResponse(
        status_code=422,
        content={
            **exc.to_dict(),
            "category": exc.category
        }
    )


# ============ Page Routes ============

@app.get("/")
async def root():
    """Serve index page"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/generate")
async def generate_page():
    """Serve generate page"""
    return FileResponse(FRONTEND_DIR / "generate.html")


@app.get("/gallery")
async def gallery_page():
    """Serve gallery page"""
    return FileResponse(FRONTEND_DIR / "gallery.html")


@app.get("/settings")
async def settings_page():
    """Serve settings page"""
    return FileResponse(FRONTEND_DIR / "settings.html")


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    favicon_path = ASSETS_DIR / "fav.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404, detail="Favicon not found")


# ============ Generation ============

def generate_filename(prefix: str, seed: int, steps: int = None) -> str:
    """Generate filename with metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [prefix, timestamp, str(seed)]
    if steps:
        parts.append(f"s{steps}")
    return "_".join(parts) + ".png"


async def process_queue_item(item) -> dict:
    """Process a single queue item"""
    return await generate_image_internal(
        prompt=item.prompt,
        negative_prompt=item.negative_prompt,
        width=item.width,
        height=item.height,
        steps=item.steps,
        cfg_scale=item.cfg_scale,
        seed=item.seed,
        style=item.style
    )


async def generate_image_internal(
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 768,
    steps: int = 35,
    cfg_scale: float = 10.0,
    seed: int = -1,
    style: str = "anime_kawai",
    nsfw_filter_enabled: bool = True,
    progress_callback=None
) -> dict:
    """Internal generation function with optional progress callback"""
    
    if pipeline_service.pipe is None:
        raise ModelNotLoadedError()
    
    # Update bot status - generating
    update_bot_status_file("generating", prompt[:50])
    
    # NSFW check
    nsfw_check = check_nsfw_prompt(prompt, nsfw_filter_enabled)
    if nsfw_check["is_unsafe"]:
        update_bot_status_file("idle")
        raise NSFWContentError(category=nsfw_check.get("category", "explicit"))
    
    # Validate parameters
    if width < 256 or width > 4096:
        raise InvalidParameterError("width", width, "Must be between 256 and 4096")
    if height < 256 or height > 4096:
        raise InvalidParameterError("height", height, "Must be between 256 and 4096")
    if steps < 1 or steps > 150:
        raise InvalidParameterError("steps", steps, "Must be between 1 and 150")
    if cfg_scale < 1 or cfg_scale > 30:
        raise InvalidParameterError("cfg_scale", cfg_scale, "Must be between 1 and 30")
    
    # VRAM pre-check and auto-adjustment
    vram_check = pipeline_service.check_vram_availability(width, height, steps)
    if not vram_check["can_generate"]:
        if vram_check.get("adjusted_params"):
            # Use adjusted parameters
            adj = vram_check["adjusted_params"]
            width, height, steps = adj["width"], adj["height"], adj["steps"]
            logger.warning(f"Auto-adjusted params due to VRAM: {width}x{height}, {steps} steps")
        else:
            raise VRAMExhaustedError(
                required_gb=vram_check.get("estimated_vram_gb", 0),
                available_gb=vram_check.get("available_vram_gb", 0)
            )
    else:
        # Still apply safe params
        width, height, steps = pipeline_service.get_safe_params(width, height, steps)
    
    # Prepare seed
    if seed == -1:
        seed = np.random.randint(0, 2147483647)
    
    generator = torch.Generator(pipeline_service.device).manual_seed(seed)
    
    # Build prompt
    if style == "pixel_art":
        full_prompt = f"pixel art, 16-bit style, {prompt}"
        neg_prompt = f"smooth, anti-aliased, {negative_prompt}"
    else:
        full_prompt = f"masterpiece, best quality, {prompt}"
        neg_prompt = negative_prompt or "lowres, bad anatomy, worst quality"
    
    start_time = time.time()
    
    if pipeline_service.device == "cuda":
        torch.cuda.empty_cache()
    
    # Get the event loop BEFORE entering the thread executor
    main_loop = asyncio.get_event_loop()
    
    # Progress callback wrapper for diffusers
    def diffusers_callback(pipe, step_index, timestep, callback_kwargs):
        if progress_callback:
            progress = int((step_index + 1) / steps * 100)
            try:
                # Use the main loop reference captured before thread execution
                main_loop.call_soon_threadsafe(
                    lambda p=progress, s=step_index, ts=timestep: asyncio.ensure_future(
                        progress_callback({
                            "step": s + 1,
                            "total_steps": steps,
                            "progress": p,
                            "timestep": float(ts) if ts is not None else 0
                        }),
                        loop=main_loop
                    )
                )
            except Exception:
                pass  # Ignore callback errors
        return callback_kwargs
    
    try:
        # Run generation in thread pool to not block the event loop
        # Prepare kwargs with callback if supported
        pipe_kwargs = {
            "prompt": full_prompt,
            "negative_prompt": neg_prompt,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "width": width,
            "height": height,
            "generator": generator
        }
        
        # Add callback for progress updates (diffusers >= 0.25.0)
        if progress_callback:
            pipe_kwargs["callback_on_step_end"] = diffusers_callback
        
        result = await main_loop.run_in_executor(
            None,
            lambda: pipeline_service.pipe(**pipe_kwargs)
        )
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cuda" in error_msg:
            # Get VRAM info if available
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                used = torch.cuda.memory_allocated(0) / (1024**3)
                raise VRAMExhaustedError(required_gb=used + 2, available_gb=total - used)
            raise VRAMExhaustedError()
        raise GenerationError(reason=str(e), seed=seed)
    
    generation_time = time.time() - start_time
    image = result.images[0]
    
    # Save
    filename = generate_filename("gen", seed, steps)
    os.makedirs("outputs", exist_ok=True)
    image.save(f"outputs/{filename}")
    
    # Log to history
    generation_history.add(
        filename=filename,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        style=style,
        model=pipeline_service.current_model or "anime_kawai",
        generation_time=generation_time
    )
    
    # Update stats
    generation_stats["total_images"] += 1
    generation_stats["total_time"] += generation_time
    
    # Update bot status - back to idle
    update_bot_status_file("idle", f"Generated: {filename}")
    
    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "success": True,
        "image": f"data:image/png;base64,{img_str}",
        "seed": seed,
        "generation_time": round(generation_time, 2),
        "filename": filename,
        "width": width,
        "height": height
    }


# ============ WebSocket Generation ============

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for real-time generation with progress updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info("WebSocket client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            await websocket.send_json({"type": "started", "message": "Generation started"})
            
            try:
                # NSFW check
                nsfw_check = check_nsfw_prompt(
                    request_data.get("prompt", ""),
                    request_data.get("nsfw_filter_enabled", True)
                )
                if nsfw_check["is_unsafe"]:
                    await websocket.send_json({
                        "type": "error",
                        "message": nsfw_check["message"],
                        "nsfw_blocked": True
                    })
                    continue
                
                # Progress callback for real-time updates
                async def send_progress(progress_data):
                    try:
                        await websocket.send_json({
                            "type": "progress",
                            **progress_data
                        })
                    except Exception:
                        pass  # Client may have disconnected
                
                # Generate with progress callback
                result = await generate_image_internal(
                    prompt=request_data.get("prompt", ""),
                    negative_prompt=request_data.get("negative_prompt", ""),
                    width=request_data.get("width", 512),
                    height=request_data.get("height", 768),
                    steps=request_data.get("steps", 35),
                    cfg_scale=request_data.get("cfg_scale", 10.0),
                    seed=request_data.get("seed", -1),
                    style=request_data.get("style", "anime_kawai"),
                    nsfw_filter_enabled=request_data.get("nsfw_filter_enabled", True),
                    progress_callback=send_progress
                )
                
                await websocket.send_json({
                    "type": "completed",
                    **result
                })
                
                # Broadcast to gallery
                for client in gallery_clients:
                    try:
                        await client.send_json({
                            "status": "complete",
                            "filename": result["filename"]
                        })
                    except:
                        pass
                
            except Exception as e:
                error_msg = str(e)
                error_type = "generic"
                
                if "out of memory" in error_msg.lower():
                    error_type = "cuda_oom"
                    if pipeline_service.device == "cuda":
                        torch.cuda.empty_cache()
                
                await websocket.send_json({
                    "type": "error",
                    "error_type": error_type,
                    "message": error_msg
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


@app.websocket("/ws/gallery-progress")
async def websocket_gallery_progress(websocket: WebSocket):
    """WebSocket for gallery progress updates"""
    await websocket.accept()
    gallery_clients.append(websocket)
    
    try:
        while True:
            await websocket.receive_text()
            await asyncio.sleep(1)
    except:
        pass
    finally:
        if websocket in gallery_clients:
            gallery_clients.remove(websocket)


# ============ Legacy API Endpoints ============
# These maintain backward compatibility with existing frontend

from pydantic import BaseModel
from typing import Optional


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    style: str = "anime_kawai"
    seed: Optional[int] = -1
    steps: int = 35
    cfg_scale: float = 10.0
    width: int = 512
    height: int = 768
    nsfw_filter_enabled: Optional[bool] = True


# Try to import limiter for rate limiting
try:
    from src.api.middleware.rate_limit import limiter, RATE_LIMITS
    _has_rate_limit = True
except ImportError:
    _has_rate_limit = False


@app.post("/api/generate")
async def api_generate(request: GenerationRequest, req: Request):
    """REST API generation endpoint"""
    # Apply rate limit if available
    if _has_rate_limit:
        await limiter.check(RATE_LIMITS["generate"], req)
    
    try:
        result = await generate_image_internal(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            style=request.style,
            nsfw_filter_enabled=request.nsfw_filter_enabled
        )
        return result
    except (ModelNotLoadedError, NSFWContentError, VRAMExhaustedError, 
            InvalidParameterError, GenerationError) as e:
        # Custom exceptions are handled by exception handlers
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prompts-history")
async def get_prompts_history(limit: int = 50):
    """Get prompt history from prompts_history.json"""
    try:
        history_file = BASE_DIR / "static" / "data" / "prompts_history.json"
        if not history_file.exists():
            return {"history": []}
        
        with open(history_file, 'r', encoding='utf-8') as f:
            all_history = json.load(f)
        
        # Deduplicate by prompt and get unique entries (most recent first)
        seen_prompts = set()
        unique_history = []
        
        # Reverse to get most recent first
        for entry in reversed(all_history):
            prompt = entry.get('prompt', '').strip()
            if prompt and prompt not in seen_prompts:
                seen_prompts.add(prompt)
                # Normalize the entry format
                settings = entry.get('settings', {})
                unique_history.append({
                    "prompt": prompt,
                    "negativePrompt": settings.get('negative_prompt', ''),
                    "seed": settings.get('seed'),
                    "steps": settings.get('steps'),
                    "cfg": settings.get('cfg_scale'),
                    "width": settings.get('width'),
                    "height": settings.get('height'),
                    "style": settings.get('style'),
                    "timestamp": entry.get('timestamp')
                })
                if len(unique_history) >= limit:
                    break
        
        return {"history": unique_history}
    except Exception as e:
        logger.error(f"Failed to load prompts history: {e}")
        return {"history": []}


@app.get("/api/stats")
async def get_stats():
    """Get generation statistics"""
    return {
        "total_images": generation_stats["total_images"],
        "total_time": round(generation_stats["total_time"], 2)
    }


# ============ Settings API ============

# Settings storage (in-memory, persisted to file)
SETTINGS_FILE = BASE_DIR / "settings.json"
app_settings = {
    "dramEnabled": False,
    "vramThreshold": 6,
    "maxDram": 16,
    "nsfwEnabled": True,
    "nsfwStrength": 2,
    "discordEnabled": False
}

def load_settings_from_file():
    """Load settings from file on startup"""
    global app_settings
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, 'r') as f:
                app_settings.update(json.load(f))
            logger.info("Settings loaded from file")
            
            # Apply NSFW filter settings
            from src.api.services.nsfw_filter import set_filter_enabled, set_filter_strength
            set_filter_enabled(app_settings.get("nsfwEnabled", True))
            set_filter_strength(app_settings.get("nsfwStrength", 2))
    except Exception as e:
        logger.warning(f"Could not load settings: {e}")

def save_settings_to_file():
    """Save settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(app_settings, f, indent=2)
        logger.info("Settings saved to file")
    except Exception as e:
        logger.warning(f"Could not save settings: {e}")

# Load settings on module import
load_settings_from_file()


class SettingsRequest(BaseModel):
    dramEnabled: Optional[bool] = False
    vramThreshold: Optional[int] = 6
    maxDram: Optional[int] = 16
    nsfwEnabled: Optional[bool] = True
    nsfwStrength: Optional[int] = 2
    discordEnabled: Optional[bool] = False


@app.get("/api/settings")
async def get_settings():
    """Get current application settings"""
    return {
        "success": True,
        "settings": app_settings
    }


@app.post("/api/settings")
async def save_settings(request: SettingsRequest):
    """Save application settings"""
    global app_settings
    
    # Check if Discord setting changed
    discord_changed = app_settings.get("discordEnabled") != request.discordEnabled
    
    app_settings.update({
        "dramEnabled": request.dramEnabled,
        "vramThreshold": request.vramThreshold,
        "maxDram": request.maxDram,
        "nsfwEnabled": request.nsfwEnabled,
        "nsfwStrength": request.nsfwStrength,
        "discordEnabled": request.discordEnabled
    })
    
    # Apply DRAM extension setting to pipeline
    if request.dramEnabled:
        pipeline_service.dram_config["enabled"] = True
        pipeline_service.dram_config["vram_threshold_gb"] = request.vramThreshold
        pipeline_service.dram_config["max_dram_gb"] = request.maxDram
        logger.info(f"DRAM extension enabled (threshold: {request.vramThreshold}GB, max: {request.maxDram}GB)")
    else:
        pipeline_service.dram_config["enabled"] = False
    
    # Apply NSFW filter settings
    from src.api.services.nsfw_filter import set_filter_enabled, set_filter_strength
    set_filter_enabled(request.nsfwEnabled)
    set_filter_strength(request.nsfwStrength)
    strength_names = {1: "Relaxed", 2: "Standard", 3: "Strict"}
    logger.info(f"NSFW filter: {'Enabled' if request.nsfwEnabled else 'Disabled'} (strength: {strength_names.get(request.nsfwStrength, 'Standard')})")
    
    # Apply Discord Bot setting - start/stop bot when setting changes
    if discord_changed:
        discord_result = await handle_discord_bot(request.discordEnabled)
        logger.info(f"Discord bot: {'Starting' if request.discordEnabled else 'Stopping'} - {discord_result.get('message', '')}")
    
    # Save to file
    save_settings_to_file()
    
    return {
        "success": True,
        "message": "Settings saved successfully"
    }


# Discord Bot Process Management
# Note: _discord_bot_process is defined at module level for shutdown handling

async def handle_discord_bot(enabled: bool):
    """Start or stop the Discord bot based on settings"""
    global _discord_bot_process
    
    if enabled:
        # Check if bot is already running
        if _discord_bot_process is not None and _discord_bot_process.poll() is None:
            logger.info("Discord bot is already running")
            return {"success": True, "message": "Bot already running"}
        
        # Check if bot token is configured
        bot_token = os.getenv('DISCORD_BOT_TOKEN')
        logger.info(f"Discord bot token from env: {'Found' if bot_token else 'Not found'}")
        
        if not bot_token:
            token_file = BASE_DIR / "static" / "config" / "bot_token.txt"
            if token_file.exists():
                with open(token_file, 'r') as f:
                    bot_token = f.read().strip()
                logger.info("Discord bot token loaded from file")
        
        if not bot_token:
            logger.warning("Discord bot token not configured - bot cannot start")
            return {"success": False, "message": "Bot token not configured"}
        
        # Start the bot in a subprocess
        try:
            import sys
            bot_script = BASE_DIR / "src" / "services" / "bot.py"
            logger.info(f"Starting Discord bot from: {bot_script}")
            
            if not bot_script.exists():
                logger.error(f"Bot script not found: {bot_script}")
                return {"success": False, "message": "Bot script not found"}
            
            # Start bot in same terminal (output visible), but in new process group
            # so Ctrl+C doesn't propagate to it
            if os.name == 'nt':
                _discord_bot_process = subprocess.Popen(
                    [sys.executable, str(bot_script)],
                    cwd=str(BASE_DIR),
                    stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                _discord_bot_process = subprocess.Popen(
                    [sys.executable, str(bot_script)],
                    cwd=str(BASE_DIR),
                    stdin=subprocess.DEVNULL,
                    start_new_session=True
                )
            
            logger.success(f"Discord bot started (PID: {_discord_bot_process.pid})")
            return {"success": True, "message": "Bot started", "pid": _discord_bot_process.pid}
        except Exception as e:
            logger.error(f"Failed to start Discord bot: {e}")
            return {"success": False, "message": str(e)}
    else:
        # Force stop the bot
        if _discord_bot_process is not None:
            pid = _discord_bot_process.pid
            logger.info(f"Force stopping Discord bot (PID: {pid})...")
            try:
                if os.name == 'nt':
                    # Windows: Force kill immediately with taskkill
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                                  capture_output=True, timeout=5)
                else:
                    # Unix: SIGKILL
                    _discord_bot_process.kill()
                    _discord_bot_process.wait(timeout=2)
                logger.success(f"Discord bot stopped (PID: {pid})")
            except Exception as e:
                logger.error(f"Error stopping Discord bot: {e}")
            finally:
                _discord_bot_process = None
        return {"success": True, "message": "Bot stopped"}


class DiscordBotRequest(BaseModel):
    enabled: bool


@app.post("/api/discord-bot")
async def control_discord_bot(request: DiscordBotRequest):
    """Start or stop the Discord bot"""
    result = await handle_discord_bot(request.enabled)
    
    # Update settings
    app_settings["discordEnabled"] = request.enabled
    save_settings_to_file()
    
    return result


@app.get("/api/discord-bot/status")
async def get_discord_bot_status():
    """Get Discord bot status"""
    global _discord_bot_process
    
    if _discord_bot_process is None:
        return {"running": False, "status": "stopped"}
    
    poll_result = _discord_bot_process.poll()
    if poll_result is None:
        return {"running": True, "status": "running", "pid": _discord_bot_process.pid}
    else:
        _discord_bot_process = None
        return {"running": False, "status": "stopped", "exit_code": poll_result}


@app.get("/api/vram-status")
async def get_vram_status():
    """Get current VRAM status and availability"""
    return pipeline_service.get_vram_status()


@app.post("/api/vram-check")
async def check_vram_for_generation(width: int = 512, height: int = 768, steps: int = 35):
    """Pre-check if generation parameters will fit in VRAM"""
    return pipeline_service.check_vram_availability(width, height, steps)


# ============ Upscale API ============

class UpscaleRequest(BaseModel):
    filename: str
    scale: Optional[int] = 2
    method: Optional[str] = "realesrgan"


@app.post("/api/upscale")
async def upscale_image(request: UpscaleRequest):
    """Upscale an image using various methods with smart VRAM management
    
    Features:
    - Automatic VRAM cleanup before upscaling (moves diffusion model to CPU)
    - Tiled processing for lower VRAM usage
    - CPU fallback if CUDA OOM occurs
    - Automatic restoration of diffusion model after upscaling
    """
    from PIL import Image
    import cv2
    
    # Validate scale
    if request.scale not in [2, 4, 8]:
        raise HTTPException(status_code=400, detail="Scale must be 2, 4, or 8")
    
    # Find the image
    image_path = Path("outputs") / request.filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    vram_freed = False
    
    try:
        # Load image
        img = Image.open(image_path)
        original_size = img.size
        
        # Upscale based on method
        method = request.method.lower()
        
        if method == "lanczos":
            # Simple Lanczos upscaling (always available, no GPU needed)
            new_width = img.width * request.scale
            new_height = img.height * request.scale
            upscaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            used_method = "lanczos"
            
        elif method == "realesrgan":
            upscaled, used_method = await _upscale_with_smart_vram(
                img, request.scale, "realesrgan", _load_upscaler, 
                lambda: pipeline_service.upscaler
            )
            vram_freed = True
                
        elif method == "anime_v3":
            upscaled, used_method = await _upscale_with_smart_vram(
                img, request.scale, "anime_v3", _load_upscaler_anime_v3,
                lambda: pipeline_service.upscaler_anime
            )
            vram_freed = True
                
        elif method == "bsrgan":
            upscaled, used_method = await _upscale_with_smart_vram(
                img, request.scale, "bsrgan", _load_upscaler_bsrgan,
                lambda: pipeline_service.upscaler_bsrgan
            )
            vram_freed = True
                
        else:
            # Default to Lanczos
            new_width = img.width * request.scale
            new_height = img.height * request.scale
            upscaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            used_method = "lanczos"
        
        # Generate new filename
        base_name = request.filename.rsplit('.', 1)[0]
        new_filename = f"up{request.scale}x_{base_name}.png"
        new_path = Path("outputs") / new_filename
        
        # Save upscaled image
        upscaled.save(new_path, "PNG")
        
        # Convert to base64 for response
        buffered = io.BytesIO()
        upscaled.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        logger.info(f"Upscaled {request.filename} from {original_size} to {upscaled.size} using {used_method}")
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_str}",
            "filename": new_filename,
            "method": used_method,
            "original_size": list(original_size),
            "new_size": list(upscaled.size)
        }
        
    except Exception as e:
        logger.error(f"Upscale failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Always restore diffusion model to GPU after upscaling
        if vram_freed:
            _restore_vram_after_upscaling()


async def _upscale_with_smart_vram(img, scale: int, method_name: str, loader_func, get_upscaler_func):
    """Smart upscaling with VRAM management and fallbacks
    
    Strategy:
    1. Try with tiled processing on GPU (default)
    2. If OOM: Free VRAM (move diffusion model to CPU) and retry with smaller tiles
    3. If still OOM: Use CPU mode for upscaling
    4. If all fails: Fall back to Lanczos
    """
    import cv2
    from PIL import Image
    
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Strategy 1: Try with default tiled processing (256px tiles)
    try:
        upscaler = get_upscaler_func()
        if not upscaler:
            await loader_func(use_cpu=False, tile_size=256)
            upscaler = get_upscaler_func()
        
        if upscaler:
            output, _ = upscaler.enhance(img_array, outscale=scale)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            return Image.fromarray(output), method_name
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            logger.warning(f"{method_name} OOM with tiles, trying with VRAM cleanup...")
        else:
            raise
    
    # Strategy 2: Free VRAM and retry with smaller tiles
    try:
        _free_vram_for_upscaling()
        
        # Reload upscaler with smaller tiles
        await loader_func(use_cpu=False, tile_size=128)
        upscaler = get_upscaler_func()
        
        if upscaler:
            output, _ = upscaler.enhance(img_array, outscale=scale)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            return Image.fromarray(output), method_name
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            logger.warning(f"{method_name} still OOM, trying CPU mode...")
        else:
            raise
    
    # Strategy 3: Use CPU mode (slower but guaranteed to work)
    try:
        await loader_func(use_cpu=True, tile_size=0)
        upscaler = get_upscaler_func()
        
        if upscaler:
            logger.info(f"Using CPU mode for {method_name} (this may take a while)...")
            output, _ = upscaler.enhance(img_array, outscale=scale)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            return Image.fromarray(output), f"{method_name}_cpu"
            
    except Exception as e:
        logger.warning(f"{method_name} CPU mode failed: {e}")
    
    # Strategy 4: Fall back to Lanczos
    logger.warning(f"All {method_name} attempts failed, using Lanczos fallback")
    new_width = img.width * scale
    new_height = img.height * scale
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS), "lanczos"


# ============ Server Control ============

@app.post("/api/server/restart")
async def restart_server():
    """Trigger server restart by touching a file (for uvicorn --reload)"""
    import asyncio
    
    # Update bot status
    update_bot_status_file("restarting")
    
    logger.info("Server restart requested...")
    
    # Touch this file to trigger uvicorn reload
    async def do_restart():
        await asyncio.sleep(0.5)  # Give time for response to be sent
        # Touch server.py to trigger reload
        Path(__file__).touch()
    
    asyncio.create_task(do_restart())
    
    return {"success": True, "message": "Server restarting..."}
