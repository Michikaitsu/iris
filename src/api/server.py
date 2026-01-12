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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI startup and shutdown"""
    logger.info("Starting I.R.I.S. Server...")
    logger.info("=" * 70)
    
    try:
        # Detect device
        pipeline_service.detect_device()
        
        # Load default model
        await pipeline_service.load_model("anime_kawai")
        
        # Setup queue callback
        generation_queue.set_process_callback(process_queue_item)
        
        logger.info("=" * 70)
        logger.info("Server ready at http://localhost:8000")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    yield
    
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
    
    # NSFW check
    nsfw_check = check_nsfw_prompt(prompt, nsfw_filter_enabled)
    if nsfw_check["is_unsafe"]:
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
    
    # Progress callback wrapper for diffusers
    def diffusers_callback(pipe, step_index, timestep, callback_kwargs):
        if progress_callback:
            progress = int((step_index + 1) / steps * 100)
            try:
                # Call the async callback in a thread-safe way
                asyncio.get_event_loop().call_soon_threadsafe(
                    lambda: asyncio.create_task(progress_callback({
                        "step": step_index + 1,
                        "total_steps": steps,
                        "progress": progress,
                        "timestep": float(timestep) if timestep is not None else 0
                    }))
                )
            except Exception:
                pass  # Ignore callback errors
        return callback_kwargs
    
    try:
        # Run generation in thread pool to not block the event loop
        loop = asyncio.get_event_loop()
        
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
        
        result = await loop.run_in_executor(
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
async def get_prompts_history():
    """Legacy endpoint for prompt history"""
    items = generation_history.get_all(limit=50)
    return {"history": items}


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
            
            # Apply NSFW filter strength
            from src.api.services.nsfw_filter import set_filter_strength
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
    
    app_settings.update({
        "dramEnabled": request.dramEnabled,
        "vramThreshold": request.vramThreshold,
        "maxDram": request.maxDram,
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
    
    # Apply NSFW filter strength
    from src.api.services.nsfw_filter import set_filter_strength
    set_filter_strength(request.nsfwStrength)
    strength_names = {1: "Relaxed", 2: "Standard", 3: "Strict"}
    logger.info(f"NSFW filter strength set to: {strength_names.get(request.nsfwStrength, 'Standard')}")
    
    # Note: Discord Bot is now controlled via /api/discord-bot endpoint
    
    # Save to file
    save_settings_to_file()
    
    return {
        "success": True,
        "message": "Settings saved successfully"
    }


# Discord Bot Process Management
_discord_bot_process = None

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
        if not bot_token:
            token_file = BASE_DIR / "static" / "config" / "bot_token.txt"
            if token_file.exists():
                with open(token_file, 'r') as f:
                    bot_token = f.read().strip()
        
        if not bot_token:
            logger.warning("Discord bot token not configured - bot cannot start")
            return {"success": False, "message": "Bot token not configured"}
        
        # Start the bot in a subprocess
        try:
            import sys
            bot_script = BASE_DIR / "src" / "services" / "bot.py"
            _discord_bot_process = subprocess.Popen(
                [sys.executable, str(bot_script)],
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.success(f"Discord bot started (PID: {_discord_bot_process.pid})")
            return {"success": True, "message": "Bot started", "pid": _discord_bot_process.pid}
        except Exception as e:
            logger.error(f"Failed to start Discord bot: {e}")
            return {"success": False, "message": str(e)}
    else:
        # Stop the bot if running
        if _discord_bot_process is not None:
            try:
                _discord_bot_process.terminate()
                _discord_bot_process.wait(timeout=5)
                logger.info("Discord bot stopped")
            except subprocess.TimeoutExpired:
                _discord_bot_process.kill()
                logger.warning("Discord bot force-killed")
            except Exception as e:
                logger.error(f"Error stopping Discord bot: {e}")
                return {"success": False, "message": str(e)}
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
    """Upscale an image using various methods"""
    from PIL import Image
    
    # Validate scale
    if request.scale not in [2, 4, 8]:
        raise HTTPException(status_code=400, detail="Scale must be 2, 4, or 8")
    
    # Find the image
    image_path = Path("outputs") / request.filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # Load image
        img = Image.open(image_path)
        original_size = img.size
        
        # Upscale based on method
        method = request.method.lower()
        
        if method == "lanczos":
            # Simple Lanczos upscaling (always available)
            new_width = img.width * request.scale
            new_height = img.height * request.scale
            upscaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            used_method = "lanczos"
            
        elif method == "realesrgan":
            # Try Real-ESRGAN, fallback to Lanczos
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                import numpy as np
                import cv2
                
                # Initialize Real-ESRGAN
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                upsampler = RealESRGANer(
                    scale=4,
                    model_path='weights/RealESRGAN_x4plus.pth',
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=True if torch.cuda.is_available() else False
                )
                
                # Convert to numpy
                img_array = np.array(img)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Upscale
                output, _ = upsampler.enhance(img_array, outscale=request.scale)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                upscaled = Image.fromarray(output)
                used_method = "realesrgan"
                
            except Exception as e:
                logger.warning(f"Real-ESRGAN failed: {e}, using Lanczos")
                new_width = img.width * request.scale
                new_height = img.height * request.scale
                upscaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                used_method = "lanczos"
                
        elif method == "swinir":
            # SwinIR is complex, fallback to Lanczos for now
            logger.info("SwinIR requested, using Lanczos (SwinIR requires additional setup)")
            new_width = img.width * request.scale
            new_height = img.height * request.scale
            upscaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            used_method = "lanczos"
            
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
