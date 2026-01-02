from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler
)
from PIL import Image
import os
import time
import sys
import random
import numpy as np
import io
import json
from datetime import datetime
import asyncio
import threading
import gc
import re
import base64
import subprocess

from src.core.config import Config
from src.utils.logger import create_logger
from src.utils.file_manager import FileManager



BASE_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = BASE_DIR / "assets"

print(">>> ASSETS DIR:", ASSETS_DIR)
print(">>> ASSETS EXISTS:", ASSETS_DIR.exists())

logger = create_logger("IRISWebServer")

PROMPTS_LOG_FILE = Config.DATA_DIR / "prompts_history.json"

# Real-ESRGAN import handling
REALESRGAN_AVAILABLE = False
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
    logger.success("Real-ESRGAN libraries available")
except ImportError as e:
    logger.warning(f"Real-ESRGAN not available: {e}")
    logger.info("Will use Lanczos upscaling as fallback")

connected_clients = []
gallery_clients = []
generation_stats = {
    "total_images": 0,
    "total_time": 0
}

# Global variables
pipe = None
img2img_pipe = None
device = None
upscaler = None
discord_bot_process = None
discord_bot_thread = None

# DRAM Extension Configuration
dram_extension_config = {
    "enabled": False,
    "vram_threshold_gb": 6,
    "max_dram_gb": 16
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI startup and shutdown"""
    logger.info("Starting I.R.I.S. Server...")
    
    # Startup: Load models
    await load_models()
    
    yield  # Added yield to make this a proper context manager
    
    # Shutdown: Cleanup
    logger.info("Shutting down I.R.I.S. Server...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

async def load_models():
    """Load models on startup"""
    global pipe, img2img_pipe, device, upscaler
    
    logger.log_session_start()
    logger.info("=" * 70)
    logger.info("Starting AI Image Generator Backend...")
    logger.info("=" * 70)
    
    dtype = torch.float32
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.success(f"NVIDIA GPU detected: {gpu_name}")
        
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.gpu_info(gpu_name, vram_total_gb, 0, 0)
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        if vram_total_gb <= dram_extension_config["vram_threshold_gb"]:
            logger.info(f"DRAM Extension available: You can enable up to +{dram_extension_config['max_dram_gb']}GB system RAM for VRAM overflow")
            logger.info(f"Auto-enabling DRAM Extension for {vram_total_gb:.1f}GB VRAM card")
            dram_extension_config["enabled"] = True
        
        has_tensor_cores = False
        if "RTX" in gpu_name or "A100" in gpu_name or "V100" in gpu_name or "T4" in gpu_name:
            has_tensor_cores = True
            logger.success("Tensor Cores detected! Using float16 for optimal performance")
            dtype = torch.float16
        else:
            logger.warning("No Tensor Cores detected (GTX series). Using float32 for stability")
            dtype = torch.float32
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        logger.success("Apple Silicon detected")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"
        dtype = torch.float32
        logger.success("Intel Arc detected")
    else:
        device = "cpu"
        dtype = torch.float32
        logger.warning("No GPU detected, using CPU (this will be SLOW!)")
    
    logger.model_load_start("Ojimi/anime-kawai-diffusion")
    logger.info("This takes 5-10 minutes on first start (model will be downloaded)...")
    
    # === UPDATED MODEL CONFIGS ===
    model_configs = {
        "anime_kawai": {
            "id": "Ojimi/anime-kawai-diffusion",
            "description": "Anime & Kawai Style"
        },
        "stable_diffusion_2_1": {
            "id": "stabilityai/stable-diffusion-2-1",
            "description": "Realistic Photo Style"
        },
        "stable_diffusion_3_5": {
            "id": "stabilityai/stable-diffusion-3.5-medium",
            "description": "High-Quality Realistic Style"
        },
        "flux_1_fast": {
            "id": "black-forest-labs/FLUX.1-schnell",
            "description": "Fast & Efficient Style"
        },
        "openjourney": {
            "id": "prompthero/openjourney",
            "description": "Artistic Illustration Style"
        },
        "pixel_art": {
            "id": "nitrosocke/pixel-art-diffusion",
            "description": "Pixel Art & Retro Style"
        },
        "pony_diffusion": {
        "id": "AstraliteHeart/pony-diffusion-v6-xl",
        "description": "High-End Anime & Character Style (SDXL)"
        },
        "anything_v5": {
         "id": "stablediffusionapi/anything-v5",
            "description": "Classic Flat Anime Style (Fast)"
        },
        "animagine_xl": {
            "id": "CagliostroResearchGroup/animagine-xl-3.1",
            "description": "High-Quality Modern Anime (SDXL)"
        },
        "aom3": {
         "id": "WarriorMama777/AbyssOrangeMix3",
         "description": "Semi-Realistic Anime Style"
        },
        "counterfeit_v3": {
         "id": "stablediffusionapi/counterfeit-v30",
         "description": "Detailed Digital Illustration Style"
         }
    }
    
    # Use the new key 'anime_kawai'
    model_id = model_configs["anime_kawai"]["id"]

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    if device == "cuda":
        logger.success("Enabling memory optimizations...")
        pipe.enable_attention_slicing(slice_size=1)
        
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
            logger.success("VAE slicing enabled (reduces memory for image processing)")
        
        if hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()
            logger.success("VAE tiling enabled (handles larger resolutions)")
        
        if dram_extension_config["enabled"]:
            apply_dram_extension(pipe, img2img_pipe)
        
        if dtype == torch.float16:
            pipe.vae.to(torch.float32)
            logger.success("Mixed Precision: Model float16 + VAE float32 (optimal)")
        
        torch.cuda.empty_cache()
        logger.success("CUDA cache cleared")
        logger.success("CUDA optimizations enabled")
    
    logger.info("Loading Image-to-Image pipeline...")
    img2img_pipe = StableDiffusionImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )
    
    if not dram_extension_config["enabled"]:
        img2img_pipe = img2img_pipe.to(device)
    
    if device == "cuda":
        img2img_pipe.enable_attention_slicing(slice_size=1)
        if hasattr(img2img_pipe, 'enable_vae_slicing'):
            img2img_pipe.enable_vae_slicing()
        if hasattr(img2img_pipe, 'enable_vae_tiling'):
            img2img_pipe.enable_vae_tiling()
        
        if dram_extension_config["enabled"]:
            apply_dram_extension(pipe, img2img_pipe)

    logger.success("Image-to-Image pipeline ready")
    
    logger.success("Model loaded successfully!")
    logger.info("=" * 70)
    logger.info(f"Server ready at http://localhost:8000")
    logger.info("=" * 70)
    
    upscaler = None
    if REALESRGAN_AVAILABLE and device == "cuda":
        try:
            logger.info("Loading Real-ESRGAN upscaler...")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upscaler = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=(dtype == torch.float16),
                device=device
            )
            logger.success("Real-ESRGAN upscaler loaded successfully!")
        except Exception as e:
            logger.warning(f"Could not load Real-ESRGAN: {e}")
            logger.info("Will use Lanczos upscaling as fallback")
            import traceback
            logger.debug(traceback.format_exc())
            upscaler = None
    elif REALESRGAN_AVAILABLE and device != "cuda":
        logger.info(f"Real-ESRGAN requires CUDA GPU (current device: {device})")
    elif not REALESRGAN_AVAILABLE:
        logger.info("Real-ESRGAN libraries not available - using Lanczos upscaling")

    if device == "cuda":
        torch.cuda.empty_cache()

def apply_dram_extension(txt2img_pipe=None, img2img_pipe_obj=None):
    """Enable DRAM as VRAM extension using sequential CPU offload"""
    global pipe, img2img_pipe
    
    if device != "cuda":
        logger.warning("DRAM extension only works with CUDA devices")
        return
    
    try:
        logger.info("Enabling DRAM Extension (VRAM + System RAM)...")
        logger.info("   This allows using system RAM to supplement VRAM")
        logger.info("   Model components will move between VRAM and RAM as needed")
        
        current_pipe = txt2img_pipe or pipe
        current_img2img = img2img_pipe_obj or img2img_pipe
        
        if current_pipe is not None:
            if hasattr(current_pipe, 'enable_sequential_cpu_offload'):
                logger.info("   Applying sequential CPU offload to text-to-image pipeline...")
                current_pipe.enable_sequential_cpu_offload()
                logger.success("   Text-to-image pipeline will use VRAM + System RAM")
            elif hasattr(current_pipe, 'enable_model_cpu_offload'):
                logger.info("   Applying model CPU offload to text-to-image pipeline...")
                current_pipe.enable_model_cpu_offload()
                logger.success("   Text-to-image pipeline will offload to RAM when needed")
        
        if current_img2img is not None:
            if hasattr(current_img2img, 'enable_sequential_cpu_offload'):
                logger.info("   Applying sequential CPU offload to image-to-image pipeline...")
                current_img2img.enable_sequential_cpu_offload()
                logger.success("   Image-to-image pipeline will use VRAM + System RAM")
            elif hasattr(current_img2img, 'enable_model_cpu_offload'):
                logger.info("   Applying model CPU offload to image-to-image pipeline...")
                current_img2img.enable_model_cpu_offload()
                logger.success("   Image-to-image pipeline will offload to RAM when needed")
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.success(f"DRAM Extension enabled! VRAM: {vram_gb:.1f}GB + System RAM available")
        
    except Exception as e:
        logger.error(f"Failed to enable DRAM extension: {e}")
        import traceback
        logger.debug(traceback.format_exc())

def log_prompt(prompt: str, settings: dict):
    """Log prompt to JSON file"""
    FileManager.log_prompt(prompt, settings)

def check_nsfw_prompt(prompt: str, nsfw_filter_enabled: bool = True) -> dict:
    """
    Check if prompt contains NSFW-related keywords/content
    Returns: {"is_unsafe": bool, "reason": str, "message": str}
    """
    if not nsfw_filter_enabled:
        return {"is_unsafe": False, "reason": "", "message": "Filter disabled"}
    
    # List of NSFW keywords/patterns to block
    nsfw_keywords = [
        # Explicit sexual content
        "nude", "naked", "xxx", "porn", "sex", "sexual",
        "penis", "vagina", "breast", "nipple", "testicle",
        "intercourse", "blowjob", "cumshot", "ejaculation",
        "horny", "erection", "orgasm", "masturbat",
        
        # Violence/Gore
        "gore", "blood", "mutilat", "severed", "decapitat",
        "dismember", "torture", "execut", "killing spree",
        
        # Drugs/Illegal content
        "cocaine", "heroin", "meth", "drug deal", "illegal drug",
        
        # Hate/Discrimination
        "racist", "sexist", "homophob", "hate crime",
    ]
    
    prompt_lower = prompt.lower()
    
    for keyword in nsfw_keywords:
        if keyword in prompt_lower:
            return {
                "is_unsafe": True,
                "reason": keyword,
                "message": f"❌ Prompt contains inappropriate content: '{keyword}'\n\nPlease modify your prompt to remove explicit or harmful descriptions."
            }
    
    return {"is_unsafe": False, "reason": "", "message": "Prompt is safe"}

def generate_filename(prefix: str, seed: int, steps: int = None, scale: int = None) -> str:
    """Generate filename with metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [prefix, timestamp, str(seed)]
    
    if steps:
        parts.append(f"s{steps}")
    if scale:
        parts.append(f"x{scale}")
    
    return "_".join(parts) + ".png"

def extract_metadata_from_filename(filename: str) -> dict:
    """Extract metadata from filename"""
    parts = filename.replace(".png", "").split("_")
    metadata = {
        "type": parts[0] if len(parts) > 0 else "unknown",
        "timestamp": parts[1] if len(parts) > 1 else None,
        "seed": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None,
        "steps": int(parts[3][1:]) if len(parts) > 3 and parts[3].startswith("s") else None,
        "scale": int(parts[4][1:]) if len(parts) > 4 and parts[4].startswith("x") else None
    }
    return metadata

def log_prompt_history(filename: str, seed: int, prompt: str, steps: int):
    """Log prompt with associated image"""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "seed": seed,
        "prompt": prompt,
        "steps": steps
    }
    
    FileManager.log_prompt(prompt, log_data)

# ───────────────────────────────
# INITIALIZE FASTAPI WITH LIFESPAN
# ───────────────────────────────
app = FastAPI(title="I.R.I.S. API", version="1.0.0", lifespan=lifespan)

# ───────────────────────────────
# MOUNT STATIC FILES
# ───────────────────────────────
app.mount(
    "/assets",
    StaticFiles(directory=str(ASSETS_DIR)),
    name="assets"
)

print(">>> ROUTES AFTER MOUNT:")
for r in app.routes:
    print("   ", r)

# ───────────────────────────────
# TEST ROUTE
# ───────────────────────────────
@app.get("/__test_static")
def test_static():
    return {"ok": True}

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")
# app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

FRONTEND_DIR = Config.BASE_DIR / "frontend"

@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/generate")
async def generate_page():
    return FileResponse(FRONTEND_DIR / "generate.html")

@app.get("/settings")
async def settings_page():
    return FileResponse(FRONTEND_DIR / "settings.html")

@app.get("/gallery")
async def gallery_page():
    return FileResponse(FRONTEND_DIR / "gallery.html")

@app.get("/favicon.ico")
async def favicon():
    favicon = ASSETS_DIR / "favico.png"
    if favicon.exists():
        return FileResponse(favicon)
    raise HTTPException(status_code=404)

@app.get("/__debug_assets")
def debug_assets():
    base = ASSETS_DIR
    thumbs = base / "thumbnails"
    return {
        "assets_exists": base.exists(),
        "assets_path": str(base),
        "thumbnails_exists": thumbs.exists(),
        "thumbnails_files": (
            [p.name for p in thumbs.iterdir()] if thumbs.exists() else []
        )
    }

# ========== MODELS ==========
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, out of focus, duplicate, ugly, morbid, mutilated, mutated hands, poorly drawn face, mutation, deformed, dehydrated, bad proportions, gross proportions, malformed limbs, long neck"
    style: str = "anime_kawai"
    seed: Optional[int] = None
    steps: int = 35
    cfg_scale: float = 10.0
    width: int = 512
    height: int = 768
    batch_size: int = 1
    dram_extension_enabled: Optional[bool] = False
    nsfw_filter_enabled: Optional[bool] = True

class UpscaleRequest(BaseModel):
    filename: str
    scale: int = 2

class VariationRequest(BaseModel):
    filename: str
    strength: float = 0.5
    prompt: Optional[str] = ""

class SystemInfo(BaseModel):
    gpu_name: str
    device: str
    vram_total: float
    vram_used: float
    gpu_temp: float
    dram_extension_enabled: bool
    dram_extension_available: bool

# Helper function to auto-adjust parameters based on VRAM
def get_safe_generation_params(width: int, height: int, steps: int, vram_gb: float):
    """
    Automatically adjust generation parameters to prevent OOM on low VRAM systems
    Returns safe (width, height, steps) based on available VRAM
    """
    if dram_extension_config["enabled"]:
        logger.info(f"   🔄 DRAM Extension active - using full parameters: {width}x{height}, {steps} steps")
        return width, height, steps
    
    total_pixels = width * height
    
    # VRAM requirements (rough estimates):
    # 512x512 (20 steps) = ~2.5GB
    # 512x768 (20 steps) = ~3.2GB
    # 512x768 (35 steps) = ~3.8GB
    # 720x1280 (35 steps) = ~5.0GB  // Added HD mobile wallpaper
    # 768x768 (35 steps) = ~5.5GB
    # 1080x1920 (35 steps) = ~7.5GB
    
    if vram_gb <= 4:
        # Very conservative for 4GB cards WITHOUT DRAM extension
        if total_pixels > 512 * 512:
            # Force smaller resolution
            width = 512
            height = 512
            logger.warning(f"⚠️  Auto-adjusted resolution to {width}x{height} for 4GB VRAM (enable DRAM extension to use higher)")
        
        if steps > 25:
            steps = 25
            logger.warning(f"⚠️  Auto-adjusted steps to {steps} for 4GB VRAM (enable DRAM extension to use 35+)")
            
    elif vram_gb <= 6:
        # Conservative for 6GB cards
        if total_pixels > 720 * 1280:
            width = 720
            height = 1280
            logger.warning(f"⚠️  Auto-adjusted resolution to {width}x{height} for 6GB VRAM")
        
        if steps > 35:
            steps = 35
            logger.warning(f"⚠️  Auto-adjusted steps to {steps} for 6GB VRAM")
    
    elif vram_gb <= 8:
        # Moderate limits for 8GB cards - can handle Full HD mobile wallpaper
        if total_pixels > 1080 * 1920:
            # Scale down proportionally
            scale_factor = (1080 * 1920 / total_pixels) ** 0.5
            width = int(width * scale_factor / 64) * 64  # Round to multiple of 64
            height = int(height * scale_factor / 64) * 64
            logger.warning(f"⚠️  Auto-adjusted resolution to {width}x{height} for 8GB VRAM")
    
    elif vram_gb > 8:
        # High VRAM cards (10GB+) can handle mobile wallpaper and larger resolutions
        if total_pixels > 1920 * 1080: # Check for common high-res like 1920x1080
            scale_factor = (1920 * 1080 / total_pixels) ** 0.5
            width = int(width * scale_factor / 64) * 64
            height = int(height * scale_factor / 64) * 64
            logger.warning(f"⚠️  Auto-adjusted resolution to {width}x{height} for {vram_gb:.0f}GB VRAM")
    
    return width, height, steps

# ========== ROUTES ==========

@app.get("/")
async def root():
    """Serve the HTML interface"""
    return FileResponse(str(Path(__file__).parent.parent / "frontend" / "index.html"))

@app.get("/generate")
async def generate_page():
    """Serve the image generation page"""
    return FileResponse(str(Path(__file__).parent.parent / "frontend" / "generate.html"))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "device": device
    }

@app.get("/api/system")
async def get_system_info():
    """Get system information"""
    info = {
        "gpu_name": "Unknown",
        "device": device,
        "vram_total": 0.0,
        "vram_used": 0.0,
        "gpu_temp": 0.0,
        "dram_extension_enabled": dram_extension_config["enabled"],
        "dram_extension_available": False
    }
    
    if device == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info["vram_total"] = vram_total
        info["vram_used"] = torch.cuda.memory_allocated(0) / 1024**3
        
        info["dram_extension_available"] = vram_total <= dram_extension_config["vram_threshold_gb"]
        
        # Get GPU temp (requires nvidia-smi)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=2
            )
            info["gpu_temp"] = float(result.stdout.strip())
        except:
            info["gpu_temp"] = 0
    
    return info

@app.get("/api/stats")
async def get_stats():
    """Get generation statistics"""
    # Placeholder for actual stats if needed, currently not tracked globally
    return {
        "total_images": generation_stats["total_images"],
        "total_time": round(generation_stats["total_time"], 2),
        # Placeholder for GPU temp and VRAM used, would require separate calls or periodic updates
        "gpu_temp": 0, 
        "vram_used": 0
    }

@app.get("/api/version")
async def get_version_info():
    """Get version information for I.R.I.S. and dependencies"""
    import sys
    
    version_info = {
        "iris_version": "1.0.0",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "sd_model": "Ojimi/anime-kawai-diffusion",
        "realesrgan_available": REALESRGAN_AVAILABLE
    }
    
    return version_info

@app.post("/api/generate")
async def generate_image(request: GenerationRequest):
    """Generate image (non-streaming version for API calls)"""
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Log prompt before generation starts
        log_prompt(request.prompt, request.model_dump())

        if device == "cuda":
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            request.width, request.height, request.steps = get_safe_generation_params(
                request.width, request.height, request.steps, vram_total
            )
        
        # Prepare seed
        seed = request.seed if request.seed is not None else np.random.randint(0, 2147483647)
        generator = torch.Generator(device).manual_seed(seed)
        
        # Adjust prompt based on style
        if request.style == "pixel_art": # UPDATED KEY
            full_prompt = f"pixel art, 16-bit style, {request.prompt}"
            neg_prompt = f"smooth, anti-aliased, {request.negative_prompt}"
        else:
            full_prompt = f"masterpiece, best quality, {request.prompt}"
            neg_prompt = request.negative_prompt or "lowres, bad anatomy, bad hands, worst quality"
        
        start_time = time.time()
        
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Generate
        result = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg_scale,
            width=request.width,
            height=request.height,
            generator=generator
        )
        
        generation_time = time.time() - start_time
        
        # Convert image to base64
        image = result.images[0]
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        now = datetime.now()
        filename = generate_filename("generated", seed, request.steps)
        os.makedirs("outputs", exist_ok=True)
        image.save(f"outputs/{filename}")
        
        # Call new prompt logging function
        log_prompt_history(filename, seed, request.prompt, request.steps)
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_str}",
            "seed": seed,
            "generation_time": round(generation_time, 2),
            "filename": filename
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            error_msg = (
                "💥 CUDA Out of Memory!\n\n"
                "Your GPU ran out of memory during generation. "
                "Try reducing resolution, steps, or enabling DRAM Extension in advanced settings."
            )
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        else:
            raise e
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for real-time generation updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    logger.websocket_connect()
    
    try:
        while True:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            data = await websocket.receive_text()
            logger.info(f"📨 Received data: {data[:100]}...")
            
            request_data = json.loads(data)
            
            # Log prompt before generation starts via WebSocket
            log_prompt(request_data.get('prompt', ''), request_data)

            if device == "cuda":
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                original_width = request_data.get("width", 512)
                original_height = request_data.get("height", 768)
                original_steps = request_data.get("steps", 35)
                
                adjusted_width, adjusted_height, adjusted_steps = get_safe_generation_params(
                    original_width, original_height, original_steps, vram_total
                )
                
                request_data["width"] = adjusted_width
                request_data["height"] = adjusted_height
                request_data["steps"] = adjusted_steps
                
                if (adjusted_width != original_width or adjusted_height != original_height or adjusted_steps != original_steps):
                    await websocket.send_json({
                        "type": "warning",
                        "message": f"Parameters auto-adjusted for low VRAM: {adjusted_width}x{adjusted_height}, {adjusted_steps} steps"
                    })
            
            logger.info("🎨 Starting generation...")
            logger.info(f"   Prompt: {request_data.get('prompt', 'N/A')[:60]}...")
            logger.info(f"   Style: {request_data.get('style', 'anime_kawai')}") # Updated default
            logger.info(f"   Steps: {request_data.get('steps', 35)} (requested)")
            
            if device == "cuda":
                available_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                logger.info(f"   Available VRAM: {available_vram:.2f} GB")
            
            websocket_active = True
            
            async def safe_send(data):
                """Safely send data through WebSocket with error handling"""
                nonlocal websocket_active
                if not websocket_active:
                    return False
                try:
                    await websocket.send_json(data)
                    return True
                except Exception as e:
                    logger.error(f"WebSocket send failed (client likely disconnected): {type(e).__name__}")
                    websocket_active = False
                    return False
            
            async def broadcast_to_gallery(data):
                """Broadcast generation progress to gallery clients"""
                disconnected = []
                for client in gallery_clients:
                    try:
                        await client.send_json(data)
                    except Exception as e:
                        logger.debug(f"Gallery client send failed: {e}")
                        disconnected.append(client)
                
                for client in disconnected:
                    if client in gallery_clients:
                        gallery_clients.remove(client)
            
            if not await safe_send({"type": "started", "message": "Generation started"}):
                break
            
            await broadcast_to_gallery({
                "status": "generating",
                "progress": 0,
                "step": 0,
                "total_steps": request_data.get("steps", 35),
                "total_images": 1,
                "current_image": 1
            })
            
            seed = request_data.get("seed") if request_data.get("seed") is not None else np.random.randint(0, 2147483647)
            generator = torch.Generator(device).manual_seed(seed)
            
            style = request_data.get("style", "anime_kawai") # Updated default
            prompt = request_data.get("prompt", "")
            nsfw_filter_enabled = request_data.get("nsfw_filter_enabled", True)
            
            # Check if prompt contains NSFW content BEFORE generation
            nsfw_check = check_nsfw_prompt(prompt, nsfw_filter_enabled)
            if nsfw_check["is_unsafe"]:
                logger.warning(f"NSFW prompt blocked: {nsfw_check['reason']}")
                await safe_send({
                    "type": "error",
                    "message": nsfw_check["message"],
                    "nsfw_blocked": True
                })
                break
            
            if style == "pixel_art": # UPDATED KEY
                full_prompt = f"pixel art, 16-bit style, {prompt}"
                neg_prompt = "smooth, anti-aliased, blurry, 3d render"
            else:
                full_prompt = f"masterpiece, best quality, {prompt}"
                neg_prompt = "lowres, bad anatomy, worst quality, blurry"
            
            user_neg = request_data.get("negative_prompt", "")
            if user_neg:
                neg_prompt = f"{neg_prompt}, {user_neg}"
            
            start_time = time.time()
            total_steps = request_data.get("steps", 35)
            
            step_times = []
            
            def progress_callback(pipe_obj, step: int, timestep: int, callback_kwargs: dict):
                nonlocal websocket_active
                if not websocket_active:
                    return callback_kwargs
                
                current_time = time.time()
                step_times.append(current_time)
                
                progress = (step + 1) / total_steps * 100
                
                avg_step_time = 0
                eta = 0
                if len(step_times) > 1:
                    avg_step_time = (current_time - start_time) / len(step_times)
                    remaining_steps = total_steps - (step + 1)
                    eta = avg_step_time * remaining_steps
                
                gpu_temp = 0
                vram_used = 0
                if device == "cuda":
                    try:
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=temperature.gpu,memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=1
                        )
                        temp_str, vram_str = result.stdout.strip().split(',')
                        gpu_temp = float(temp_str)
                        vram_used = float(vram_str) / 1024
                    except:
                        try:
                            vram_used = torch.cuda.memory_allocated(0) / 1024**3
                        except:
                            vram_used = 0
                
                bar_length = 50
                filled_length = int(bar_length * progress // 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                elapsed_time = current_time - start_time
                print(f"\r🎨 [{bar}] {progress:.0f}% │ {step + 1}/{total_steps} │ ⏱️ {elapsed_time:.0f}s │ ETA: {eta:.0f}s │ 🌡️ {gpu_temp:.0f}°C │ 💾 {vram_used:.1f}GB", end='', flush=True)
                
                async def send_progress():
                    progress_data = {
                        "type": "progress",
                        "step": step + 1,
                        "total_steps": total_steps,
                        "progress": round(progress, 1),
                        "eta": round(eta, 1),
                        "gpu_temp": round(gpu_temp, 1),
                        "vram_used": round(vram_used, 2),
                        "avg_step_time": round(avg_step_time, 2)
                    }
                    await safe_send(progress_data)
                    
                    await broadcast_to_gallery({
                        "status": "generating",
                        "progress": round(progress, 1),
                        "step": step + 1,
                        "total_steps": total_steps,
                        "total_images": 1,
                        "current_image": 1
                    })
                
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(send_progress())
                except RuntimeError:
                    pass
                
                return callback_kwargs

            try:
                # Check if model is loaded
                if pipe is None:
                    error_msg = "Model not loaded yet. Please wait for the server to finish loading models on startup."
                    logger.error(error_msg)
                    await safe_send({
                        "type": "error",
                        "error_type": "model_not_loaded",
                        "message": error_msg
                    })
                    break
                
                # Prepare kwargs for pipe call
                pipe_kwargs = {
                    "prompt": full_prompt,
                    "negative_prompt": neg_prompt,
                    "num_inference_steps": total_steps,
                    "guidance_scale": request_data.get("cfg_scale", 10.0),
                    "width": request_data.get("width", 512),
                    "height": request_data.get("height", 768),
                    "generator": generator
                }
                
                # Only add callback if pipe supports it
                try:
                    if pipe is not None and (hasattr(pipe, 'callback_on_step_end') or 'callback_on_step_end' in pipe.__call__.__code__.co_varnames):
                        pipe_kwargs["callback_on_step_end"] = progress_callback
                except (AttributeError, TypeError):
                    # Callback not supported, continue without it
                    pass
                
                result = pipe(**pipe_kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device == "cuda" else 0
                    error_msg = (
                        "💥 CUDA Out of Memory!\n\n"
                        f"Your GPU has {vram_gb:.1f}GB VRAM but this generation requires more.\n\n"
                        "💡 Solutions (try in order):\n"
                        "1. ✅ Enable DRAM Extension in Advanced Settings\n"
                        "   → This adds 8GB system RAM to supplement your VRAM\n"
                        "2. 📐 Use smaller resolution:\n"
                        "   → 512x512 (safest for 4GB VRAM)\n"
                        "   → 512x768 (safe for 6GB VRAM)\n"
                        "3. ⚡ Reduce steps:\n"
                        "   → Use 'Fast' preset (20 steps)\n"
                        "   → Or manually set to 20-25 steps\n"
                        "4. 🔄 Restart the server:\n"
                        "   → This clears cached memory\n"
                        "5. 🎯 Alternative workflow:\n"
                        "   → Generate at 512x512, then Upscale 2x\n"
                        "   → Much faster and uses less VRAM!\n\n"
                        "Recommended settings for your GPU:\n"
                        f"• Resolution: {'512x512' if vram_gb <= 4 else '512x768' if vram_gb <= 6 else '768x768'}\n"
                        "• Steps: 20-28\n"
                        "• Quality: Fast or Balanced\n"
                        "• DRAM Extension: Enabled"
                    )
                    logger.error(error_msg)
                    await safe_send({
                        "type": "error",
                        "error_type": "cuda_oom",
                        "message": error_msg
                    })
                    
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        try:
                            gc.collect()
                        except:
                            pass
                    
                    break
                else:
                    raise e
            
            generation_time = time.time() - start_time
            print()
            logger.success(f"Generation completed in {generation_time:.1f}s")
            
            image = result.images[0]
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            filename = generate_filename("generated", seed, total_steps)
            os.makedirs("outputs", exist_ok=True)
            image.save(f"outputs/{filename}")
            
            logger.info(f"💾 Saved as: {filename}")
            
            success = await safe_send({
                "type": "completed",
                "image": f"data:image/png;base64,{img_str}",
                "seed": seed,
                "generation_time": round(generation_time, 2),
                "filename": filename,
                "width": request_data.get("width", 512),
                "height": request_data.get("height", 768)
            })
            
            await broadcast_to_gallery({
                "status": "complete",
                "progress": 100,
                "filename": filename,
                "image": f"data:image/png;base64,{img_str}"
            })
            
            if not success:
                logger.warning("⚠️  Client disconnected, but image was saved successfully")
                break
            
            # Update global stats (this might need to be more robust for concurrent access)
            generation_stats["total_images"] += 1
            generation_stats["total_time"] += generation_time
            
            if device == "cuda":
                torch.cuda.empty_cache()
                logger.info("🧹 CUDA cache cleared - Memory freed")
            
            log_prompt_history(filename, seed, prompt, total_steps)
            
    except WebSocketDisconnect:
        logger.websocket_disconnect()
    except asyncio.CancelledError:
        logger.websocket_disconnect()
    except Exception as e:
        error_name = type(e).__name__
        if "Disconnect" not in error_name and "Closed" not in error_name:
            logger.error(f"WebSocket error: {error_name}: {e}")

            error_type = "unknown"
            error_message = str(e)
            
            if "out of memory" in error_message.lower() or "cuda" in error_message.lower():
                error_type = "cuda_oom"
                logger.error("💥 CUDA Out of Memory Error!")
                logger.error("Solutions:")
                logger.error("  • Enable DRAM Extension in Advanced Settings")
                logger.error("  • Use 512x512 resolution")
                logger.error("  • Reduce steps to 20-25")
                logger.error("  • Select 'Fast' quality preset")
                logger.error("  • Restart server to clear memory")
            elif "connection" in error_message.lower() or "refused" in error_message.lower() or "reset by peer" in error_message.lower():
                error_type = "connection"
                logger.error("🔌 Connection Error!")
                logger.error("Solutions:")
                logger.error("  • Check if backend server is running")
                logger.error("  • Verify port 8000 is not blocked")
                logger.error("  • Check firewall settings")
            else:
                error_type = "generic"
                logger.error(f"❌ Error: {error_message}")
            
            async def send_error_to_client():
                try:
                    if websocket_active:
                        await websocket.send_json({
                            "type": "error",
                            "error_type": error_type,
                            "message": error_message
                        })
                except RuntimeError as e:
                    logger.debug(f"Could not send error to client (WebSocket already closed): {e}")
            
            try:
                asyncio.create_task(send_error_to_client())
            except RuntimeError:
                pass
            
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.websocket_disconnect()

@app.websocket("/ws/gallery-progress")
async def websocket_gallery_progress(websocket: WebSocket):
    """WebSocket endpoint for gallery to receive generation progress updates"""
    await websocket.accept()
    gallery_clients.append(websocket)
    
    logger.info("📸 Gallery client connected for progress updates")
    
    try:
        while True:
            try:
                await websocket.receive_text()
            except:
                break
            await asyncio.sleep(1)
                
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Gallery WebSocket error: {e}")
    finally:
        if websocket in gallery_clients:
            gallery_clients.remove(websocket)
        logger.info("📸 Gallery client disconnected")

@app.get("/api/output-gallery")
async def get_output_gallery():
    """Get list of all images from outputs directory"""
    try:
        os.makedirs("outputs", exist_ok=True)
        files = [f for f in os.listdir("outputs") if f.endswith((".png", ".jpg", ".jpeg"))]
        files.sort(key=lambda x: os.path.getmtime(f"outputs/{x}"), reverse=True)
        return {"images": files, "count": len(files)}
    except Exception as e:
        logger.error(f"Error loading output gallery: {e}")
        return {"images": [], "count": 0, "error": str(e)}

@app.get("/api/output-image/{filename}")
async def get_output_image(filename: str):
    """Get a specific image from outputs directory"""
    filepath = f"outputs/{filename}"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)

@app.post("/api/recreate-from-output")
async def recreate_from_output(request: dict):
    """Extract settings from output filename for recreation"""
    try:
        filename = request.get("filename", "")
        
        metadata = extract_metadata_from_filename(filename)
        
        if metadata["seed"] is not None:
            return {
                "success": True,
                "seed": metadata["seed"],
                "message": f"Extracted seed: {metadata['seed']}, type: {metadata['type']}"
            }
        
        parts = filename.replace(".png", "").split("_")
        
        seed = None
        steps = None
        
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) >= 4:
                if seed is None:
                    seed = int(part)
                else:
                    steps = int(part)
        
        if seed is None:
            return {"success": False, "error": "Could not extract seed from filename"}
        
        return {
            "success": True,
            "seed": seed,
            "steps": steps,
            "message": f"Extracted seed: {seed}" + (f", steps: {steps}" if steps else "")
        }
        
    except Exception as e:
        logger.error(f"Recreation failed: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/gallery")
async def get_gallery():
    """Get list of generated images"""
    os.makedirs("outputs", exist_ok=True)
    files = sorted(
        [f for f in os.listdir("outputs") if f.endswith(".png")],
        key=lambda x: os.path.getmtime(f"outputs/{x}"),
        reverse=True
    )[:12]
    
    return {"images": files}

@app.post("/api/upscale")
async def upscale_image(request: UpscaleRequest):
    """Upscale image using Real-ESRGAN (AI) or enhanced Lanczos interpolation"""
    try:
        filepath = f"outputs/{request.filename}"
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Image not found")
        
        scale = request.scale
        if scale not in [2, 4, 8]:
            scale = 2
        
        logger.upscale_start(request.filename, scale)
        
        image = Image.open(filepath).convert("RGB")
        original_size = image.size
        
        start_time = time.time()
        
        if REALESRGAN_AVAILABLE and upscaler is not None:
            logger.info(f"Using AI Upscaling (Real-ESRGAN) for {original_size} at {scale}x...")
            
            try:
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                image_np = np.array(image)
                
                if scale == 2:
                    upscaled_4x, _ = upscaler.enhance(image_np, outscale=4)
                    upscaled_img = Image.fromarray(upscaled_4x.astype('uint8'))
                    new_size = (original_size[0] * 2, original_size[1] * 2)
                    upscaled = upscaled_img.resize(new_size, Image.Resampling.LANCZOS)
                    
                elif scale == 4:
                    upscaled_4x, _ = upscaler.enhance(image_np, outscale=4)
                    upscaled = Image.fromarray(upscaled_4x.astype('uint8'))
                    new_size = (original_size[0] * 4, original_size[1] * 4)
                    
                elif scale == 8:
                    logger.info("Performing two-pass upscale for 8x...")
                    upscaled_4x, _ = upscaler.enhance(image_np, outscale=4)
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    upscaled_8x, _ = upscaler.enhance(upscaled_4x, outscale=4)
                    upscaled_img = Image.fromarray(upscaled_8x.astype('uint8'))
                    new_size = (original_size[0] * 8, original_size[1] * 8)
                    upscaled = upscaled_img.resize(new_size, Image.Resampling.LANCZOS)
                
                logger.success(f"Real-ESRGAN {scale}x upscaling completed!")
                
            except Exception as e:
                logger.warning(f"Real-ESRGAN failed: {e}. Falling back to enhanced Lanczos...")
                new_size = (original_size[0] * scale, original_size[1] * scale)
                upscaled = image.resize(new_size, Image.Resampling.LANCZOS)
                from PIL import ImageFilter, ImageEnhance
                upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                enhancer = ImageEnhance.Contrast(upscaled)
                upscaled = enhancer.enhance(1.1)
        else:
            logger.warning("Real-ESRGAN not available or failed - using enhanced Lanczos")
            new_size = (original_size[0] * scale, original_size[1] * scale)
            upscaled = image.resize(new_size, Image.Resampling.LANCZOS)
            
            from PIL import ImageFilter, ImageEnhance
            upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            enhancer = ImageEnhance.Contrast(upscaled)
            upscaled = enhancer.enhance(1.1)

        upscale_time = time.time() - start_time
        
        original_metadata = extract_metadata_from_filename(request.filename)
        original_seed = original_metadata["seed"] if original_metadata["seed"] is not None else int(time.time())
        
        new_filename = generate_filename("upscale", original_seed, scale=scale)
        upscaled.save(f"outputs/{new_filename}")
        
        buffered = io.BytesIO()
        upscaled.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        logger.success(f"Upscaled: {original_size} → {new_size} ({scale}x) in {upscale_time:.1f}s")
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_str}",
            "filename": new_filename,
            "width": new_size[0],
            "height": new_size[1],
            "scale": scale,
            "method": "Real-ESRGAN" if (REALESRGAN_AVAILABLE and upscaler) else "Lanczos"
        }
        
    except Exception as e:
        logger.error(f"Upscaling failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/variation")
async def create_variation(request: VariationRequest):
    """Create variation of existing image using img2img"""
    try:
        filepath = f"outputs/{request.filename}"
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Image not found")
        
        logger.variation_start(request.filename, request.strength)
        
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        init_image = Image.open(filepath).convert("RGB")
        
        if device == "cuda":
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_total <= 4:
                max_size = 512
                if init_image.width > max_size or init_image.height > max_size:
                    init_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    logger.warning(f"Image resized to {init_image.size} for 4GB VRAM")
        
        prompt = request.prompt or "anime girl, high quality, detailed"
        full_prompt = f"masterpiece, best quality, {prompt}"
        neg_prompt = "lowres, bad anatomy, worst quality, blurry"
        
        seed = np.random.randint(0, 2147483647)
        generator = torch.Generator(device).manual_seed(seed)
        
        start_time = time.time()
        
        inference_steps = 25 if device == "cuda" and torch.cuda.get_device_properties(0).total_memory / 1024**3 <= 4 else 30
        
        result = img2img_pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            image=init_image,
            strength=request.strength,
            num_inference_steps=inference_steps,
            guidance_scale=10.0,
            generator=generator
        )
        
        generation_time = time.time() - start_time
        
        new_filename = generate_filename("variation", seed, inference_steps)
        
        image = result.images[0]
        image.save(f"outputs/{new_filename}")
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        logger.success(f"Variation created in {generation_time:.1f}s")
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
        log_prompt_history(new_filename, seed, prompt, inference_steps)
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_str}",
            "filename": new_filename,
            "seed": seed,
            "generation_time": round(generation_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Variation failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/dram-extension")
async def toggle_dram_extension(request: dict):
    """Toggle DRAM extension with configurable max DRAM allocation"""
    enabled = request.get("enabled", False)
    max_dram = request.get("max_dram_gb", 8)
    
    max_dram = max(8, min(16, max_dram))

    if device != "cuda":
        return {
            "success": False,
            "error": "DRAM extension only works on CUDA devices."
        }

    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    threshold = dram_extension_config["vram_threshold_gb"]

    if vram_total > threshold and enabled:
        return {
            "success": False,
            "error": f"DRAM extension only allowed for GPUs with ≤ {threshold}GB VRAM. Yours has {vram_total:.1f}GB."
        }

    was_enabled = dram_extension_config["enabled"]
    dram_extension_config["enabled"] = enabled
    dram_extension_config["max_dram_gb"] = max_dram

    if enabled:
        apply_dram_extension(pipe, img2img_pipe)
        message = f"DRAM Extension ENABLED. Your {vram_total:.1f}GB VRAM can now use up to +{max_dram}GB system RAM. You can use 35 steps and higher resolutions!"

    else:
        logger.info("🔧 Disabling DRAM Extension…")

        try:
            if pipe is not None:
                pipe.to("cuda")
                logger.success("   ✅ Text-to-image pipeline moved back to VRAM-only mode")
        except Exception as e:
            logger.error(f"Could not reset text-to-image pipeline: {e}")

        try:
            if img2img_pipe is not None:
                img2img_pipe.to("cuda")
                logger.success("   ✅ Image-to-image pipeline moved back to VRAM-only mode")
        except Exception as e:
            logger.error(f"Could not reset image-to-image pipeline: {e}")

        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        torch.cuda.empty_cache()

        message = "DRAM Extension DISABLED. Using VRAM only mode. Use lower settings to avoid OOM errors."

    logger.info(f"🔧 {message}")

    return {
        "success": True,
        "enabled": enabled,
        "max_dram_gb": max_dram,
        "message": message
    }


@app.get("/api/dram-status")
async def get_dram_status():
    """Get current DRAM extension status"""
    return {
        "enabled": dram_extension_config["enabled"],
        "max_dram_gb": dram_extension_config["max_dram_gb"],
        "vram_threshold_gb": dram_extension_config["vram_threshold_gb"]
    }

@app.get("/settings")
async def settings_page():
    """Serve settings page"""
    return FileResponse(str(Path(__file__).parent.parent / "frontend" / "settings.html"))


# ========== MAIN ==========
if __name__ == "__main__":
    import uvicorn
    
    logger.info("\n" + "=" * 70)
    logger.info("🎨 AI Image Generator - FastAPI Backend")
    logger.info("=" * 70)
    logger.info("\n📝 Requirements:")
    logger.info("   pip install fastapi uvicorn websockets pillow numpy torch diffusers")
    logger.info("\n📁 Files needed:")
    logger.info("   - server.py (this file)")
    logger.info("   - index.html (in frontend/ folder)")
    logger.info("   - generate.html (in frontend/ folder)")
    logger.info("   - settings.html (in frontend/ folder)")
    logger.info("   - gallery.html (in frontend/ folder)")
    logger.info("   - favicon.png (in frontend/assets/ folder)")
    logger.info("\n🚀 Starting server...")
    logger.info("=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")