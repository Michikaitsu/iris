from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import io
import base64
import json
import asyncio
import time
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.backend.logger import create_logger
import threading
import re  # <UPDATE>

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
    print("[SUCCESS] Real-ESRGAN libraries loaded successfully!")
except ImportError as e:
    REALESRGAN_AVAILABLE = False
    print(f"[WARNING] Real-ESRGAN not available: {e}")
    print("[INFO] Install with: pip install realesrgan basicsr torch torchvision")

logger = create_logger("ImageGeneratorServer")

pipe = None
img2img_pipe = None
device = None
upscaler = None  # Global variable for Real-ESRGAN upscaler

connected_clients = []
generation_stats = {
    "total_images": 0,
    "total_time": 0,
    "gpu_temp": 0,
    "vram_used": 0
}

dram_extension_config = {
    "enabled": False,
    "max_dram_gb": 8,
    "vram_threshold_gb": 4
}

discord_bot_process = None
discord_bot_thread = None

# <UPDATE>
def generate_filename(file_type: str, seed: int, steps: int = None, scale: int = None) -> str:
    """
    Generate clean, sortable filename with consistent structure
    
    Format: {type}_{YYYY-MM-DD}_{HHMMSS}_{seed}.png
    
    Examples:
        gen_2025-11-15_142530_661119903.png (Generated)
        var_2025-11-15_143015_882200119.png (Variation)
        up2x_2025-11-15_143120_original-seed.png (Upscale 2x)
        up4x_2025-11-15_143125_original-seed.png (Upscale 4x)
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")
    
    if file_type == "generated":
        prefix = "gen"
    elif file_type == "variation":
        prefix = "var"
    elif file_type == "upscale":
        prefix = f"up{scale}x"
    else:
        prefix = file_type
    
    filename = f"{prefix}_{date_str}_{time_str}_{seed}.png"
    return filename

def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extract metadata from new filename format
    
    Returns: {type, date, time, seed, scale}
    """
    # Remove .png extension
    name = filename.replace(".png", "")
    
    # Split by underscore
    parts = name.split("_")
    
    metadata = {
        "type": "unknown",
        "date": None,
        "time": None,
        "seed": None,
        "scale": None
    }
    
    if len(parts) >= 4:
        # Extract type
        type_prefix = parts[0]
        if type_prefix == "gen":
            metadata["type"] = "generated"
        elif type_prefix == "var":
            metadata["type"] = "variation"
        elif type_prefix.startswith("up"):
            metadata["type"] = "upscale"
            # Extract scale (e.g., "up2x" -> 2)
            scale_match = re.match(r"up(\d+)x", type_prefix)
            if scale_match:
                metadata["scale"] = int(scale_match.group(1))
        
        # Extract date, time, seed
        metadata["date"] = parts[1] if len(parts) > 1 else None
        metadata["time"] = parts[2] if len(parts) > 2 else None
        metadata["seed"] = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
    
    return metadata
# </UPDATE>

@app.post("/api/variation")
async def create_variation(request: VariationRequest):
    """Create variation of existing image using img2img"""
    try:
        
        prompt = request.prompt or "anime girl, high quality, detailed"
        full_prompt = f"masterpiece, best quality, {prompt}"
        neg_prompt = "lowres, bad anatomy, worst quality, blurry"
        
        neg_prompt = f"{neg_prompt}, {NSFW_NEGATIVE_PROMPT}"
        
    except Exception as e:
        logger.error(f"Error creating variation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global pipe, img2img_pipe, device, discord_bot_process, discord_bot_thread, upscaler
    
    logger.log_session_start()
    logger.info("=" * 70)
    logger.info("🚀 Starting AI Image Generator Backend...")
    logger.info("=" * 70)
    
    dtype = torch.float32  # Default
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.success(f"NVIDIA GPU detected: {gpu_name}")
        
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.gpu_info(gpu_name, vram_total_gb, 0, 0)
        
        # Set PyTorch memory allocator for better fragmentation handling
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        if vram_total_gb <= dram_extension_config["vram_threshold_gb"]:
            logger.info(f"💡 DRAM Extension available: You can enable up to +{dram_extension_config['max_dram_gb']}GB system RAM for VRAM overflow")
            logger.info(f"🔧 Auto-enabling DRAM Extension for {vram_total_gb:.1f}GB VRAM card")
            dram_extension_config["enabled"] = True
        
        # Check if GPU has Tensor Cores (RTX series)
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
    
    # Load default model (anime)
    logger.model_load_start("Ojimi/anime-kawai-diffusion")
    logger.info("⏳ This will take 5-10 minutes on first run (downloading model)...")
    
    model_id = "Ojimi/anime-kawai-diffusion"
    
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
        pipe.enable_attention_slicing(slice_size=1)  # Maximum memory saving
        
        # Enable VAE slicing for encoding/decoding images in smaller chunks
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
            logger.success("VAE slicing enabled (reduces memory for image processing)")
        
        # Enable VAE tiling for very large images
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
    
    # Load img2img pipeline (reuse components to save memory)
    logger.info("📦 Loading Image-to-Image pipeline...")
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
    
    logger.success("✅ Model loaded successfully!")
    logger.info("=" * 70)
    logger.info(f"🌐 Server ready at http://localhost:8000")
    logger.info("=" * 70)
    
    upscaler = None
    if REALESRGAN_AVAILABLE and device == "cuda":
        try:
            logger.info("📦 Loading Real-ESRGAN upscaler...")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upscaler = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=(dtype == torch.float16),  # Use half precision only if model uses it
                device=device
            )
            logger.success("✅ Real-ESRGAN upscaler loaded successfully!")
        except Exception as e:
            logger.warning(f"⚠️  Could not load Real-ESRGAN: {e}")
            logger.info("📐 Will use Lanczos upscaling as fallback")
            import traceback
            logger.debug(traceback.format_exc())
            upscaler = None
    elif REALESRGAN_AVAILABLE and device != "cuda":
        logger.info(f"Real-ESRGAN requires CUDA GPU (current device: {device})")
    elif not REALESRGAN_AVAILABLE:
        logger.info("Real-ESRGAN libraries not available - using Lanczos upscaling")

    logger.info("")
    logger.discord_bot_start()
    try:
        def run_bot_process():
            """Run bot in background and capture output"""
            try:
                bot_path = Path(__file__).parent / "discord_bot.py"
                process = subprocess.Popen(
                    [sys.executable, "-u", str(bot_path)],  # -u for unbuffered output
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    # Run from project root (3 levels up)
                    cwd=str(Path(__file__).parent.parent.parent)
                )
                
                # Stream all output to console
                for line in process.stdout:
                    if line:
                        print(f"[BOT] {line.rstrip()}")
                    
            except Exception as e:
                print(f"[BOT ERROR] Failed to run bot: {e}", file=sys.stderr)
        
        # Keep track of the thread
        discord_bot_thread = threading.Thread(target=run_bot_process, daemon=True)
        discord_bot_thread.start()
        
        logger.success("Discord bot started in background")
        logger.info("Bot will monitor outputs/ folder and send images to Discord")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Failed to start Discord bot: {e}")
        logger.warning("Server will continue without Discord bot")

    yield  # Server runs here
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down...")
    
    if discord_bot_thread and discord_bot_thread.is_alive():
        logger.info("Stopping Discord bot thread...")
        # Note: Directly killing threads is generally discouraged.
        # A more robust solution would involve signaling the bot thread to exit gracefully.
        # For this example, we'll rely on the daemon=True for automatic termination on main thread exit.
        # If explicit termination is needed, a shared event or flag would be required.
        logger.success("Discord bot thread will terminate automatically as it's a daemon.")

    if device == "cuda":
        torch.cuda.empty_cache()
    logger.log_session_end()

def apply_dram_extension(txt2img_pipe=None, img2img_pipe_obj=None):
    """Enable DRAM as VRAM extension using sequential CPU offload - moves model layers dynamically"""
    global pipe, img2img_pipe
    
    if device != "cuda":
        logger.warning("DRAM extension only works with CUDA devices")
        return
    
    try:
        logger.info("🔧 Enabling DRAM Extension (VRAM + System RAM)...")
        logger.info("   This allows using system RAM to supplement VRAM")
        logger.info("   Model components will move between VRAM and RAM as needed")
        
        # Use the most aggressive offloading: sequential_cpu_offload
        # This moves model layers one-by-one between CUDA and CPU
        # Allows running on very low VRAM (even 2GB) but uses system RAM
        
        current_pipe = txt2img_pipe or pipe
        current_img2img = img2img_pipe_obj or img2img_pipe
        
        if current_pipe is not None:
            if hasattr(current_pipe, 'enable_sequential_cpu_offload'):
                logger.info("   🔄 Applying sequential CPU offload to text-to-image pipeline...")
                current_pipe.enable_sequential_cpu_offload()
                logger.success("   ✅ Text-to-image pipeline will use VRAM + System RAM")
            elif hasattr(current_pipe, 'enable_model_cpu_offload'):
                logger.info("   🔄 Applying model CPU offload to text-to-image pipeline...")
                current_pipe.enable_model_cpu_offload()
                logger.success("   ✅ Text-to-image pipeline will offload to RAM when needed")
        
        if current_img2img is not None:
            if hasattr(current_img2img, 'enable_sequential_cpu_offload'):
                logger.info("   🔄 Applying sequential CPU offload to image-to-image pipeline...")
                current_img2img.enable_sequential_cpu_offload()
                logger.success("   ✅ Image-to-image pipeline will use VRAM + System RAM")
            elif hasattr(current_img2img, 'enable_model_cpu_offload'):
                logger.info("   🔄 Applying model CPU offload to image-to-image pipeline...")
                current_img2img.enable_model_cpu_offload()
                logger.success("   ✅ Image-to-image pipeline will offload to RAM when needed")
        
        # Additional memory optimizations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.success("DRAM Extension ENABLED!")
        logger.info(f"   Your {vram_gb:.1f}GB VRAM can now use up to +{dram_extension_config['max_dram_gb']}GB system RAM")
        logger.info("   You can now use 35 steps and higher resolutions")
        logger.info("   Note: Generation may be slightly slower but won't crash")
        
    except Exception as e:
        logger.error(f"Failed to enable DRAM extension: {e}")
        import traceback
        traceback.print_exc()

def log_prompt_history(filename: str, seed: int, prompt: str, steps: int):
    """Log prompt and seed to history file for future variations"""
    try:
        history_file = Path("static") / "data" / "prompts_history.txt"
        Path("static/data").mkdir(parents=True, exist_ok=True)
        with open(history_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | Seed: {seed} | Steps: {steps} | Prompt: {prompt} | File: {filename}\n")
        logger.debug(f"Logged prompt history: {filename}")
    except Exception as e:
        logger.error(f"Error logging prompt: {e}")

app = FastAPI(title="AI Image Generator API", lifespan=lifespan)

# CORS für Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ========== MODELS ==========
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "lowres, bad anatomy, bad hands, worst quality, low quality, blurry, nsfw, nude"
    style: str = "anime"
    seed: Optional[int] = None
    steps: int = 35
    cfg_scale: float = 10.0
    width: int = 512
    height: int = 768
    batch_size: int = 1
    dram_extension_enabled: Optional[bool] = False

NSFW_NEGATIVE_PROMPT = "nsfw, nude, naked, explicit, adult, sexual, pornographic, x-rated, 18+, underage, minor, loli, shota, child"

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
    
    Supported resolutions:
    - Portrait: 512x768, 768x1024, 1024x1536, 1536x2048, 1920x2688
    - Mobile Wallpaper: 720x1280 (9:16 HD), 1080x1920 (9:16 Full HD)
    - Square: 512x512, 768x768, 1024x1024, 1536x1536, 2048x2048
    - Landscape: 768x512, 1024x768, 1536x1024, 1920x1536, 2688x1920
    """
    if dram_extension_config["enabled"]:
        logger.info(f"   🔄 DRAM Extension active - using full parameters: {width}x{height}, {steps} steps")
        return width, height, steps
    
    total_pixels = width * height
    
    # VRAM requirements (rough estimates):
    # 512x512 (20 steps) = ~2.5GB
    # 512x768 (20 steps) = ~3.2GB
    # 512x768 (35 steps)
