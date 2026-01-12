"""
System Routes - Health, GPU info, version endpoints
"""
import subprocess
import torch
from fastapi import APIRouter

from src.api.services.pipeline import pipeline_service, MODEL_CONFIGS
from src.utils.logger import create_logger

logger = create_logger("SystemRoutes")
router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipeline_service.pipe is not None,
        "device": pipeline_service.device
    }


@router.get("/system")
async def get_system_info():
    """Get system information"""
    info = {
        "gpu_name": "Unknown",
        "device": pipeline_service.device,
        "vram_total": 0.0,
        "vram_used": 0.0,
        "gpu_temp": 0.0,
        "dram_extension_enabled": pipeline_service.dram_config["enabled"],
        "dram_extension_available": False
    }
    
    if pipeline_service.device == "cuda":
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info["vram_total"] = vram_total
            info["vram_used"] = torch.cuda.memory_allocated(0) / 1024**3
            info["dram_extension_available"] = vram_total <= pipeline_service.dram_config["vram_threshold_gb"]
            
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=2
                )
                info["gpu_temp"] = float(result.stdout.strip())
            except:
                info["gpu_temp"] = 0
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
    
    return info


@router.get("/gpu-info")
async def get_gpu_info():
    """Get detailed GPU information"""
    info = {
        "gpu_name": "Unknown",
        "vram_total": 0.0,
        "vram_used": 0.0,
        "vram_free": 0.0,
        "vram_percent": 0.0,
        "gpu_temp": 0.0,
        "power_draw": 0.0,
        "gpu_utilization": 0.0,
        "status": "unknown"
    }
    
    if pipeline_service.device == "cuda":
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram_used = torch.cuda.memory_allocated(0) / 1024**3
            vram_free = vram_total - vram_used
            
            info["vram_total"] = round(vram_total, 2)
            info["vram_used"] = round(vram_used, 2)
            info["vram_free"] = round(vram_free, 2)
            info["vram_percent"] = round((vram_used / vram_total) * 100, 1) if vram_total > 0 else 0
            info["status"] = "success"
            
            # GPU stats from nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2
                )
                parts = result.stdout.strip().split(',')
                if len(parts) >= 5:
                    info["gpu_temp"] = float(parts[0].strip())
                    info["power_draw"] = float(parts[1].strip())
                    info["gpu_utilization"] = float(parts[2].strip())
                    # Use nvidia-smi values for more accurate VRAM
                    info["vram_used"] = round(float(parts[3].strip()) / 1024, 2)
                    info["vram_total"] = round(float(parts[4].strip()) / 1024, 2)
                    info["load"] = info["gpu_utilization"]
                    info["temp"] = info["gpu_temp"]
                    info["memoryUsed"] = float(parts[3].strip())
                    info["memoryTotal"] = float(parts[4].strip())
            except:
                pass
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            info["status"] = "error"
    else:
        info["status"] = "no_gpu"
    
    return {"status": info["status"], "gpu": info}


@router.get("/version")
async def get_version_info():
    """Get version information"""
    import sys
    import platform
    
    # Detect OS
    os_name = platform.system()
    if os_name == "Darwin":
        os_name = "macOS"
    
    return {
        "iris_version": "1.2.0",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "CPU",
        "current_model": pipeline_service.current_model,
        "device": pipeline_service.device,
        "os": os_name
    }


@router.get("/models")
async def get_available_models():
    """Get list of available models"""
    models = []
    for key, config in MODEL_CONFIGS.items():
        models.append({
            "id": key,
            "name": config["id"].split("/")[-1],
            "description": config["description"],
            "huggingface_id": config["id"],
            "is_loaded": pipeline_service.current_model == key
        })
    return {"models": models}


@router.get("/rpc-status")
async def get_rpc_status():
    """Get RPC status (placeholder)"""
    return {
        "connected": True,
        "status": "ready",
        "details": "Ready for requests"
    }
