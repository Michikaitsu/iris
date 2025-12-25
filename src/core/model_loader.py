"""Model loading and management for I.R.I.S."""
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from pathlib import Path
import os

from ..utils.logger import logger


class ModelLoader:
    """Handles loading and managing AI models"""
    
    def __init__(self):
        self.pipe = None
        self.img2img_pipe = None
        self.upscaler = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
    async def load_text2img_pipeline(self, model_id: str = "Ojimi/anime-kawai-diffusion"):
        """Load the text-to-image pipeline"""
        logger.info(f"Loading text-to-image model: {model_id}")
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None
            )
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            if torch.cuda.is_available():
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_slicing()
                self.pipe.enable_vae_tiling()
                
                # DRAM extension for low VRAM GPUs
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 6:
                    self.pipe.enable_model_cpu_offload()
                    logger.info(f"DRAM Extension enabled for {vram_gb:.1f}GB VRAM GPU")
                    
            logger.info("Text-to-image pipeline loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load text-to-image pipeline: {e}")
            return False
            
    async def load_img2img_pipeline(self):
        """Load the image-to-image pipeline"""
        if not self.pipe:
            logger.error("Text-to-image pipeline must be loaded first")
            return False
            
        try:
            logger.info("Loading image-to-image pipeline...")
            self.img2img_pipe = StableDiffusionImg2ImgPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,
                feature_extractor=self.pipe.feature_extractor
            )
            self.img2img_pipe = self.img2img_pipe.to(self.device)
            
            logger.info("Image-to-image pipeline loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load image-to-image pipeline: {e}")
            return False
            
    async def load_upscaler(self):
        """Load Real-ESRGAN upscaler if available"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upscaler = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus_anime_6B.pth',
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=self.dtype == torch.float16,
                device=self.device
            )
            logger.info("Real-ESRGAN upscaler loaded successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"Real-ESRGAN not available: {e}")
            logger.warning("Will use Lanczos upscaling as fallback")
            self.upscaler = None
            return False
        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN: {e}")
            self.upscaler = None
            return False
            
    def get_pipelines(self):
        """Get loaded pipelines"""
        return {
            "text2img": self.pipe,
            "img2img": self.img2img_pipe,
            "upscaler": self.upscaler
        }
