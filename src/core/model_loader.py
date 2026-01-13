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
        self.upscaler = None  # Real-ESRGAN
        self.upscaler_bsrgan = None  # BSRGAN
        self.upscaler_anime = None  # Real-ESRGAN Anime v3
        self.swinir_model = None
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
            # Apply torchvision compatibility fix
            self._patch_torchvision_compat()
            
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
    
    async def load_upscaler_bsrgan(self):
        """Load BSRGAN upscaler - optimized for degraded/compressed images (no Tensor Cores needed)"""
        try:
            # Apply torchvision compatibility fix
            self._patch_torchvision_compat()
            
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # BSRGAN uses same RRDB architecture but different weights
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upscaler_bsrgan = RealESRGANer(
                scale=4,
                model_path='https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth',
                model=model,
                tile=400,  # Tile for memory efficiency
                tile_pad=10,
                pre_pad=0,
                half=self.dtype == torch.float16,
                device=self.device
            )
            logger.info("BSRGAN upscaler loaded (optimized for degraded images)")
            return True
            
        except Exception as e:
            logger.warning(f"BSRGAN not available: {e}")
            self.upscaler_bsrgan = None
            return False
    
    async def load_upscaler_anime_v3(self):
        """Load Real-ESRGAN Anime Video v3 - best for anime/illustrations (no Tensor Cores needed)"""
        try:
            # Apply torchvision compatibility fix
            self._patch_torchvision_compat()
            
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # Anime v3 model - smaller and faster, optimized for anime
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            self.upscaler_anime = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=self.dtype == torch.float16,
                device=self.device
            )
            logger.info("Real-ESRGAN Anime v3 upscaler loaded (fast anime upscaling)")
            return True
            
        except Exception as e:
            logger.warning(f"Real-ESRGAN Anime v3 not available: {e}")
            self.upscaler_anime = None
            return False
    
    def _patch_torchvision_compat(self):
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
    
    async def load_swinir(self):
        """Load SwinIR upscaler for superior quality"""
        try:
            import timm
            from timm.models import swin_transformer
            
            logger.info("Loading SwinIR model...")
            
            # Load pretrained SwinIR model from HuggingFace
            model_url = "https://huggingface.co/lixiaoqiang/SwinIR/resolve/main/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
            
            # Import SwinIR architecture
            from .swinir_arch import SwinIR
            
            # Initialize model
            self.swinir_model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='1conv'
            )
            
            # Load weights
            pretrained_model = torch.hub.load_state_dict_from_url(
                model_url,
                map_location=self.device,
                progress=True
            )
            
            self.swinir_model.load_state_dict(pretrained_model['params'], strict=True)
            self.swinir_model = self.swinir_model.to(self.device).eval()
            
            logger.info("SwinIR loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"SwinIR not available: {e}")
            logger.info("SwinIR requires additional dependencies. Real-ESRGAN will be used as fallback.")
            self.swinir_model = None
            return False
            
    def get_pipelines(self):
        """Get loaded pipelines"""
        return {
            "text2img": self.pipe,
            "img2img": self.img2img_pipe,
            "upscaler": self.upscaler,
            "upscaler_bsrgan": self.upscaler_bsrgan,
            "upscaler_anime": self.upscaler_anime,
            "swinir": self.swinir_model
        }
