"""Image upscaling service for I.R.I.S."""
from PIL import Image
import torch
from pathlib import Path

from ..utils.logger import logger


class UpscalerService:
    """Handles image upscaling operations"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        
    def upscale_with_realesrgan(self, image: Image.Image, scale: int = 2) -> Image.Image:
        """Upscale using Real-ESRGAN"""
        if not self.model_loader.upscaler:
            raise RuntimeError("Real-ESRGAN not available")
            
        import numpy as np
        import cv2
        
        # Convert PIL to numpy
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Upscale
        output, _ = self.model_loader.upscaler.enhance(img_array, outscale=scale)
        
        # Convert back to PIL
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output)
        
    def upscale_with_lanczos(self, image: Image.Image, scale: int = 2) -> Image.Image:
        """Upscale using Lanczos interpolation (fallback)"""
        new_width = image.width * scale
        new_height = image.height * scale
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    def upscale(self, image: Image.Image, scale: int = 2, use_ai: bool = True) -> Image.Image:
        """Upscale image using best available method"""
        if use_ai and self.model_loader.upscaler:
            try:
                return self.upscale_with_realesrgan(image, scale)
            except Exception as e:
                logger.warning(f"Real-ESRGAN upscaling failed: {e}, falling back to Lanczos")
                return self.upscale_with_lanczos(image, scale)
        else:
            return self.upscale_with_lanczos(image, scale)
