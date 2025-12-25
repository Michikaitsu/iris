"""Image generation logic for I.R.I.S."""
import torch
from PIL import Image
import io
import base64
from datetime import datetime
from typing import Optional, Dict, Any

from ..utils.logger import logger
from ..utils.file_manager import generate_filename, save_image


class ImageGenerator:
    """Handles image generation operations"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        
    async def generate_text2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 768,
        steps: int = 28,
        cfg_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate image from text prompt"""
        
        pipe = self.model_loader.pipe
        if not pipe:
            raise RuntimeError("Text-to-image pipeline not loaded")
            
        # Set seed
        generator = torch.Generator(device=self.model_loader.device)
        if seed is None:
            seed = generator.seed()
        else:
            generator = generator.manual_seed(seed)
            
        # Generate image
        logger.info(f"Generating image with seed {seed}")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        )
        
        image = result.images[0]
        
        # Save image
        filename = generate_filename(seed)
        filepath = save_image(image, filename)
        
        return {
            "image": image,
            "filename": filename,
            "filepath": filepath,
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale
        }
        
    async def generate_img2img(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.75,
        steps: int = 28,
        cfg_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate image from image and prompt"""
        
        pipe = self.model_loader.img2img_pipe
        if not pipe:
            raise RuntimeError("Image-to-image pipeline not loaded")
            
        # Set seed
        generator = torch.Generator(device=self.model_loader.device)
        if seed is None:
            seed = generator.seed()
        else:
            generator = generator.manual_seed(seed)
            
        # Generate image
        logger.info(f"Generating variation with seed {seed}")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        )
        
        output_image = result.images[0]
        
        # Save image
        filename = generate_filename(seed)
        filepath = save_image(output_image, filename)
        
        return {
            "image": output_image,
            "filename": filename,
            "filepath": filepath,
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "steps": steps,
            "cfg_scale": cfg_scale
        }
