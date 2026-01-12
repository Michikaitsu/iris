"""
Prompt Templates Service - Pre-defined prompt building blocks
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.core.config import Config
from src.utils.logger import create_logger

logger = create_logger("TemplatesService")

TEMPLATES_FILE = Config.DATA_DIR / "prompt_templates.json"

# Default built-in templates
DEFAULT_TEMPLATES = {
    "styles": {
        "anime": {
            "name": "Anime Style",
            "prefix": "anime style, anime art,",
            "suffix": "anime aesthetic, japanese animation style",
            "negative": "realistic, photorealistic, 3d render"
        },
        "realistic": {
            "name": "Photorealistic",
            "prefix": "photorealistic, ultra realistic, 8k uhd,",
            "suffix": "professional photography, sharp focus",
            "negative": "anime, cartoon, illustration, painting"
        },
        "pixel_art": {
            "name": "Pixel Art",
            "prefix": "pixel art, 16-bit style, retro game art,",
            "suffix": "pixelated, sprite art, game asset",
            "negative": "smooth, anti-aliased, realistic, 3d"
        },
        "oil_painting": {
            "name": "Oil Painting",
            "prefix": "oil painting, classical art, masterpiece,",
            "suffix": "brush strokes, canvas texture, fine art",
            "negative": "digital art, photo, anime"
        },
        "watercolor": {
            "name": "Watercolor",
            "prefix": "watercolor painting, soft colors,",
            "suffix": "watercolor texture, artistic, flowing colors",
            "negative": "digital, sharp edges, photorealistic"
        },
        "cyberpunk": {
            "name": "Cyberpunk",
            "prefix": "cyberpunk style, neon lights, futuristic,",
            "suffix": "sci-fi, dystopian, high tech low life",
            "negative": "medieval, fantasy, nature, pastoral"
        },
        "fantasy": {
            "name": "Fantasy Art",
            "prefix": "fantasy art, magical, ethereal,",
            "suffix": "epic fantasy, mystical atmosphere",
            "negative": "modern, sci-fi, realistic photo"
        },
        "minimalist": {
            "name": "Minimalist",
            "prefix": "minimalist art, simple, clean design,",
            "suffix": "negative space, geometric, modern",
            "negative": "detailed, complex, busy, cluttered"
        }
    },
    "quality": {
        "masterpiece": {
            "name": "Masterpiece Quality",
            "prefix": "masterpiece, best quality, ultra-detailed,",
            "suffix": "highly detailed, intricate details",
            "negative": "lowres, bad quality, worst quality"
        },
        "high_res": {
            "name": "High Resolution",
            "prefix": "8k uhd, high resolution, sharp focus,",
            "suffix": "detailed textures, crisp details",
            "negative": "blurry, low resolution, pixelated"
        },
        "cinematic": {
            "name": "Cinematic",
            "prefix": "cinematic lighting, dramatic lighting,",
            "suffix": "movie scene, film grain, depth of field",
            "negative": "flat lighting, amateur"
        }
    },
    "subjects": {
        "portrait": {
            "name": "Portrait",
            "prefix": "portrait, face focus,",
            "suffix": "beautiful face, detailed eyes, expressive",
            "negative": "bad anatomy, deformed face, ugly"
        },
        "landscape": {
            "name": "Landscape",
            "prefix": "landscape, scenery, wide shot,",
            "suffix": "beautiful nature, atmospheric",
            "negative": "person, character, portrait"
        },
        "character": {
            "name": "Full Character",
            "prefix": "full body, character design,",
            "suffix": "detailed outfit, dynamic pose",
            "negative": "cropped, partial body, bad anatomy"
        },
        "environment": {
            "name": "Environment",
            "prefix": "environment concept art, detailed background,",
            "suffix": "atmospheric, immersive scene",
            "negative": "character focus, portrait"
        }
    },
    "lighting": {
        "golden_hour": {
            "name": "Golden Hour",
            "prefix": "golden hour lighting, warm sunlight,",
            "suffix": "sunset colors, soft shadows",
            "negative": "harsh lighting, cold colors"
        },
        "studio": {
            "name": "Studio Lighting",
            "prefix": "studio lighting, professional lighting,",
            "suffix": "soft box, rim light, key light",
            "negative": "natural lighting, outdoor"
        },
        "neon": {
            "name": "Neon Glow",
            "prefix": "neon lighting, glowing lights,",
            "suffix": "vibrant colors, light reflections",
            "negative": "natural lighting, daylight"
        },
        "dramatic": {
            "name": "Dramatic",
            "prefix": "dramatic lighting, chiaroscuro,",
            "suffix": "strong shadows, contrast, moody",
            "negative": "flat lighting, even lighting"
        }
    }
}


@dataclass
class PromptTemplate:
    """Represents a prompt template"""
    id: str
    name: str
    category: str
    prefix: str
    suffix: str
    negative: str
    is_custom: bool = False


class PromptTemplates:
    """Manages prompt templates"""
    
    def __init__(self):
        self.templates: Dict = {}
        self.custom_templates: Dict = {}
        self._load()
    
    def _load(self):
        """Load templates"""
        # Load defaults
        self.templates = DEFAULT_TEMPLATES.copy()
        
        # Load custom templates
        try:
            if TEMPLATES_FILE.exists():
                with open(TEMPLATES_FILE, 'r', encoding='utf-8') as f:
                    self.custom_templates = json.load(f)
                logger.info(f"Loaded custom templates")
        except Exception as e:
            logger.error(f"Failed to load custom templates: {e}")
            self.custom_templates = {}
    
    def _save_custom(self):
        """Save custom templates"""
        try:
            os.makedirs(TEMPLATES_FILE.parent, exist_ok=True)
            with open(TEMPLATES_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.custom_templates, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save custom templates: {e}")
    
    def get_categories(self) -> List[str]:
        """Get all template categories"""
        categories = list(self.templates.keys())
        if "custom" in self.custom_templates:
            categories.append("custom")
        return categories
    
    def get_by_category(self, category: str) -> Dict:
        """Get all templates in a category"""
        if category == "custom":
            return self.custom_templates.get("custom", {})
        return self.templates.get(category, {})
    
    def get_template(self, category: str, template_id: str) -> Optional[Dict]:
        """Get a specific template"""
        if category == "custom":
            return self.custom_templates.get("custom", {}).get(template_id)
        return self.templates.get(category, {}).get(template_id)
    
    def get_all(self) -> Dict:
        """Get all templates including custom"""
        all_templates = self.templates.copy()
        if self.custom_templates:
            all_templates["custom"] = self.custom_templates.get("custom", {})
        return all_templates
    
    def add_custom(self, 
                   template_id: str,
                   name: str,
                   prefix: str = "",
                   suffix: str = "",
                   negative: str = "") -> bool:
        """Add a custom template"""
        if "custom" not in self.custom_templates:
            self.custom_templates["custom"] = {}
        
        self.custom_templates["custom"][template_id] = {
            "name": name,
            "prefix": prefix,
            "suffix": suffix,
            "negative": negative
        }
        
        self._save_custom()
        logger.info(f"Added custom template: {template_id}")
        return True
    
    def update_custom(self,
                      template_id: str,
                      name: str = None,
                      prefix: str = None,
                      suffix: str = None,
                      negative: str = None) -> bool:
        """Update a custom template"""
        if "custom" not in self.custom_templates:
            return False
        
        if template_id not in self.custom_templates["custom"]:
            return False
        
        template = self.custom_templates["custom"][template_id]
        
        if name is not None:
            template["name"] = name
        if prefix is not None:
            template["prefix"] = prefix
        if suffix is not None:
            template["suffix"] = suffix
        if negative is not None:
            template["negative"] = negative
        
        self._save_custom()
        return True
    
    def delete_custom(self, template_id: str) -> bool:
        """Delete a custom template"""
        if "custom" not in self.custom_templates:
            return False
        
        if template_id in self.custom_templates["custom"]:
            del self.custom_templates["custom"][template_id]
            self._save_custom()
            return True
        
        return False
    
    def apply_template(self, 
                       prompt: str,
                       category: str,
                       template_id: str,
                       include_negative: bool = True) -> Dict:
        """Apply a template to a prompt"""
        template = self.get_template(category, template_id)
        
        if not template:
            return {
                "prompt": prompt,
                "negative_prompt": ""
            }
        
        # Build enhanced prompt
        parts = []
        
        if template.get("prefix"):
            parts.append(template["prefix"])
        
        parts.append(prompt)
        
        if template.get("suffix"):
            parts.append(template["suffix"])
        
        enhanced_prompt = " ".join(parts)
        
        return {
            "prompt": enhanced_prompt,
            "negative_prompt": template.get("negative", "") if include_negative else ""
        }
    
    def combine_templates(self,
                          prompt: str,
                          template_selections: List[Dict]) -> Dict:
        """
        Combine multiple templates
        template_selections: [{"category": "styles", "id": "anime"}, ...]
        """
        prefixes = []
        suffixes = []
        negatives = []
        
        for selection in template_selections:
            template = self.get_template(
                selection.get("category", ""),
                selection.get("id", "")
            )
            
            if template:
                if template.get("prefix"):
                    prefixes.append(template["prefix"])
                if template.get("suffix"):
                    suffixes.append(template["suffix"])
                if template.get("negative"):
                    negatives.append(template["negative"])
        
        # Build combined prompt
        parts = prefixes + [prompt] + suffixes
        enhanced_prompt = " ".join(parts)
        
        # Combine negatives (remove duplicates)
        combined_negative = ", ".join(set(", ".join(negatives).split(", ")))
        
        return {
            "prompt": enhanced_prompt,
            "negative_prompt": combined_negative
        }


# Global instance
prompt_templates = PromptTemplates()
