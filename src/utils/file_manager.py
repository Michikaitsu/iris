"""
File management utilities for I.R.I.S.
"""
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, Dict, List
from src.core.config import Config

class FileManager:
    """Manage file operations and naming"""
    
    @staticmethod
    def generate_filename(
        prefix: str = "img",
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        extension: str = "png"
    ) -> str:
        """
        Generate a standardized filename
        Format: {prefix}_{timestamp}_{seed}_{width}x{height}.{extension}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        parts = [prefix, timestamp]
        
        if seed is not None:
            parts.append(str(seed))
        
        if width and height:
            parts.append(f"{width}x{height}")
        
        filename = "_".join(parts) + f".{extension}"
        return filename
    
    @staticmethod
    def save_json(filepath: Path, data: Dict or List):
        """Save data to JSON file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_json(filepath: Path) -> Dict or List:
        """Load data from JSON file"""
        if not filepath.exists():
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def log_prompt(prompt: str, settings: Dict):
        """Log prompt to prompts_history.json"""
        log_file = Config.DATA_DIR / "prompts_history.json"
        
        # Load existing prompts
        prompts = []
        if log_file.exists():
            prompts = FileManager.load_json(log_file)
            if not isinstance(prompts, list):
                prompts = []
        
        # Add new prompt
        prompts.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "settings": settings
        })
        
        # Save
        FileManager.save_json(log_file, prompts)
    
    @staticmethod
    def log_sent_image(filename: str, message_link: str):
        """Log sent image to img_send.json"""
        log_file = Config.DATA_DIR / "img_send.json"
        
        # Load existing
        sent_images = FileManager.load_json(log_file)
        if not isinstance(sent_images, dict):
            sent_images = {}
        
        # Add new entry
        sent_images[filename] = {
            "message_link": message_link,
            "sent_at": datetime.now().isoformat()
        }
        
        # Save
        FileManager.save_json(log_file, sent_images)
    
    @staticmethod
    def get_sent_images() -> Dict:
        """Get all sent images"""
        log_file = Config.DATA_DIR / "img_send.json"
        sent_images = FileManager.load_json(log_file)
        return sent_images if isinstance(sent_images, dict) else {}
