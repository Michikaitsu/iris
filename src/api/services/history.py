"""
Generation History Service - Tracks all generations with full metadata
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from src.core.config import Config
from src.utils.logger import create_logger

logger = create_logger("HistoryService")

HISTORY_FILE = Config.DATA_DIR / "generation_history.json"


@dataclass
class HistoryEntry:
    """Represents a single generation history entry"""
    id: str
    filename: str
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    steps: int
    cfg_scale: float
    style: str
    model: str
    generation_time: float
    created_at: str
    tags: List[str] = None
    favorite: bool = False
    notes: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class GenerationHistory:
    """Manages generation history with full metadata"""
    
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.history: List[Dict] = []
        self._load()
    
    def _load(self):
        """Load history from file"""
        try:
            if HISTORY_FILE.exists():
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info(f"Loaded {len(self.history)} history entries")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            self.history = []
    
    def _save(self):
        """Save history to file"""
        try:
            os.makedirs(HISTORY_FILE.parent, exist_ok=True)
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def add(self,
            filename: str,
            prompt: str,
            negative_prompt: str,
            seed: int,
            width: int,
            height: int,
            steps: int,
            cfg_scale: float,
            style: str,
            model: str,
            generation_time: float) -> Dict:
        """Add a new entry to history"""
        
        entry = {
            "id": f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{seed}",
            "filename": filename,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "style": style,
            "model": model,
            "generation_time": generation_time,
            "created_at": datetime.now().isoformat(),
            "tags": [],
            "favorite": False,
            "notes": ""
        }
        
        self.history.insert(0, entry)
        
        # Trim if exceeds max
        if len(self.history) > self.max_entries:
            self.history = self.history[:self.max_entries]
        
        self._save()
        logger.info(f"Added to history: {filename}")
        
        return entry
    
    def get(self, entry_id: str) -> Optional[Dict]:
        """Get a specific entry by ID"""
        for entry in self.history:
            if entry.get("id") == entry_id:
                return entry
        return None
    
    def get_by_filename(self, filename: str) -> Optional[Dict]:
        """Get entry by filename"""
        for entry in self.history:
            if entry.get("filename") == filename:
                return entry
        return None
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get paginated history"""
        return self.history[offset:offset + limit]
    
    def get_favorites(self) -> List[Dict]:
        """Get all favorited entries"""
        return [e for e in self.history if e.get("favorite")]
    
    def search(self, query: str, limit: int = 50) -> List[Dict]:
        """Search history by prompt text"""
        query_lower = query.lower()
        results = []
        
        for entry in self.history:
            prompt = entry.get("prompt", "").lower()
            tags = " ".join(entry.get("tags", [])).lower()
            notes = entry.get("notes", "").lower()
            
            if query_lower in prompt or query_lower in tags or query_lower in notes:
                results.append(entry)
                if len(results) >= limit:
                    break
        
        return results
    
    def filter_by_style(self, style: str) -> List[Dict]:
        """Filter history by style"""
        return [e for e in self.history if e.get("style") == style]
    
    def filter_by_seed(self, seed: int) -> List[Dict]:
        """Filter history by seed"""
        return [e for e in self.history if e.get("seed") == seed]
    
    def toggle_favorite(self, entry_id: str) -> bool:
        """Toggle favorite status"""
        for entry in self.history:
            if entry.get("id") == entry_id:
                entry["favorite"] = not entry.get("favorite", False)
                self._save()
                return entry["favorite"]
        return False
    
    def add_tag(self, entry_id: str, tag: str) -> bool:
        """Add a tag to an entry"""
        for entry in self.history:
            if entry.get("id") == entry_id:
                if "tags" not in entry:
                    entry["tags"] = []
                if tag not in entry["tags"]:
                    entry["tags"].append(tag)
                    self._save()
                return True
        return False
    
    def remove_tag(self, entry_id: str, tag: str) -> bool:
        """Remove a tag from an entry"""
        for entry in self.history:
            if entry.get("id") == entry_id:
                if "tags" in entry and tag in entry["tags"]:
                    entry["tags"].remove(tag)
                    self._save()
                return True
        return False
    
    def update_notes(self, entry_id: str, notes: str) -> bool:
        """Update notes for an entry"""
        for entry in self.history:
            if entry.get("id") == entry_id:
                entry["notes"] = notes
                self._save()
                return True
        return False
    
    def delete(self, entry_id: str) -> bool:
        """Delete an entry from history"""
        for i, entry in enumerate(self.history):
            if entry.get("id") == entry_id:
                self.history.pop(i)
                self._save()
                return True
        return False
    
    def get_stats(self) -> Dict:
        """Get history statistics"""
        total = len(self.history)
        if total == 0:
            return {
                "total": 0,
                "favorites": 0,
                "styles": {},
                "avg_generation_time": 0
            }
        
        styles = {}
        total_time = 0
        favorites = 0
        
        for entry in self.history:
            style = entry.get("style", "unknown")
            styles[style] = styles.get(style, 0) + 1
            total_time += entry.get("generation_time", 0)
            if entry.get("favorite"):
                favorites += 1
        
        return {
            "total": total,
            "favorites": favorites,
            "styles": styles,
            "avg_generation_time": round(total_time / total, 2)
        }
    
    def export(self, filepath: str) -> bool:
        """Export history to a file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to export history: {e}")
            return False


# Global instance
generation_history = GenerationHistory()
