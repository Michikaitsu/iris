"""
Generation Queue Service - Manages prompt queue for batch processing
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger import create_logger

logger = create_logger("QueueService")


class QueueItemStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueItem:
    """Represents a single item in the generation queue"""
    id: str
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 768
    steps: int = 35
    cfg_scale: float = 10.0
    seed: int = -1
    style: str = "anime_kawai"
    status: QueueItemStatus = QueueItemStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_filename: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
            "style": self.style,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result_filename": self.result_filename,
            "error_message": self.error_message,
            "progress": self.progress
        }


class GenerationQueue:
    """Manages a queue of generation requests"""
    
    def __init__(self, max_size: int = 50):
        self.queue: List[QueueItem] = []
        self.max_size = max_size
        self.is_processing = False
        self._process_callback: Optional[Callable] = None
        self._progress_callback: Optional[Callable] = None
    
    def set_process_callback(self, callback: Callable):
        """Set the callback function for processing items"""
        self._process_callback = callback
    
    def set_progress_callback(self, callback: Callable):
        """Set the callback function for progress updates"""
        self._progress_callback = callback
    
    def add(self, 
            prompt: str,
            negative_prompt: str = "",
            width: int = 512,
            height: int = 768,
            steps: int = 35,
            cfg_scale: float = 10.0,
            seed: int = -1,
            style: str = "anime_kawai") -> Optional[QueueItem]:
        """Add a new item to the queue"""
        if len(self.queue) >= self.max_size:
            logger.warning(f"Queue is full ({self.max_size} items)")
            return None
        
        item = QueueItem(
            id=str(uuid.uuid4())[:8],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            style=style
        )
        
        self.queue.append(item)
        logger.info(f"Added to queue: {item.id} - {prompt[:50]}...")
        
        return item
    
    def remove(self, item_id: str) -> bool:
        """Remove an item from the queue"""
        for i, item in enumerate(self.queue):
            if item.id == item_id:
                if item.status == QueueItemStatus.PROCESSING:
                    logger.warning(f"Cannot remove item {item_id} - currently processing")
                    return False
                self.queue.pop(i)
                logger.info(f"Removed from queue: {item_id}")
                return True
        return False
    
    def cancel(self, item_id: str) -> bool:
        """Cancel a pending item"""
        for item in self.queue:
            if item.id == item_id and item.status == QueueItemStatus.PENDING:
                item.status = QueueItemStatus.CANCELLED
                logger.info(f"Cancelled: {item_id}")
                return True
        return False
    
    def get_item(self, item_id: str) -> Optional[QueueItem]:
        """Get a specific item by ID"""
        for item in self.queue:
            if item.id == item_id:
                return item
        return None
    
    def get_pending(self) -> List[QueueItem]:
        """Get all pending items"""
        return [item for item in self.queue if item.status == QueueItemStatus.PENDING]
    
    def get_all(self) -> List[Dict]:
        """Get all items as dictionaries"""
        return [item.to_dict() for item in self.queue]
    
    def clear_completed(self):
        """Remove all completed/failed/cancelled items"""
        self.queue = [
            item for item in self.queue 
            if item.status in [QueueItemStatus.PENDING, QueueItemStatus.PROCESSING]
        ]
    
    def clear_all(self):
        """Clear entire queue (except processing items)"""
        self.queue = [
            item for item in self.queue 
            if item.status == QueueItemStatus.PROCESSING
        ]
    
    def move_up(self, item_id: str) -> bool:
        """Move an item up in the queue"""
        for i, item in enumerate(self.queue):
            if item.id == item_id and i > 0:
                if item.status == QueueItemStatus.PENDING:
                    self.queue[i], self.queue[i-1] = self.queue[i-1], self.queue[i]
                    return True
        return False
    
    def move_down(self, item_id: str) -> bool:
        """Move an item down in the queue"""
        for i, item in enumerate(self.queue):
            if item.id == item_id and i < len(self.queue) - 1:
                if item.status == QueueItemStatus.PENDING:
                    self.queue[i], self.queue[i+1] = self.queue[i+1], self.queue[i]
                    return True
        return False
    
    async def process_next(self) -> Optional[QueueItem]:
        """Process the next pending item"""
        if self.is_processing:
            return None
        
        pending = self.get_pending()
        if not pending:
            return None
        
        item = pending[0]
        item.status = QueueItemStatus.PROCESSING
        item.started_at = datetime.now()
        self.is_processing = True
        
        try:
            if self._process_callback:
                result = await self._process_callback(item)
                item.result_filename = result.get("filename")
                item.status = QueueItemStatus.COMPLETED
            else:
                item.status = QueueItemStatus.FAILED
                item.error_message = "No process callback configured"
        except Exception as e:
            item.status = QueueItemStatus.FAILED
            item.error_message = str(e)
            logger.error(f"Queue item {item.id} failed: {e}")
        finally:
            item.completed_at = datetime.now()
            self.is_processing = False
        
        return item
    
    async def process_all(self):
        """Process all pending items in sequence"""
        while self.get_pending():
            await self.process_next()
            await asyncio.sleep(0.1)  # Small delay between items
    
    def get_stats(self) -> Dict:
        """Get queue statistics"""
        return {
            "total": len(self.queue),
            "pending": len([i for i in self.queue if i.status == QueueItemStatus.PENDING]),
            "processing": len([i for i in self.queue if i.status == QueueItemStatus.PROCESSING]),
            "completed": len([i for i in self.queue if i.status == QueueItemStatus.COMPLETED]),
            "failed": len([i for i in self.queue if i.status == QueueItemStatus.FAILED]),
            "cancelled": len([i for i in self.queue if i.status == QueueItemStatus.CANCELLED]),
            "max_size": self.max_size
        }


# Global instance
generation_queue = GenerationQueue()
