"""
Dependency Injection for I.R.I.S.
Provides clean dependency management for FastAPI
"""
from typing import Optional, Generator
from functools import lru_cache

from src.core.config import Config


class AppState:
    """Application state container for dependency injection"""
    
    _instance: Optional["AppState"] = None
    
    def __init__(self):
        self._pipeline_service = None
        self._generation_queue = None
        self._nsfw_filter = None
        self._history_service = None
        self._rate_limiter = None
    
    @classmethod
    def get_instance(cls) -> "AppState":
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def pipeline_service(self):
        """Lazy-load pipeline service"""
        if self._pipeline_service is None:
            from src.api.services.pipeline import PipelineService
            self._pipeline_service = PipelineService()
        return self._pipeline_service
    
    @property
    def generation_queue(self):
        """Lazy-load generation queue"""
        if self._generation_queue is None:
            from src.api.services.queue import GenerationQueue
            self._generation_queue = GenerationQueue()
        return self._generation_queue
    
    @property
    def nsfw_filter(self):
        """Lazy-load NSFW filter"""
        if self._nsfw_filter is None:
            from src.api.services.nsfw_filter import NSFWFilterService
            self._nsfw_filter = NSFWFilterService()
        return self._nsfw_filter
    
    @property
    def history_service(self):
        """Lazy-load history service"""
        if self._history_service is None:
            from src.api.services.history import GenerationHistory
            self._history_service = GenerationHistory()
        return self._history_service
    
    def cleanup(self):
        """Cleanup all services"""
        if self._pipeline_service:
            self._pipeline_service.cleanup()
        self._instance = None


# Dependency functions for FastAPI
def get_app_state() -> AppState:
    """Get application state"""
    return AppState.get_instance()


def get_pipeline_service():
    """Dependency for pipeline service"""
    return get_app_state().pipeline_service


def get_generation_queue():
    """Dependency for generation queue"""
    return get_app_state().generation_queue


def get_nsfw_filter():
    """Dependency for NSFW filter"""
    return get_app_state().nsfw_filter


def get_history_service():
    """Dependency for history service"""
    return get_app_state().history_service


@lru_cache()
def get_config() -> Config:
    """Cached config instance"""
    return Config
