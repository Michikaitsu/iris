"""
Tests for Custom Exceptions
"""
import pytest
from src.core.exceptions import (
    IRISException,
    ModelLoadError,
    VRAMExhaustedError,
    NSFWContentError,
    GenerationError,
    QueueFullError,
    InvalidParameterError,
    ModelNotLoadedError
)


class TestIRISException:
    """Test base exception class"""
    
    def test_basic_exception(self):
        exc = IRISException("Test error", "test_code")
        assert exc.message == "Test error"
        assert exc.code == "test_code"
        assert str(exc) == "Test error"
    
    def test_to_dict(self):
        exc = IRISException("Test error", "test_code")
        result = exc.to_dict()
        assert result["error"] == "test_code"
        assert result["message"] == "Test error"


class TestModelLoadError:
    """Test model loading errors"""
    
    def test_model_load_error(self):
        exc = ModelLoadError("anime_kawai", "File not found")
        assert "anime_kawai" in exc.message
        assert "File not found" in exc.message
        assert exc.code == "model_load_error"
        assert exc.model_name == "anime_kawai"


class TestVRAMExhaustedError:
    """Test VRAM exhaustion errors"""
    
    def test_basic_vram_error(self):
        exc = VRAMExhaustedError()
        assert exc.code == "vram_exhausted"
        assert "memory" in exc.message.lower()
    
    def test_vram_error_with_details(self):
        exc = VRAMExhaustedError(required_gb=8.0, available_gb=4.0)
        assert "8.0" in exc.message
        assert "4.0" in exc.message
        assert exc.required_gb == 8.0
        assert exc.available_gb == 4.0


class TestNSFWContentError:
    """Test NSFW content errors"""
    
    def test_nsfw_error(self):
        exc = NSFWContentError(category="sexual")
        assert exc.code == "nsfw_blocked"
        assert exc.category == "sexual"


class TestGenerationError:
    """Test generation errors"""
    
    def test_generation_error(self):
        exc = GenerationError("Pipeline failed", seed=12345)
        assert "Pipeline failed" in exc.message
        assert exc.seed == 12345
        assert exc.code == "generation_error"


class TestQueueFullError:
    """Test queue full errors"""
    
    def test_queue_full_error(self):
        exc = QueueFullError(max_size=50)
        assert "50" in exc.message
        assert exc.max_size == 50
        assert exc.code == "queue_full"


class TestInvalidParameterError:
    """Test invalid parameter errors"""
    
    def test_invalid_param_error(self):
        exc = InvalidParameterError("width", 5000, "Must be <= 4096")
        assert "width" in exc.message
        assert "5000" in exc.message
        assert exc.param == "width"
        assert exc.value == 5000


class TestModelNotLoadedError:
    """Test model not loaded errors"""
    
    def test_model_not_loaded(self):
        exc = ModelNotLoadedError()
        assert exc.code == "model_not_loaded"
        assert "model" in exc.message.lower()
