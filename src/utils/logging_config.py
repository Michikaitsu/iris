"""
Improved Logging Configuration for I.R.I.S.
Features: Log rotation, structured logging, performance tracking
"""
import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import time


@dataclass
class LogConfig:
    """Logging configuration"""
    log_dir: Path = field(default_factory=lambda: Path("Logs"))
    max_file_size_mb: int = 10
    backup_count: int = 5
    console_level: int = logging.INFO
    file_level: int = logging.DEBUG
    enable_json_logs: bool = False
    enable_performance_tracking: bool = True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console output formatter"""
    
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
        "SUCCESS": "\033[92m",    # Bright Green
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color based on level
        color = self.COLORS.get(record.levelname, "")
        
        # Custom format for different message types
        if hasattr(record, "msg_type"):
            if record.msg_type == "generation":
                prefix = "🎨"
            elif record.msg_type == "model":
                prefix = "🧠"
            elif record.msg_type == "gpu":
                prefix = "💻"
            elif record.msg_type == "websocket":
                prefix = "🔌"
            else:
                prefix = "•"
        else:
            prefix = "•"
        
        formatted = f"{color}{prefix} {record.getMessage()}{self.RESET}"
        return formatted


class PerformanceTracker:
    """Track performance metrics for operations"""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
    
    def record(self, operation: str, duration: float, metadata: Dict = None):
        """Record a performance metric"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append({
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        
        # Keep only last 100 entries per operation
        if len(self.metrics[operation]) > 100:
            self.metrics[operation] = self.metrics[operation][-100:]
    
    def get_stats(self, operation: str) -> Dict:
        """Get statistics for an operation"""
        if operation not in self.metrics or not self.metrics[operation]:
            return {"count": 0}
        
        durations = [m["duration"] for m in self.metrics[operation]]
        return {
            "count": len(durations),
            "avg_ms": sum(durations) / len(durations) * 1000,
            "min_ms": min(durations) * 1000,
            "max_ms": max(durations) * 1000,
            "last_ms": durations[-1] * 1000
        }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all operations"""
        return {op: self.get_stats(op) for op in self.metrics}


class IRISLogger:
    """
    Enhanced logger for I.R.I.S. with rotation and structured logging
    """
    
    _instances: Dict[str, "IRISLogger"] = {}
    _performance_tracker: Optional[PerformanceTracker] = None
    
    def __init__(self, name: str, config: Optional[LogConfig] = None):
        self.name = name
        self.config = config or LogConfig()
        self.config.log_dir.mkdir(exist_ok=True)
        
        # Initialize performance tracker (shared across all loggers)
        if IRISLogger._performance_tracker is None:
            IRISLogger._performance_tracker = PerformanceTracker()
        self.perf = IRISLogger._performance_tracker
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # Add custom SUCCESS level
        if not hasattr(logging, "SUCCESS"):
            logging.SUCCESS = 25
            logging.addLevelName(logging.SUCCESS, "SUCCESS")
        
        self._setup_handlers()
        self.session_start = datetime.now()
    
    def _setup_handlers(self):
        """Setup file and console handlers with rotation"""
        
        # Rotating file handler (main log)
        main_log_path = self.config.log_dir / "iris.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_path,
            maxBytes=self.config.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(self.config.file_level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(file_handler)
        
        # JSON log file (if enabled)
        if self.config.enable_json_logs:
            json_log_path = self.config.log_dir / "iris.json.log"
            json_handler = logging.handlers.RotatingFileHandler(
                json_log_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count,
                encoding="utf-8"
            )
            json_handler.setLevel(logging.DEBUG)
            json_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(json_handler)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.config.console_level)
        
        # Use colored formatter if terminal supports it
        if sys.stdout.isatty():
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        
        # Fix encoding on Windows
        if sys.platform == "win32" and hasattr(sys.stdout, "encoding"):
            if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
                try:
                    console_handler.stream = open(
                        sys.stdout.fileno(), 
                        mode="w", 
                        encoding="utf-8", 
                        buffering=1
                    )
                except:
                    pass
        
        self.logger.addHandler(console_handler)
    
    @classmethod
    def get_logger(cls, name: str, config: Optional[LogConfig] = None) -> "IRISLogger":
        """Get or create a logger instance (singleton per name)"""
        if name not in cls._instances:
            cls._instances[name] = cls(name, config)
        return cls._instances[name]
    
    def _log(self, level: int, message: str, msg_type: str = None, extra_data: Dict = None):
        """Internal logging method"""
        record = self.logger.makeRecord(
            self.name, level, "", 0, message, (), None
        )
        if msg_type:
            record.msg_type = msg_type
        if extra_data:
            record.extra_data = extra_data
        self.logger.handle(record)
    
    # Standard logging methods
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, f"⚠️  {message}", **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, f"❌ {message}", **kwargs)
    
    def success(self, message: str, **kwargs):
        self._log(logging.SUCCESS, f"✅ {message}", **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, f"🚨 {message}", **kwargs)
    
    # Specialized logging methods
    def generation_start(self, prompt: str, settings: Dict):
        """Log generation start"""
        self._log(
            logging.INFO,
            f"Generation started: {prompt[:50]}...",
            msg_type="generation",
            extra_data={"prompt": prompt, "settings": settings}
        )
    
    def generation_complete(self, filename: str, duration: float, seed: int):
        """Log generation completion"""
        self.perf.record("generation", duration, {"filename": filename, "seed": seed})
        self._log(
            logging.SUCCESS,
            f"Generated: {filename} ({duration:.1f}s, seed: {seed})",
            msg_type="generation"
        )
    
    def model_loading(self, model_name: str):
        """Log model loading start"""
        self._log(logging.INFO, f"Loading model: {model_name}...", msg_type="model")
    
    def model_loaded(self, model_name: str, device: str, duration: float = 0):
        """Log model loaded"""
        if duration:
            self.perf.record("model_load", duration, {"model": model_name})
        self._log(
            logging.SUCCESS,
            f"Model loaded: {model_name} on {device}",
            msg_type="model"
        )
    
    def gpu_status(self, name: str, vram_used: float, vram_total: float, temp: float = 0):
        """Log GPU status"""
        temp_str = f", {temp}°C" if temp else ""
        self._log(
            logging.INFO,
            f"GPU: {name} | VRAM: {vram_used:.1f}/{vram_total:.1f}GB{temp_str}",
            msg_type="gpu"
        )
    
    def websocket_event(self, event: str, client_count: int = 0):
        """Log WebSocket events"""
        self._log(
            logging.DEBUG,
            f"WebSocket: {event} (clients: {client_count})",
            msg_type="websocket"
        )
    
    @contextmanager
    def timed_operation(self, operation_name: str):
        """Context manager for timing operations"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            if self.config.enable_performance_tracking:
                self.perf.record(operation_name, duration)
            self.debug(f"{operation_name} completed in {duration*1000:.1f}ms")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return self.perf.get_all_stats()


def create_logger(name: str = "IRIS") -> IRISLogger:
    """Factory function to create a logger instance"""
    return IRISLogger.get_logger(name)
