import logging
import os
from datetime import datetime
from pathlib import Path
import sys

class AIGeneratorLogger:
    """
    Custom logger for AI Image Generator
    Creates timestamped log files in Logs folder
    """
    
    def __init__(self, name: str = "AIGenerator"):
        self.name = name
        self.logs_dir = Path("Logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create session-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"session_{timestamp}.log"
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler (detailed logs)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(
            self.log_file, 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler (user-friendly output) - Added encoding handling for Windows
        console_formatter = logging.Formatter(
            '%(message)s'
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        # Fix encoding issues on Windows
        if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
            console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Session start
        self.session_start_time = datetime.now()
        self.log_session_start()
    
    def log_session_start(self):
        """Log session start with system info"""
        separator = "=" * 70
        self.logger.info(separator)
        self.logger.info("AI Image Generator Session Started")
        self.logger.info(f"Date: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Log File: {self.log_file.name}")
        self.logger.info(separator)
    
    def log_session_end(self):
        """Log session end with statistics"""
        separator = "=" * 70
        session_end_time = datetime.now()
        duration = session_end_time - self.session_start_time
        
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        
        self.logger.info("")
        self.logger.info(separator)
        self.logger.info("AI Image Generator Session Ended")
        self.logger.info(f"End Time: {session_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Duration: {hours}h {minutes}m {seconds}s")
        self.logger.info(separator)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(f"[WARNING] {message}")
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(f"[ERROR] {message}")
    
    def success(self, message: str):
        """Log success message"""
        self.logger.info(f"[SUCCESS] {message}")
    
    def generation_start(self, prompt: str, settings: dict):
        """Log image generation start"""
        self.logger.info("")
        self.logger.info("─" * 70)
        self.logger.info("Generation Started")
        self.logger.debug(f"   Prompt: {prompt[:60]}...")
        self.logger.debug(f"   Resolution: {settings.get('width')}x{settings.get('height')}")
        self.logger.debug(f"   Steps: {settings.get('steps')}")
        self.logger.debug(f"   CFG Scale: {settings.get('cfg_scale')}")
        self.logger.debug(f"   Seed: {settings.get('seed', 'Random')}")
    
    def generation_progress(self, step: int, total_steps: int, eta: float):
        """Log generation progress"""
        progress = (step / total_steps) * 100
        eta_min = int(eta // 60)
        eta_sec = int(eta % 60)
        self.logger.debug(
            f"   Progress: Step {step}/{total_steps} ({progress:.1f}%) - "
            f"ETA: {eta_min}m {eta_sec}s"
        )
    
    def generation_complete(self, filename: str, generation_time: float, seed: int):
        """Log generation completion"""
        minutes = int(generation_time // 60)
        seconds = int(generation_time % 60)
        self.logger.info(f"[SUCCESS] Generation Complete: {filename}")
        self.logger.info(f"   Time: {minutes}m {seconds}s | Seed: {seed}")
        self.logger.info("─" * 70)
    
    def gpu_info(self, gpu_name: str, vram_total: float, vram_used: float, temp: float):
        """Log GPU information"""
        self.logger.info(f"GPU: {gpu_name}")
        self.logger.info(f"   VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB")
        if temp > 0:
            temp_status = "[HOT]" if temp > 80 else "[WARM]" if temp > 70 else "[COOL]"
            self.logger.info(f"   Temperature: {temp}C {temp_status}")
    
    def upscale_start(self, filename: str, scale: int):
        """Log upscale operation start"""
        self.logger.info(f"Upscaling: {filename} ({scale}x)")
    
    def upscale_complete(self, new_filename: str, new_size: tuple):
        """Log upscale completion"""
        self.logger.info(f"[SUCCESS] Upscaled: {new_filename} ({new_size[0]}x{new_size[1]})")
    
    def variation_start(self, filename: str, strength: float):
        """Log variation creation start"""
        self.logger.info(f"Creating Variation: {filename} (strength: {strength})")
    
    def variation_complete(self, new_filename: str, generation_time: float):
        """Log variation completion"""
        self.logger.info(f"[SUCCESS] Variation Complete: {new_filename} ({generation_time:.1f}s)")
    
    def websocket_connect(self):
        """Log WebSocket connection"""
        self.logger.debug("WebSocket Client Connected")
    
    def websocket_disconnect(self):
        """Log WebSocket disconnection"""
        self.logger.debug("WebSocket Client Disconnected")
    
    def model_load_start(self, model_name: str):
        """Log model loading start"""
        self.logger.info(f"Loading Model: {model_name}")
        self.logger.info("This may take 5-10 minutes on first run...")
    
    def model_load_complete(self, model_name: str, device: str):
        """Log model loading completion"""
        self.logger.info(f"[SUCCESS] Model Loaded: {model_name} on {device}")
    
    def nsfw_blocked(self, reason: str):
        """Log NSFW content block"""
        self.logger.warning(f"Content Blocked: {reason}")
    
    def discord_bot_start(self):
        """Log Discord bot start"""
        self.logger.info("Discord Bot Starting...")
    
    def discord_bot_ready(self, bot_name: str):
        """Log Discord bot ready"""
        self.logger.info(f"[SUCCESS] Discord Bot Ready: {bot_name}")
    
    def discord_image_sent(self, filename: str, channel: str):
        """Log image sent to Discord"""
        self.logger.debug(f"Sent to Discord: {filename} -> #{channel}")
    
    def cleanup(self):
        """Clean up and close logger"""
        self.log_session_end()
        # Close all handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


def create_logger(name: str = "AIGenerator") -> AIGeneratorLogger:
    """
    Factory function to create a logger instance
    Usage: logger = create_logger()
    """
    return AIGeneratorLogger(name)


# Example usage and testing
if __name__ == "__main__":
    # Create logger
    logger = create_logger()
    
    # Test various log types
    logger.info("Testing logger functionality")
    logger.debug("This is a debug message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    logger.success("This is a success message")
    
    # Test generation logs
    logger.generation_start(
        "anime girl with blue hair", 
        {"width": 512, "height": 768, "steps": 30, "cfg_scale": 10, "seed": 12345}
    )
    logger.generation_progress(15, 30, 45.5)
    logger.generation_complete("test_image.png", 90.5, 12345)
    
    # Test GPU info
    logger.gpu_info("NVIDIA GTX 1650", 4.0, 2.5, 65)
    
    # End session
    logger.cleanup()
