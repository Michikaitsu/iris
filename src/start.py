#!/usr/bin/env python3
"""
I.R.I.S. Universal Starter
==========================
Intelligent Rendering & Image Synthesis

Starts Web UI and optionally Discord Bot based on settings.json

Usage:
    python src/start.py           # Auto-start based on settings.json
    python src/start.py --no-bot  # Force start without Discord Bot

Press CTRL+C to exit the program gracefully
"""

import sys
import subprocess
import os
import signal
import time
import json
from pathlib import Path

current_processes = []
shutdown_in_progress = False

def signal_handler(sig, frame):
    """Handle CTRL+C gracefully"""
    global shutdown_in_progress
    
    if shutdown_in_progress:
        print("\n[IRIS] Force shutdown...")
        for process in current_processes:
            try:
                process.kill()
            except:
                pass
        sys.exit(1)
    
    shutdown_in_progress = True
    print("\n\n[IRIS] Shutting down all services...")
    print("[IRIS] Press CTRL+C again to force immediate shutdown")
    
    for process in current_processes:
        try:
            process.send_signal(signal.SIGINT)
        except:
            pass
    
    start_wait = time.time()
    all_stopped = False
    
    while time.time() - start_wait < 10:
        all_stopped = True
        for process in current_processes:
            if process.poll() is None:
                all_stopped = False
                break
        
        if all_stopped:
            break
        time.sleep(0.5)
    
    if not all_stopped:
        print("[IRIS] Forcing shutdown...")
        for process in current_processes:
            try:
                process.kill()
            except:
                pass
    
    print("[IRIS] All services stopped")
    sys.exit(0)

def print_banner():
    """Print I.R.I.S. startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║              I.R.I.S. v1.0.0                     ║
    ║   Intelligent Rendering & Image Synthesis        ║
    ║                                                  ║
    ║   Local AI Image Generation                      ║
    ║   Powered by Stable Diffusion                    ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    
    [TIP] Press CTRL+C to exit the program
    """
    print(banner)

def load_settings():
    """Load settings from settings.json"""
    project_root = Path(__file__).resolve().parents[1]
    settings_path = project_root / "settings.json"
    
    default_settings = {
        "discordEnabled": False,
        "dramEnabled": False,
        "vramThreshold": 6,
        "maxDram": 8,
        "nsfwStrength": 2
    }
    
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                return {**default_settings, **settings}
        except Exception as e:
            print(f"[WARN] Could not load settings.json: {e}")
    
    return default_settings

def start_web_server():
    """Start FastAPI Web UI Server"""
    print("\n[WEB] Starting Web UI Server...")
    print("      Access at: http://localhost:8000\n")

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    process = subprocess.Popen([
        sys.executable,
        "-m", "uvicorn",
        "src.api.server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ])

    current_processes.append(process)
    return process

def start_discord_bot():
    """Start Discord Bot with Rich Presence"""
    print("\n[BOT] Starting Discord Bot...")
    print("      Rich Presence will show generation status\n")
    
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    
    process = subprocess.Popen([sys.executable, "src/services/bot.py"])
    current_processes.append(process)
    return process

def main():
    """Main entry point"""
    print_banner()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check for --no-bot flag
    no_bot = "--no-bot" in sys.argv
    
    # Load settings
    settings = load_settings()
    discord_enabled = settings.get("discordEnabled", False) and not no_bot
    
    # Show startup info
    print(f"    [CONFIG] Discord Bot: {'Enabled' if discord_enabled else 'Disabled'}")
    print(f"    [CONFIG] DRAM Extension: {'Enabled' if settings.get('dramEnabled') else 'Disabled'}")
    print()
    
    try:
        # Always start web server
        web_process = start_web_server()
        
        # Start Discord bot if enabled
        bot_process = None
        if discord_enabled:
            time.sleep(2)  # Wait a bit for web server to initialize
            bot_process = start_discord_bot()
        
        # Wait for processes
        web_process.wait()
        if bot_process:
            bot_process.wait()
            
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
