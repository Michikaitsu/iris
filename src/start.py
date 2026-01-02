#!/usr/bin/env python3
"""
I.R.I.S. Universal Starter
==========================
Intelligent Rendering & Image Synthesis

Start Web UI, Discord Bot, or both

Usage:
    python src/start.py web    # Start Web UI only
    python src/start.py bot    # Start Discord Bot only
    python src/start.py all    # Start both services

Press CTRL+C to exit the program gracefully
"""

import sys
import subprocess
import os
import signal
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

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
            if process.poll() is None:  # Still running
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

def start_web_server():
    """Start FastAPI Web UI Server"""
    print("\n[WEB] Starting Web UI Server...")
    print("      Access at: http://localhost:8000\n")

    # 🔒 WICHTIG: Projekt-Root korrekt setzen
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    process = subprocess.Popen([
        sys.executable,
        "-m", "uvicorn",
        "src.api.server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        # ❗ KEIN --reload
    ])

    current_processes.append(process)
    return process

def start_discord_bot():
    """Start Discord Bot with Rich Presence"""
    print("\n[BOT] Starting Discord Bot...")
    print("      Rich Presence will show generation status\n")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    process = subprocess.Popen([sys.executable, "src/services/bot.py"])
    current_processes.append(process)
    return process

def start_all():
    """Start both Web Server and Discord Bot in parallel"""
    import threading
    
    print("\n[IRIS] Starting ALL services...\n")
    
    web_thread = threading.Thread(target=start_web_server, daemon=False)
    web_thread.start()
    
    bot_process = start_discord_bot()
    
    try:
        web_thread.join()
        bot_process.wait()
    except KeyboardInterrupt:
        pass

def main():
    """Main entry point"""
    print_banner()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    if len(sys.argv) < 2:
        print("[ERROR] Missing argument\n")
        print("Usage:")
        print("  python src/start.py web    # Start Web UI only")
        print("  python src/start.py bot    # Start Discord Bot only")
        print("  python src/start.py all    # Start both services")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    try:
        if mode == "web":
            process = start_web_server()
            process.wait()
        elif mode == "bot":
            process = start_discord_bot()
            process.wait()
        elif mode == "all":
            start_all()
        else:
            print(f"[ERROR] Unknown mode '{mode}'")
            print("Valid modes: web, bot, all")
            sys.exit(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
