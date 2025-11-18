#!/usr/bin/env python3
"""
L.O.O.M. Universal Starter
==========================
Start Web UI, Discord Bot, or both

Usage:
    python src/start.py web    # Start Web UI only
    python src/start.py bot    # Start Discord Bot only
    python src/start.py all    # Start both services
"""

import sys
import subprocess
import os
from pathlib import Path

def print_banner():
    """Print L.O.O.M. startup banner"""
    banner = """
    ╔══════════════════════════════════════╗
    ║                                      ║
    ║           L.O.O.M. v2.0             ║
    ║   Local Operator of Open Minds      ║
    ║                                      ║
    ╚══════════════════════════════════════╝
    """
    print(banner)

def start_web_server():
    """Start FastAPI Web UI Server"""
    print("\n🌐 Starting Web UI Server...")
    print("   Access at: http://localhost:8000\n")
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Start web server
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "src.backend.web_server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

def start_discord_bot():
    """Start Discord Bot"""
    print("\n🤖 Starting Discord Bot...")
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Start discord bot
    subprocess.run([sys.executable, "src/backend/discord_bot.py"])

def start_all():
    """Start both Web Server and Discord Bot in parallel"""
    import threading
    
    print("\n🚀 Starting ALL services...\n")
    
    # Start web server in thread
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    
    # Start discord bot in main thread
    try:
        start_discord_bot()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down all services...")
        sys.exit(0)

def main():
    """Main entry point"""
    print_banner()
    
    if len(sys.argv) < 2:
        print("❌ Error: Missing argument\n")
        print("Usage:")
        print("  python src/start.py web    # Start Web UI only")
        print("  python src/start.py bot    # Start Discord Bot only")
        print("  python src/start.py all    # Start both services")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "web":
        start_web_server()
    elif mode == "bot":
        start_discord_bot()
    elif mode == "all":
        start_all()
    else:
        print(f"❌ Error: Unknown mode '{mode}'")
        print("Valid modes: web, bot, all")
        sys.exit(1)

if __name__ == "__main__":
    main()
