import discord
from discord.ext import commands, tasks
import os
import asyncio
from datetime import datetime
from pathlib import Path
import sys
import json
import threading
from dotenv import load_dotenv

# Calculate path to .env file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = BASE_DIR / '.env'

# Load .env file
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

sys.path.insert(0, str(BASE_DIR))
from src.utils.logger import create_logger

logger = create_logger("IRISDiscordBot")

# Paths for data
SENT_IMAGES_FILE = BASE_DIR / "static" / "data" / "img_send.json"
PROMPTS_LOG_FILE = BASE_DIR / "static" / "data" / "prompts_history.json"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Discord Rich Presence Setup
try:
    from pypresence import Presence
    DISCORD_RPC_AVAILABLE = True
except ImportError:
    DISCORD_RPC_AVAILABLE = False
    print("[INFO] pypresence not installed. Discord RPC disabled.")

# Configuration functions
def read_config_file(filename):
    """Read configuration from a file in static/config folder"""
    filepath = BASE_DIR / "static" / "config" / filename
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            if not content:
                logger.error(f"{filepath} is empty!")
                return None
            return content
    except FileNotFoundError:
        logger.error(f"{filepath} not found!")
        return None

def get_env_id(key, default=0):
    """Helper function to safely load IDs as integers from .env"""
    val = os.getenv(key)
    if not val:
        logger.warning(f"Variable {key} not found in .env. Using default: {default}")
        return default
    try:
        return int(val)
    except ValueError:
        logger.error(f"Invalid ID for {key} in .env (must be a number)!")
        return default

# Bot Configuration
logger.info("Loading Discord Bot Configuration...")
BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN') or read_config_file("bot_token.txt")
BOT_OWNER_ID = os.getenv('DISCORD_BOT_OWNER_ID') or read_config_file("bot_owner_id.txt")
BOT_ID = os.getenv('DISCORD_BOT_ID') or read_config_file("bot_id.txt")

CHANNEL_NEW_IMAGES = get_env_id('DISCORD_CHANNEL_NEW_IMAGES')
CHANNEL_VARIATIONS = get_env_id('DISCORD_CHANNEL_VARIATIONS')
CHANNEL_UPSCALED = get_env_id('DISCORD_CHANNEL_UPSCALED')

# Discord RPC ID
DISCORD_RPC_CLIENT_ID = read_config_file("rpc_client_id.txt") or "YOUR_DISCORD_APPLICATION_ID"

rpc = None
rpc_connected = False
generation_count = 0
current_status = "idle"

def init_discord_rpc():
    """Initialize Discord Rich Presence"""
    global rpc, rpc_connected
    if not DISCORD_RPC_AVAILABLE or DISCORD_RPC_CLIENT_ID == "YOUR_DISCORD_APPLICATION_ID":
        return False
    
    try:
        rpc = Presence(DISCORD_RPC_CLIENT_ID)
        rpc.connect()
        rpc_connected = True
        logger.success("Discord Rich Presence connected!")
        update_rpc_status("idle")
        return True
    except Exception as e:
        logger.warning(f"Discord RPC connection failed: {e}")
        rpc_connected = False
        return False

def update_rpc_status(status: str, details: str = None, progress: int = None):
    """Update status in Discord Rich Presence"""
    global rpc, rpc_connected, current_status
    if not rpc_connected or rpc is None:
        return

    current_status = status
    try:
        state_text = f"{generation_count} Images generated"
        if status == "idle":
            details_text = "Ready to create"
            small_image, small_text = "idle", "Ready"
        elif status == "generating":
            details_text = details or "Generating image..."
            if progress is not None: details_text = f"Generating... {progress}%"
            small_image, small_text = "generating", "Generating"
        elif status == "upscaling":
            details_text = details or "Upscaling..."
            small_image, small_text = "upscaling", "Upscaling"
        else:
            details_text = "Working..."
            small_image, small_text = "idle", status.capitalize()

        rpc.update(
            details=details_text,
            state=state_text,
            large_image="iris_logo",
            large_text="I.R.I.S. - Intelligent Rendering & Image Synthesis",
            small_image=small_image,
            small_text=small_text,
            start=int(datetime.now().timestamp())
        )
    except Exception as e:
        logger.warning(f"RPC Update failed: {e}")
        rpc_connected = False

def increment_generation_count():
    global generation_count
    generation_count += 1
    if rpc_connected:
        update_rpc_status(current_status)

def disconnect_rpc():
    global rpc, rpc_connected
    if rpc and rpc_connected:
        try:
            rpc.close()
            logger.info("Discord RPC disconnected")
        except: pass
    rpc_connected = False

# Image Tracking Functions
sent_images_dict = {}
currently_processing = set()

def load_sent_images():
    """Load sent images from JSON file"""
    global sent_images_dict
    if SENT_IMAGES_FILE.exists():
        try:
            with open(SENT_IMAGES_FILE, 'r', encoding='utf-8') as f:
                sent_images_dict = json.load(f)
            logger.info(f"{len(sent_images_dict)} previously sent images loaded")
        except Exception as e:
            logger.error(f"Error loading sent images: {e}")
            sent_images_dict = {}

def save_sent_image(filename, message_link):
    """Save sent image to JSON file"""
    sent_images_dict[filename] = {
        "message_link": message_link,
        "sent_at": datetime.now().isoformat()
    }
    try:
        SENT_IMAGES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SENT_IMAGES_FILE, 'w', encoding='utf-8') as f:
            json.dump(sent_images_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving sent image: {e}")

# Bot Setup
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

monitor_lock = asyncio.Lock()

@bot.event
async def on_ready():
    logger.discord_bot_ready(bot.user.name)
    logger.info("=" * 70)
    logger.info(f"Connected as: {bot.user.name}")
    
    channels = {
        "New": (bot.get_channel(CHANNEL_NEW_IMAGES), CHANNEL_NEW_IMAGES),
        "Variations": (bot.get_channel(CHANNEL_VARIATIONS), CHANNEL_VARIATIONS),
        "Upscaled": (bot.get_channel(CHANNEL_UPSCALED), CHANNEL_UPSCALED)
    }

    all_found = True
    for name, (channel, channel_id) in channels.items():
        if channel:
            logger.success(f"{name} Channel found: #{channel.name}")
        else:
            logger.error(f"{name} Channel ID {channel_id} invalid!")
            all_found = False

    if all_found:
        init_discord_rpc()
        load_sent_images()
        if not monitor_images.is_running():
            monitor_images.start()
            logger.success("Image monitoring started")
    else:
        logger.error("MONITORING NOT STARTED: Please check the .env file.")

@tasks.loop(seconds=3.0)
async def monitor_images():
    async with monitor_lock:
        try:
            if not OUTPUTS_DIR.exists(): return
            
            chan_new = bot.get_channel(CHANNEL_NEW_IMAGES)
            chan_var = bot.get_channel(CHANNEL_VARIATIONS)
            chan_up = bot.get_channel(CHANNEL_UPSCALED)
            
            if not all([chan_new, chan_var, chan_up]): return

            image_files = sorted(OUTPUTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime)
            
            for image_path in image_files:
                filename = image_path.name
                if filename in sent_images_dict or filename in currently_processing:
                    continue
                
                currently_processing.add(filename)
                
                await asyncio.sleep(12)
                
                if filename in sent_images_dict:
                    currently_processing.discard(filename)
                    continue
                
                is_up = filename.startswith(("up", "upscaled"))
                is_var = filename.startswith(("var", "variation"))
                target_channel = chan_up if is_up else (chan_var if is_var else chan_new)

                try:
                    if image_path.stat().st_size > 8 * 1024 * 1024:
                        logger.warning(f"Image too large for Discord: {filename}")
                        currently_processing.discard(filename)
                        continue

                    seed = "Unknown"
                    parts = filename.replace(".png", "").split("_")
                    if len(parts) >= 4 and parts[3].isdigit(): seed = parts[3]

                    embed = discord.Embed(
                        title="I.R.I.S. Rendering",
                        description=f"**Seed:** `{seed}`",
                        color=0x14b8a6 if is_up else 0x06b6d4,
                        timestamp=datetime.now()
                    )
                    embed.set_footer(text="I.R.I.S. - Intelligent Rendering & Image Synthesis")
                    
                    with open(image_path, 'rb') as f:
                        df = discord.File(f, filename=filename)
                        embed.set_image(url=f"attachment://{filename}")
                        msg = await target_channel.send(file=df, embed=embed)
                        
                        from src.utils.file_manager import FileManager
                        FileManager.log_sent_image(filename, msg.jump_url)
                        
                        save_sent_image(filename, msg.jump_url)
                        logger.success(f"Image sent: {filename}")
                        if not is_up: increment_generation_count()

                except Exception as e:
                    logger.error(f"Error sending {filename}: {e}")
                finally:
                    currently_processing.discard(filename)
        except Exception as e:
            logger.error(f"Monitor error: {e}")

@bot.command(name='iris')
async def iris_info(ctx):
    embed = discord.Embed(title="I.R.I.S. Info", color=0x06b6d4)
    embed.add_field(name="Generated (Session)", value=str(generation_count))
    embed.add_field(name="Status", value=current_status.capitalize())
    await ctx.send(embed=embed)

@bot.command(name='cleanup')
async def cleanup_channels(ctx):
    """Remove all images from Discord channels that are not in img_send.json"""
    if str(ctx.author.id) != str(BOT_OWNER_ID):
        await ctx.send("You don't have permission to use this command.")
        return
    
    await ctx.send("Starting channel cleanup... This may take a while.")
    
    try:
        # Load sent images from JSON
        load_sent_images()
        valid_filenames = set(sent_images_dict.keys())
        
        channels = [
            (bot.get_channel(CHANNEL_NEW_IMAGES), "New Images"),
            (bot.get_channel(CHANNEL_VARIATIONS), "Variations"),
            (bot.get_channel(CHANNEL_UPSCALED), "Upscaled")
        ]
        
        total_deleted = 0
        
        for channel, channel_name in channels:
            if not channel:
                continue
                
            deleted_count = 0
            async for message in channel.history(limit=None):
                # Check if message has attachments
                if not message.attachments:
                    continue
                
                # Check if message is from this bot
                if message.author.id != bot.user.id:
                    continue
                
                # Get filename from attachment
                for attachment in message.attachments:
                    filename = attachment.filename
                    
                    # If filename not in img_send.json, delete the message
                    if filename not in valid_filenames:
                        try:
                            await message.delete()
                            deleted_count += 1
                            total_deleted += 1
                            logger.info(f"Deleted: {filename} from {channel_name}")
                            await asyncio.sleep(1)  # Rate limiting
                        except Exception as e:
                            logger.error(f"Failed to delete {filename}: {e}")
            
            logger.info(f"Cleaned {deleted_count} messages from {channel_name}")
        
        await ctx.send(f"Cleanup complete! Deleted {total_deleted} messages total.")
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        await ctx.send(f"Cleanup failed: {e}")

@bot.command(name='help')
async def help_command(ctx):
    embed = discord.Embed(title="I.R.I.S. Help", color=0x06b6d4)
    embed.add_field(name="!iris", value="Show bot status", inline=False)
    embed.add_field(name="!cleanup", value="Remove images not in img_send.json (Owner only)", inline=False)
    embed.add_field(name="!help", value="Show this help message", inline=False)
    await ctx.send(embed=embed)

def main():
    if not BOT_TOKEN:
        logger.error("No Bot Token found!")
        return
    try:
        import signal
        
        def shutdown_handler(signum, frame):
            logger.info("Received shutdown signal, closing bot...")
            disconnect_rpc()
            bot.loop.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        
        bot.run(BOT_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
    finally:
        disconnect_rpc()

if __name__ == "__main__":
    main()
