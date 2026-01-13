import discord
from discord.ext import commands, tasks
import os
import asyncio
from datetime import datetime
from pathlib import Path
import sys
import json
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
BOT_STATUS_FILE = BASE_DIR / "static" / "data" / "bot_status.json"
OUTPUTS_DIR = BASE_DIR / "outputs"

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

# Status tracking
current_status = "idle"
generation_count = 0

async def update_bot_status(status: str, details: str = None):
    """Update the bot's Discord status/activity"""
    global current_status
    current_status = status
    
    if status == "idle":
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="for new images 👀"
        )
    elif status == "monitoring":
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name=f"outputs folder | {generation_count} sent"
        )
    elif status == "sending":
        activity = discord.Activity(
            type=discord.ActivityType.playing,
            name=f"Sending: {details}" if details else "Sending image..."
        )
    elif status == "generating":
        activity = discord.Activity(
            type=discord.ActivityType.playing,
            name=f"🎨 Generating..."
        )
    elif status == "restarting":
        activity = discord.Activity(
            type=discord.ActivityType.playing,
            name="🔄 Server restarting..."
        )
    else:
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="I.R.I.S. Server"
        )
    
    try:
        await bot.change_presence(activity=activity, status=discord.Status.online)
    except Exception as e:
        logger.error(f"Failed to update status: {e}")

@bot.event
async def on_ready():
    logger.discord_bot_ready(bot.user.name)
    logger.info("=" * 70)
    logger.info(f"Connected as: {bot.user.name}")
    
    # Set initial status
    await update_bot_status("idle")
    
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
        load_sent_images()
        if not monitor_images.is_running():
            monitor_images.start()
            logger.success("Image monitoring started")
        if not monitor_server_status.is_running():
            monitor_server_status.start()
            logger.success("Server status monitoring started")
        await update_bot_status("monitoring")
    else:
        logger.error("MONITORING NOT STARTED: Please check the .env file.")

@tasks.loop(seconds=2.0)
async def monitor_server_status():
    """Monitor server status file for generation updates"""
    try:
        if not BOT_STATUS_FILE.exists():
            return
        
        with open(BOT_STATUS_FILE, 'r') as f:
            status_data = json.load(f)
        
        server_status = status_data.get("status", "idle")
        details = status_data.get("details", "")
        
        global current_status
        if server_status == "generating" and current_status != "generating":
            await update_bot_status("generating", details)
        elif server_status == "idle" and current_status == "generating":
            await update_bot_status("monitoring")
        elif server_status == "restarting":
            await update_bot_status("restarting")
            
    except Exception as e:
        pass  # Ignore errors reading status file

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
                    
                    # Update status while sending
                    await update_bot_status("sending", filename)
                    
                    with open(image_path, 'rb') as f:
                        df = discord.File(f, filename=filename)
                        embed.set_image(url=f"attachment://{filename}")
                        msg = await target_channel.send(file=df, embed=embed)
                        
                        from src.utils.file_manager import FileManager
                        FileManager.log_sent_image(filename, msg.jump_url)
                        
                        save_sent_image(filename, msg.jump_url)
                        logger.success(f"Image sent: {filename}")
                        
                        # Update generation count and status
                        global generation_count
                        generation_count += 1
                        await update_bot_status("monitoring")

                except Exception as e:
                    logger.error(f"Error sending {filename}: {e}")
                finally:
                    currently_processing.discard(filename)
        except Exception as e:
            logger.error(f"Monitor error: {e}")

@bot.command(name='iris')
async def iris_info(ctx):
    embed = discord.Embed(title="I.R.I.S. Bot Status", color=0x06b6d4)
    embed.add_field(name="📊 Sent (Session)", value=str(generation_count), inline=True)
    embed.add_field(name="📁 Total Tracked", value=str(len(sent_images_dict)), inline=True)
    embed.add_field(name="🔄 Status", value=current_status.capitalize(), inline=True)
    embed.set_footer(text="I.R.I.S. - Intelligent Rendering & Image Synthesis")
    await ctx.send(embed=embed)

@bot.command(name='cleanup')
async def cleanup_channels(ctx):
    """Remove all images from Discord channels that are not in img_send.json"""
    # Permission check
    if str(ctx.author.id) != str(BOT_OWNER_ID):
        await ctx.send("❌ Du hast keine Berechtigung für diesen Command.")
        return
    
    # Load sent images
    load_sent_images()
    valid_filenames = set(sent_images_dict.keys())
    
    if not valid_filenames:
        await ctx.send("⚠️ Keine Bilder in img_send.json gefunden. Abbruch.")
        return
    
    # Confirmation message
    channels = [
        (bot.get_channel(CHANNEL_NEW_IMAGES), "New Images"),
        (bot.get_channel(CHANNEL_VARIATIONS), "Variations"),
        (bot.get_channel(CHANNEL_UPSCALED), "Upscaled")
    ]
    
    # Count messages to delete first
    embed = discord.Embed(
        title="🧹 Channel Cleanup",
        description="Scanne Channels nach zu löschenden Nachrichten...",
        color=0xfbbf24
    )
    status_msg = await ctx.send(embed=embed)
    
    try:
        # Collect all message links from img_send.json
        valid_message_links = set()
        for filename, data in sent_images_dict.items():
            if isinstance(data, dict) and "message_link" in data:
                valid_message_links.add(data["message_link"])
        
        if not valid_message_links:
            await ctx.send("⚠️ Keine Message-Links in img_send.json gefunden. Abbruch.")
            return
        
        to_delete = []
        
        for channel, channel_name in channels:
            if not channel:
                logger.warning(f"Channel {channel_name} nicht gefunden")
                continue
            
            # Check bot permissions
            permissions = channel.permissions_for(channel.guild.me)
            if not permissions.read_message_history or not permissions.manage_messages:
                await ctx.send(f"❌ Fehlende Berechtigungen in {channel_name}")
                continue
            
            async for message in channel.history(limit=None):
                # Only check messages with attachments
                if not message.attachments:
                    continue
                
                # Check if message link is in img_send.json
                if message.jump_url not in valid_message_links:
                    filename = message.attachments[0].filename if message.attachments else "unknown"
                    to_delete.append((message, filename, channel_name))
        
        if not to_delete:
            embed = discord.Embed(
                title="✅ Cleanup Abgeschlossen",
                description="Keine zu löschenden Nachrichten gefunden.",
                color=0x10b981
            )
            await status_msg.edit(embed=embed)
            return
        
        # Confirmation
        embed = discord.Embed(
            title="⚠️ Bestätigung erforderlich",
            description=f"**{len(to_delete)} Nachrichten** werden gelöscht.\n\nReagiere mit ✅ zum Bestätigen oder ❌ zum Abbrechen.",
            color=0xef4444
        )
        await status_msg.edit(embed=embed)
        await status_msg.add_reaction('✅')
        await status_msg.add_reaction('❌')
        
        def check(reaction, user):
            return user == ctx.author and str(reaction.emoji) in ['✅', '❌'] and reaction.message.id == status_msg.id
        
        try:
            reaction, user = await bot.wait_for('reaction_add', timeout=30.0, check=check)
            
            if str(reaction.emoji) == '❌':
                embed = discord.Embed(
                    title="❌ Cleanup Abgebrochen",
                    description="Keine Nachrichten wurden gelöscht.",
                    color=0x6b7280
                )
                await status_msg.edit(embed=embed)
                await status_msg.clear_reactions()
                return
        
        except asyncio.TimeoutError:
            embed = discord.Embed(
                title="⏱️ Timeout",
                description="Keine Reaktion erhalten. Cleanup abgebrochen.",
                color=0x6b7280
            )
            await status_msg.edit(embed=embed)
            await status_msg.clear_reactions()
            return
        
        # Delete messages
        await status_msg.clear_reactions()
        total_deleted = 0
        failed = 0
        
        for i, (message, filename, channel_name) in enumerate(to_delete, 1):
            try:
                # Update status every 5 messages
                if i % 5 == 0 or i == len(to_delete):
                    embed = discord.Embed(
                        title="🧹 Cleanup läuft...",
                        description=f"Fortschritt: {i}/{len(to_delete)} ({int(i/len(to_delete)*100)}%)\n"
                                  f"✅ Gelöscht: {total_deleted}\n"
                                  f"❌ Fehler: {failed}",
                        color=0x3b82f6
                    )
                    await status_msg.edit(embed=embed)
                
                await message.delete()
                total_deleted += 1
                logger.info(f"Gelöscht: {filename} aus {channel_name}")
                
                # Rate limiting: 1 deletion per second
                await asyncio.sleep(1.0)
                
            except discord.errors.NotFound:
                logger.warning(f"Nachricht bereits gelöscht: {filename}")
                failed += 1
            except discord.errors.Forbidden:
                logger.error(f"Keine Berechtigung zum Löschen: {filename}")
                failed += 1
            except Exception as e:
                logger.error(f"Fehler beim Löschen {filename}: {e}")
                failed += 1
        
        # Final report
        embed = discord.Embed(
            title="✅ Cleanup Abgeschlossen",
            description=f"**Ergebnis:**\n"
                       f"✅ Erfolgreich gelöscht: {total_deleted}\n"
                       f"❌ Fehlgeschlagen: {failed}\n"
                       f"📊 Gesamt verarbeitet: {len(to_delete)}",
            color=0x10b981
        )
        await status_msg.edit(embed=embed)
        
        logger.success(f"Cleanup abgeschlossen: {total_deleted} gelöscht, {failed} Fehler")
        
    except Exception as e:
        logger.error(f"Cleanup Fehler: {e}")
        embed = discord.Embed(
            title="❌ Cleanup Fehler",
            description=f"Ein Fehler ist aufgetreten: {str(e)}",
            color=0xef4444
        )
        await status_msg.edit(embed=embed)

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
            try:
                if bot.loop and bot.loop.is_running():
                    asyncio.run_coroutine_threadsafe(bot.close(), bot.loop)
                else:
                    sys.exit(0)
            except:
                sys.exit(0)
        
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        if os.name == 'nt':
            try:
                signal.signal(signal.SIGBREAK, shutdown_handler)
            except:
                pass
        
        bot.run(BOT_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except SystemExit:
        logger.info("Bot shutdown complete")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Bot process ending")

if __name__ == "__main__":
    main()
