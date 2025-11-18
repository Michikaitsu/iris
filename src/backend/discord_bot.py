import discord
from discord.ext import commands, tasks
import os
import asyncio
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.backend.logger import create_logger

logger = create_logger("DiscordBot")

SENT_IMAGES_FILE = Path("static") / "data" / "sent_images.txt"
PROMPTS_LOG_FILE = Path("static") / "data" / "prompts_history.txt"  # Fixed path to match server.py logging location

# Read configuration from files
def read_config_file(filename):
    """Read configuration from a file in the static/config folder"""
    filepath = Path("static") / "config" / filename
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            if not content:
                logger.error(f"{filepath} is empty!")
                return None
            return content
    except FileNotFoundError:
        logger.error(f"{filepath} not found!")
        logger.info(f"Please create the file with the required information.")
        return None

def read_channel_ids():
    """Read Discord channel IDs from config file"""
    filepath = Path("static") / "config" / "channel_ids.txt"
    channel_ids = {
        'new': None,
        'variations': None,
        'upscaled': None
    }
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key in channel_ids:
                        try:
                            channel_ids[key] = int(value)
                        except ValueError:
                            logger.error(f"Invalid channel ID for {key}: {value}")
        
        # Validate all channels are set
        if all(v is None for v in channel_ids.values()):
            logger.error(f"No valid channel IDs found in {filepath}")
            return None
        
        return channel_ids
    except FileNotFoundError:
        logger.error(f"{filepath} not found!")
        logger.info("Please create the file with channel IDs:")
        logger.info("   new=CHANNEL_ID")
        logger.info("   variations=CHANNEL_ID")
        logger.info("   upscaled=CHANNEL_ID")
        return None

# Load bot configuration
logger.info("Loading Discord bot configuration...")
BOT_TOKEN = read_config_file("bot_token.txt")
BOT_OWNER_ID = read_config_file("bot_owner_id.txt")
BOT_ID = read_config_file("bot_id.txt")

CHANNEL_IDS = read_channel_ids()

if CHANNEL_IDS:
    CHANNEL_NEW_IMAGES = CHANNEL_IDS['new']
    CHANNEL_VARIATIONS = CHANNEL_IDS['variations']
    CHANNEL_UPSCALED = CHANNEL_IDS['upscaled']
else:
    CHANNEL_NEW_IMAGES = None
    CHANNEL_VARIATIONS = None
    CHANNEL_UPSCALED = None

sent_images_dict = {}  # Format: {"filename.png": "message_link"}

def load_sent_images():
    """Load sent images tracking from file"""
    global sent_images_dict
    if SENT_IMAGES_FILE.exists():
        try:
            with open(SENT_IMAGES_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '|' in line:
                        filename, message_link = line.split('|', 1)
                        sent_images_dict[filename] = message_link
            logger.info(f"Loaded {len(sent_images_dict)} previously sent images")
        except Exception as e:
            logger.error(f"Error loading sent images: {e}")
    else:
        logger.info("No previous sent images found, starting fresh")

def save_sent_image(filename, message_link):
    """Save a sent image to the tracking file"""
    sent_images_dict[filename] = message_link
    try:
        with open(SENT_IMAGES_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{filename}|{message_link}\n")
    except Exception as e:
        logger.error(f"Error saving sent image: {e}")

def log_prompt_to_file(filename, seed):
    """Log prompt and seed to history file for future variations"""
    try:
        Path("static/data").mkdir(parents=True, exist_ok=True)
        with open(PROMPTS_LOG_FILE, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | {filename} | Seed: {seed}\n")
        logger.debug(f"Logged: {filename}")
    except Exception as e:
        logger.error(f"Error logging prompt: {e}")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

@bot.event
async def on_ready():
    """Called when the bot is ready"""
    logger.discord_bot_ready(bot.user.name)
    logger.info("=" * 70)
    logger.info(f"Bot User: {bot.user.name} (ID: {bot.user.id})")
    logger.info(f"Bot Owner ID: {BOT_OWNER_ID}")
    logger.info(f"Connected to {len(bot.guilds)} guild(s)")
    
    for guild in bot.guilds:
        logger.info(f"  - {guild.name} (ID: {guild.id})")
    
    logger.info("=" * 70)
    
    logger.info("Validating Discord channels...")
    
    channel_new = bot.get_channel(CHANNEL_NEW_IMAGES)
    channel_var = bot.get_channel(CHANNEL_VARIATIONS)
    channel_upscaled = bot.get_channel(CHANNEL_UPSCALED)
    
    if channel_new:
        logger.success(f"Channel 'new': #{channel_new.name} ({channel_new.guild.name}) [ID: {CHANNEL_NEW_IMAGES}]")
    else:
        logger.error(f"Cannot access channel with ID {CHANNEL_NEW_IMAGES} (new images)")
    
    if channel_var:
        logger.success(f"Channel 'variations': #{channel_var.name} ({channel_var.guild.name}) [ID: {CHANNEL_VARIATIONS}]")
    else:
        logger.error(f"Cannot access channel with ID {CHANNEL_VARIATIONS} (variations)")
    
    if channel_upscaled:
        logger.success(f"Channel 'upscaled': #{channel_upscaled.name} ({channel_upscaled.guild.name}) [ID: {CHANNEL_UPSCALED}]")
    else:
        logger.error(f"Cannot access channel with ID {CHANNEL_UPSCALED} (upscaled)")
    
    if channel_new and channel_var and channel_upscaled:
        logger.success("=" * 70)
        logger.success("🤖 Discord bot is ONLINE and ready!")
        logger.success("=" * 70)
        
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            outputs_dir.mkdir(exist_ok=True)
        
        load_sent_images()
        
        if not monitor_images.is_running():
            monitor_images.start()
            logger.success("Image monitoring started")
    else:
        logger.error("=" * 70)
        logger.error("Missing one or more Discord channels!")
        logger.error("Please check your channel IDs in static/config/channel_ids.txt")
        logger.error("=" * 70)

@bot.event
async def on_error(event, *args, **kwargs):
    """Handle bot errors"""
    import traceback
    logger.error(f"Error in {event}: {traceback.format_exc()}")

@bot.event
async def on_disconnect():
    """Handle disconnection"""
    logger.warning("Bot disconnected from Discord")

@bot.event
async def on_resumed():
    """Handle reconnection"""
    logger.success("Bot reconnected to Discord")

@tasks.loop(seconds=5)
async def monitor_images():
    """Monitor the outputs folder for new images"""
    try:
        outputs_dir = Path("outputs")
        
        if not outputs_dir.exists():
            return
        
        channel_new = bot.get_channel(CHANNEL_NEW_IMAGES)
        channel_var = bot.get_channel(CHANNEL_VARIATIONS)
        channel_upscaled = bot.get_channel(CHANNEL_UPSCALED)
        
        if not all([channel_new, channel_var, channel_upscaled]):
            return
        
        image_files = sorted(outputs_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        
        for image_path in image_files:
            filename = image_path.name
            
            if filename in sent_images_dict:
                continue
            
            is_upscaled = filename.startswith("upscaled")
            is_variation = filename.startswith("variation")
            
            target_channel = channel_upscaled if is_upscaled else (channel_var if is_variation else channel_new)
            
            try:
                seed_info = ""
                if "seed" in filename:
                    try:
                        seed = int(filename.split("seed")[1].split("_")[0])
                        seed_info = f"\n🎲 Seed: `{seed}`"
                        log_prompt_to_file(filename, seed)
                    except:
                        pass
                
                creation_time = datetime.fromtimestamp(image_path.stat().st_mtime)
                time_str = creation_time.strftime("%Y-%m-%d %H:%M:%S")
                
                msg_type = "📐 Upscaled" if is_upscaled else ("🔀 Variation" if is_variation else "🎨 New Image")
                
                message = await target_channel.send(
                    f"{msg_type}\n📅 {time_str}{seed_info}",
                    file=discord.File(image_path)
                )
                
                save_sent_image(filename, message.jump_url)
                logger.success(f"Sent to Discord: {filename}")
                
            except discord.Forbidden:
                logger.error(f"Missing permissions for {target_channel.name}")
            except Exception as e:
                logger.error(f"Failed to send {filename}: {e}")
            
            await asyncio.sleep(0.5)
    
    except Exception as e:
        logger.error(f"Monitor error: {e}")

@monitor_images.before_loop
async def before_monitor():
    """Wait until the bot is ready before starting the monitoring loop"""
    await bot.wait_until_ready()
    logger.debug("Monitor task ready to start")

@bot.command(name="status")
async def status_command(ctx):
    """Check bot status and image count"""
    if str(ctx.author.id) != BOT_OWNER_ID:
        await ctx.send("❌ Only the bot owner can use this command.")
        return
    
    outputs_dir = Path("outputs")
    total_images = len(list(outputs_dir.glob("*.png"))) if outputs_dir.exists() else 0
    
    embed = discord.Embed(
        title="🤖 Bot Status",
        color=discord.Color.green(),
        timestamp=datetime.now()
    )
    embed.add_field(name="Images Sent", value=str(len(sent_images_dict)), inline=True)
    embed.add_field(name="Total Images", value=str(total_images), inline=True)
    embed.add_field(name="Servers", value=str(len(bot.guilds)), inline=True)
    embed.add_field(name="Ping", value=f"{round(bot.latency * 1000)}ms", inline=True)
    
    await ctx.send(embed=embed)
    logger.info(f"Status command executed by {ctx.author.name}")

@bot.command(name="resendall")
async def resend_all_command(ctx):
    """Clear channels and resend all images from oldest to newest"""
    if str(ctx.author.id) != BOT_OWNER_ID:
        await ctx.send("❌ Only the bot owner can use this command.")
        return
    
    await ctx.send("🔄 Starting resend process...")
    logger.info(f"Resend all command initiated by {ctx.author.name}")
    
    try:
        # Get channels
        channel_new = bot.get_channel(CHANNEL_NEW_IMAGES)
        channel_var = bot.get_channel(CHANNEL_VARIATIONS)
        channel_upscaled = bot.get_channel(CHANNEL_UPSCALED)
        
        if not channel_new or not channel_var or not channel_upscaled:
            await ctx.send("❌ Could not find one or more channels!")
            return
        
        # Clear channels
        await ctx.send("🧹 Clearing channels...")
        
        deleted_new = 0
        deleted_var = 0
        deleted_upscaled = 0
        
        # Delete messages in new images channel
        async for message in channel_new.history(limit=None):
            if message.author == bot.user:
                await message.delete()
                deleted_new += 1
                await asyncio.sleep(0.3)
        
        # Delete messages in variations channel
        async for message in channel_var.history(limit=None):
            if message.author == bot.user:
                await message.delete()
                deleted_var += 1
                await asyncio.sleep(0.3)
        
        async for message in channel_upscaled.history(limit=None):
            if message.author == bot.user:
                await message.delete()
                deleted_upscaled += 1
                await asyncio.sleep(0.3)
        
        await ctx.send(f"✅ Cleared {deleted_new} messages from new images channel")
        await ctx.send(f"✅ Cleared {deleted_var} messages from variations channel")
        await ctx.send(f"✅ Cleared {deleted_upscaled} messages from upscaled images channel")
        
        sent_images_dict.clear()
        with open(SENT_IMAGES_FILE, 'w', encoding='utf-8') as f:
            pass  # Empty the file
        
        await ctx.send("📂 Cleared tracking file")
        
        # Get all images sorted by modification time (oldest first)
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            await ctx.send("❌ Outputs folder not found!")
            return
        
        image_files = sorted(outputs_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        
        if not image_files:
            await ctx.send("❌ No images found in outputs folder!")
            return
        
        await ctx.send(f"📤 Resending {len(image_files)} images (oldest to newest)...")
        
        # Resend all images
        sent_count = 0
        for image_path in image_files:
            filename = image_path.name
            
            is_upscaled = filename.startswith("upscaled")
            is_variation = filename.startswith("variation")
            
            if is_upscaled:
                target_channel = channel_upscaled
            elif is_variation:
                target_channel = channel_var
            else:
                target_channel = channel_new
            
            try:
                seed_info = ""
                seed = None
                if "seed" in filename:
                    try:
                        seed = int(filename.split("seed")[1].split("_")[0])
                        seed_info = f"\n🎲 Seed: `{seed}`"
                        log_prompt_to_file(filename, seed)
                    except:
                        pass
                
                creation_time = datetime.fromtimestamp(image_path.stat().st_mtime)
                time_str = creation_time.strftime("%Y-%m-%d %H:%M:%S")
                
                msg_type = "📐 Upscaled" if is_upscaled else ("🔀 Variation" if is_variation else "🎨 New Image")
                
                message = await target_channel.send(
                    f"{msg_type}\n📅 {time_str}{seed_info}",
                    file=discord.File(image_path)
                )
                
                save_sent_image(filename, message.jump_url)
                
                sent_count += 1
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to resend image {filename}: {e}")
                await ctx.send(f"⚠️  Failed to send {filename}: {e}")
        
        await ctx.send(f"✅ Resend complete! Sent {sent_count}/{len(image_files)} images")
        logger.success(f"Resend completed: {sent_count}/{len(image_files)} images")
        
    except Exception as e:
        await ctx.send(f"❌ Error during resend: {e}")
        logger.error(f"Resend error: {e}")

@bot.command(name="clearall")
async def clear_all_command(ctx):
    """Clear all images from channels and tracking file without resending"""
    if str(ctx.author.id) != BOT_OWNER_ID:
        await ctx.send("❌ Only the bot owner can use this command.")
        return
    
    # Ask for confirmation
    await ctx.send("⚠️  Are you sure you want to clear all images from channels? Type `yes` to confirm.")
    
    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() == 'yes'
    
    try:
        await bot.wait_for('message', check=check, timeout=30.0)
    except asyncio.TimeoutError:
        await ctx.send("❌ Confirmation timed out. Clear cancelled.")
        return
    
    await ctx.send("🧹 Starting clear process...")
    logger.info(f"Clear all command initiated by {ctx.author.name}")
    
    try:
        # Get channels
        channel_new = bot.get_channel(CHANNEL_NEW_IMAGES)
        channel_var = bot.get_channel(CHANNEL_VARIATIONS)
        channel_upscaled = bot.get_channel(CHANNEL_UPSCALED)
        
        if not channel_new or not channel_var or not channel_upscaled:
            await ctx.send("❌ Could not find one or more channels!")
            return
        
        deleted_new = 0
        deleted_var = 0
        deleted_upscaled = 0
        
        # Delete messages in new images channel
        async for message in channel_new.history(limit=None):
            if message.author == bot.user:
                await message.delete()
                deleted_new += 1
                await asyncio.sleep(0.3)
        
        # Delete messages in variations channel
        async for message in channel_var.history(limit=None):
            if message.author == bot.user:
                await message.delete()
                deleted_var += 1
                await asyncio.sleep(0.3)
        
        async for message in channel_upscaled.history(limit=None):
            if message.author == bot.user:
                await message.delete()
                deleted_upscaled += 1
                await asyncio.sleep(0.3)
        
        # Clear tracking file
        sent_images_dict.clear()
        with open(SENT_IMAGES_FILE, 'w', encoding='utf-8') as f:
            pass  # Empty the file
        
        await ctx.send(f"✅ Cleared {deleted_new} messages from new images channel")
        await ctx.send(f"✅ Cleared {deleted_var} messages from variations channel")
        await ctx.send(f"✅ Cleared {deleted_upscaled} messages from upscaled images channel")
        await ctx.send("✅ Cleared tracking file")
        await ctx.send("🎉 All images cleared! New images will be sent as they are generated.")
        
        logger.success("All channels cleared successfully")
        
    except Exception as e:
        await ctx.send(f"❌ Error during clear: {e}")
        logger.error(f"Clear error: {e}")

@bot.command(name="rescan")
async def rescan_command(ctx):
    """Force rescan of outputs folder (for testing)"""
    if str(ctx.author.id) != BOT_OWNER_ID:
        await ctx.send("❌ Only the bot owner can use this command.")
        return
    
    await ctx.send("🔄 Rescanning outputs folder...")
    logger.info(f"Rescan command executed by {ctx.author.name}")

@bot.command(name="bothelp")
async def bot_help_command(ctx):
    """Show bot help and troubleshooting info"""
    embed = discord.Embed(
        title="🤖 L.O.O.M. Discord Bot Help",
        description="This bot automatically posts AI-generated images to Discord channels.",
        color=discord.Color.blue()
    )
    
    if str(ctx.author.id) == BOT_OWNER_ID:
        embed.add_field(
            name="📝 Owner Commands",
            value=(
                "`!status` - Check bot status\n"
                "`!resendall` - Clear and resend all images\n"
                "`!clearall` - Clear all images from channels\n"
                "`!rescan` - Force rescan outputs folder\n"
                "`!bothelp` - Show this help message"
            ),
            inline=False
        )
    
    embed.add_field(
        name="🔧 Required Permissions",
        value=(
            "• View Channels\n"
            "• Send Messages\n"
            "• Attach Files\n"
            "• Read Message History\n"
            "• Manage Messages (for clear commands)"
        ),
        inline=False
    )
    
    embed.add_field(
        name="📁 Monitored Channels",
        value=(
            f"• New Images: <#{CHANNEL_NEW_IMAGES}>\n"
            f"• Variations: <#{CHANNEL_VARIATIONS}>\n"
            f"• Upscaled: <#{CHANNEL_UPSCALED}>"
        ),
        inline=False
    )
    
    embed.add_field(
        name="⚙️ Configuration",
        value=(
            "Edit channel IDs in: `static/config/channel_ids.txt`\n"
            "Format:\n"
            "\`\`\`\n"
            "new=CHANNEL_ID\n"
            "variations=CHANNEL_ID\n"
            "upscaled=CHANNEL_ID\n"
            "\`\`\`"
        ),
        inline=False
    )
    
    embed.set_footer(text="L.O.O.M. - Local Operator of Open Minds")
    
    await ctx.send(embed=embed)

# Main execution
if __name__ == "__main__":
    if not BOT_TOKEN:
        logger.error("=" * 70)
        logger.error("BOT TOKEN NOT FOUND!")
        logger.error("=" * 70)
        logger.info("")
        logger.info("Please create: static/config/bot_token.txt")
        logger.info("   Add your Discord bot token on the first line")
        logger.info("")
        logger.info("Please create: static/config/bot_owner_id.txt")
        logger.info("   Add your Discord user ID on the first line")
        logger.info("")
        logger.info("Please create: static/config/bot_id.txt")
        logger.info("   Add the bot's user ID on the first line")
        logger.info("=" * 70)
        exit(1)
    
    logger.info("")
    logger.discord_bot_start()
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info(f"   Token: {'✅ Loaded' if BOT_TOKEN else '❌ Missing'}")
    logger.info(f"   Owner ID: {BOT_OWNER_ID if BOT_OWNER_ID else '❌ Missing'}")
    logger.info(f"   Bot ID: {BOT_ID if BOT_ID else '❌ Missing'}")
    logger.info("")
    logger.info("Connecting to Discord...")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        bot.run(BOT_TOKEN)
    except discord.LoginFailure:
        logger.error("")
        logger.error("Failed to login! Please check your bot token in static/config/bot_token.txt")
        logger.info("Make sure the token is valid and not expired")
    except discord.PrivilegedIntentsRequired:
        logger.error("")
        logger.error("Missing privileged intents!")
        logger.info("Go to Discord Developer Portal > Your Bot > Bot Settings")
        logger.info("Enable 'Message Content Intent' and 'Server Members Intent'")
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.cleanup()
