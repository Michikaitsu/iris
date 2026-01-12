# 📖 Setup Guide
### Complete Installation Guide for I.R.I.S.

*Intelligent Rendering & Image Synthesis*

---

## 💻 System Requirements

### Hardware

| Tier | GPU | VRAM | RAM | Storage | Performance |
|------|-----|------|-----|---------|-------------|
| **Minimum** | GTX 1650 | 4GB | 8GB | 20GB | ~6 min / 512×768 |
| **Recommended** | RTX 3060 | 12GB | 16GB | 50GB | ~2 min / 512×768 |
| **High-End** | RTX 4090 | 24GB | 32GB | 100GB | ~30 sec / 512×768 |
| **CPU-Only** | 8+ cores | N/A | 16GB+ | 20GB | 30-60 min / image |

### Software

**Required:**
- Python 3.9, 3.10, or 3.11 (3.12 not yet fully supported)
- CUDA 11.8 or 12.1 (if using NVIDIA GPU)
- Git

**Optional:**
- Discord Account (for bot integration)

---

## 📦 Prerequisites

### Install Python

**Windows:**
```bash
winget install Python.Python.3.11
python --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
python3.11 --version
```

**macOS:**
```bash
brew install python@3.11
python3.11 --version
```

### Install CUDA (NVIDIA GPU Only)

1. Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Install CUDA 11.8 or 12.1
3. Verify: `nvcc --version`

---

## 🚀 Installation

```bash
# 1. Clone repository
git clone https://github.com/KaiTooast/iris.git
cd iris

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) Copy environment template
cp .env.example .env
```

### Manual PyTorch Installation (if needed)

```bash
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision torchaudio

# Verify
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## ⚙️ Configuration

### settings.json

Located in project root. Controls runtime behavior:

```json
{
  "dramEnabled": true,
  "vramThreshold": 6,
  "maxDram": 8,
  "nsfwStrength": 2,
  "discordEnabled": false
}
```

| Setting | Description |
|---------|-------------|
| `dramEnabled` | Use system RAM when VRAM is low |
| `vramThreshold` | VRAM threshold (GB) to enable DRAM Extension |
| `maxDram` | Maximum system RAM to use (GB) |
| `nsfwStrength` | 1=Minimal, 2=Standard, 3=Strict |
| `discordEnabled` | Auto-start Discord bot |

### .env (Optional)

```env
HOST=0.0.0.0
PORT=8000
DEFAULT_MODEL=anime_kawai

# Discord Bot (optional)
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_BOT_ID=your_bot_id
DISCORD_BOT_OWNER_ID=your_user_id

DISCORD_CHANNEL_NEW_IMAGES=channel_id
DISCORD_CHANNEL_VARIATIONS=channel_id
DISCORD_CHANNEL_UPSCALED=channel_id
```

---

## 🎬 Running

```bash
# Make sure venv is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Start (auto-detects Discord from settings.json)
python src/start.py

# Start without Discord bot
python src/start.py --no-bot
```

### First Run

On first run, the AI model will download (~5GB):

```
Loading Model: Ojimi/anime-kawai-diffusion
Downloading model components... 100%
✅ Model loaded successfully!
```

> ⏰ This only happens once per model.

### Access Web UI

```
Server ready at http://localhost:8000
```

🌐 Open: **http://localhost:8000**

---

## 🔧 Discord Bot Setup (Optional)

### 1. Create Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Go to "Bot" tab → "Add Bot"
4. Copy the **Bot Token**

### 2. Get Channel IDs

1. Enable Developer Mode: User Settings → Advanced → Developer Mode
2. Right-click channel → Copy ID
3. You need three channels:
   - New images
   - Variations
   - Upscaled images

### 3. Configure

Add to `.env`:
```env
DISCORD_BOT_TOKEN=your_token_here
DISCORD_CHANNEL_NEW_IMAGES=123456789
DISCORD_CHANNEL_VARIATIONS=123456789
DISCORD_CHANNEL_UPSCALED=123456789
```

Set in `settings.json`:
```json
{
  "discordEnabled": true
}
```

### 4. Invite Bot

1. Go to OAuth2 → URL Generator
2. Select: `bot`
3. Permissions: `Send Messages`, `Attach Files`
4. Copy URL and open in browser

---

## 🛠️ Troubleshooting

### "Out of Memory" Error
- Enable DRAM Extension in Settings
- Use smaller resolution (512×512)
- Reduce steps (20-30)
- Close other GPU applications

### "CUDA not found"
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available())"
```

### Discord Bot Not Sending Images
- Verify Bot Token is correct
- Check Channel IDs are valid
- Ensure bot has permissions: Send Messages, Attach Files
- Check bot is in the server

### Model Download Stuck
- Check internet connection
- Try again (downloads resume)
- Manually clear `~/.cache/huggingface` and retry

---

## 📁 File Structure

```
iris/
├── src/
│   ├── api/server.py          # FastAPI server
│   ├── core/generator.py      # Generation logic
│   ├── services/bot.py        # Discord bot
│   └── start.py               # Entry point
├── frontend/                  # HTML pages
├── static/
│   ├── css/                   # Stylesheets
│   ├── js/                    # JavaScript
│   └── data/                  # History files
├── outputs/                   # Generated images
├── Logs/                      # Log files
├── settings.json              # Runtime config
└── .env                       # Environment variables
```

---

## 💬 Support

Still stuck? Open an issue on GitHub with:
- OS and Python version
- GPU model and VRAM
- Full error message
- Steps to reproduce
