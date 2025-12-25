<div align="center">

# 📖 Setup Guide
### Complete Installation Guide for I.R.I.S.

*Intelligent Rendering & Image Synthesis*

</div>

---

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [First Run](#-first-run)
- [Troubleshooting](#-troubleshooting)

---

## 💻 System Requirements

### Hardware

<table>
<tr>
<th>Tier</th>
<th>GPU</th>
<th>VRAM</th>
<th>RAM</th>
<th>Storage</th>
<th>Performance</th>
</tr>
<tr>
<td><strong>Minimum</strong></td>
<td>GTX 1650</td>
<td>4GB</td>
<td>8GB</td>
<td>20GB</td>
<td>~6 min / 512x768</td>
</tr>
<tr>
<td><strong>Recommended</strong></td>
<td>RTX 3060</td>
<td>12GB</td>
<td>16GB</td>
<td>50GB</td>
<td>~2 min / 512x768</td>
</tr>
<tr>
<td><strong>High-End</strong></td>
<td>RTX 4090</td>
<td>24GB</td>
<td>32GB</td>
<td>100GB</td>
<td>~30 sec / 512x768</td>
</tr>
<tr>
<td><strong>CPU-Only</strong></td>
<td>8+ cores</td>
<td>N/A</td>
<td>16GB+</td>
<td>20GB</td>
<td>30-60 min / image</td>
</tr>
</table>

### Software

**Required:**
- Python 3.9, 3.10, or 3.11 (3.12 not yet fully supported)
- CUDA 11.8 or 12.1 (if using NVIDIA GPU)
- Git

**Optional:**
- Discord Account (for bot integration)
- nvidia-smi (included with CUDA)

---

## 📦 Prerequisites

### 1. Install Python

<details>
<summary><strong>Windows</strong></summary>

```bash
# Download from python.org
# Or use winget:
winget install Python.Python.3.11

# Verify installation
python --version
```

</details>

<details>
<summary><strong>Linux (Ubuntu/Debian)</strong></summary>

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
python3.11 --version
```

</details>

<details>
<summary><strong>macOS</strong></summary>

```bash
# Using Homebrew
brew install python@3.11
python3.11 --version
```

</details>

### 2. Install CUDA (NVIDIA GPU Only)

<details>
<summary><strong>Windows</strong></summary>

1. Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Install CUDA 11.8 or 12.1
3. Verify: `nvcc --version`

</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
# Ubuntu 22.04 example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
# ... (follow NVIDIA's installation instructions)

# Verify
nvcc --version
nvidia-smi
```

</details>

### 3. Install Git

```bash
# Windows
winget install Git.Git

# Linux
sudo apt install git

# macOS
brew install git
```

---

## 🚀 Installation

### Method 1: Quick Install (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/iris-image-synthesis.git
cd iris-image-synthesis

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Method 2: Manual Installation

<details>
<summary><strong>Step-by-Step Manual Setup</strong></summary>

```bash
# 1. Clone repository
git clone https://github.com/yourusername/iris-image-synthesis.git
cd iris-image-synthesis

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install PyTorch
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (slow!):
pip install torch torchvision torchaudio

# 6. Install other dependencies
pip install -r requirements.txt

# 7. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

</details>

---

## ⚙️ Configuration

### 1. Basic Configuration

> ✅ **No configuration needed for basic web UI!**

Just run:

```bash
python src/start.py web
```

### 2. Environment Variables (Recommended)

Create a `.env` file in the project root:

```env
# ============================================
# Server Settings
# ============================================
HOST=0.0.0.0
PORT=8000

# ============================================
# Discord Bot Configuration (Optional)
# ============================================
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_BOT_ID=your_bot_id_here
DISCORD_BOT_OWNER_ID=your_discord_user_id_here

# Discord Channel IDs
DISCORD_CHANNEL_NEW_IMAGES=1234567890123456789
DISCORD_CHANNEL_VARIATIONS=9876543210987654321
DISCORD_CHANNEL_UPSCALED=1234567890123456789

# ============================================
# Model Settings
# ============================================
DEFAULT_MODEL=anime

# ============================================
# DRAM Extension (for low VRAM GPUs)
# ============================================
DRAM_EXTENSION_ENABLED=false
VRAM_THRESHOLD_GB=6
MAX_DRAM_GB=16
```

### 3. Discord Bot Setup (Optional)

<details>
<summary><strong>Step 1: Create Discord Bot</strong></summary>

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Name your bot (e.g., "I.R.I.S. Bot")
4. Go to "Bot" tab → "Add Bot"
5. Copy the **Bot Token**

</details>

<details>
<summary><strong>Step 2: Get Channel IDs</strong></summary>

1. Enable Developer Mode in Discord:
   - User Settings → Advanced → Developer Mode (ON)
2. Right-click your channel → Copy ID
3. You need **THREE** channel IDs:
   - One for new images (`DISCORD_CHANNEL_NEW_IMAGES`)
   - One for variations (`DISCORD_CHANNEL_VARIATIONS`)
   - One for upscaled images (`DISCORD_CHANNEL_UPSCALED`)

</details>

<details>
<summary><strong>Step 3: Get Your User ID and Bot ID</strong></summary>

1. Right-click your username → Copy ID (for `DISCORD_BOT_OWNER_ID`)
2. Right-click the bot username → Copy ID (for `DISCORD_BOT_ID`)

</details>

<details>
<summary><strong>Step 4: Configure .env File</strong></summary>

Add all Discord settings to your `.env` file:

```env
# Discord Bot Configuration
DISCORD_BOT_TOKEN=MTM3OTU1MjI2MjA2OTgxMzI0OA.GjAJ2-.example_token
DISCORD_BOT_ID=1379552262069813248
DISCORD_BOT_OWNER_ID=918149823587307580

# Discord Channel IDs
DISCORD_CHANNEL_NEW_IMAGES=1442114035368591462
DISCORD_CHANNEL_VARIATIONS=1442114068445003839
DISCORD_CHANNEL_UPSCALED=1442114100095090822
```

</details>

---

## 🎬 First Run

### 1. Start Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Start web UI only
python src/start.py web

# Or start bot only
python src/start.py bot

# Or start both
python src/start.py all
```

### 2. Model Download (First Time Only)

On first run, the AI model will be downloaded (~5GB):

```
================================================================
Starting AI Image Generator Backend...
================================================================
✅ NVIDIA GPU detected: NVIDIA GeForce RTX 3060

Loading Model: Ojimi/anime-kawai-diffusion
This may take 5-10 minutes on first run...
Downloading model components... 100%
✅ Model loaded successfully!
```

> ⏰ **Be patient!** This only happens once.

### 3. Open Web Interface

```
================================================================
Server ready at http://localhost:8000
================================================================
```

> 🌐 Open browser: **http://localhost:8000**

### 4. Generate First Image

1. Enter a prompt: `"anime girl with blue hair"`
2. Click **"Generate Image"**
3. Watch the progress bar
4. See your generated image!

---

## 📁 File Structure

```
iris-image-synthesis/
├── 📂 src/
│   ├── 📂 api/
│   │   └── server.py              # FastAPI server
│   ├── 📂 core/
│   │   ├── config.py              # Configuration
│   │   ├── model_loader.py        # Model loading
│   │   └── generator.py           # Generation logic
│   ├── 📂 services/
│   │   ├── bot.py                 # Discord bot
│   │   └── upscaler.py            # Upscaling
│   ├── 📂 utils/
│   │   ├── logger.py              # Logging
│   │   └── file_manager.py        # File operations
│   ├── 📂 frontend/
│   │   ├── index.html             # Landing page
│   │   ├── generate.html          # Generation UI
│   │   ├── gallery.html           # Image gallery
│   │   └── settings.html          # Settings
│   └── start.py                   # Entry point
├── 📂 static/
│   ├── 📂 data/
│   │   ├── prompts_history.json   # ✅ Prompt logging
│   │   └── img_send.json          # ✅ Image tracking
│   ├── 📂 assets/
│   ├── 📂 css/
│   └── 📂 js/
├── 📂 outputs/                    # Generated images
├── 📂 docs/
│   ├── SETUP.md                   # This file
│   └── ARTIFACTS.md
├── .env                           # ✅ Environment variables
└── requirements.txt
```

---

## 🛠️ Troubleshooting

### Common Errors

<details>
<summary><strong>1. "ModuleNotFoundError: No module named 'discord.ext'"</strong></summary>

**Cause**: File naming conflict.

**Solution**: Already fixed! The bot is now named `bot.py`.

```bash
python src/start.py bot
```

</details>

<details>
<summary><strong>2. "RuntimeError: File at path static/settings.html does not exist"</strong></summary>

**Cause**: HTML files are in `src/frontend/`, not `static/`.

**Solution**: Already fixed in `server.py`.

</details>

<details>
<summary><strong>3. Discord Bot Not Sending Images</strong></summary>

**Solution**: Ensure your `.env` file has the correct variable names:

```env
DISCORD_BOT_TOKEN=your_token_here
DISCORD_BOT_ID=your_bot_id
DISCORD_BOT_OWNER_ID=your_user_id
DISCORD_CHANNEL_NEW_IMAGES=channel_id_here
DISCORD_CHANNEL_VARIATIONS=channel_id_here
DISCORD_CHANNEL_UPSCALED=channel_id_here
```

Check:
- ✅ Bot has "Send Messages" and "Attach Files" permissions
- ✅ All channel IDs are valid
- ✅ Bot is invited to your server

</details>

<details>
<summary><strong>4. Images Sending Twice to Discord</strong></summary>

**Solution**: Fixed with 6-second buffer delay. Update to latest version.

</details>

<details>
<summary><strong>5. Favicon Not Loading</strong></summary>

**Solution**: Already fixed. Favicon path corrected to `static/assets/favico.png`.

</details>

<details>
<summary><strong>6. "Out of Memory" Error</strong></summary>

**Solutions:**
1. Enable DRAM Extension in Settings
2. Use smaller resolution (512x512)
3. Reduce steps (20-30 instead of 50)
4. Close other GPU applications

</details>

### Installation Issues

<details>
<summary><strong>"Python not found"</strong></summary>

```bash
# Windows: Add Python to PATH during installation
# Or reinstall with "Add to PATH" checked

# Linux: Install python3
sudo apt install python3.11

# Verify
python --version
```

</details>

<details>
<summary><strong>"CUDA not found"</strong></summary>

```bash
# 1. Check NVIDIA driver
nvidia-smi

# 2. Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Verify
python -c "import torch; print(torch.cuda.is_available())"
```

</details>

---

## 🎯 Next Steps

<div align="center">

| Status | Next Step |
|--------|-----------|
| ✅ Installation complete | → Start generating images! |
| ✅ Want Discord integration | → Complete Discord Bot Setup |
| ✅ Having issues | → Check Troubleshooting |
| ✅ Want to contribute | → Read [CONTRIBUTING.md](../CONTRIBUTING.md) |

</div>

---

## 💬 Support

Still stuck? Open an issue on GitHub with:

- Your OS and Python version
- GPU model and VRAM
- Full error message
- Steps to reproduce

---

<div align="center">

**Made with ❤️ using Stable Diffusion**

[⬆ Back to Top](#-setup-guide)

</div>
