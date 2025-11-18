# 🛠️ Setup Guide

Complete installation guide for AI Image Generator Pro.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [First Run](#first-run)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

#### Minimum (GTX 1650 / 4GB VRAM)
- **GPU**: NVIDIA GTX 1650 or better
- **RAM**: 8GB system RAM
- **Storage**: 20GB free space
- **Performance**: ~6 minutes per 512x768 image

#### Recommended (RTX 3060 / 12GB VRAM)
- **GPU**: NVIDIA RTX 3060 or better
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space
- **Performance**: ~2 minutes per 512x768 image

#### High-End (RTX 4090 / 24GB VRAM)
- **GPU**: NVIDIA RTX 4090
- **RAM**: 32GB system RAM
- **Storage**: 100GB free space
- **Performance**: ~30 seconds per 512x768 image

#### CPU-Only (Not Recommended)
- **CPU**: 8+ cores recommended
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB free space
- **Performance**: 30-60 minutes per image

### Software

#### Required
- **Python**: 3.9, 3.10, or 3.11 (3.12 not yet fully supported)
- **CUDA**: 11.8 or 12.1 (if using NVIDIA GPU)
- **Git**: For cloning repository

#### Optional
- **Discord Account**: For bot integration
- **nvidia-smi**: For GPU monitoring (included with CUDA)

---

## Prerequisites

### 1. Install Python

#### Windows
```bash
# Download from python.org
# Or use winget:
winget install Python.Python.3.11

# Verify installation
python --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
python3.11 --version
```

#### macOS
```bash
# Using Homebrew
brew install python@3.11
python3.11 --version
```

### 2. Install CUDA (NVIDIA GPU Only)

#### Windows
1. Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Install CUDA 11.8 or 12.1
3. Verify: `nvcc --version`

#### Linux
```bash
# Ubuntu 22.04 example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Verify
nvcc --version
nvidia-smi
```

### 3. Install Git

#### Windows
```bash
winget install Git.Git
```

#### Linux
```bash
sudo apt install git
```

#### macOS
```bash
brew install git
```

---

## Installation

### Method 1: Automated Installer (Recommended)

#### Windows
```bash
# 1. Clone repository
git clone https://github.com/yourusername/ai-image-generator.git
cd ai-image-generator

# 2. Run installer
scripts\install.bat

# 3. Follow prompts
```

#### Linux/macOS
```bash
# 1. Clone repository
git clone https://github.com/yourusername/ai-image-generator.git
cd ai-image-generator

# 2. Make installer executable
chmod +x scripts/install.sh

# 3. Run installer
./scripts/install.sh

# 4. Follow prompts
```

### Method 2: Manual Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ai-image-generator.git
cd ai-image-generator

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install PyTorch (CUDA)
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

---

## Configuration

### 1. Basic Configuration

No configuration needed for basic usage! The web interface works out of the box.

### 2. Discord Bot Setup (Optional)

#### Step 1: Create Discord Bot
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Name your bot (e.g., "AI Art Bot")
4. Go to "Bot" tab → "Add Bot"
5. Copy the **Bot Token**

#### Step 2: Get Channel IDs
1. Enable Developer Mode in Discord:
   - User Settings → Advanced → Developer Mode (ON)
2. Right-click your channel → Copy ID
3. You need TWO channel IDs:
   - One for new images
   - One for variations

#### Step 3: Get Your User ID
1. Right-click your username → Copy ID

#### Step 4: Configure Files
```bash
# Copy example files
cp static/bot_token.txt.example static/bot_token.txt
cp static/bot_owner_id.txt.example static/bot_owner_id.txt
cp static/bot_id.txt.example static/bot_id.txt

# Edit files
# Windows:
notepad static\bot_token.txt
# Linux/macOS:
nano static/bot_token.txt
```

Add your credentials:
```
# bot_token.txt
your_bot_token_here

# bot_owner_id.txt
your_discord_user_id

# bot_id.txt
bot_discord_user_id
```

#### Step 5: Update Channel IDs
Edit `src/discord_bot.py`:
```python
# Line 20-21
CHANNEL_NEW_IMAGES = 1234567890123456789  # Your channel ID
CHANNEL_VARIATIONS = 9876543210987654321  # Your channel ID
```

#### Step 6: Invite Bot to Server
1. Go to Discord Developer Portal → Your App → OAuth2 → URL Generator
2. Select scopes: `bot`
3. Select permissions: 
   - Send Messages
   - Attach Files
   - Read Message History
4. Copy generated URL and open in browser
5. Select your server and authorize

---

## First Run

### 1. Start Server
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Start server
python src/server.py
```

### 2. Model Download (First Time Only)
On first run, the AI model will be downloaded (~5GB):
```
================================================================
🚀 Starting AI Image Generator Backend...
================================================================
✅ NVIDIA GPU detected: NVIDIA GeForce RTX 3060

📦 Loading Anime model...
⏳ This will take 5-10 minutes on first run (downloading model)...
Downloading (…)ain/model_index.json: 100%|██████████| 5.2G/5.2G [05:23<00:00, 16.0MB/s]
✅ Model loaded successfully!
```

**Be patient!** This only happens once.

### 3. Open Web Interface
```
🌐 Server ready at http://localhost:8000
📱 Open your browser and navigate to the URL above
```

Open browser: `http://localhost:8000`

### 4. Generate First Image
1. Enter a prompt: "anime girl with blue hair"
2. Click "🚀 Generate Image"
3. Watch the progress bar
4. See your generated image!

---

## GPU-Specific Setup

### GTX Series (No Tensor Cores)
Edit `src/server.py`, line ~150:
```python
# Use float32 for stability
dtype = torch.float32
```

### RTX Series (Tensor Cores)
```python
# Use float16 for speed
dtype = torch.float16
```

Auto-detection is enabled by default!

---

## Troubleshooting

### Installation Issues

#### "Python not found"
```bash
# Windows: Add Python to PATH during installation
# Or reinstall with "Add to PATH" checked

# Linux: Install python3
sudo apt install python3.11

# Verify
python --version
```

#### "CUDA not found" / "torch.cuda.is_available() = False"
```bash
# 1. Check NVIDIA driver
nvidia-smi

# 2. Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Verify
python -c "import torch; print(torch.cuda.is_available())"
```

#### "Out of Memory" Error
```python
# Reduce resolution or steps
width = 512  # Instead of 1024
steps = 20   # Instead of 50

# Or enable model CPU offloading in server.py:
pipe.enable_model_cpu_offload()
```

### Runtime Issues

#### "Model download stuck"
```bash
# Check internet connection
# Or download manually from HuggingFace

# Clear cache and retry
rm -rf ~/.cache/huggingface/
python src/server.py
```

#### "WebSocket connection failed"
```bash
# Check if port 8000 is in use
# Windows:
netstat -ano | findstr :8000
# Linux:
lsof -i :8000

# Change port in server.py if needed
uvicorn.run(app, host="0.0.0.0", port=8001)
```

#### Discord Bot Not Starting
```bash
# Check bot_token.txt exists and contains valid token
cat static/bot_token.txt

# Check bot permissions in Discord server
# Ensure bot has "Send Messages" and "Attach Files"

# Check console for error messages
```

### Performance Issues

#### "Generation too slow"
```bash
# 1. Check GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# 2. Lower settings
Steps: 20-30 (instead of 50)
Resolution: 512x512 (instead of 1024x1024)

# 3. Close other GPU applications
# - Chrome/Firefox with hardware acceleration
# - Other AI tools
# - Games
```

#### "GPU temperature too high (>85°C)"
```bash
# 1. Check GPU fans
nvidia-smi

# 2. Clean GPU heatsink
# 3. Improve case airflow
# 4. Lower power limit (if needed)
nvidia-smi -pl 150  # 150W limit example
```

---

## Advanced Configuration

### Custom Model Path
```python
# src/server.py, line ~140
model_id = "Ojimi/anime-kawai-diffusion"  # Default

# Change to local path or different model:
model_id = "/path/to/your/model"
model_id = "stabilityai/stable-diffusion-2-1"
```

### Environment Variables
Create `.env` file:
```bash
CUDA_VISIBLE_DEVICES=0    # Use first GPU only
MODEL_CACHE_DIR=/custom/path
API_HOST=0.0.0.0
API_PORT=8000
```

### Multi-GPU Setup
```python
# src/server.py
device = "cuda:0"  # First GPU
device = "cuda:1"  # Second GPU
```

---

## Next Steps

- ✅ Installation complete? → Read [CONFIGURATION.md](CONFIGURATION.md)
- ✅ Want to use the API? → Read [API.md](API.md)
- ✅ Having issues? → Read [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- ✅ Want to contribute? → Read [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## Support

Still stuck? Open an issue on GitHub:
https://github.com/yourusername/ai-image-generator/issues

Include:
- Your OS and Python version
- GPU model and VRAM
- Full error message
- Steps to reproduce