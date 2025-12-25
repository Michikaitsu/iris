<div align="center">

# 🎨 I.R.I.S.
### Intelligent Rendering & Image Synthesis

*Local AI Image Generation Powered by Stable Diffusion*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GPU Support](https://img.shields.io/badge/GPU-NVIDIA%20%7C%20AMD%20%7C%20Intel%20%7C%20Apple-green.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B%20%7C%2012.1%2B-76B900.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Bot%20Ready-7289da.svg)](https://discord.com/)

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🖼️ **Core Features**
- **Web UI** - Modern, responsive interface
- **Discord Bot** - Auto-posting with Rich Presence
- **Multiple Styles** - Anime, realistic, abstract, pixel art
- **Image-to-Image** - Create variations
- **AI Upscaling** - Real-ESRGAN (2x, 4x, 8x)
- **Custom Resolutions** - Set any resolution you want!

</td>
<td width="50%">

### ⚡ **Advanced Features**
- **Extended Resolution Range** - 256x256 to 4096x4096
- **Precise CFG Control** - 0.1 step increments
- **DRAM Extension** - Use system RAM for low VRAM GPUs
- **Real-time Progress** - WebSocket live updates
- **Prompt History** - JSON logging of all prompts
- **Image Tracking** - Auto-logging sent images
- **Multi-GPU Support** - NVIDIA, AMD, Intel Arc, Apple Silicon

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

```bash
✅ Python 3.9, 3.10, or 3.11
✅ 8GB+ System RAM (16GB+ recommended)

GPU Support (choose one):
🟢 NVIDIA GPU with 4GB+ VRAM (CUDA 11.8 or 12.1)
   → GTX 1050 Ti / 1650 or newer (GTX 1660+ recommended)
   
🟡 AMD GPU with 6GB+ VRAM
   → Linux: RX 6600 or newer with ROCm 5.4+
   → Windows: RX 6600 or newer with DirectML (slower)
   
🟠 Intel Arc GPU with 8GB+ VRAM (experimental)
   → Arc A750/A770 with latest drivers
   
🟢 Apple Silicon with 8GB+ unified memory
   → M1/M2/M3/M4 (any variant) on macOS 12.3+
   
💾 OR CPU-only mode (very slow, not recommended for regular use)
```

> 💡 **Recommendation**: For best experience, use NVIDIA RTX 3050 (8GB) or higher

### Installation

```bash
# 1. Clone repository
git clone https://github.com/KaiTooast/iris-image-synthesis.git
cd iris-image-synthesis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install GPU-specific packages (choose one):

# For NVIDIA (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For AMD (ROCm - Linux only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# For AMD (DirectML - Windows)
pip install torch-directml

# For Intel Arc (DirectML - Windows)
pip install torch-directml

# For Intel Arc (XPU - Linux)
pip install intel-extension-for-pytorch

# For Apple Silicon (included in base PyTorch)
# MPS support is built-in, no additional installation needed
```

> 💡 **Note**: For AMD and Intel GPUs on Windows, ensure you have the latest drivers installed.

### Running

```bash
# Start Web UI only
python src/start.py web

# Start Discord Bot only
python src/start.py bot

# Start both services
python src/start.py all
```

> 🌐 Access the Web UI at: **http://localhost:8000**

---

## 📁 Project Structure

```
iris-image-synthesis/
├── 📂 src/
│   ├── 📂 api/                     # API Layer
│   │   └── server.py               # FastAPI server
│   ├── 📂 core/                    # Core Logic
│   │   ├── config.py               # Configuration management
│   │   ├── model_loader.py         # AI model loading
│   │   └── generator.py            # Generation logic
│   ├── 📂 services/                # External Services
│   │   ├── bot.py                  # Discord bot integration
│   │   └── upscaler.py             # Upscaling service
│   ├── 📂 utils/                   # Utilities
│   │   ├── logger.py               # Logging system
│   │   └── file_manager.py         # File operations
│   ├── 📂 frontend/                # Web Interface
│   │   ├── index.html              # Landing page
│   │   ├── generate.html           # Generation interface
│   │   ├── gallery.html            # Image gallery
│   │   └── settings.html           # Settings panel
│   └── start.py                    # Entry point
├── 📂 static/
│   ├── 📂 assets/                  # Images & icons
│   ├── 📂 data/                    # Data files
│   │   ├── prompts_history.json    # Prompt logging
│   │   └── img_send.json           # Sent images tracking
│   ├── 📂 css/                     # Stylesheets
│   └── 📂 js/                      # Scripts
├── 📂 outputs/                     # Generated images
├── 📂 docs/                        # Documentation
│   ├── SETUP.md                    # Installation guide
│   └── ARTIFACTS.md                # Common issues
├── .env                            # Environment variables (create this!)
└── requirements.txt                # Python dependencies
```

---

## ⚙️ Configuration

### Environment Variables (.env)

Create a `.env` file in the project root:

```env
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
# Server Settings
# ============================================
HOST=0.0.0.0
PORT=8000

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

> 💡 **Tip**: See [docs/SETUP.md](docs/SETUP.md) for detailed Discord bot setup instructions.

---

## 🎯 Key Features Explained

### 🎨 Custom Resolutions

You can now use **any custom resolution** you want!

<table>
<tr>
<td>

**Preset Resolutions:**
- 256x256 → 4096x4096
- Portrait, Landscape, Square
- Mobile wallpapers
- HD to 4K options

</td>
<td>

**Custom Resolutions:**
- Enter any size: `512x812`, `1337x1920`, etc.
- Perfect for unique aspect ratios
- Ideal for specific use cases
- GPU requirements shown automatically

</td>
</tr>
</table>

### ⚡ DRAM Extension

For GPUs with limited VRAM (4GB-6GB):

- ✅ Automatically enables for GPUs with ≤6GB VRAM
- ✅ Configurable up to 16GB additional RAM
- ✅ Enables higher resolutions and more steps
- ✅ Toggle in Settings page or via API

### 📝 Prompt Logging

All prompts automatically logged to `static/data/prompts_history.json`:

```json
[
  {
    "timestamp": "2025-12-25T12:53:15.123456",
    "prompt": "anime girl with blue hair",
    "settings": {
      "seed": 12345,
      "steps": 28,
      "width": 512,
      "height": 768,
      "cfg_scale": 10.5
    }
  }
]
```

### 📊 Image Tracking

Sent images tracked in `static/data/img_send.json`:

```json
{
  "gen_2025-12-25_125315_98765.png": {
    "message_link": "https://discord.com/channels/.../...",
    "sent_at": "2025-12-25T12:53:20.123456"
  }
}
```

### 📦 File Naming Convention

```
{type}_{date}_{time}_{seed}_{steps}.png

Examples:
gen_20251225_125315_98765_s28.png       # Generated image
var_20251225_130122_54321_s25.png       # Variation
upscale_20251225_131045_98765_x4.png    # Upscaled 4x
```

---

## 🔌 API Endpoints

### Generation Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/generate` | Generate image (non-streaming) |
| `WebSocket` | `/ws/generate` | Generate with real-time progress |
| `POST` | `/api/variation` | Create image variation |
| `POST` | `/api/upscale` | Upscale image |

### Gallery Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/output-gallery` | List all generated images |
| `GET` | `/api/output-image/{filename}` | Get specific image |
| `WebSocket` | `/ws/gallery-progress` | Gallery progress updates |

### System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/system` | System information |
| `GET` | `/api/stats` | Generation statistics |
| `GET` | `/api/version` | Version information |

### DRAM Extension Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/dram-status` | Get DRAM extension status |
| `POST` | `/api/dram-extension` | Toggle DRAM extension |

---

## 💻 System Requirements

### 🎮 GPU Compatibility with Stable Diffusion

<table>
<tr>
<td width="25%">

**🟢 NVIDIA (Fully Supported)**
- ✅ GTX 1000 series (1050 Ti+)
- ✅ GTX 1600 series
- ✅ RTX 2000/3000/4000
- ✅ CUDA 11.8 or 12.1
- ✅ Best performance & stability
- ✅ All features work

</td>
<td width="25%">

**🟡 AMD (Partial Support)**
- ⚠️ RX 6000/7000 series
- ⚠️ Linux: ROCm 5.4+ (better)
- ⚠️ Windows: DirectML (slower)
- ⚠️ 70-80% NVIDIA performance
- ⚠️ Some features limited
- ✅ Functional for most tasks

</td>
<td width="25%">

**🟠 Intel Arc (Experimental)**
- ⚠️ Arc A-series (A380/A750/A770)
- ⚠️ DirectML (Windows)
- ⚠️ XPU (Linux, unstable)
- ⚠️ 50-60% NVIDIA performance
- ⚠️ Driver updates required
- ⚠️ May have compatibility issues

</td>
<td width="25%">

**🟢 Apple Silicon (Well Supported)**
- ✅ M1/M2/M3/M4 (all variants)
- ✅ MPS Backend (macOS 12.3+)
- ✅ 8GB+ unified memory
- ✅ Good performance for power
- ✅ Most features work
- ⚠️ Slower than equivalent NVIDIA

</td>
</tr>
</table>

### 🚫 Not Compatible / Not Recommended

- ❌ NVIDIA GPUs older than GTX 1000 series (insufficient CUDA support)
- ❌ AMD GPUs older than RX 5000 series (no ROCm/DirectML support)
- ❌ Integrated GPUs (Intel UHD, AMD Vega) - too slow, use CPU instead
- ❌ GPUs with less than 4GB VRAM (will fail on most resolutions)

### ⚙️ Real-World Performance Examples

<table>
<tr>
<th>GPU</th>
<th>VRAM</th>
<th>Platform</th>
<th>512x768<br/>(28 steps)</th>
<th>1024x1024<br/>(28 steps)</th>
<th>Notes</th>
</tr>
<tr>
<td><strong>RTX 4090</strong></td>
<td>24GB</td>
<td>NVIDIA/CUDA</td>
<td>~15-20 sec</td>
<td>~35-45 sec</td>
<td>✅ Best choice, all features</td>
</tr>
<tr>
<td><strong>RTX 3060</strong></td>
<td>12GB</td>
<td>NVIDIA/CUDA</td>
<td>~1-2 min</td>
<td>~3-4 min</td>
<td>✅ Recommended mid-range</td>
</tr>
<tr>
<td><strong>RTX 3050</strong></td>
<td>8GB</td>
<td>NVIDIA/CUDA</td>
<td>~2-3 min</td>
<td>~5-7 min</td>
<td>✅ Good budget option</td>
</tr>
<tr>
<td><strong>GTX 1650</strong></td>
<td>4GB</td>
<td>NVIDIA/CUDA</td>
<td>~4-6 min</td>
<td>⚠️ OOM likely</td>
<td>⚠️ Need DRAM extension</td>
</tr>
<tr>
<td><strong>RX 7900 XTX</strong></td>
<td>24GB</td>
<td>AMD/ROCm</td>
<td>~25-35 sec</td>
<td>~1-1.5 min</td>
<td>✅ Linux only for best speed</td>
</tr>
<tr>
<td><strong>RX 6700 XT</strong></td>
<td>12GB</td>
<td>AMD/DirectML</td>
<td>~2-3 min</td>
<td>~5-8 min</td>
<td>⚠️ Windows slower than Linux</td>
</tr>
<tr>
<td><strong>Arc A770</strong></td>
<td>16GB</td>
<td>Intel/DirectML</td>
<td>~3-5 min</td>
<td>~8-12 min</td>
<td>⚠️ Experimental, needs updates</td>
</tr>
<tr>
<td><strong>M2 Max</strong></td>
<td>32GB unified</td>
<td>Apple/MPS</td>
<td>~2-3 min</td>
<td>~4-6 min</td>
<td>✅ Good for MacBooks</td>
</tr>
<tr>
<td><strong>M1</strong></td>
<td>8GB unified</td>
<td>Apple/MPS</td>
<td>~4-5 min</td>
<td>⚠️ Memory limited</td>
<td>⚠️ Use 512x512 max</td>
</tr>
<tr>
<td><strong>CPU (i7-12700K)</strong></td>
<td>-</td>
<td>x86_64</td>
<td>~20-30 min</td>
<td>~60-90 min</td>
<td>❌ Very slow, last resort</td>
</tr>
</table>

<sup>⚠️ Performance times are **estimates** based on community reports. Your results may vary based on:</sup>
- System configuration (CPU, RAM speed, cooling)
- Background processes and system load
- Driver versions and optimizations
- Specific Stable Diffusion model used
- Operating system (Linux typically faster than Windows for AMD)

### 💾 CPU Fallback Mode

Don't have a compatible GPU? No problem!

- ✅ Full functionality on CPU
- ⚠️ **Very slow** (~20-30 min per 512x768 image)
- ✅ Good for testing and low-priority generation
- ✅ Requires 16GB+ RAM for stable operation
- ⚠️ Not recommended for regular use

### 🎯 GPU Recommendations by Budget

<table>
<tr>
<th>Budget</th>
<th>Best Choice</th>
<th>Alternative</th>
<th>VRAM</th>
<th>Why?</th>
</tr>
<tr>
<td><strong>Budget<br/>($150-250)</strong></td>
<td>GTX 1660 Super</td>
<td>RTX 3050</td>
<td>6GB / 8GB</td>
<td>Best value, reliable CUDA support</td>
</tr>
<tr>
<td><strong>Mid-Range<br/>($300-400)</strong></td>
<td>RTX 3060 12GB</td>
<td>RX 6700 XT (Linux)</td>
<td>12GB</td>
<td>Sweet spot for SD, handles 1024px</td>
</tr>
<tr>
<td><strong>High-End<br/>($600-800)</strong></td>
<td>RTX 4070 Ti</td>
<td>RX 7900 XT</td>
<td>12-16GB</td>
<td>Fast generation, future-proof</td>
</tr>
<tr>
<td><strong>Enthusiast<br/>($1500+)</strong></td>
<td>RTX 4090</td>
<td>-</td>
<td>24GB</td>
<td>Fastest available, handles anything</td>
</tr>
<tr>
<td><strong>Mac Users</strong></td>
<td>M2 Pro/Max</td>
<td>M3 Pro/Max</td>
<td>16GB+ unified</td>
<td>Native MPS support, portable</td>
</tr>
</table>

> 💡 **Buying advice**: Don't buy Intel Arc specifically for AI generation yet. AMD is okay on Linux, but NVIDIA offers the best compatibility and community support.

### 📝 Important Compatibility Notes

**Before installing, please note:**

1. **NVIDIA GPUs**: ✅ Best supported, most stable, fastest. If you're buying a GPU specifically for Stable Diffusion, choose NVIDIA.

2. **AMD GPUs**: ⚠️ Work well on **Linux with ROCm**, but on Windows performance is significantly slower due to DirectML. Expect 30-50% slower than equivalent NVIDIA GPUs.

3. **Intel Arc GPUs**: ⚠️ **Experimental support only**. Requires frequent driver updates. May have bugs and compatibility issues. Not recommended as primary GPU for AI generation.

4. **Apple Silicon**: ✅ Works well with MPS backend on macOS 12.3+. Good performance per watt, but generally slower than desktop NVIDIA GPUs. Great for MacBook users who want local generation.

5. **Minimum VRAM**: 4GB VRAM can work with DRAM extension, but 6GB+ is recommended for comfortable usage.

6. **Mixed GPU Systems**: If you have both integrated and dedicated GPUs, IRIS will automatically select the best one. You can verify GPU selection in the logs.

---

## 🛠️ Troubleshooting

### 🎮 GPU-Specific Issues

<details>
<summary><strong>NVIDIA: CUDA Out of Memory</strong></summary>

**Solutions**:
1. Enable DRAM Extension in Settings
2. Use smaller resolution (512x512 instead of 1024x1024)
3. Reduce steps (20-30 instead of 50)
4. Update to latest NVIDIA drivers
5. Close other GPU applications

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

</details>

<details>
<summary><strong>AMD: ROCm Not Detected</strong></summary>

**Solutions** (Linux):
1. Install ROCm 5.4+: https://rocm.docs.amd.com/
2. Add user to video group: `sudo usermod -a -G video $USER`
3. Reboot system
4. Verify: `rocm-smi`

**Solutions** (Windows - DirectML):
1. Update AMD Adrenalin drivers
2. Install DirectML: `pip install torch-directml`
3. Restart application

</details>

<details>
<summary><strong>Intel Arc: GPU Not Recognized</strong></summary>

**Solutions**:
1. Update Intel Arc drivers (latest version)
2. Windows: Install DirectML support
3. Linux: Install Intel Extension for PyTorch
4. Enable ReBAR in BIOS if available
5. Check GPU detection:

```bash
# Windows (DirectML)
python -c "import torch_directml; print(torch_directml.is_available())"

# Linux (XPU)
python -c "import intel_extension_for_pytorch as ipex; print(ipex.xpu.is_available())"
```

</details>

<details>
<summary><strong>Apple Silicon: MPS Backend Issues</strong></summary>

**Solutions**:
1. Ensure macOS 12.3+ (Monterey or newer)
2. Update to latest PyTorch version
3. Check MPS availability:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

4. If MPS fails, application will fall back to CPU automatically
5. For M1/M2 with 8GB: Use lower resolutions (512x512) and enable memory optimization

</details>

<details>
<summary><strong>Discord Bot Not Sending Images</strong></summary>

**Solution**: Ensure your `.env` file has the correct variable names:

```env
DISCORD_BOT_TOKEN=your_token_here
DISCORD_CHANNEL_NEW_IMAGES=channel_id_here
DISCORD_CHANNEL_VARIATIONS=channel_id_here
DISCORD_CHANNEL_UPSCALED=channel_id_here
```

Check that:
- ✅ All channel IDs are valid
- ✅ Bot has proper permissions
- ✅ Bot is invited to your server

</details>

<details>
<summary><strong>ModuleNotFoundError: No module named 'discord.ext'</strong></summary>

**Solution**: The bot file has been renamed from `discord.py` to `bot.py` to avoid conflicts.

```bash
python src/start.py bot  # Now uses bot.py
```

</details>

<details>
<summary><strong>"Out of Memory" Error</strong></summary>

**Solutions**:
1. Enable DRAM Extension in Settings
2. Use smaller resolution (512x512 instead of 1024x1024)
3. Reduce steps (20-30 instead of 50)
4. Close other GPU applications
5. For Apple Silicon: Ensure at least 8GB unified memory free

</details>

<details>
<summary><strong>Images Sending Twice to Discord</strong></summary>

**Solution**: Fixed with 6-second buffer delay. The bot now:
1. Waits 6 seconds after detecting new file
2. Double-checks if already sent
3. Sends only once per image

</details>

<details>
<summary><strong>Slow Performance on Any GPU</strong></summary>

**Solutions**:
1. Check if GPU is actually being used (check task manager/activity monitor)
2. Update to latest GPU drivers
3. Close background applications using GPU
4. Enable hardware acceleration in system settings
5. Verify correct PyTorch version for your GPU is installed
6. For laptops: Ensure power mode is set to "High Performance"
7. Check thermals - overheating GPUs throttle performance

</details>

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [📖 Setup Guide](docs/SETUP.md) | Complete installation and configuration |
| [🎭 Artifacts Guide](docs/ARTIFACTS.md) | Common generation issues and fixes |
| [🤝 Contributing](CONTRIBUTING.md) | How to contribute to the project |

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

<table>
<tr>
<td>

**AI & Models**
- [Stability AI](https://stability.ai/) - Stable Diffusion
- [Hugging Face](https://huggingface.co/) - Diffusers library
- [xinntao](https://github.com/xinntao/Real-ESRGAN) - Real-ESRGAN

</td>
<td>

**Frameworks**
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Discord.py](https://discordpy.readthedocs.io/) - Discord integration
- [PyTorch](https://pytorch.org/) - Deep learning

</td>
</tr>
</table>

---

## 💬 Support

<div align="center">

**Need help?** We're here for you!

[![GitHub Issues](https://img.shields.io/badge/Issues-Report%20Bug-red?style=for-the-badge&logo=github)](https://github.com/KaiTooast/iris-image-synthesis/issues)
[![GitHub Discussions](https://img.shields.io/badge/Discussions-Ask%20Questions-blue?style=for-the-badge&logo=github)](https://github.com/KaiTooast/iris-image-synthesis/discussions)
[![Documentation](https://img.shields.io/badge/Docs-Read%20More-green?style=for-the-badge&logo=readthedocs)](docs/SETUP.md)

</div>

---

<div align="center">

**Made with ❤️ using Stable Diffusion**

⭐ Star this repo if you find it useful!

</div>
