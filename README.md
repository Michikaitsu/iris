<div align="center">

# 🎨 I.R.I.S.
### Intelligent Rendering & Image Synthesis

*Local AI Image Generation Powered by Stable Diffusion*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)
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

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

```bash
✅ Python 3.9, 3.10, or 3.11
✅ NVIDIA GPU with 4GB+ VRAM (or CPU mode)
✅ CUDA 11.8 or 12.1 (for GPU acceleration)
```

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
```

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

## 🛠️ Troubleshooting

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

</details>

<details>
<summary><strong>Images Sending Twice to Discord</strong></summary>

**Solution**: Fixed with 6-second buffer delay. The bot now:
1. Waits 6 seconds after detecting new file
2. Double-checks if already sent
3. Sends only once per image

</details>

---

## 💻 System Requirements

<table>
<tr>
<th>Tier</th>
<th>GPU</th>
<th>VRAM</th>
<th>RAM</th>
<th>Performance</th>
</tr>
<tr>
<td><strong>Minimum</strong></td>
<td>GTX 1650</td>
<td>4GB</td>
<td>8GB</td>
<td>~6 min per 512x768</td>
</tr>
<tr>
<td><strong>Recommended</strong></td>
<td>RTX 3060</td>
<td>12GB</td>
<td>16GB</td>
<td>~2 min per 512x768</td>
</tr>
<tr>
<td><strong>High-End</strong></td>
<td>RTX 4090</td>
<td>24GB</td>
<td>32GB</td>
<td>~30 sec per 512x768</td>
</tr>
</table>

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

[![GitHub Issues](https://img.shields.io/badge/Issues-Report%20Bug-red?style=for-the-badge&logo=github)](https://github.com/KaiTooast/Iris-Image-Synthesis/issues)
[![GitHub Discussions](https://img.shields.io/badge/Discussions-Ask%20Questions-blue?style=for-the-badge&logo=github)](https://github.com/KaiTooast/iris-image-synthesis/discussions)
[![Documentation](https://img.shields.io/badge/Docs-Read%20More-green?style=for-the-badge&logo=readthedocs)](docs/SETUP.md)

</div>

---

<div align="center">

**Made with ❤️ using Stable Diffusion**

⭐ Star this repo if you find it useful!

</div>
