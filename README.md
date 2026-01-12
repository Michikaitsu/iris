# 🌌 I.R.I.S.
### Intelligent Rendering & Image Synthesis

**A modular, local-first AI image generation engine — built to be forked, extended, and owned.**

I.R.I.S. is an **open-source AI image generation platform** designed as a **foundation**, not a locked product.  
Think of it as **Linux for AI image generation**:

> You get a fully working system —  
> but *you* decide how it evolves.

⚠️ **Runs entirely on your own hardware**  
No cloud. No accounts. No telemetry. No vendor lock-in.

---

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-green)
![WebSockets](https://img.shields.io/badge/WebSockets-realtime-purple)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/status-active%20development-orange)

---

## ✨ Core Philosophy

- 🧠 **Local-first** — everything runs on your machine
- 🔓 **Open Source** — modify, fork, redistribute
- 🧩 **Modular architecture** — UI, backend, models are replaceable
- 🧪 **Experiment-friendly** — designed for tinkering & research
- 🚀 **Production-capable** — APIs, WebSockets, scaling-ready

This repository provides a **fully functional reference implementation**, not a closed product.

---

## 🖼️ Feature Overview

### Core Features
- Modern **Web UI** (Generate, Gallery, Settings)
- **Multiple AI models** (anime, realistic, pixel art, SDXL)
- **Text-to-Image** generation
- **Real-time progress streaming** (WebSockets)
- **Prompt & image history logging**
- **NSFW prompt filtering** (configurable strength)
- **CPU & low-VRAM GPU support**

### Advanced Features
- **DRAM Extension** (system RAM fallback for low VRAM GPUs)
- **Custom resolutions** (512×512 → 1080×1920)
- **CFG scale & sampling control**
- **Real-ESRGAN & Lanczos upscaling** (2× / 4×)
- **Live gallery with session history**
- **Automatic VRAM safety adjustments**
- **Optional Discord bot integration**

---

## 🚀 Quick Start

### Requirements
- Python 3.9 – 3.11
- GPU recommended (4 GB VRAM minimum)
- CUDA 11.8+ (optional, CPU mode supported)

### Installation
```bash
git clone https://github.com/KaiTooast/iris.git
cd iris

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt

# Optional: Copy environment template
cp .env.example .env
```

### Run
```bash
# Auto-start based on settings.json
python src/start.py

# Force start without Discord bot
python src/start.py --no-bot
```

🌐 Open: **[http://localhost:8000](http://localhost:8000)**

---

## 🧩 Project Structure

```
iris/
├── src/                    # Backend & core logic
│   ├── api/                # FastAPI server & routes
│   │   ├── server.py       # Main server
│   │   ├── middleware/     # Rate limiting
│   │   ├── routes/         # API endpoints
│   │   └── services/       # NSFW filter, pipeline
│   ├── core/               # Model loading & generation
│   │   ├── config.py       # Configuration
│   │   ├── model_loader.py # Model management
│   │   ├── generator.py    # Image generation
│   │   └── exceptions.py   # Custom exceptions
│   ├── services/           # Optional services
│   │   ├── upscaler.py     # Image upscaling
│   │   └── bot.py          # Discord bot
│   ├── utils/              # Utilities
│   │   ├── logger.py       # Logging
│   │   └── file_manager.py # File operations
│   └── start.py            # Entry point
│
├── frontend/               # Web UI
│   ├── index.html          # Landing page
│   ├── generate.html       # Generation UI
│   ├── gallery.html        # Image gallery
│   └── settings.html       # Settings page
│
├── static/                 # Static assets
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript
│   ├── config/             # Bot config files
│   └── data/               # Runtime data (history)
│
├── assets/                 # UI assets
│   ├── fav.ico             # Favicon
│   └── thumbnails/         # Model previews
│
├── tests/                  # Test suite
├── outputs/                # Generated images
├── Logs/                   # Runtime logs
├── docs/                   # Documentation
│
├── .env.example            # Environment template
├── settings.json           # Runtime settings
└── requirements.txt        # Python dependencies
```

> ⚠️ **Note**
>
> This project structure was taken **directly from the active development and testing environment**.
> It reflects the real layout used during day-to-day coding, experimentation and debugging.
>
> Some folders (e.g. logs, outputs, cached data) are intentionally kept in the repository
> to show how the system behaves in practice and how components interact at runtime.
> 
> The structure is intentionally not over-simplified.
> It represents a real-world, evolving codebase rather than a polished showcase.

---

## ⚙️ Configuration

### settings.json
```json
{
  "dramEnabled": true,
  "vramThreshold": 6,
  "maxDram": 8,
  "nsfwStrength": 2,
  "discordEnabled": false
}
```

### .env (optional)
```env
HOST=0.0.0.0
PORT=8000
DEFAULT_MODEL=anime_kawai

# Discord Bot (optional)
DISCORD_BOT_TOKEN=your_token
DISCORD_CHANNEL_NEW_IMAGES=channel_id
DISCORD_CHANNEL_VARIATIONS=channel_id
DISCORD_CHANNEL_UPSCALED=channel_id
```

---

## 🖥️ Hardware Reference

| Tier | GPU | VRAM | Notes |
|------|-----|------|-------|
| **Minimum** | NVIDIA GTX 1650 | 4 GB | The birthplace. Small models, DRAM Extension recommended. |
| **Sweet Spot** | **Intel Arc B580** | 12 GB | **Best value for money.** |
| **Advanced** | NVIDIA RTX 4070 Super | 12 GB | Faster inference, still VRAM-limited. |
| **Professional** | NVIDIA RTX 3090 Ti / 4090 | 24 GB | No-compromise local AI & SDXL. |
| **God Tier** | **NVIDIA RTX 5090** | 32 GB | Industrial scale. (Overkill for most) |

> 💡 **Developer Note:** I.R.I.S. was **developed and tested on a GTX 1650**, proving functionality on low-end hardware. We optimize for best hardware per dollar, not expensive branding.

---

## 🔌 API & WebSocket Support

- REST API for generation, gallery, system info
- WebSocket streams for:
  - Generation progress
  - Gallery updates
  - Multi-page synchronization

Perfect for **custom frontends**, automation, or external clients.

---

## 🛡️ Safety

- Prompt-based NSFW filtering
- Three strength levels (Minimal, Standard, Strict)
- Category-based detection
- Easily extendable or disableable

---

## 🧠 Designed for Modification

You are explicitly encouraged to:

* Replace the frontend entirely
* Add your own models or pipelines
* Build a token or subscription system
* Deploy in a private or public datacenter
* Run on NVIDIA, AMD, or Intel GPUs (experimental)
* Fork this into a commercial or closed product

**I.R.I.S. does not enforce a business model.**

---

## � License

**Creative Commons Attribution 4.0 (CC BY 4.0)**

You may use, modify, redistribute, and commercialize this project —
**attribution is required.**

See `LICENSE` for details.

---

## 🤝 Contributing

Contributions are welcome — from small fixes to major architectural changes.

Please read **CONTRIBUTING.md** before submitting a pull request.

---

## 🌍 Final Note

I.R.I.S. is not built to compete with cloud AI platforms.

It exists to **give control back** to developers and creators.

If you value:
- ownership over subscriptions
- experimentation over lock-in
- transparency over black boxes

then this project is for you.
