# I.R.I.S. v1.0.0 — First Stable Release

## 🎉 Summary

**I.R.I.S.** (Intelligent Rendering & Image Synthesis) is a local-first, open-source AI image generation platform. This is the first stable release, featuring a complete generation pipeline with dual frontend support.

**Key Highlights:**
- 🖼️ **Dual Frontend** — Modern React UI + Classic HTML UI
- 🎮 **Multi-GPU Support** — NVIDIA, AMD, Intel Arc, Apple Silicon, CPU
- 🔧 **4 Upscaler Methods** — Real-ESRGAN, Anime v3, Tile Mode, Lanczos
- 📊 **Hardware Monitoring** — CPU, RAM, GPU power tracking
- 🤖 **Discord Integration** — Bot + Rich Presence
- 🛡️ **Configurable NSFW Filter** — Can be disabled
- 💾 **DRAM Extension** — Run on 4GB VRAM GPUs

---

## 📥 Installation

```bash
git clone https://github.com/KaiTooast/iris.git
cd iris
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
.\venv\Scripts\python.exe src/start.py
```

Open: **http://localhost:8000**

---

## 🆕 What's New

### Frontends
- ✅ React 18 frontend with Vite + Tailwind CSS
- ✅ Zustand state management
- ✅ Real-time WebSocket progress streaming
- ✅ Continuous generation timer
- ✅ Benchmark-based ETA estimation

### Backend
- ✅ Multi-GPU device detection and switching
- ✅ Hardware monitoring API (CPU, RAM, GPU power)
- ✅ Multiple upscaler methods with on-demand loading
- ✅ Server-side prompt history
- ✅ Discord bot start/stop from UI
- ✅ NSFW filter toggle (can be disabled)

### Upscaling
- ✅ Real-ESRGAN — Best quality
- ✅ Anime v3 — Fast, anime-optimized
- ✅ Tile Mode — For compressed/JPEG images
- ✅ Lanczos — CPU fallback

### Hardware Support
- ✅ NVIDIA CUDA (full support)
- ✅ AMD ROCm (experimental)
- ✅ Intel Arc XPU (experimental)
- ✅ Apple Silicon MPS (experimental)
- ✅ CPU fallback (always available)

---

## 📋 Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.10-3.11 |
| GPU VRAM | 4 GB | 8+ GB |
| System RAM | 8 GB | 16 GB |
| Storage | 20 GB | 50 GB |

---

## 🙏 Credits

Developed and tested on **NVIDIA GTX 1650** (4GB VRAM).

Built with Stable Diffusion, Diffusers, Real-ESRGAN, FastAPI, React, and Tailwind CSS.

---

## 📄 License

**CC BY 4.0** — Use, modify, redistribute with attribution.
