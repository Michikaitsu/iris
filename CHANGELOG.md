# Changelog

All notable changes to I.R.I.S. will be documented in this file.

---

## [1.0.0] - 2026-01-13

### 🎉 First Stable Release

This is the first stable release of I.R.I.S., featuring a complete local AI image generation platform with dual frontend support.

---

### ✨ New Features

#### Dual Frontend System
- **React Frontend** — Modern, responsive UI built with React 18, Vite, and Tailwind CSS
- **HTML Frontend** — Classic lightweight UI for maximum compatibility
- Both frontends share the same backend API and stay synchronized

#### Multi-GPU Support
- **NVIDIA CUDA** — Full support with automatic detection
- **AMD ROCm** — Experimental support for AMD GPUs
- **Intel Arc XPU** — Experimental support via oneAPI
- **Apple Silicon MPS** — Experimental support for M1/M2/M3 chips
- **CPU Fallback** — Always available when no GPU is detected
- **Runtime Device Switching** — Switch between GPU and CPU without restart

#### Advanced Upscaling System
- **Real-ESRGAN** — AI-enhanced upscaling, best quality
- **Anime v3** — Fast upscaling optimized for anime images
- **Tile Mode** — Tile-based processing for JPEG/compressed images
- **Lanczos** — CPU-based fast fallback
- Support for 2x and 4x upscaling

#### Hardware Monitoring
- Real-time CPU usage and frequency
- RAM usage (used/total/percentage)
- GPU power draw monitoring
- VRAM usage tracking

#### Discord Integration
- **Discord Bot** — Auto-post generated images to Discord channels
- **Rich Presence** — Show generation progress in Discord status
- Start/stop bot from Settings page
- Separate channels for new images, variations, and upscaled images

#### NSFW Filter System
- Configurable prompt-based filtering
- Three strength levels (Minimal, Standard, Strict)
- Can be completely disabled via Settings
- Filter status synced between frontends

#### Generation Features
- Real-time progress streaming via WebSocket
- Continuous generation timer (counts total time)
- ETA calculation based on progress
- Benchmark system for time estimation
- Quality presets (Draft, Standard, High Quality, Ultra)
- Custom resolution support (256×256 to 2048×2048)
- Seed locking and random seed generation

#### Prompt History
- Server-side persistent history (`prompts_history.json`)
- Deduplicated entries (unique prompts only)
- Click to restore previous generation settings
- Shared between both frontends

#### DRAM Extension
- Automatic system RAM fallback for low-VRAM GPUs
- Configurable VRAM threshold
- Sequential CPU offloading for 4GB GPUs
- Enables generation on GTX 1650 and similar cards

---

### 🛠️ Technical Improvements

- FastAPI backend with async support
- WebSocket-based real-time communication
- Zustand state management in React frontend
- Tailwind CSS with custom IRIS theme
- Proper error handling and logging
- torchvision compatibility patch for Real-ESRGAN

---

### 📦 Supported Models

- **Anime Kawai** — Ojimi/anime-kawai-diffusion
- **Stable Diffusion 2.1** — stabilityai/stable-diffusion-2-1
- **Stable Diffusion 3.5** — stabilityai/stable-diffusion-3.5-medium
- **FLUX.1 Schnell** — black-forest-labs/FLUX.1-schnell
- **OpenJourney** — prompthero/openjourney
- **Pixel Art** — nitrosocke/pixel-art-diffusion
- **Pony Diffusion v6** — AstraliteHeart/pony-diffusion-v6-xl
- **Anything v5** — stablediffusionapi/anything-v5
- **Animagine XL 3.1** — CagliostroResearchGroup/animagine-xl-3.1
- **AbyssOrangeMix3** — WarriorMama777/AbyssOrangeMix3
- **Counterfeit v3** — stablediffusionapi/counterfeit-v30

---

### 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8+ GB |
| System RAM | 8 GB | 16 GB |
| Python | 3.9 | 3.10-3.11 |
| Storage | 20 GB | 50 GB |

---

### 📝 Notes

- Developed and tested on NVIDIA GTX 1650 (4GB VRAM)
- DRAM Extension enables generation on low-VRAM GPUs
- First generation may take longer due to model download
- React frontend requires `npm install` in `frontend-react/`

---

### 🙏 Acknowledgments

Built with:
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Diffusers](https://github.com/huggingface/diffusers)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
