<div align="center">

# 🧠 L.O.O.M.

### Local Operator of Open Minds

<p>
  <strong>AI Image Generation System with Web UI and Discord Bot Integration</strong>
</p>

<p>
  <a href="#features"><img src="https://img.shields.io/badge/Stable_Diffusion-XL-blue?style=for-the-badge" alt="Stable Diffusion XL"></a>
  <a href="#features"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="#features"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="#features"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
</p>

<p>
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#project-structure">Structure</a>
</p>

---

</div>

## ⚡ Quick Start

<table>
<tr>
<td width="33%">

### 🌐 Web UI Only

```bash
python src/start.py web
```

Access at: [localhost:8000](http://localhost:8000)

</td>
<td width="33%">

### 🤖 Discord Bot Only

```bash
python src/start.py bot
```

</td>
<td width="33%">

### 🚀 Both Services

```bash
python src/start.py all
```

</td>
</tr>
</table>

---

## ✨ Features

<table>
<tr>
<td>🎨 <strong>Stable Diffusion XL</strong></td>
<td>State-of-the-art image generation</td>
</tr>
<tr>
<td>🌐 <strong>Modern Web UI</strong></td>
<td>Clean, responsive interface</td>
</tr>
<tr>
<td>🤖 <strong>Discord Integration</strong></td>
<td>Generate images directly in Discord</td>
</tr>
<tr>
<td>📊 <strong>Real-time Progress</strong></td>
<td>Live generation tracking</td>
</tr>
<tr>
<td>🖼️ <strong>Image Gallery</strong></td>
<td>Browse with full metadata</td>
</tr>
<tr>
<td>⬆️ <strong>Real-ESRGAN Upscaling</strong></td>
<td>Enhance image resolution</td>
</tr>
<tr>
<td>🎯 <strong>Quality Presets</strong></td>
<td>Multiple generation settings</td>
</tr>
<tr>
<td>📱 <strong>Mobile Wallpapers</strong></td>
<td>Optimized aspect ratios</td>
</tr>
</table>

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/KaiTooast/loom.git

# Navigate to directory
cd loom

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
├── src/
│   ├── start.py              # Universal starter script
│   ├── backend/              # Python backend services
│   │   ├── web_server.py     # FastAPI web server
│   │   ├── discord_bot.py    # Discord bot
│   │   └── logger.py         # Logging utilities
│   └── frontend/             # HTML frontend
│       ├── index.html        # Main generator UI
│       └── gallery.html      # Image gallery
├── static/                   # Static assets
├── outputs/                  # Generated images
└── requirements.txt          # Python dependencies
```

---

<div align="center">

**Made with ❤️ for the AI Art Community**

</div>
