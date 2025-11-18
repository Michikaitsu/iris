# L.O.O.M. - Local Operator of Open Minds

AI Image Generation System with Web UI and Discord Bot Integration

## Quick Start

### Start Web UI Only
\`\`\`bash
python src/start.py web
\`\`\`
Access at: http://localhost:8000

### Start Discord Bot Only
\`\`\`bash
python src/start.py bot
\`\`\`

### Start Both Services
\`\`\`bash
python src/start.py all
\`\`\`

## Project Structure

\`\`\`
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
\`\`\`

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Features

- 🎨 Stable Diffusion XL image generation
- 🌐 Modern web interface
- 🤖 Discord bot integration
- 📊 Real-time progress tracking
- 🖼️ Image gallery with metadata
- ⬆️ Real-ESRGAN upscaling
- 🎯 Multiple quality presets
- 📱 Mobile wallpaper support
