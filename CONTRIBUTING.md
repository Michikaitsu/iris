# Contributing to L.O.O.M. (Local Operator of Open Minds)

First off, thank you for considering contributing! It's people like you that make this project better.

---

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Setup](#development-setup)
4. [Pull Request Process](#pull-request-process)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Configuration & Discord Setup](#configuration--discord-setup)

---

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

**TL;DR**: Be respectful, inclusive, and professional. L.O.O.M. is dedicated to anime art generation and has built-in safety filters for NSFW content.

---

## How Can I Contribute?

### Bugs & Features

**Before submitting:**
- Check existing [Issues](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/issues)
- Test with latest version
- Gather relevant information (OS, GPU, Python version, CUDA version)

**Bug Report Template:**
\`\`\`markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g., Windows 11]
- GPU: [e.g., RTX 3060]
- Python Version: [e.g., 3.11]
- CUDA Version: [e.g., 12.1]
- VRAM: [e.g., 6GB]

**Additional context**
Any other relevant information.
\`\`\`

**Feature Request Template:**
\`\`\`markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Related to L.O.O.M.**
How does this feature fit with anime art generation and L.O.O.M.'s goals?

**Additional context**
Mockups, examples, or references.
\`\`\`

### Code Contributions

We accept contributions for:
- Bug fixes
- New features (discuss in Issues first)
- Performance improvements
- UI/UX enhancements
- Better Discord integration
- Anime model improvements
- Memory optimization

**We do NOT accept:**
- NSFW content generation features
- Removal of safety filters
- Cryptocurrency mining
- Malicious code
- Code without tests (for core features)

### Documentation Improvements

Documentation improvements are always welcome!
- Fix typos
- Clarify instructions
- Add examples
- Document GPU compatibility
- Improve setup guides

---

## Development Setup

### 1. Fork & Clone
\`\`\`bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Local-Operator-of-Open-Minds.git
cd Local-Operator-of-Open-Minds
git remote add upstream https://github.com/KaiTooast/Local-Operator-of-Open-Minds.git
\`\`\`

### 2. Create Branch
\`\`\`bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
\`\`\`

**Branch naming:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring
- `test/` - Adding tests

### 3. Install Dependencies
\`\`\`bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
\`\`\`

### 4. Configure for Development
\`\`\`bash
# Create config files
mkdir -p static/config
mkdir -p static/data

# Add your bot token (get from Discord Developer Portal)
echo "YOUR_BOT_TOKEN_HERE" > static/config/bot_token.txt

# Add your Discord user ID
echo "YOUR_DISCORD_ID_HERE" > static/config/bot_owner_id.txt

# Add bot's user ID
echo "BOT_USER_ID_HERE" > static/config/bot_id.txt

# Configure Discord channel IDs (see Configuration & Discord Setup section)
echo "new=CHANNEL_ID_HERE" > static/config/channel_ids.txt
echo "variations=CHANNEL_ID_HERE" >> static/config/channel_ids.txt
echo "upscaled=CHANNEL_ID_HERE" >> static/config/channel_ids.txt
\`\`\`

### 5. Test Your Setup
\`\`\`bash
# Start the web UI only
python src/start.py web

# Or start bot only
python src/start.py bot

# Or start everything
python src/start.py all
\`\`\`

---

## Configuration & Discord Setup

### Discord Bot Setup

1. **Create Application** at [Discord Developer Portal](https://discord.com/developers/applications)
2. **Enable Intents:**
   - Message Content Intent
   - Server Members Intent
   - Guild Members Intent

3. **Get IDs:**
   - Application ID (use as Bot User ID)
   - Your Personal Discord ID (right-click user → Copy User ID)

4. **Create Discord Server** for testing (if needed)

5. **Create Channels:**
   - `#generated-images` (for new images)
   - `#variations` (for image variations)
   - `#upscaled` (for upscaled images)

6. **Configure L.O.O.M:**
   \`\`\`bash
   # Edit static/config/channel_ids.txt
   new=1234567890          # Replace with #generated-images channel ID
   variations=0987654321   # Replace with #variations channel ID
   upscaled=1122334455     # Replace with #upscaled channel ID
   \`\`\`

7. **Get Channel IDs:**
   - Enable Developer Mode in Discord (User Settings → App Settings → Advanced)
   - Right-click channel → Copy Channel ID

### Configuration Files

All configuration is in `static/config/`:
\`\`\`
static/config/
├── bot_token.txt        # Discord bot token
├── bot_owner_id.txt     # Your Discord user ID
├── bot_id.txt           # Bot's user ID
└── channel_ids.txt      # Discord channel IDs (configurable)
\`\`\`

**channel_ids.txt format:**
\`\`\`ini
# New generated images
new=123456789012345678

# Variations of existing images
variations=234567890123456789

# Upscaled images
upscaled=345678901234567890
\`\`\`

---

## Pull Request Process

### 1. Commit Your Changes
\`\`\`bash
git add .
git commit -m "feat: add anime model selector"
\`\`\`

**Commit message format:**
\`\`\`
<type>: <short summary>

<optional detailed description>

Fixes #<issue_number> (if applicable)
\`\`\`

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
\`\`\`bash
feat: add support for custom negative prompts
fix: resolve Discord channel validation issue
docs: update GPU compatibility guide
refactor: optimize NSFW filter performance
\`\`\`

### 2. Push to Your Fork
\`\`\`bash
git push origin feature/your-feature-name
\`\`\`

### 3. Create Pull Request

Go to GitHub and create a PR with:

**Title:** Clear, descriptive (e.g., "Add anime model selector to UI")

**Description:**
\`\`\`markdown
## Description
Brief explanation of changes and why they're needed.

## Related Issue
Fixes #<issue_number> (if applicable)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How you tested this:
- [ ] Tested locally on Windows/Linux/Mac
- [ ] Tested with GPU
- [ ] Tested Discord integration
- [ ] Added unit tests

## Screenshots
If UI changes, add before/after screenshots.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed the code
- [ ] Commented hard-to-understand areas
- [ ] Updated documentation
- [ ] No new warnings
- [ ] NSFW safety filters still active
- [ ] Discord channel configuration is flexible
\`\`\`

### 4. Code Review
- Respond to feedback promptly
- Make requested changes
- Push updates to same branch

### 5. Merge
Once approved, maintainers will merge your PR!

---

## Coding Standards

### Python Style Guide

**Follow PEP 8** with these specifics:

\`\`\`python
# Good - Clear, well-documented
def apply_nsfw_filter(prompt: str, negative_prompt: str) -> str:
    """
    Apply NSFW safety filter to prompt and negative prompt.
    
    Always appends safety keywords to negative prompt to prevent
    unsafe content generation.
    
    Args:
        prompt: Main generation prompt
        negative_prompt: Safety filter prompt
    
    Returns:
        Filtered negative prompt with safety keywords appended
    """
    nsfw_filter = "nsfw, nude, naked, explicit, sexual"
    return f"{negative_prompt}, {nsfw_filter}".strip()

# Bad - Unclear, no documentation
def filter_prompt(p, n):
    return f"{n}, nsfw, nude".strip()
\`\`\`

**Key Rules:**
- Use type hints
- Write docstrings for functions/classes
- Meaningful variable names
- Max line length: 100 characters
- Use f-strings for formatting
- Imports: stdlib → third-party → local
- Always include NSFW safety considerations

### File Organization

\`\`\`
src/backend/
├── discord_bot.py       # Discord integration (channel IDs configurable)
├── web_server.py        # FastAPI backend with safety filters
├── logger.py            # Logging system
└── __init__.py

src/frontend/
├── index.html           # Main UI
└── gallery.html         # Image gallery

static/config/
├── bot_token.txt        # Discord bot token (not in Git)
├── bot_owner_id.txt     # Owner Discord ID (not in Git)
├── bot_id.txt           # Bot user ID (not in Git)
└── channel_ids.txt      # Channel IDs (not in Git)

tests/
├── test_generation.py   # Image generation tests
├── test_discord.py      # Discord integration tests
└── test_safety.py       # NSFW filter tests
\`\`\`

### Key Features to Preserve

- **NSFW Safety:** Always verify safety filters are active
- **Configurable Discord Channels:** Channel IDs must be read from `static/config/channel_ids.txt`
- **Memory Optimization:** DRAM extension for low-VRAM GPUs
- **GPU Compatibility:** Support RTX, GTX, and custom GPUs

---

## Testing Guidelines

### Unit Tests

\`\`\`python
# tests/test_safety.py
import pytest
from src.backend.web_server import NSFW_NEGATIVE_PROMPT

def test_nsfw_negative_prompt_exists():
    """Test that NSFW negative prompt is defined"""
    assert NSFW_NEGATIVE_PROMPT is not None
    assert "nsfw" in NSFW_NEGATIVE_PROMPT.lower()
    assert "nude" in NSFW_NEGATIVE_PROMPT.lower()

def test_discord_channel_ids_configurable():
    """Test that channel IDs are read from config"""
    # Should not have hardcoded channel IDs
    # Channel IDs should come from static/config/channel_ids.txt
    pass
\`\`\`

### Integration Tests

\`\`\`python
# tests/test_discord.py
import pytest

def test_channel_configuration():
    """Test Discord channel configuration"""
    # Verify channels are read from config file
    # Verify all three channels are configured
    pass

def test_image_sending():
    """Test image sending to Discord"""
    # Mock Discord client
    # Verify images are sent to correct channels
    pass
\`\`\`

### Manual Testing

\`\`\`bash
# Test generation with NSFW filter active
python src/start.py web
# Navigate to http://localhost:8000
# Try various prompts, verify NSFW filter is applied

# Test Discord bot with custom channels
python src/start.py bot
# Verify bot connects to configured Discord channels
# Verify images are sent to correct channels

# Test channel ID configuration
# Edit static/config/channel_ids.txt
# Verify bot uses new channels
\`\`\`

### Running Tests

\`\`\`bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_safety.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v
\`\`\`

---

## Project Structure

L.O.O.M. is structured for easy maintenance and GPU compatibility:

\`\`\`
Local-Operator-of-Open-Minds/
├── src/
│   ├── backend/
│   │   ├── discord_bot.py      # Reads channel IDs from config
│   │   ├── web_server.py        # FastAPI + NSFW safety filters
│   │   └── logger.py            # Centralized logging
│   ├── frontend/
│   │   └── index.html           # Web UI
│   └── start.py                 # Universal starter
├── static/
│   ├── config/                  # Not in Git (user-specific)
│   ├── data/                    # Logs and history
│   ├── assets/                  # Images and styling
│   └── css/js/                  # Frontend files
├── outputs/                     # Generated images
├── Logs/                        # Session logs
├── docs/                        # Documentation
├── requirements.txt             # Python dependencies
├── CONTRIBUTING.md              # This file
├── LICENSE                      # MIT License
└── README.md                    # Project overview
\`\`\`

---

## GPU Compatibility

L.O.O.M. supports various GPUs:

- **NVIDIA:**
  - RTX Series (RTX 2060+): Float16 support
  - GTX Series (GTX 1060+): Float32 mode
  - A100, V100, T4: Tensor cores + optimizations

- **AMD:** Limited (Compute Capability)
- **Apple Silicon:** MPS acceleration
- **Intel Arc:** XPU support
- **CPU Mode:** Supported (slow)

### VRAM Requirements

- **4GB VRAM:** 512x512 with DRAM extension
- **6GB VRAM:** 512x768 or 720x1280
- **8GB VRAM:** 1024x768 or higher
- **10GB+:** 1080x1920 mobile wallpapers

---

## Performance Tips

1. **DRAM Extension:** Enable for GPUs with ≤4GB VRAM
2. **Batch Processing:** Process multiple generations efficiently
3. **Memory Optimization:** VAE slicing/tiling enabled by default
4. **GPU Monitoring:** Real-time temp and VRAM monitoring

---

## Getting Help

**Stuck?** We're here to help!

- Chat in [GitHub Discussions](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/discussions)
- Report in [Issues](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/issues)
- Check [docs/SETUP.md](docs/SETUP.md) for installation help
- Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues

---

## Recognition

Contributors are recognized in:
- README.md Contributors section
- Release notes
- GitHub Contributors graph

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Additional Resources

- [L.O.O.M. README](README.md)
- [Setup Guide](docs/SETUP.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Discord.py Documentation](https://discordpy.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Diffusers Library](https://huggingface.co/docs/diffusers/)

---

Thank you for contributing to L.O.O.M.! Every contribution helps improve anime art generation for everyone. Let's build something amazing together! 🎨
