# Contributing to I.R.I.S. (Intelligent Rendering & Image Synthesis)

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

**TL;DR**: Be respectful, inclusive, and professional. I.R.I.S. is an AI image generation platform focused on creative content.

---

## How Can I Contribute?

### Bugs & Features

**Before submitting:**
- Check existing [Issues](https://github.com/yourusername/IRIS/issues)
- Test with latest version
- Gather relevant information (OS, GPU, Python version, CUDA version)

**Bug Report Template:**
```markdown
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
```

**Feature Request Template:**
```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Related to I.R.I.S.**
How does this feature fit with I.R.I.S.'s goals?

**Additional context**
Mockups, examples, or references.
```

### Code Contributions

We accept contributions for:
- Bug fixes
- New features (discuss in Issues first)
- Performance improvements
- UI/UX enhancements
- Better Discord integration
- Model improvements
- Memory optimization

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
```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/IRIS.git
cd IRIS
git remote add upstream https://github.com/ORIGINAL_OWNER/IRIS.git
```

### 2. Create Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

**Branch naming:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring
- `test/` - Adding tests

### 3. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 4. Configure for Development
Create a `.env` file in the root directory:

```env
# Discord Bot Configuration
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_BOT_OWNER_ID=your_discord_user_id
DISCORD_BOT_ID=bot_user_id

# Discord Channel IDs
DISCORD_CHANNEL_NEW_IMAGES=channel_id_for_new_images
DISCORD_CHANNEL_VARIATIONS=channel_id_for_variations
DISCORD_CHANNEL_UPSCALED=channel_id_for_upscaled

# Server Configuration
SERVER_PORT=8000
SERVER_HOST=0.0.0.0
```

### 5. Test Your Setup
```bash
# Start the web UI only
python src/start.py web

# Or start bot only
python src/start.py bot

# Or start everything
python src/start.py all
```

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

6. **Get Channel IDs:**
   - Enable Developer Mode in Discord (User Settings → App Settings → Advanced)
   - Right-click channel → Copy Channel ID
   - Add IDs to `.env` file

---

## Pull Request Process

### 1. Commit Your Changes
```bash
git add .
git commit -m "feat: add custom resolution support"
```

**Commit message format:**
```
<type>: <short summary>

<optional detailed description>

Fixes #<issue_number> (if applicable)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```bash
feat: add support for custom resolutions
fix: resolve Discord channel validation issue
docs: update GPU compatibility guide
refactor: optimize image sending performance
```

### 2. Push to Your Fork
```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

Go to GitHub and create a PR with:

**Title:** Clear, descriptive (e.g., "Add custom resolution support")

**Description:**
```markdown
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
```

---

## Coding Standards

### Python Style Guide

**Follow PEP 8** with these specifics:

```python
# Good - Clear, well-documented
def send_image_to_discord(filename: str, channel_id: int) -> bool:
    """
    Send generated image to specified Discord channel.
    
    Args:
        filename: Name of the image file to send
        channel_id: Discord channel ID to send to
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Implementation
        return True
    except Exception as e:
        logger.error(f"Failed to send {filename}: {e}")
        return False

# Bad - Unclear, no documentation
def send_img(f, c):
    return True
```

**Key Rules:**
- Use type hints
- Write docstrings for functions/classes
- Meaningful variable names
- Max line length: 100 characters
- Use f-strings for formatting
- Imports: stdlib → third-party → local

### File Organization

```
src/
├── api/
│   └── server.py           # FastAPI backend
├── services/
│   └── bot.py              # Discord bot with configurable channels
├── core/
│   ├── config.py           # Configuration management
│   ├── generator.py        # Image generation logic
│   └── model_loader.py     # AI model loading
├── utils/
│   ├── logger.py           # Logging system
│   └── file_manager.py     # File operations
└── start.py                # Entry point

static/
├── data/
│   ├── prompts_history.json    # Prompt logging
│   └── img_send.json           # Sent images tracking
└── config/                     # User configuration files

tests/
├── test_generation.py      # Image generation tests
├── test_discord.py         # Discord integration tests
└── test_api.py            # API endpoint tests
```

---

## Testing Guidelines

### Unit Tests

```python
# tests/test_discord.py
import pytest
from src.services.bot import load_sent_images, save_sent_image

def test_load_sent_images():
    """Test loading sent images from JSON"""
    load_sent_images()
    # Verify data loaded correctly
    pass

def test_save_sent_image():
    """Test saving sent image to JSON"""
    save_sent_image("test.png", "https://discord.com/channels/...")
    # Verify file was saved
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_discord.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v
```

---

## Project Structure

I.R.I.S. is structured for easy maintenance and GPU compatibility:

```
IRIS/
├── src/
│   ├── api/                    # API layer
│   ├── services/               # External services (Discord)
│   ├── core/                   # Core business logic
│   ├── utils/                  # Utilities
│   └── start.py                # Entry point
├── static/
│   ├── data/                   # JSON logs
│   ├── config/                 # Configuration (not in Git)
│   └── assets/                 # Images and styling
├── outputs/                    # Generated images
├── Logs/                       # Application logs
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in Git)
├── CONTRIBUTING.md             # This file
├── LICENSE                     # License
└── README.md                   # Project overview
```

---

## Getting Help

**Stuck?** We're here to help!

- Chat in [GitHub Discussions](https://github.com/yourusername/IRIS/discussions)
- Report in [Issues](https://github.com/yourusername/IRIS/issues)
- Check [docs/SETUP.md](docs/SETUP.md) for installation help
- Check [README.md](README.md) for project overview

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

Thank you for contributing to I.R.I.S.! Every contribution helps improve AI image generation for everyone. Let's build something amazing together!
