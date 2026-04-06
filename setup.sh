#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Bike Fit Analyzer — Setup Script (macOS, Apple Silicon)
# Run once: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "🚴  Bike Fit Analyzer — Setup"
echo "────────────────────────────────────────────────────────────"

# Check Python version
PYTHON=$(which python3)
PY_VER=$($PYTHON --version 2>&1)
echo "✅  Python: $PY_VER"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "🔧  Creating virtual environment..."
    $PYTHON -m venv .venv
fi

# Activate venv
source .venv/bin/activate
echo "✅  Virtual environment activated"

# Upgrade pip
pip install --upgrade pip --quiet

# Install dependencies
echo "📦  Installing dependencies (this may take a minute)..."
pip install -r requirements.txt --quiet

echo ""
echo "✅  Setup complete!"
echo ""
echo "────────────────────────────────────────────────────────────"
echo "  To launch the desktop app:"
echo "    source .venv/bin/activate"
echo "    python main.py"
echo ""
echo "  To run headless on a video:"
echo "    python main.py --cli path/to/video.mp4"
echo "────────────────────────────────────────────────────────────"
echo ""
