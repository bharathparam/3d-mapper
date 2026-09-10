#!/usr/bin/env bash
# Quick setup for macOS + Apple Silicon
set -e

echo "=== Drone 3D Reconstruction — Setup ==="
echo ""

# Check Homebrew
if ! command -v brew &>/dev/null; then
    echo "❌ Homebrew not found. Install from https://brew.sh"
    exit 1
fi

# System deps
echo "[1/4] Installing system dependencies..."
brew install colmap ffmpeg

# Python venv
echo "[2/4] Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Frontend
echo "[3/4] Installing frontend dependencies..."
cd frontend && npm install && cd ..

# Projects directory
echo "[4/4] Creating runtime directories..."
mkdir -p projects

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start:"
echo "  source .venv/bin/activate"
echo "  python run.py                    # starts backend"
echo "  cd frontend && npm run dev       # starts frontend (new terminal)"
echo "  Open: http://localhost:5173"
