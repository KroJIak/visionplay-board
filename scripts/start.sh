#!/bin/bash

# VisionPlay Board - Quick Start Script with proper pyenv initialization

echo "Starting VisionPlay Board..."

# Initialize pyenv properly
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

# Initialize pyenv shell integration
eval "$(pyenv init -)"
eval "$(pyenv init --path)"

# Fix Wayland warning for OpenCV
export QT_QPA_PLATFORM=xcb

# Set Python version
pyenv shell 3.11.11

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python not found. Please check pyenv installation."
    exit 1
fi

echo "Using Python version: $(python --version)"

# Run the application
echo "Starting application..."
cd "$(dirname "$0")/.."
python main.py
