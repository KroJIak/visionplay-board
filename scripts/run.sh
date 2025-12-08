#!/bin/bash

# VisionPlay Board - Quick Start Script

echo "Starting VisionPlay Board..."

# Initialize pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Fix Wayland warning for OpenCV
export QT_QPA_PLATFORM=xcb

# Ensure pyenv shell integration is enabled
eval "$(pyenv init --path)"

# Check if Python 3.11.11 is available
if ! pyenv versions | grep -q "3.11.11"; then
    echo "Python 3.11.11 not found. Installing..."
    pyenv install 3.11.11
fi

# Ensure we're using the correct Python version
pyenv shell 3.11.11

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python not found. Please check pyenv installation."
    exit 1
fi

echo "Using Python version: $(python --version)"

# Change to project root directory
cd "$(dirname "$0")/.."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the application
echo "Starting application..."
python main.py
