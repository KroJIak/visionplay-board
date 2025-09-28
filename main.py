#!/usr/bin/env python3
"""
VisionPlay Board - Interactive Computer Vision Game Application
Main entry point for the application.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix Wayland warning for OpenCV
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.app import VisionPlayApp

def main():
    """Main entry point for the application."""
    try:
        app = VisionPlayApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
