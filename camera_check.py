#!/usr/bin/env python3
"""
Simple camera check without OpenCV dependency.
"""

import os
import glob
import subprocess
import time

def check_video_devices():
    """Check available video devices."""
    print("🔍 Checking video devices...")
    
    video_devices = glob.glob('/dev/video*')
    print(f"Found {len(video_devices)} video devices: {video_devices}")
    
    for device in video_devices:
        try:
            # Check if device is readable
            with open(device, 'rb') as f:
                f.read(1)  # Try to read 1 byte
            print(f"  ✅ {device}: Readable")
        except PermissionError:
            print(f"  ❌ {device}: Permission denied")
        except Exception as e:
            print(f"  ❌ {device}: Error - {e}")

def check_v4l2_devices():
    """Check devices using v4l2-ctl."""
    print("\n📋 Checking with v4l2-ctl...")
    
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("Available devices:")
            print(result.stdout)
        else:
            print("❌ v4l2-ctl failed")
    except FileNotFoundError:
        print("❌ v4l2-ctl not installed. Install with: sudo apt install v4l-utils")
    except Exception as e:
        print(f"❌ v4l2-ctl error: {e}")

def check_usb_devices():
    """Check USB devices."""
    print("\n📱 Checking USB devices...")
    
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("USB devices:")
            for line in result.stdout.split('\n'):
                if 'logitech' in line.lower() or 'camera' in line.lower() or 'webcam' in line.lower():
                    print(f"  📹 {line}")
        else:
            print("❌ lsusb failed")
    except FileNotFoundError:
        print("❌ lsusb not available")
    except Exception as e:
        print(f"❌ lsusb error: {e}")

def check_processes_using_cameras():
    """Check if cameras are being used by other processes."""
    print("\n🔍 Checking for processes using cameras...")
    
    try:
        result = subprocess.run(['lsof', '/dev/video*'], 
                            capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            print("⚠️  Camera devices in use:")
            print(result.stdout)
        else:
            print("✅ No processes using camera devices")
    except FileNotFoundError:
        print("ℹ️  lsof not available")
    except Exception as e:
        print(f"ℹ️  lsof error: {e}")

def check_dmesg_for_errors():
    """Check dmesg for USB/camera errors."""
    print("\n📋 Checking dmesg for USB errors...")
    
    try:
        result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            usb_errors = [line for line in lines if 'usb' in line.lower() and ('error' in line.lower() or 'fail' in line.lower())]
            if usb_errors:
                print("⚠️  USB errors found:")
                for error in usb_errors[-5:]:  # Show last 5 errors
                    print(f"  {error}")
            else:
                print("✅ No recent USB errors found")
        else:
            print("❌ dmesg failed")
    except Exception as e:
        print(f"❌ dmesg error: {e}")

def main():
    """Main camera check function."""
    print("🎥 Simple Camera Check (No OpenCV required)")
    print("=" * 50)
    
    check_video_devices()
    check_v4l2_devices()
    check_usb_devices()
    check_processes_using_cameras()
    check_dmesg_for_errors()
    
    print("\n💡 TROUBLESHOOTING TIPS:")
    print("1. If camera not detected: try unplugging and reconnecting")
    print("2. If permission denied: check user groups: groups $USER")
    print("3. If camera in use: close other applications using camera")
    print("4. For Logitech: try 'sudo modprobe uvcvideo'")
    print("5. Check if camera works with: cheese, guvcview, or vlc")
    
    print("\n🎯 Camera check completed!")

if __name__ == "__main__":
    main()

