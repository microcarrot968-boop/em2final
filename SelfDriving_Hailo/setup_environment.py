#!/usr/bin/env python3
"""
Environment Setup Module
Handles package installation and environment checks for the Smart Vision System.
"""

import sys
import subprocess
import pkgutil


def install_packages():
    """
    Install required Python packages for the Smart Vision System.
    
    Returns:
        bool: True if all packages are installed successfully, False otherwise.
    """
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'scipy', 
        'opencv-python', 'tflite-runtime', 'picamera2'
    ]
    
    def pip_install(pkgs):
        """Install packages using pip."""
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade'] + pkgs)
    
    # Check which packages need to be installed
    to_install = [p for p in required_packages if not pkgutil.find_loader(p.split('==')[0])]
    
    if to_install:
        print('Installing missing packages:', to_install)
        try:
            pip_install(to_install)
            print('All packages installed successfully!')
            return True
        except Exception as e:
            print('Error installing packages:', str(e))
            print('If Picamera2 fails via pip, install system package: sudo apt install -y python3-picamera2')
            return False
    else:
        print('All required packages are already installed!')
        return True


def check_camera_availability():
    """
    Check if camera is available and working.
    
    Returns:
        bool: True if camera is available, False otherwise.
    """
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={'size': (640, 480), 'format': 'RGB888'})
        picam2.configure(config)
        picam2.start()
        frame = picam2.capture_array()
        picam2.stop()
        print(f"Camera test successful! Frame shape: {frame.shape}")
        return True
    except Exception as e:
        print(f"Camera test failed: {str(e)}")
        print("Please check camera connection and run: rpicam-hello -t 3000")
        return False


if __name__ == "__main__":
    print("=== Smart Vision System - Environment Setup ===")
    
    # Install packages
    if install_packages():
        print("\n=== Testing Camera ===")
        check_camera_availability()
    else:
        print("Failed to install required packages. Please check the error messages above.")
