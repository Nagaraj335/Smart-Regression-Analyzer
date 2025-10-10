import sys
import os

print("=== Python Environment Debug Info ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

print("\n=== Testing OpenCV Import ===")
try:
    import cv2
    print(f"✅ OpenCV imported successfully!")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV location: {cv2.__file__}")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")
    print("Available packages:")
    import pkg_resources
    installed_packages = [d.project_name for d in pkg_resources.working_set]
    opencv_packages = [pkg for pkg in installed_packages if 'opencv' in pkg.lower()]
    print(f"OpenCV packages found: {opencv_packages}")