import cv2
import numpy as np

print("🔍 Testing OpenCV...")
print(f"📦 OpenCV Version: {cv2.__version__}")

# Your image path
image_path = r"C:\Users\ASUS\OneDrive\Pictures\Camera Roll\WIN_20250814_23_12_31_Pro.jpg"

# Test if image exists and can be read
image = cv2.imread(image_path)

if image is None:
    print("❌ Could not read the image. Check if the file exists and path is correct.")
    print(f"📂 Tried to read: {image_path}")
else:
    print("✅ Image loaded successfully!")
    print(f"📊 Image shape: {image.shape}")
    print(f"📏 Dimensions: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"🎨 Channels: {image.shape[2]}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"⚫ Grayscale shape: {gray.shape}")
    
    # Save a small version to confirm it's working
    small = cv2.resize(image, (300, 200))
    cv2.imwrite('test_output.jpg', small)
    print("💾 Saved resized image as 'test_output.jpg'")
    
    print("🎉 OpenCV is working perfectly!")