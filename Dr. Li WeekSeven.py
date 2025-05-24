"""
adaptive_thresholding.py

This script implements adaptive thresholding to segment three types of images:
- Indoor scene
- Outdoor scenery
- Close-up of a single object

Adaptive Mean and Adaptive Gaussian thresholding techniques are used.
The segmented images are saved for each scene.

Usage:
    python adaptive_thresholding.py
"""

import cv2
import numpy as np

# List of image filenames and scene descriptions
images = {
    "indoor": "indoor.jpg",
    "outdoor": "outdoor.jpg",
    "closeup": "closeup.jpg"
}

# Adaptive Thresholding parameters
block_size = 11  # Neighborhood size
C = 2            # Constant to subtract from the mean or weighted mean

# Process each image
for scene, filename in images.items():
    # Load and convert to grayscale
    img = cv2.imread(filename)
    if img is None:
        print(f"Image '{filename}' not found. Please check the path.")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive Mean Thresholding
    adaptive_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, block_size, C
    )

    # Adaptive Gaussian Thresholding
    adaptive_gaussian = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )

    # Save the results
    cv2.imwrite(f"{scene}_adaptive_mean.jpg", adaptive_mean)
    cv2.imwrite(f"{scene}_adaptive_gaussian.jpg", adaptive_gaussian)

    # Optional: Display results for quick check
    cv2.imshow(f"{scene} - Original", gray)
    cv2.imshow(f"{scene} - Adaptive Mean", adaptive_mean)
    cv2.imshow(f"{scene} - Adaptive Gaussian", adaptive_gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()
