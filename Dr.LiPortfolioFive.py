import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded latent fingerprint image
image = cv2.imread('/mnt/data/A_grayscale_photograph_of_a_latent_fingerprint_is_.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Threshold the image to binary
_, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Define morphological kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Apply morphological operations
dilated = cv2.dilate(binary, kernel, iterations=1)
eroded = cv2.erode(binary, kernel, iterations=1)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Save the processed images (optional)
cv2.imwrite('/mnt/data/dilated.png', dilated)
cv2.imwrite('/mnt/data/eroded.png', eroded)
cv2.imwrite('/mnt/data/opened.png', opened)
cv2.imwrite('/mnt/data/closed.png', closed)

# Display results using matplotlib
titles = ['Binary', 'Dilated', 'Eroded', 'Opened', 'Closed']
images = [binary, dilated, eroded, opened, closed]

plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
