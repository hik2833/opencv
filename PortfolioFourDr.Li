import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "Mod4CT1.jpg"  # Ensure this image is in your working directory
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define Gaussian sigmas
sigma1 = 0.5
sigma2 = 1.5

# Define kernel sizes
kernel_sizes = [3, 5, 7]

# Initialize figure for plotting
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
filter_names = ["Mean", "Median", f"Gaussian σ={sigma1}", f"Gaussian σ={sigma2}"]
row_labels = ["3x3 Kernel", "5x5 Kernel", "7x7 Kernel"]

# Apply filters and plot results
for i, k in enumerate(kernel_sizes):
    mean_filtered = cv2.blur(image_rgb, (k, k))
    median_filtered = cv2.medianBlur(image_rgb, k)
    gaussian1_filtered = cv2.GaussianBlur(image_rgb, (k, k), sigma1)
    gaussian2_filtered = cv2.GaussianBlur(image_rgb, (k, k), sigma2)
    
    filters = [mean_filtered, median_filtered, gaussian1_filtered, gaussian2_filtered]
    
    for j, filtered_img in enumerate(filters):
        ax = axes[i, j]
        ax.imshow(filtered_img)
        ax.axis("off")
        if i == 0:
            ax.set_title(filter_names[j], fontsize=12)
        if j == 0:
            ax.set_ylabel(row_labels[i], fontsize=12)

plt.tight_layout()
plt.savefig("Image_Filter_Comparison.png")
plt.show()
