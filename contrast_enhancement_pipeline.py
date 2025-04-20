# Student 2- Ruckshada Khan
# Contrast Enhancement Pipeline

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

# Convert RGB to grayscale using the given luminosity formula
def convert_rgb_to_grayscale(img):
    #Extract rgb channels
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b #perform grayscale conversion
    return grayscale.astype(np.uint8)

# Normalize RGB channels
def normalize_channels(img):
    #extract rbg as floats
    r = img[:, :, 2].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 0].astype(np.float32)
    total = r + g + b + 1e-8 #Add 1e-8 to avoid dividing by 0
    r_norm = r / total #Perform normalization of each channel
    g_norm = g / total
    b_norm = b / total
    return np.stack((b_norm, g_norm, r_norm), axis=-1) #recombine channels

# Gamma correction on the difference between grayscale and normalized image
def gamma_correction_combo(gray, norm_rgb, gamma=0.8):
    # Normalize grayscale to [0, 1]
    gray_norm = gray.astype(np.float32) / 255.0

    # Convert normalized RGB to grayscale-equivalent shape using average
    norm_gray_like = norm_rgb.mean(axis=2)

    # Combine (difference then scale)
    combo = np.abs(gray_norm - norm_gray_like) #difference
    combo = np.power(combo, gamma) #gamma enhance for more contrast in darker areas
    return (combo * 255).astype(np.uint8) #scale to keep within 0-255

# Alpha blending
def alpha_blend(img1, img2, alpha=0.5):
    blended = alpha * img1 + (1 - alpha) * img2 #formula
    return np.clip(blended, 0, 255).astype(np.uint8) #clip to stay within 0-255

# Piecewise contrast stretching based off student 1 (Mahim's) code
def optimal_piecewise_contrast(img: np.ndarray, t: int, L_min=0, L_max=255) -> np.ndarray:
    Imin, Imax = int(img.min()), int(img.max())
    img = img.astype(np.float32)
    stretched = np.zeros_like(img)

    low = (img <= t)
    high = (img > t)

    if t > Imin:
        stretched[low] = (img[low] - Imin) / (t - Imin) * t
    else:
        stretched[low] = img[low]

    if Imax > t:
        stretched[high] = ((img[high] - t) / (Imax - t)) * (Imax - t) + t
    else:
        stretched[high] = img[high]

    return np.clip(stretched, L_min, L_max).astype(np.uint8)

# Combine each step to create Full pipeline
def full_contrast_pipeline(image_path, threshold=100, alpha=0.6, gamma=0.8):
    img = cv.imread(image_path)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    gray = convert_rgb_to_grayscale(img_rgb) # Step 1a: output a grayscale version of the original
    norm_rgb = normalize_channels(img_rgb) # Step 1b: output a normalized version of the original
    gamma_img = gamma_correction_combo(gray, norm_rgb, gamma) # Step 2: Difference combine and gamma correction of the grayscale and normalized imgs
    blended = alpha_blend(gray, gamma_img, alpha) # Step 3: Alpha blend the grayscale img with the gamma corrected combined (gray + normalized) img
    enhanced = optimal_piecewise_contrast(blended, threshold) # Step 4: Perform piecewise contrast enhancement on the final blended output

    return img_rgb, enhanced

# Batch process all .jpgs in dataset/
def process_all_images(dataset_folder="dataset", output_folder="student2_enhanced_results", threshold=100):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob(os.path.join(dataset_folder, "*.jpg"))

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")

        original, enhanced = full_contrast_pipeline(img_path, threshold)

        # Save the enhanced image
        output_path = os.path.join(output_folder, f"enhanced_{filename}")
        cv.imwrite(output_path, cv.cvtColor(enhanced, cv.COLOR_RGB2BGR))

        # Plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(original)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Enhanced")
        plt.imshow(enhanced, cmap='gray')
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# Run the batch processor
process_all_images()
