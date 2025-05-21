# Student 2: Ruckshada Khan
# Adaptive Local Thresholding

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Input and output folders
input_folder = "dataset"
output_folder = os.path.join(os.path.dirname(__file__), "student2_adaptive_local_results")
os.makedirs(output_folder, exist_ok=True)

# Calculate PSNR func.
def compute_psnr(original, processed):
    mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

# Combined preprocessing func.
def pre_processing(img):
    # Median filtering
    filtered = cv2.medianBlur(img, 5)

    # Histogram equalization
    equalized = cv2.equalizeHist(filtered)

    # Sobel edge detection
    grad_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    edges = cv2.convertScaleAbs(grad)

    return edges

# Optimized adaptive local thresholding
def adaptive_local_threshold(img, k, window_size=15):
    img = img.astype(np.float32)
    mean = cv2.blur(img, (window_size, window_size))
    local_max = cv2.dilate(img, np.ones((window_size, window_size), np.uint8))
    local_min = cv2.erode(img, np.ones((window_size, window_size), np.uint8))
    T = k * (mean + (local_max - local_min) / 255.0)
    binary = np.where(img > T, 255, 0).astype(np.uint8)
    return binary

# Process all JPEG images in dataset
csv_path = os.path.join(output_folder, "student2_psnr_report.csv")
with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image Name", "PSNR (Original vs Preprocessed)", "PSNR (Preprocessed vs Adapative Local Thresholded)", "PSNR (Original vs Adaptive Local Thresholded)"])

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply combined preprocessing
        preprocessed = pre_processing(gray)

        # Adaptive thresholding
        binary = adaptive_local_threshold(preprocessed, k=0.5, window_size=15)

        # Save result
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, binary)
        
        # Compute PSNRs
        psnr_op = compute_psnr(gray, preprocessed)
        psnr_pt = compute_psnr(preprocessed, binary)
        psnr_ot = compute_psnr(gray, binary)

        # Append results to CSV
        with open(csv_path, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, f"{psnr_op:.2f}", f"{psnr_pt:.2f}", f"{psnr_ot:.2f}"])
        
        # Print msg to notify each image processing completion
        print(f"{filename} done")

        

# TESTING DIFFERENT K-VALUES USING 1ST IMAGE IN DATASET
dataset_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
if not dataset_files:
    raise FileNotFoundError("No images found in the dataset folder.")

first_img_path = os.path.join(input_folder, dataset_files[0])
first_img = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)

preprocessed_img = pre_processing(first_img)

k_values = [0.1, 0.3, 0.5, 0.7, 0.9]

plt.figure(figsize=(15, 4))
for i, k in enumerate(k_values):
    binary_img = adaptive_local_threshold(preprocessed_img, k)
    psnr_value = compute_psnr(first_img, binary_img)
    print("Image done")
    plt.subplot(1, len(k_values), i+1)
    plt.imshow(binary_img, cmap='gray')
    plt.title(f'k={k}\nPSNR={psnr_value:.2f} dB')
    plt.axis('off')

plt.tight_layout()
plt.show()

print("Processing complete. JPG AND CSV Results saved to 'student2_adaptive_local_results/' folder.")
