import os  # Provides functions for interacting with the operating system
import cv2  # OpenCV library for computer vision tasks (like image processing)
import numpy as np  # Provides support for large, multi-dimensional arrays and matrices
import matplotlib.pyplot as plt  # For plotting images and visualizations
import csv  # For reading and writing CSV files to save results

# PARAMETERS
DATASET_DIR = 'dataset'  # Folder where the 10 images are stored 
RESULTS_CSV = 'thresholding_results.csv'  # CSV file to store the results of thresholding

# Function to calculate PSNR between two images
def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)  # Mean Squared Error (MSE)
    if mse == 0:  # If MSE is 0, the images are identical, so PSNR is infinity
        return float('inf')
    max_pixel = 255.0  # Max pixel value for an 8-bit image
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Open the CSV file in write mode, to save the results of our thresholding process
with open(RESULTS_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)  # Create a CSV writer object to write data into the file
    # Write the header row in the CSV file (names of columns)
    writer.writerow(['Image', 'Otsu_Threshold', 'Adaptive_Threshold_Mean', 'Adaptive_Threshold_Gaussian', 'PSNR_Otsu', 'PSNR_Adaptive_Mean', 'PSNR_Adaptive_Gaussian'])

    # Loop through each image in the dataset folder
    for fname in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, fname)  # Construct the full file path of the image
        img = cv2.imread(path)  # Read the image using OpenCV (cv2)

        # Convert the image to grayscale, since thresholding requires a single channel (gray) image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 1: Preprocessing - Improving the image for better thresholding results

        # 1.1 Median filtering to reduce noise (helpful to smooth the image by removing small noise)
        gray = cv2.medianBlur(gray, 3)

        # 1.2 Contrast enhancement using Histogram Equalization
        gray = cv2.equalizeHist(gray)

        # 1.3 Edge enhancement and preservation using Canny Edge Detection
        edges = cv2.Canny(gray, 100, 200)  # Canny edge detection with lower and upper thresholds

        # Step 2: Apply Otsu's Thresholding (a global thresholding method)
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 3: Apply Adaptive Thresholding Methods:
        # - Adaptive Mean Thresholding: Applies a threshold based on the mean of local pixel values in a region.
        adaptive_mean_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # - Adaptive Gaussian Thresholding: Similar to the mean method but with a Gaussian weighted sum for the threshold calculation
        adaptive_gaussian_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Step 4: Calculate PSNR for each thresholding result
        psnr_otsu = calculate_psnr(gray, otsu_thresh)
        psnr_adaptive_mean = calculate_psnr(gray, adaptive_mean_thresh)
        psnr_adaptive_gaussian = calculate_psnr(gray, adaptive_gaussian_thresh)

        # Step 5: Save the results into the CSV file
        writer.writerow([fname, otsu_thresh.mean(), adaptive_mean_thresh.mean(), adaptive_gaussian_thresh.mean(),
                         psnr_otsu, psnr_adaptive_mean, psnr_adaptive_gaussian])

        # Step 6: Visualize the results (side-by-side comparison of different thresholding effects)

        # Create a new figure for plotting images (set figure size)
        plt.figure(figsize=(12, 8))
        
        # Plot the original (grayscale) image in the first subplot
        plt.subplot(2, 3, 1)
        plt.imshow(gray, cmap='gray')  # Display the image in grayscale
        plt.title(f"Original Image: {fname}")  # Title for the subplot
        plt.axis('off')  # Hide axes for better viewing

        # Plot the result of Otsu's thresholding in the second subplot
        plt.subplot(2, 3, 2)
        plt.imshow(otsu_thresh, cmap='gray')
        plt.title(f"Otsu Thresholding\nPSNR: {psnr_otsu:.2f}")
        plt.axis('off')

        # Plot the result of Adaptive Mean Thresholding in the third subplot
        plt.subplot(2, 3, 3)
        plt.imshow(adaptive_mean_thresh, cmap='gray')
        plt.title(f"Adaptive Mean Thresholding\nPSNR: {psnr_adaptive_mean:.2f}")
        plt.axis('off')

        # Plot the result of Adaptive Gaussian Thresholding in the fourth subplot
        plt.subplot(2, 3, 4)
        plt.imshow(adaptive_gaussian_thresh, cmap='gray')
        plt.title(f"Adaptive Gaussian Thresholding\nPSNR: {psnr_adaptive_gaussian:.2f}")
        plt.axis('off')

        # Plot the edge detection result from Canny in the fifth subplot
        plt.subplot(2, 3, 5)
        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edge Detection")
        plt.axis('off')

        # Adjust layout to make sure all images fit properly
        plt.tight_layout()
        
        # Save the figure as an image with a filename that corresponds to the original image name
        plt.savefig(f"threshold_comparison_{os.path.splitext(fname)[0]}.png")
        plt.close()  # Close the plot to avoid memory issues

# Final message to indicate the process is complete
print("Thresholding Process Complete")
