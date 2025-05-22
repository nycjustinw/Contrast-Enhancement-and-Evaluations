# Justin Wang
# Student 3

import os
import cv2
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu, threshold_sauvola, threshold_niblack
from skimage.metrics import peak_signal_noise_ratio as psnr

def iterative_global_threshold(img, tol=0.5):
    """Perform iterative global thresholding."""
    # initial threshold: mean of min and max
    T_prev = (img.min() + img.max()) / 2.0
    while True:
        # split into two groups
        G1 = img[img > T_prev]
        G2 = img[img <= T_prev]
        μ1 = G1.mean() if len(G1) > 0 else 0
        μ2 = G2.mean() if len(G2) > 0 else 0
        T_new = 0.5 * (μ1 + μ2)
        if abs(T_new - T_prev) < tol:
            break
        T_prev = T_new
    return (img > T_new).astype(np.uint8) * 255

def process_student3(dataset_dir, output_dir):
    methods = {
        "iterative_global": lambda im: iterative_global_threshold(im),
        "otsu": lambda im: (im > threshold_otsu(im)).astype(np.uint8) * 255,
        "niblack": lambda im: (im > threshold_niblack(im, window_size=25, k=0.2)).astype(np.uint8) * 255,
        "sauvola": lambda im: (im > threshold_sauvola(im, window_size=25, k=0.2, r=128)).astype(np.uint8) * 255,
    }
    # ensure output folders
    for m in methods:
        os.makedirs(os.path.join(output_dir, m), exist_ok=True)

    results = []
    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.lower().endswith(('.png','.jpg','jpeg','bmp','tif','tiff')):
            continue
        path = os.path.join(dataset_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        for m, func in methods.items():
            bin_img = func(img)
            out_path = os.path.join(output_dir, m, fname)
            cv2.imwrite(out_path, bin_img)
            score = psnr(img, bin_img, data_range=255)
            results.append({
                "filename": fname,
                "method": m,
                "psnr": round(score, 2)
            })
            print(f"{fname} | {m} | PSNR: {score:.2f}")
    # save CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "student3_psnr_comparison.csv"), index=False)
    print("Saved PSNR comparison CSV to", os.path.join(output_dir, "student3_psnr_comparison.csv"))

if __name__ == "__main__":
    dataset = "dataset"
    output = "student3_thresholding_results"
    process_student3(dataset, output)

