# Mahim Ali Student 1
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# PARAMETERS
DATASET_DIR = 'dataset'      # folder with 10 images
BLOCK_SIZE  = (64, 64)       # size of blocks for EME
C_CONST     = 1.0            # small constant in EME formula

# Prepares CSV to record results
with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image', 'EME_before', 'EME_after', 't_opt'])

    # Processes each image file
    for fname in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, fname)
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)

        # 1) Computes original EME
        H, W = gray.shape
        eme_before = 0.0
        cnt = 0
        for i in range(0, H, BLOCK_SIZE[0]):
            for j in range(0, W, BLOCK_SIZE[1]):
                blk = gray[i:i+BLOCK_SIZE[0], j:j+BLOCK_SIZE[1]]
                wmax, wmin = blk.max(), blk.min()
                eme_before += np.log((wmax + C_CONST)/(wmin + C_CONST))
                cnt += 1
        eme_before /= cnt

        # 2) Sweeps thresholds to find t*
        Imin, Imax = int(gray.min()), int(gray.max())
        thresholds  = np.arange(Imin+1, Imax)
        eme_vals    = []

        for t in thresholds:
            stretched = np.zeros_like(gray, dtype=np.float32)
            low  = (gray <= t)
            high = (gray >  t)
            # piecewises stretch
            stretched[low]  = (gray[low]  - Imin)/(t - Imin)*t
            stretched[high] = ((gray[high] - t)/(Imax - t))*(Imax - t) + t

            # Computes EME on stretched
            eme_sum, cnt2 = 0.0, 0
            for i in range(0, H, BLOCK_SIZE[0]):
                for j in range(0, W, BLOCK_SIZE[1]):
                    blk = stretched[i:i+BLOCK_SIZE[0], j:j+BLOCK_SIZE[1]]
                    wmax, wmin = blk.max(), blk.min()
                    eme_sum += np.log((wmax + C_CONST)/(wmin + C_CONST))
                    cnt2 += 1
            eme_vals.append(eme_sum/cnt2)

        eme_vals = np.array(eme_vals)
        opt_idx  = np.argmax(eme_vals)
        t_opt    = thresholds[opt_idx]
        eme_after= eme_vals[opt_idx]

        # Records results
        writer.writerow([fname, f"{eme_before:.4f}", f"{eme_after:.4f}", t_opt])

        # Plots EME vs t
        plt.figure(figsize=(6,3))
        plt.plot(thresholds, eme_vals, label='EME(t)')
        plt.axvline(t_opt, color='r', linestyle='--', label=f't* = {t_opt}')
        plt.title(fname)
        plt.xlabel('Threshold t')
        plt.ylabel('EME')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plot_{os.path.splitext(fname)[0]}.png")
        plt.close()

        # Saves before/after thumbnail
        stretched_opt = np.zeros_like(gray, dtype=np.float32)
        stretched_opt[gray<=t_opt] = (gray[gray<=t_opt]-Imin)/(t_opt-Imin)*t_opt
        stretched_opt[gray> t_opt] = ((gray[gray> t_opt]-t_opt)/(Imax-t_opt))*(Imax-t_opt)+t_opt

        both = np.hstack([
            cv2.resize(gray,        (256,256)),
            cv2.resize(stretched_opt.astype(np.uint8), (256,256))
        ])
        cv2.imwrite(f"comp_{os.path.splitext(fname)[0]}.png", both)

print("Done")
