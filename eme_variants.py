import os
import cv2
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

ORIGINAL_DIR = 'dataset'
ENHANCED_DIR = os.path.join('results', 'comps')
BLOCK_SIZE   = (64, 64)
ALPHA        = 1.0
EPSILON      = 1e-6
CSV_FILE     = 'student3_results.csv'
PLOTS_DIR    = 'plots'
METRICS      = ['EME', 'EMEE', 'Visibility', 'AME', 'AMEE']

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def compute_contrast_metrics(img, block_size, alpha, eps):
    H, W = img.shape
    eme_sum = emee_sum = vis_sum = ame_sum = amee_sum = 0.0
    cnt = 0
    for i in range(0, H, block_size[0]):
        for j in range(0, W, block_size[1]):
            blk = img[i:i+block_size[0], j:j+block_size[1]]
            Imax, Imin = float(blk.max()), float(blk.min())
            Imax_corr, Imin_corr = Imax + eps, Imin + eps

            # 1) EME & 2) EMEE
            with np.errstate(divide='ignore'):
                eme_sum  += 20.0 * np.log(Imax_corr / Imin_corr)
                emee_sum += alpha  * np.log(Imax_corr / Imin_corr)

            # 3) Michelson visibility
            vis_sum += (Imax_corr - Imin_corr) / (Imax_corr + Imin_corr)

            # 4) AME & 5) AMEE
            ratio = (Imax_corr - Imin_corr) / (Imax_corr + Imin_corr)
            with np.errstate(divide='ignore', invalid='ignore'):
                ame_sum  += -np.log(ratio + eps)
                amee_sum += -np.log(ratio**alpha + eps)

            cnt += 1

    return (
        eme_sum  / cnt,
        emee_sum / cnt,
        vis_sum  / cnt,
        ame_sum  / cnt,
        amee_sum / cnt,
    )

def plot_per_image_bars(df):
    ensure_dir(PLOTS_DIR)
    for m in METRICS:
        pivot = df.pivot(index='Image', columns='Type', values=m)
        ax = pivot.plot(kind='bar', figsize=(10,5))
        ax.set_title(f'{m}: Before vs. After Enhancement')
        ax.set_ylabel(m)
        ax.set_xlabel('Image')
        ax.legend(title='')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'bar_{m}.png'))
        plt.close()

def plot_average_improvement(df):
    diffs = {}
    for m in METRICS:
        orig = df[df.Type=='Original'][m].values
        enh  = df[df.Type=='Enhanced'][m].values
        diffs[m] = (enh - orig).mean()
    ensure_dir(PLOTS_DIR)
    plt.figure(figsize=(6,4))
    plt.bar(diffs.keys(), diffs.values())
    plt.title('Average Metric Improvement (Enhanced − Original)')
    plt.ylabel('Δ Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'avg_improvement.png'))
    plt.close()

def main():
    # 1) Compute and record metrics
    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'Type'] + METRICS)

        for fname in sorted(os.listdir(ORIGINAL_DIR)):
            base, _   = os.path.splitext(fname)
            orig_path = os.path.join(ORIGINAL_DIR, fname)
            comp_name = f"comp_{base}.png"
            enh_path  = os.path.join(ENHANCED_DIR, comp_name)

            orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            enh  = cv2.imread(enh_path,  cv2.IMREAD_GRAYSCALE)
            if orig is None or enh is None:
                print(f"[!] Missing file: {orig_path} or {enh_path}")
                continue

            orig_metrics = compute_contrast_metrics(orig, BLOCK_SIZE, ALPHA, EPSILON)
            enh_metrics  = compute_contrast_metrics(enh,  BLOCK_SIZE, ALPHA, EPSILON)

            writer.writerow([fname, 'Original'] + [f"{m:.4f}" for m in orig_metrics])
            writer.writerow([fname, 'Enhanced'] + [f"{m:.4f}" for m in enh_metrics])

    print(f"Metrics saved to {CSV_FILE}")

    # 2) Load results into DataFrame
    df = pd.read_csv(CSV_FILE)

    # 3) Generate and save plots
    plot_per_image_bars(df)
    print(f"Per-image bar charts saved in {PLOTS_DIR}/")
    plot_average_improvement(df)
    print(f"Average improvement chart saved at {PLOTS_DIR}/avg_improvement.png")

if __name__ == '__main__':
    main()
