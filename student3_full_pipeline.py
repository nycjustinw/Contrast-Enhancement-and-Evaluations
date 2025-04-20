# Student 3: Justin Wang

import os, csv, glob, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────────────────────
ORIGINAL_DIR = 'dataset'
ST1_DIR      = os.path.join('results', 'comps')
ST2_DIR      = os.path.join('results', 'student2_enhanced_results')

CSV_FILE  = 'student3_results.csv'
PLOTS_DIR = os.path.join('results', 'plots')

BLOCK_SIZE = (64, 64)    # h × w of blocks used by the metrics
ALPHA      = 1.0         # entropy‑weight for EMEE / AMEE
EPS        = 1e-6        # guard against log(0)
METRICS    = ['EME', 'EMEE', 'Visibility', 'AME', 'AMEE']
# ──────────────────────────────────────────────────────────────────────────


# ─── helpers ──────────────────────────────────────────────────────────────
def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def first_existing(patterns: list[str] | str) -> str | None:
    """Return first file path that matches any glob pattern, else None."""
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None

def safe_imread(path: str) -> np.ndarray | None:
    """Try cv2.imread; fall back to Pillow (returns uint8 grayscale array)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    try:
        pil = Image.open(path).convert('L')
        return np.asarray(pil, dtype=np.uint8)
    except Exception:
        return None

def compute_metrics(img: np.ndarray) -> tuple[float, float, float, float, float]:
    """Compute five contrast metrics on an 8‑bit grayscale image."""
    H, W = img.shape
    eme=emee=vis=ame=amee=0.0; cnt=0
    for i in range(0, H, BLOCK_SIZE[0]):
        for j in range(0, W, BLOCK_SIZE[1]):
            blk = img[i:i+BLOCK_SIZE[0], j:j+BLOCK_SIZE[1]]
            Imax, Imin = float(blk.max())+EPS, float(blk.min())+EPS

            with np.errstate(divide='ignore'):
                log_ratio = np.log(Imax/Imin)
            eme  += 20.0 * log_ratio
            emee += ALPHA * log_ratio

            vis_val = (Imax - Imin) / (Imax + Imin)
            vis    += vis_val
            with np.errstate(divide='ignore', invalid='ignore'):
                ame  += -np.log(vis_val + EPS)
                amee += -np.log(vis_val**ALPHA + EPS)
            cnt += 1
    return (eme/cnt, emee/cnt, vis/cnt, ame/cnt, amee/cnt)


# ─── version → path resolver dict ────────────────────────────────────────
DASH, ENDASH = '-', '–'
VERSIONS = {
    'Original': lambda b: first_existing([
        f'{ORIGINAL_DIR}/{b}.jpg', f'{ORIGINAL_DIR}/{b}.png']),
    'Student1': lambda b: first_existing([
        f'{ST1_DIR}/comp_{b}.png', f'{ST1_DIR}/comp_{b}.jpg']),
    'Student2': lambda b: first_existing([
        f'{ST2_DIR}/enhanced_{b}.png', f'{ST2_DIR}/enhanced_{b}.jpg',
        f'{ST2_DIR}/enhanced_{b.replace(DASH, ENDASH)}.png',
        f'{ST2_DIR}/enhanced_{b.replace(DASH, ENDASH)}.jpg']),
}


# ─── gather metrics ──────────────────────────────────────────────────────
def gather_scores() -> None:
    missing = {v:0 for v in VERSIONS}
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Version', *METRICS])

        for fname in sorted(os.listdir(ORIGINAL_DIR)):
            base, _ = os.path.splitext(fname)        # "1-014"
            for label, path_fn in VERSIONS.items():
                path = path_fn(base)
                if not path or not os.path.isfile(path):
                    missing[label] += 1; continue
                img = safe_imread(path)
                if img is None:
                    missing[label] += 1; continue
                writer.writerow([f'{base}.jpg', label,
                                 *[f'{m:.4f}' for m in compute_metrics(img)]])

    print(f'✓ metrics saved → {CSV_FILE}')
    for ver, n in missing.items():
        if n:
            print(f'[info] {n} missing/unreadable file(s) for {ver}')


# ─── plotting utilities ──────────────────────────────────────────────────
def bar_per_image(df: pd.DataFrame) -> None:
    ensure_dir(PLOTS_DIR)
    for m in METRICS:
        pivot = df.pivot(index='Image', columns='Version', values=m)
        ax = pivot.plot(kind='bar', figsize=(12,5))
        ax.set_title(f'{m}: Original vs. Student1 vs. Student2')
        ax.set_ylabel(m)
        ax.set_xlabel('Image')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'bar_{m}.png'))
        plt.close()

def avg_improvement(df: pd.DataFrame) -> None:
    ensure_dir(PLOTS_DIR)
    methods = [v for v in df.Version.unique() if v != 'Original']
    avg: dict[str, dict[str, float]] = {m:{} for m in methods}

    for meth in methods:
        merged = (df[df.Version=='Original']
                    .merge(df[df.Version==meth],
                           on='Image', suffixes=('_orig','_enh')))
        if merged.empty:   # nothing to compare
            continue
        for met in METRICS:
            avg[meth][met] = (merged[f'{met}_enh'] -
                              merged[f'{met}_orig']).mean()

    x = np.arange(len(METRICS)); w = 0.8 / max(1, len(methods))
    plt.figure(figsize=(7,4))
    for idx, meth in enumerate(methods):
        plt.bar(x + (idx-len(methods)/2)*w + w/2,
                [avg[meth].get(m,0) for m in METRICS],
                width=w, label=meth)
    plt.xticks(x, METRICS, rotation=45, ha='right')
    plt.ylabel('Average Δ (enhanced − original)')
    plt.title('Average Improvement by Method')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'avg_improvement.png'))
    plt.close()


# ─── main ────────────────────────────────────────────────────────────────
def main() -> None:
    gather_scores()
    df = pd.read_csv(CSV_FILE)
    bar_per_image(df)
    avg_improvement(df)
    print(f'✓ plots saved → {PLOTS_DIR}/')

if __name__ == '__main__':
    main()
