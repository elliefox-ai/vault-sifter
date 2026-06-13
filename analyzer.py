"""
Fast image quality metrics using pure numpy/scipy.
No neural models — just signal processing. ~15-30ms per image.
"""

import io
import numpy as np
from scipy.ndimage import convolve
from PIL import Image


def analyze_image(filepath, max_size=512):
    """
    Compute quality metrics for an image file.
    
    Returns dict with:
        sharpness: Laplacian variance (higher = sharper)
        brightness: 0.0-1.0 (mean luminance)
        contrast: 0.0+ (std dev of luminance, normalized)
        saturation: 0.0-1.0 (mean color saturation)
        width, height: original dimensions
        aspect_ratio: width/height
        entropy: 0.0-1.0 (information density)
        is_grayscale: bool
        mean_color: [r, g, b] 0-255
        unique_colors: approximate count
        file_corrupt: bool
    """
    result = {
        "sharpness": None,
        "brightness": None,
        "contrast": None,
        "saturation": None,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "entropy": None,
        "is_grayscale": False,
        "mean_color": None,
        "unique_colors": None,
        "file_corrupt": False,
    }

    try:
        img = Image.open(filepath)
        img.load()  # force decode to catch corrupt files
    except Exception:
        result["file_corrupt"] = True
        return result

    result["width"] = img.width
    result["height"] = img.height
    result["aspect_ratio"] = round(img.width / img.height, 3) if img.height else 0

    # Downscale for metric computation (faster, same signal)
    work = img.copy()
    work.thumbnail((max_size, max_size), Image.LANCZOS)

    arr = np.array(work)

    if arr.ndim == 2:
        # Grayscale
        gray = arr.astype(float)
        result["is_grayscale"] = True
        result["mean_color"] = [int(gray.mean())] * 3
        result["saturation"] = 0.0
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # RGBA — drop alpha for metrics
        rgb = arr[:, :, :3]
        gray = np.mean(rgb, axis=2, dtype=float)
        result["mean_color"] = rgb.reshape(-1, 3).mean(axis=0).astype(int).tolist()
        result["is_grayscale"] = _check_grayscale(rgb)
        if not result["is_grayscale"]:
            result["saturation"] = _compute_saturation(rgb)
        else:
            result["saturation"] = 0.0
    elif arr.ndim == 3:
        gray = np.mean(arr, axis=2, dtype=float)
        result["mean_color"] = arr.reshape(-1, 3).mean(axis=0).astype(int).tolist()
        result["is_grayscale"] = _check_grayscale(arr)
        if not result["is_grayscale"]:
            result["saturation"] = _compute_saturation(arr)
        else:
            result["saturation"] = 0.0
    else:
        result["file_corrupt"] = True
        return result

    # Sharpness: Laplacian variance
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    laplacian = convolve(gray, lap_kernel, mode="reflect")
    result["sharpness"] = round(float(laplacian.var()), 1)

    # Brightness: mean luminance normalized to 0-1
    result["brightness"] = round(float(gray.mean()) / 255.0, 3)

    # Contrast: std dev of luminance
    result["contrast"] = round(float(gray.std()) / 128.0, 3)

    # Entropy: histogram-based
    hist, _ = np.histogram(gray, bins=64, range=(0, 255))
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-8
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    result["entropy"] = round(float(entropy) / 6.0, 3)  # normalize to ~0-1 (max entropy of 64 bins ≈ 6 bits)

    # Unique colors (approximate via quantization)
    if arr.ndim == 3:
        quantized = (arr[:, :, :3] // 16).reshape(-1, 3)
        result["unique_colors"] = int(len(np.unique(quantized, axis=0)))

    return result


def _check_grayscale(rgb_arr):
    """Check if a 3-channel image is effectively grayscale."""
    r, g, b = rgb_arr[:, :, 0], rgb_arr[:, :, 1], rgb_arr[:, :, 2]
    diff = np.maximum(np.maximum(np.abs(r.astype(int) - g.astype(int)),
                                  np.abs(g.astype(int) - b.astype(int))),
                       np.abs(r.astype(int) - b.astype(int)))
    return bool(diff.max() < 10)


def _compute_saturation(rgb_arr):
    """Compute mean saturation using max-min in HSV-ish space."""
    maxc = rgb_arr.max(axis=2).astype(float)
    minc = rgb_arr.min(axis=2).astype(float)
    sat = (maxc - minc) / (maxc + 1e-8)
    return round(float(sat.mean()), 3)


# ─── Quality Score ────────────────────────────────────────────────────────────

def quality_score(metrics):
    """
    Heuristic 0-100 quality score for ranking.
    Combines sharpness, contrast, and color richness.
    Not a judgment of artistic merit — just 'did the generation produce detail.'
    """
    if metrics.get("file_corrupt"):
        return 0

    sharp = metrics.get("sharpness", 0) or 0
    contrast = metrics.get("contrast", 0) or 0
    sat = metrics.get("saturation", 0) or 0
    entropy = metrics.get("entropy", 0) or 0

    # Normalize sharpness (log scale — values range ~100-50000)
    import math
    sharp_norm = min(1.0, math.log10(max(1, sharp)) / 4.5)

    # Weighted combination
    score = (
        sharp_norm * 35 +     # sharpness is the biggest signal
        contrast * 20 +       # contrast matters
        min(1.0, entropy) * 25 +  # information density
        min(1.0, sat * 2) * 20     # color richness
    )

    return round(score, 1)


if __name__ == "__main__":
    import sys
    import time
    import glob

    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/shared/comfy-output/"
    files = sorted(glob.glob(f"{path}/*.png"))

    print(f"Analyzing {len(files)} images...\n")
    start = time.time()
    results = []
    for f in files[:50]:
        m = analyze_image(f)
        qs = quality_score(m)
        results.append((qs, m, f))

    results.sort(key=lambda x: -x[0])

    print(f"{'SCORE':>5}  {'SHARP':>8}  {'BRI':>5}  {'CON':>5}  {'SAT':>5}  {'ENT':>5}  FILE")
    print("-" * 80)
    for qs, m, f in results[:20]:
        name = f.split("/")[-1][:40]
        print(f"{qs:>5.1f}  {m['sharpness']:>8.1f}  {m['brightness']:>5.2f}  {m['contrast']:>5.2f}  {m['saturation']:>5.3f}  {m['entropy']:>5.3f}  {name}")
    print(f"\n...(showing top 20 of {len(results)})")

    elapsed = time.time() - start
    print(f"\n{len(results)} images in {elapsed:.2f}s = {elapsed/max(1,len(results))*1000:.0f}ms/image")
