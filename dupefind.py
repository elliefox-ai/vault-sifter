#!/usr/bin/env python3
"""dupefind.py — perceptual-hash near-dupe detection core for Vault Sifter.

Tier model (from the spec):
  - byte-identical       → SHA256 (handled at pull time by dupe_pull.py)
  - near-identical       → pHash (DCT low-freq structure)
  - regional edits       → dHash / pHash tolerant distance (global layout unchanged)
  - full re-renders      → CLIP/DINO embeddings (escalation path, not implemented yet)

Pair distance = min(hamming(phash), hamming(dhash)) — if EITHER hash says two
images are close, they're candidates for human review. Loose by design: the
human makes the final call.

Data model (all cached inside the pool dir, so the pool is self-contained):
  .dupe-hashes.json   {relpath: {"ph": int, "dh": int, "mtime": float, "size": int}}
  .dupe-edges.npz     arrays: a[], b[], d[] — all pairs with dist <= MAX_DIST

Families are connected components of the pair graph at a given threshold.
"""

import json
import os
import shutil
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

HASHES_FILE = ".dupe-hashes.json"
EDGES_FILE = ".dupe-edges.npz"
MANIFEST_FILE = "manifest.json"
TRASH_DIR = ".dupe-trash"

# Hash cache version. v2 = draft-mode JPEG decode + bilinear resampling.
# Bits differ slightly from v1 (LANCZOS), so caches are never mixed.
HASHVER = 2

# Pairs beyond this hamming distance are never candidates (bounds memory).
MAX_DIST = 40
# Save the hash cache every N images so interrupted scans resume, not restart.
CHECKPOINT_EVERY = 500
# Confidence slider (0=loosest … 100=strictest) -> hamming threshold mapping.
def threshold_for_confidence(confidence):
    """confidence 100 -> threshold 0 (only near-identical), 0 -> 40 (very loose)."""
    confidence = max(0, min(100, int(confidence)))
    return round((100 - confidence) * 0.4)


def confidence_for_threshold(threshold):
    return int(100 - threshold / 0.4)


# ─── Hashing ────────────────────────────────────────────────────────────

def _dct_matrix(n):
    """DCT-II transform matrix (n x n), avoiding a scipy dependency."""
    k = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    m = np.cos(np.pi / n * (x + 0.5) * k)
    m[0] *= 1.0 / np.sqrt(2)  # orthonormalize row 0
    return m


_DCT32 = None


def _dct2d(a):
    global _DCT32
    if _DCT32 is None:
        _DCT32 = _dct_matrix(32)
    return _DCT32 @ a @ _DCT32.T


def phash(img):
    """64-bit DCT perceptual hash (imagehash-style)."""
    g = img.convert("L").resize((32, 32), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float64)
    dct = _dct2d(a)
    low = dct[:8, :8]
    med = np.median(low)
    return int(np.packbits((low > med).astype(np.uint8)).tobytes().hex(), 16)


def dhash(img):
    """64-bit gradient/difference hash — sensitive to edge layout."""
    g = img.convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(np.packbits(bits.astype(np.uint8)).tobytes().hex(), 16)


def popcount(x):
    return int(np.bitwise_count(np.uint64(x)))


def hash_image(filepath):
    """Return (phash, dhash) for an image file, EXIF-corrected."""
    with Image.open(filepath) as img:
        # JPEG fast path: request a 1/8-scale draft decode so PIL only
        # Huffman-decodes the DC + a few AC coefficients. Exact enough for
        # hashing at a fraction of the full decode cost.
        img.draft("L", (64, 64))
        img = ImageOps.exif_transpose(img)
        return phash(img), dhash(img)


def _hash_one(args):
    """Worker: hash one file off-process. Returns (rel, payload) or (rel, None, err)."""
    fp, pool_dir = args
    rel = os.path.relpath(fp, pool_dir)
    try:
        st = os.stat(fp)
        ph, dh = hash_image(fp)
        return rel, {"ph": ph, "dh": dh, "mtime": st.st_mtime,
                     "size": st.st_size, "v": HASHVER}
    except Exception as e:
        return rel, None, str(e)


# ─── Pool scanning / caching ────────────────────────────────────────────

def pool_files(pool_dir):
    """All image paths under the pool, excluding dupe metadata + trash."""
    pool = Path(pool_dir)
    out = []
    for root, dirs, files in os.walk(pool):
        # Skip internal dupe tooling dirs (dot-prefixed = tooling/trash)
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                out.append(str(Path(root) / f))
    return sorted(out)


def load_hashes(pool_dir):
    p = Path(pool_dir) / HASHES_FILE
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_hashes(pool_dir, hashes):
    Path(pool_dir).mkdir(parents=True, exist_ok=True)
    (Path(pool_dir) / HASHES_FILE).write_text(json.dumps(hashes))


def compute_hashes(pool_dir, progress=None, on_exact=None, should_stop=None):
    """Hash any new/changed images. Returns (hashes, new_count, stopped).

    Checkpoints every CHECKPOINT_EVERY images so an interrupted scan
    keeps its progress. If should_stop() returns truthy, hashing halts
    early and remaining files are simply skipped (resumable)."""
    # Drop stale-version cache entries so v1/v2 hashes never mix.
    hashes = {rel: h for rel, h in load_hashes(pool_dir).items()
              if h.get("v") == HASHVER}
    files = pool_files(pool_dir)
    new_count = 0

    # Build reverse lookup from existing hashes for instant exact-dupe detection
    seen = {}
    for rel, h in hashes.items():
        key = (h["ph"], h["dh"])
        seen.setdefault(key, []).append(rel)

    to_hash = []
    for fp in files:
        rel = os.path.relpath(fp, pool_dir)
        h = hashes.get(rel)
        try:
            st = os.stat(fp)
        except OSError as e:
            print(f"  [warn] {rel}: {e}")
            continue
        if h and h.get("mtime") == st.st_mtime and h.get("size") == st.st_size:
            continue
        to_hash.append((fp, pool_dir))

    done = len(files) - len(to_hash)
    if progress:
        progress(done, len(files))

    stopped = False

    def _absorb(result):
        nonlocal done, new_count, stopped
        rel, payload, *err = result
        if payload is None:
            print(f"  [warn] {rel}: {err[0] if err else 'unknown'}")
        else:
            hashes[rel] = payload
            new_count += 1
            key = (payload["ph"], payload["dh"])
            if key in seen and on_exact:
                on_exact(rel, list(seen[key]))
            seen.setdefault(key, []).append(rel)
        done += 1
        if progress and (done % 50 == 0 or done == len(files)):
            progress(done, len(files))

    # Parallel hashing pays off above a few dozen images; below that,
    # process spawn overhead (esp. on Windows) outweighs the gain.
    if len(to_hash) >= 32:
        nproc = min(mp.cpu_count() or 1, 8)
        try:
            with mp.get_context("spawn").Pool(nproc) as poolr:
                pending = poolr.imap_unordered(_hash_one, to_hash, chunksize=16)
                while True:
                    try:
                        result = next(pending)
                    except StopIteration:
                        break
                    _absorb(result)
                    if should_stop and should_stop():
                        stopped = True
                        break
                    if done % CHECKPOINT_EVERY < 16:
                        save_hashes(pool_dir, hashes)
            poolr.terminate()
        except (OSError, ImportError, PermissionError):
            # Sandboxed/restricted environments: fall back to serial.
            for t in to_hash:
                if should_stop and should_stop():
                    stopped = True
                    break
                _absorb(_hash_one(t))
                if done % CHECKPOINT_EVERY == 0:
                    save_hashes(pool_dir, hashes)
    else:
        for t in to_hash:
            if should_stop and should_stop():
                stopped = True
                break
            _absorb(_hash_one(t))
            if done % CHECKPOINT_EVERY == 0:
                save_hashes(pool_dir, hashes)

    save_hashes(pool_dir, hashes)
    return hashes, new_count, stopped


# ─── Pair edges ─────────────────────────────────────────────────────────

def compute_edges(hashes, progress=None, should_stop=None):
    """All pairs with dist <= MAX_DIST. Returns (a, b, d) or None if stopped."""
    rels = sorted(hashes.keys())
    n = len(rels)
    if n < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.uint8)

    ph = np.array([hashes[r]["ph"] for r in rels], dtype=np.uint64)
    dh = np.array([hashes[r]["dh"] for r in rels], dtype=np.uint64)

    a_list, b_list, d_list = [], [], []
    chunk = 512
    for i0 in range(0, n, chunk):
        if should_stop and should_stop():
            return None
        i1 = min(i0 + chunk, n)
        for j0 in range(i0, n, chunk):
            j1 = min(j0 + chunk, n)
            phx = np.bitwise_count(ph[i0:i1, None] ^ ph[None, j0:j1])
            dhx = np.bitwise_count(dh[i0:i1, None] ^ dh[None, j0:j1])
            dist = np.minimum(phx, dhx)
            # upper triangle only (i < j) to avoid duplicate pairs
            rows, cols = np.nonzero(dist <= MAX_DIST)
            rows = rows + i0
            cols = cols + j0
            keep = rows < cols
            rows, cols = rows[keep], cols[keep]
            if rows.size:
                a_list.append(rows)
                b_list.append(cols)
                d_list.append(dist[rows - i0, cols - j0])
        if progress:
            progress(min(i1, n), n)

    if not a_list:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.uint8)
    return (np.concatenate(a_list), np.concatenate(b_list), np.concatenate(d_list))


def load_edges(pool_dir):
    p = Path(pool_dir) / EDGES_FILE
    if p.exists():
        try:
            z = np.load(p)
            return z["a"], z["b"], z["d"]
        except Exception:
            return None
    return None


def save_edges(pool_dir, a, b, d):
    np.savez(Path(pool_dir) / EDGES_FILE, a=a, b=b, d=d)


# ─── Families ───────────────────────────────────────────────────────────

def families_at_threshold(a, b, d, threshold):
    """Union-find connected components of pairs with dist <= threshold."""
    mask = d <= threshold
    a, b = a[mask], b[mask]
    n_nodes = int(max(a.max(), b.max())) + 1 if a.size else 0
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for x, y in zip(a, b):
        union(int(x), int(y))

    comps = {}
    for i in range(n_nodes):
        comps.setdefault(find(i), []).append(i)
    families = [sorted(v) for v in comps.values() if len(v) >= 2]
    families.sort(key=len, reverse=True)
    return families


def family_stats(a, b, d, threshold, n_images):
    fams = families_at_threshold(a, b, d, threshold)
    return {
        "families": len(fams),
        "images_in_families": sum(len(f) for f in fams),
        "isolated": n_images - sum(len(f) for f in fams),
        "max_family_size": max((len(f) for f in fams), default=0),
    }


# ─── Resolve helpers ────────────────────────────────────────────────────

def resolve_family(pool_dir, keep_paths, reject_paths, curated_dir, delete=False):
    """Move keepers to curated_dir (structure preserved); move rejects to pool
    trash (flat) or delete. Returns {"kept": [...], "trashed": [...], "errors": [...]}."""
    pool = Path(pool_dir)
    trash = pool / TRASH_DIR
    trash.mkdir(parents=True, exist_ok=True)
    curated = Path(curated_dir)
    if keep_paths:
        curated.mkdir(parents=True, exist_ok=True)

    kept, trashed, errors = [], [], []
    for src in keep_paths:
        try:
            # Preserve the pool's relative structure in the curated folder:
            # pool/<source>/<rel> -> curated/<source>/<rel>
            rel = os.path.relpath(src, pool_dir)
            dst = curated / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst = _uniquify(dst)
            shutil.move(src, dst)  # cross-device safe (rename would EXDEV on shares)
            kept.append(str(dst))
        except Exception as e:
            errors.append(f"{src}: {e}")
    for src in reject_paths:
        try:
            if delete:
                os.unlink(src)
                trashed.append(str(src) + " (deleted)")
            else:
                dst = _uniquify(trash / Path(src).name)
                shutil.move(src, dst)
                trashed.append(str(dst))
        except Exception as e:
            errors.append(f"{src}: {e}")
    return {"kept": kept, "trashed": trashed, "errors": errors}


def _uniquify(path):
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        cand = path.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def member_min_dist(family, a, b, d):
    """For each member index in a family, its min edge distance to another
    member of the same family (None if it has no in-family edge)."""
    fam_set = set(family)
    out = {}
    for idx in family:
        mask = (a == idx) | (b == idx)
        if not mask.any():
            out[idx] = None
            continue
        a2, b2, d2 = a[mask], b[mask], d[mask]
        both = np.array([(int(x) in fam_set and int(y) in fam_set) for x, y in zip(a2, b2)])
        if not both.any():
            out[idx] = None
        else:
            out[idx] = int(d2[both].min())
    return out


def prune_hashes(pool_dir, removed_rels):
    """Remove hashes entries for resolved images so re-clustering is consistent."""
    hashes = load_hashes(pool_dir)
    for rel in removed_rels:
        hashes.pop(rel, None)
    save_hashes(pool_dir, hashes)
    # Edges cache is stale after prune; drop it so it rebuilds fresh.
    ep = Path(pool_dir) / EDGES_FILE
    if ep.exists():
        ep.unlink()
