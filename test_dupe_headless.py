#!/usr/bin/env python3
"""Headless test for dupefind/dupe_pull/dupe_cluster on synthetic data.

Builds a fake ComfyUI-ish output tree with known dupe relationships:
  - exact byte dupes
  - near-identical (resize/re-encode)
  - face-detailer style (same scene, face region replaced)
  - re-render (same scene, different lighting)  [hard tier]
  - isolated images (genuinely different motifs — must stay out)

Then pulls into a pool, clusters, and asserts the expected families.
"""

import io
import os
import random
import shutil
import sys
import tempfile

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dupefind
import dupe_pull


# ─── Synthetic scene factory ───────────────────────────────────────────

def character_pos(seed, w, h):
    """Deterministic character placement — varies with seed."""
    cx = int(w * (0.2 + 0.6 * (seed % 5) / 4))
    cy = int(h * (0.45 + 0.2 * ((seed // 5) % 3) / 2))
    return cx, cy


def make_scene(seed, w=512, h=512):
    """A character scene — genuinely different per seed (colors + placement)."""
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    # seed-dependent vertical gradient
    c1 = (rng.randint(20, 90), rng.randint(20, 90), rng.randint(40, 120))
    c2 = (rng.randint(90, 200), rng.randint(90, 200), rng.randint(110, 220))
    for y in range(h):
        t = y / h
        fill = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
        d.line([(0, y), (w, y)], fill=fill)
    # sun only for some seeds
    if seed % 3 != 0:
        d.ellipse([w * 0.6, h * 0.08, w * 0.75, h * 0.23], fill=(255, 220, 140))
    # character at seed-dependent position
    cx, cy = character_pos(seed, w, h)
    head_r = 30 + (seed % 4) * 8
    skin = (rng.randint(180, 240), rng.randint(120, 200), rng.randint(100, 180))
    d.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r], fill=skin)
    body = (rng.randint(30, 200), rng.randint(30, 120), rng.randint(30, 120))
    d.rounded_rectangle([cx - head_r * 1.4, cy + head_r, cx + head_r * 1.4, cy + head_r * 4],
                        fill=body, outline=(30, 30, 30))
    return img


def face_detailer_pass(img, seed=1):
    """Simulate a face detailer: replace the head region with a very different face."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    cx, cy = character_pos(seed, out.width, out.height)
    rng = random.Random(seed * 7 + 3)
    skin2 = (rng.randint(150, 220), rng.randint(90, 160), rng.randint(80, 150))
    d.ellipse([cx - 48, cy - 68, cx + 48, cy + 28], fill=skin2)
    d.polygon([(cx - 20, cy - 20), (cx, cy - 35), (cx + 20, cy - 20)], fill=(240, 200, 180))
    d.rectangle([cx - 30, cy - 5, cx - 15, cy + 10], fill=(20, 20, 20))
    d.rectangle([cx + 15, cy - 5, cx + 30, cy + 10], fill=(20, 20, 20))
    return out


def re_render(img, seed):
    """Simulate a re-render: same composition, shifted palette."""
    out = img.copy()
    shift = (seed % 60) - 30
    out = out.point(lambda p: max(0, min(255, p + shift)))
    return out


def make_landscape(w=512, h=512):
    """A genuinely different composition: layered mountains, no character."""
    rng = random.Random(777)
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    # sky
    for y in range(int(h * 0.6)):
        t = y / (h * 0.6)
        d.line([(0, y), (w, y)], fill=(int(200 - 80 * t), int(170 - 60 * t), int(230 - 100 * t)))
    # mountains (two overlapping triangles)
    d.polygon([(0, int(h * 0.6)), (int(w * 0.4), int(h * 0.25)), (int(w * 0.8), int(h * 0.6))],
              fill=(70, 110, 90))
    d.polygon([(int(w * 0.3), int(h * 0.6)), (int(w * 0.7), int(h * 0.3)), (w, int(h * 0.6))],
              fill=(50, 85, 70))
    # snow caps
    d.polygon([(int(w * 0.4), int(h * 0.25)), (int(w * 0.36), int(h * 0.33)), (int(w * 0.44), int(h * 0.33))],
              fill=(240, 240, 250))
    # ground
    d.rectangle([0, int(h * 0.6), w, h], fill=(60, 100, 60))
    return img


def make_isolated(w=512, h=512):
    """A completely different motif (abstract stripes) — should never join a family."""
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for x in range(0, w, 64):
        d.rectangle([x, 0, x + 32, h], fill=(20, 30, 40) if (x // 64) % 2 else (200, 190, 60))
    return img


def save(img, path):
    img.save(path)
    return path


# ─── Test ──────────────────────────────────────────────────────────────

def build_source_tree(root):
    """Create a fake ComfyUI output tree with known relationships."""
    base = make_scene(1)
    base2 = make_landscape()  # genuinely different composition

    # output/ (main output dir)
    out = os.path.join(root, "output")
    os.makedirs(out, exist_ok=True)
    save(base, os.path.join(out, "render_0001.png"))
    save(base2, os.path.join(out, "render_0002.png"))
    save(face_detailer_pass(base, seed=1), os.path.join(out, "render_0001_facefix.png"))
    save(face_detailer_pass(base, seed=1), os.path.join(out, "render_0001_facefix_v2.png"))  # byte-dupe of facefix
    save(re_render(base, 10), os.path.join(out, "render_0001_rerender.png"))

    # output/Refreshed/ subfolder
    ref = os.path.join(out, "Refreshed")
    os.makedirs(ref, exist_ok=True)
    # near-identical: resize + re-encode of base
    small = base.resize((256, 256), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "PNG")
    buf.seek(0)
    save(Image.open(buf), os.path.join(ref, "refreshed_0001.png"))
    # genuinely different image
    save(make_isolated(), os.path.join(ref, "refreshed_isolated.png"))

    return root


def main():
    tmp = tempfile.mkdtemp(prefix="vs-dupe-test-")
    src = os.path.join(tmp, "src")
    pool = os.path.join(tmp, "pool")
    os.makedirs(src)
    print(f"tmp: {tmp}")
    build_source_tree(src)

    # ── 1. Pull ──
    print("\n=== dupe_pull ===")
    sys.argv = ["dupe_pull.py", pool, src]
    dupe_pull.main()
    manifest = dupe_pull.load_manifest(pool)
    assert manifest, "manifest missing"
    files = manifest["files"]
    assert len([f for f in files if f.get("pool_path")]) == 6, f"expected 6 moved, got {len(files)}"
    exact = [f for f in files if f.get("exact_dupe_of")]
    assert len(exact) == 1, f"expected 1 exact dupe, got {len(exact)}"
    # structure preserved: pool/<source>/output/Refreshed/...
    assert os.path.isdir(os.path.join(pool, "src", "output", "Refreshed")), "subfolder not preserved"
    print("  OK: 6 moved, 1 exact-dupe skipped, structure preserved")

    # source should be empty of images now
    remaining = dupe_pull.find_images(src)
    assert len(remaining) == 1, f"source should have only the exact-dupe left, got {len(remaining)}"

    # ── 2. Cluster ──
    print("\n=== dupe_cluster ===")
    hashes, new_count = dupefind.compute_hashes(pool)
    assert len(hashes) == 6, f"expected 6 hashed, got {len(hashes)}"
    a, b, d = dupefind.compute_edges(hashes)
    dupefind.save_edges(pool, a, b, d)

    rels = sorted(hashes.keys())

    def fam_members(thresh):
        fams = dupefind.families_at_threshold(a, b, d, thresh)
        return [[rels[i] for i in f] for f in fams]

    print("  -- threshold 2 (tight) --")
    tight = fam_members(2)
    for f in tight:
        print("   ", [os.path.basename(x) for x in f])
    # near-identical resize should pair with the original even at a tight threshold
    tight_flat = [os.path.basename(x) for f in tight for x in f]
    assert "refreshed_0001.png" in tight_flat and "render_0001.png" in tight_flat, \
        "resize not paired with original at tight threshold"

    print("  -- threshold 14 (loose) --")
    loose = fam_members(14)
    for f in loose:
        print("   ", sorted(os.path.basename(x) for x in f))
    loose_flat = [sorted(os.path.basename(x) for x in f) for f in loose]
    # face-detailer family: original + facefix should join
    facefix_fam = [f for f in loose_flat if any("facefix" in x for x in f)]
    assert facefix_fam, "face-detailer family missing at loose threshold"
    assert any("render_0001.png" in f for f in facefix_fam), "original not pulled into facefix family"
    # isolated image stays alone
    assert all(not any("isolated" in x for x in f) for f in loose_flat), \
        "isolated image should never be in a family"
    # different scene stays alone
    assert all("render_0002.png" not in f for f in loose_flat), "different scene should not join"

    print("  OK: face-detailer family formed, isolated + different-scene stay out")

    # ── 3. Re-run idempotency ──
    print("\n=== idempotency ===")
    before = len(dupefind.load_hashes(pool))
    hashes2, new2 = dupefind.compute_hashes(pool)
    assert new2 == 0, f"re-run should hash nothing new, got {new2}"
    print("  OK: checkpointed, 0 new hashes on re-run")

    # ── 4. Resolve helpers ──
    print("\n=== resolve ===")
    curated = os.path.join(tmp, "curated")
    fam = loose[0]  # a real family
    keep, reject = [fam[0]], fam[1:]
    abs_paths = {r: os.path.join(pool, r) for r in fam}
    res = dupefind.resolve_family(pool, [abs_paths[keep[0]]], [abs_paths[r] for r in reject], curated)
    assert len(res["kept"]) == 1 and len(res["trashed"]) == len(reject), f"resolve mismatch: {res}"
    assert os.path.exists(res["kept"][0]), "keeper not at curated dest"
    assert os.path.isdir(os.path.join(pool, dupefind.TRASH_DIR)), "trash dir missing"
    print(f"  OK: 1 kept -> curated, {len(reject)} trashed")
    dupefind.prune_hashes(pool, keep + reject)
    assert len(dupefind.load_hashes(pool)) == before - len(fam), "prune didn't remove resolved hashes"

    print("\nALL TESTS PASSED ✅")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"cleaned {tmp}")


if __name__ == "__main__":
    main()
