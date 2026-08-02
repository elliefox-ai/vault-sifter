#!/usr/bin/env python3
"""dupe_pull.py — move a discrete set of images into a review pool.

Usage:
    python3 dupe_pull.py <pool_dir> <source_dir> [source_dir ...] [--dry-run]

Behavior:
    - MOVES (not copies) image files into <pool_dir>, preserving each source's
      relative folder structure under a top-level folder named after the source.
    - Byte-identical files (same SHA256) are not duplicated: the second
      occurrence is recorded in manifest.json as an exact dupe and left in place.
    - Writes manifest.json in the pool root tracking every origin → pool mapping
      (backup signal; the visible folder tree is the primary signal).
    - Idempotent: re-running skips files already recorded as moved.

Example:
    python3 dupe_pull.py /mnt/shared/_DupePool "/mnt/shared/_Incoming/Style Examples" /mnt/shared/_Incoming/_Caricature
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dupefind import IMAGE_EXTS, MANIFEST_FILE

try:
    from dupefind import TRASH_DIR  # reuse the same exclusion name
except ImportError:
    TRASH_DIR = ".dupe-trash"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def find_images(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                out.append(str(Path(dirpath) / fn))
    return out


def load_manifest(pool_dir):
    p = Path(pool_dir) / MANIFEST_FILE
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def save_manifest(pool_dir, manifest):
    Path(pool_dir).mkdir(parents=True, exist_ok=True)
    (Path(pool_dir) / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))


def uniquify(dest):
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 1
    while True:
        cand = dest.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def pull_into_pool(pool_dir, sources, dry_run=False):
    """Move images from source dirs into the pool, preserving structure.

    Returns a summary dict: {"pool", "moved", "already", "exact_dupes",
    "errors", "dry_run", "manifest"}. exact_dupes is a list of
    (origin, of_pool_path) tuples.
    """
    pool = Path(pool_dir).resolve()
    pool.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(pool)
    if manifest is None:
        manifest = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": [],
            "files": [],
        }

    moved_origins = {f["origin"] for f in manifest["files"] if f.get("pool_path")}
    # sha256 -> pool_path for already-moved files (exact-dupe detection)
    sha_to_pool = {f["sha256"]: f["pool_path"] for f in manifest["files"] if f.get("pool_path")}

    planned = []   # (origin, dest) to move
    exact_dupes = []  # (origin, of_pool_path)
    already = 0

    for src_arg in sources:
        src = Path(src_arg).resolve()
        if not src.is_dir():
            print(f"[warn] not a directory: {src}")
            continue
        top = src.name  # top-level folder name in the pool
        if top not in manifest["sources"]:
            manifest["sources"].append(top)

        for origin in find_images(src):
            if origin in moved_origins:
                already += 1
                continue

            size = os.path.getsize(origin)
            digest = sha256_file(origin)

            # Exact duplicate of something already in the pool?
            if digest in sha_to_pool:
                exact_dupes.append((origin, sha_to_pool[digest]))
                manifest["files"].append({
                    "origin": origin,
                    "exact_dupe_of": sha_to_pool[digest],
                    "sha256": digest,
                    "size": size,
                    "skipped_at": time.time(),
                })
                continue

            rel = Path(origin).relative_to(src)
            dest = pool / top / rel
            dest = uniquify(dest)
            planned.append((origin, str(dest)))
            sha_to_pool[digest] = str(dest)

    # ── Execute ──
    moved = 0
    errors = []
    for origin, dest in planned:
        if dry_run:
            print(f"  would move: {origin}\n           -> {dest}")
            continue
        try:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            if os.path.abspath(origin) == os.path.abspath(dest):
                continue
            shutil.move(origin, dest)
            manifest["files"].append({
                "origin": origin,
                "pool_path": dest,
                "sha256": sha256_file(dest),
                "size": os.path.getsize(dest),
                "moved_at": time.time(),
            })
            moved += 1
            if moved % 50 == 0:
                save_manifest(pool, manifest)
        except Exception as e:
            errors.append(f"{origin}: {e}")

    save_manifest(pool, manifest)
    return {
        "pool": str(pool),
        "moved": moved,
        "already": already,
        "exact_dupes": exact_dupes,
        "errors": errors,
        "dry_run": dry_run,
        "manifest": str(pool / MANIFEST_FILE),
    }


def main():
    ap = argparse.ArgumentParser(description="Pull images into a review pool (moves, not copies).")
    ap.add_argument("pool", help="Target pool directory (created if missing)")
    ap.add_argument("sources", nargs="+", help="Source directories to pull from")
    ap.add_argument("--dry-run", action="store_true", help="Report what would move without moving")
    args = ap.parse_args()

    res = pull_into_pool(args.pool, args.sources, dry_run=args.dry_run)

    print(f"Pool: {res['pool']}")
    print(f"  moved:        {res['moved']}" + (" (dry run)" if res["dry_run"] else ""))
    print(f"  already:      {res['already']}")
    print(f"  exact dupes:  {len(res['exact_dupes'])} (left in source, recorded)")
    for origin, of in res["exact_dupes"]:
        print(f"    {origin}\n      = {of}")
    print(f"  errors:       {len(res['errors'])}")
    for e in res["errors"]:
        print(f"    {e}")
    print(f"  manifest:     {res['manifest']}")


if __name__ == "__main__":
    main()
