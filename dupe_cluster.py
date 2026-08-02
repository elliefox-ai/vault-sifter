#!/usr/bin/env python3
"""dupe_cluster.py — cluster a review pool into near-dupe families.

Usage:
    python3 dupe_cluster.py <pool_dir> [--threshold N] [--report]

- Hashes every image (checkpointed in .dupe-hashes.json — re-runs only hash
  new/changed files).
- Computes pairwise distances (cached in .dupe-edges.npz; recomputed when
  hashes change).
- Reports family stats at the given threshold (default 12).

Works on any folder, but pairs naturally with dupe_pull.py pools
(pool has manifest.json + preserved folder structure).
"""

import argparse
import json
import os
import sys
import time

import dupefind


def main():
    ap = argparse.ArgumentParser(description="Cluster a pool into dupe families")
    ap.add_argument("pool", help="Pool/vault directory")
    ap.add_argument("--threshold", type=int, default=12,
                    help="Hamming distance cutoff (0=identical, 40=very loose). Default 12.")
    ap.add_argument("--report", action="store_true",
                    help="Print full family member listing")
    args = ap.parse_args()

    pool = os.path.abspath(args.pool)
    if not os.path.isdir(pool):
        print(f"Error: not a directory: {pool}")
        sys.exit(1)

    t0 = time.time()
    print(f"Pool: {pool}")
    print("Hashing images (checkpointed)...")
    hashes, new_count = dupefind.compute_hashes(pool, progress=lambda done, total: (
        print(f"  hashed {done}/{total}", end="\r", flush=True) if done % 100 == 0 or done == total else None
    ))
    print(f"  {len(hashes)} images hashed ({new_count} new)")

    # Recompute edges if hashes changed since the cache
    edges = dupefind.load_edges(pool)
    hashes_mtime = os.path.getmtime(os.path.join(pool, dupefind.HASHES_FILE))
    edges_mtime = os.path.getmtime(os.path.join(pool, dupefind.EDGES_FILE)) if edges is not None else -1
    if edges is None or edges_mtime < hashes_mtime:
        print("Computing pairwise distances...")
        a, b, d = dupefind.compute_edges(hashes)
        dupefind.save_edges(pool, a, b, d)
        print(f"  {len(d)} candidate pairs (dist <= {dupefind.MAX_DIST})")
    else:
        a, b, d = edges
        print(f"  {len(d)} candidate pairs (cached)")

    stats = dupefind.family_stats(a, b, d, args.threshold, len(hashes))
    print()
    print(f"Threshold: {args.threshold} (confidence ~{dupefind.confidence_for_threshold(args.threshold)}%)")
    print(f"  families:           {stats['families']}")
    print(f"  images in families: {stats['images_in_families']}")
    print(f"  isolated images:    {stats['isolated']}")
    print(f"  largest family:     {stats['max_family_size']}")

    if args.report:
        rels = sorted(hashes.keys())
        fams = dupefind.families_at_threshold(a, b, d, args.threshold)
        for i, fam in enumerate(fams, 1):
            print(f"\nFamily {i} ({len(fam)} members):")
            for idx in fam:
                print(f"    {rels[idx]}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
