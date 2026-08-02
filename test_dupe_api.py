#!/usr/bin/env python3
"""Integration test: dupe workflow through the real Flask routes.

Tests are intentionally ordered (test_01..test_05) because they share one
pool fixture: status -> pull -> cluster -> resolve -> trash-index check.
"""
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Point sifter at a temp DB/thumbnails before importing routes
import sifter
sifter.DB_PATH = Path(tempfile.mkdtemp(prefix="vs-db-")) / "sifter.db"
sifter.THUMB_DIR = Path(tempfile.mkdtemp(prefix="vs-thumbs-"))

import dupefind
import dupe_pull
from test_dupe_headless import make_scene, make_landscape, face_detailer_pass, re_render, save

POOL = None
SRC = None


def setUpModule():
    global POOL, SRC
    base = Path(tempfile.mkdtemp(prefix="vs-dupe-api-"))
    POOL = base / "pool"
    SRC = base / "source"
    SRC.mkdir(parents=True)
    (SRC / "output").mkdir(parents=True)
    (SRC / "output" / "Refreshed").mkdir(parents=True)

    # render_0001.png and refreshed_0001.png are byte-identical (same seed):
    # the puller records the second as an exact dupe and leaves it in source.
    save(make_scene(1, 640, 640), SRC / "output" / "render_0001.png")
    save(face_detailer_pass(make_scene(1, 640, 640), 1), SRC / "output" / "render_0001_facefix.png")
    save(re_render(make_scene(1, 640, 640), 1), SRC / "output" / "render_0001_rerender.png")
    save(make_scene(1, 640, 640), SRC / "output" / "Refreshed" / "refreshed_0001.png")  # exact dupe
    save(make_scene(2, 640, 640), SRC / "output" / "render_0002.png")
    save(face_detailer_pass(make_scene(2, 640, 640), 2), SRC / "output" / "render_0002_facefix.png")
    save(make_landscape(640, 640), SRC / "output" / "landscape_01.png")

    sifter.VAULT_ROOT = str(POOL)
    sifter.init_db()
    sifter.index_directory(POOL, force=False)


class DupeApiTest(unittest.TestCase):
    def setUp(self):
        self.client = sifter.app.test_client()

    # ── 01 · empty pool status ───────────────────────────────────────────
    def test_01_status_empty(self):
        r = self.client.get("/api/dupes/status")
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual(d["images_on_disk"], 0)
        self.assertFalse(d["has_manifest"])
        self.assertFalse(d["running"])

    # ── 02 · pull moves into pool, structure preserved, exact dupes noted ─
    def test_02_pull(self):
        r = self.client.post("/api/dupes/pull", json={"source": str(SRC)})
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual(d["moved"], 6)           # 7 files, 1 exact dupe stays
        self.assertEqual(d["exact_dupes"], 1)     # refreshed_0001 = render_0001
        self.assertEqual(d["errors"], [])
        # Structure preserved: pool/source/output/render_0001.png (moved file)
        self.assertTrue((POOL / "source" / "output" / "render_0001.png").exists())
        # The Refreshed dir had only the exact-dupe file, so nothing moved from it:
        # it legitimately does NOT exist in the pool (empty dirs aren't recreated)
        self.assertFalse((POOL / "source" / "output" / "Refreshed").exists())
        # Source keeps only the exact dupe
        remaining = list(SRC.rglob("*.png"))
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].name, "refreshed_0001.png")
        # Reindex picked up the moved files
        r2 = self.client.get("/api/images?limit=50&offset=0")
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.get_json()["total"], 6)

    # ── 03 · cluster + families with DB ids and working thumbnails ───────
    def test_03_cluster_and_families(self):
        self.client.post("/api/dupes/cluster", json={})
        for _ in range(60):
            p = self.client.get("/api/dupes/cluster/progress").get_json()
            if not p["running"]:
                break
            time.sleep(0.2)
        self.assertFalse(p["running"], f"cluster did not finish: {p}")
        self.assertEqual(p["errors"], 0)

        r = self.client.get("/api/dupes/families?confidence=70")
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertGreaterEqual(d["stats"]["families"], 1)
        # Every member maps to an indexed DB id and an existing file
        for fam in d["families"]:
            for m in fam["members"]:
                self.assertIsNotNone(m["id"], f"member {m['rel']} has no DB id")
                self.assertTrue(os.path.exists(m["filepath"]), f"member file missing: {m['filepath']}")
        # Thumbnails resolve for every member
        for fam in d["families"]:
            for m in fam["members"]:
                tr = self.client.get(f"/api/thumbnail/{m['id']}")
                self.assertEqual(tr.status_code, 200, f"thumb failed for {m['rel']}")
        # Landscape (different scene) must NOT be in any family
        for fam in d["families"]:
            for m in fam["members"]:
                self.assertNotIn("landscape_01", m["rel"], "landscape joined a family")

    # ── 04 · resolve: keeper -> curated (structure preserved), rejects -> trash
    def test_04_resolve(self):
        d = self.client.get("/api/dupes/families?confidence=70").get_json()
        self.assertGreaterEqual(len(d["families"]), 1)
        fam = d["families"][0]
        keep = [fam["members"][0]["filepath"]]
        reject = [m["filepath"] for m in fam["members"][1:]]
        self.assertTrue(reject, "family needs 2+ members to test resolve")

        r = self.client.post("/api/dupes/resolve", json={"keep": keep, "reject": reject})
        self.assertEqual(r.status_code, 200)
        res = r.get_json()
        self.assertEqual(len(res["kept"]), 1)
        self.assertEqual(len(res["trashed"]), len(reject))
        self.assertEqual(res["errors"], [])
        # Keeper moved into curated dir, pool-relative structure preserved
        curated = Path(res["curated_dir"])
        self.assertTrue(os.path.exists(res["kept"][0]))
        self.assertTrue(str(res["kept"][0]).startswith(str(curated)))
        # Rejects in trash
        for t in res["trashed"]:
            self.assertTrue(t.endswith("(deleted)") or t.startswith(str(POOL / ".dupe-trash")))
        # DB rows: rejects gone
        with sifter.db() as conn:
            for m in fam["members"][1:]:
                row = conn.execute("SELECT id FROM images WHERE filepath = ?", (m["filepath"],)).fetchone()
                self.assertIsNone(row, f"reject row still in DB: {m['filepath']}")
        # Hashes pruned -> re-cluster stays consistent
        hashes = dupefind.load_hashes(str(POOL))
        for rel in [os.path.relpath(p, POOL) for p in reject]:
            self.assertNotIn(rel, hashes)

    # ── 05 · .dupe-trash is never indexed ────────────────────────────────
    def test_05_trash_not_indexed(self):
        trash = POOL / ".dupe-trash"
        trash.mkdir(exist_ok=True)
        save(make_scene(99, 320, 320), trash / "junk.png")
        sifter.index_directory(str(POOL), force=False)
        with sifter.db() as conn:
            row = conn.execute("SELECT id FROM images WHERE filepath LIKE '%junk.png'").fetchone()
        self.assertIsNone(row, ".dupe-trash image got indexed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
