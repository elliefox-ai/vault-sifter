#!/usr/bin/env python3
"""Vault Sifter — local image vault review tool."""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from io import BytesIO

from flask import Flask, jsonify, request, send_file, send_from_directory, abort
from analyzer import analyze_image, quality_score
import dupefind
import dupe_pull

# Resolve paths for both normal and PyInstaller-bundled execution
if getattr(sys, 'frozen', False):
    # PyInstaller: static files are in sys._MEIPASS/static
    _STATIC_DIR = os.path.join(sys._MEIPASS, "static")
else:
    _STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = Flask(__name__, static_folder=_STATIC_DIR, static_url_path="/static")


# ─── Security ─────────────────────────────────────────────────────────────────

@app.before_request
def check_origin():
    """Reject cross-origin POSTs. Localhost-only tool, but defense in depth."""
    if request.method in ("POST", "PUT", "DELETE"):
        origin = request.headers.get("Origin", "")
        if origin and "localhost" not in origin and "127.0.0.1" not in origin:
            return jsonify({"error": "Cross-origin requests not allowed"}), 403


@app.before_request
def require_vault():
    """Vault-less mode: only the loader endpoints work until a folder is loaded."""
    if VAULT_ROOT is None:
        allowed = request.path == "/" or request.path.startswith("/static") \
            or request.path in ("/api/vault", "/api/load")
        if not allowed:
            return jsonify({"error": "No vault loaded — use Load Folder in the UI"}), 409

VAULT_HOME = Path.home() / ".vault-sifter"
THUMB_DIR = VAULT_HOME / "thumbnails"
VAULT_ROOT = None
DB_PATH = None  # Set per-vault in main()


# ─── Database ────────────────────────────────────────────────────────────────

def init_db():
    global DB_PATH
    # Per-vault DB: hash the vault path so each folder gets its own ratings/rejects
    vault_hash = hashlib.md5(VAULT_ROOT.encode()).hexdigest()[:16]
    db_name = f"vault_{vault_hash}.db"
    DB_PATH = VAULT_HOME / db_name
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            directory TEXT NOT NULL,
            filesize INTEGER,
            width INTEGER,
            height INTEGER,
            prompt TEXT,
            workflow TEXT,
            model TEXT,
            seed TEXT,
            file_created REAL,
            file_modified REAL,
            rating INTEGER DEFAULT 0,
            flag INTEGER DEFAULT 0,
            rejected INTEGER DEFAULT 0,
            tags TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            indexed_at REAL,
            -- Auto-analysis metrics
            sharpness REAL,
            brightness REAL,
            contrast REAL,
            saturation REAL,
            entropy REAL,
            quality_score REAL,
            is_grayscale INTEGER DEFAULT 0,
            mean_color TEXT,
            unique_colors INTEGER,
            file_corrupt INTEGER DEFAULT 0,
            analyzed_at REAL,
            UNIQUE(filepath)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn

@contextlib.contextmanager
def db():
    """Context manager that auto-closes the connection."""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


# ─── Metadata Extraction ──────────────────────────────────────────────────────

def extract_png_metadata(filepath):
    """Extract prompt/workflow metadata from PNG tEXt chunks (ComfyUI format)."""
    def _as_str(value):
        """Coerce ComfyUI graph input values to plain strings.
        Connected inputs serialize as [node_id, slot] lists; some loaders
        store arrays. SQLite can't bind lists, so flatten defensively."""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            return ", ".join(
                _as_str(v) for v in value if not isinstance(v, (list, dict))
            )
        return ""
    info = {}
    try:
        from PIL import Image, ImageOps
        img = Image.open(filepath)
        img = ImageOps.exif_transpose(img)
        info["width"] = img.width
        info["height"] = img.height

        raw_info = img.info or {}

        # ComfyUI stores 'prompt' and 'workflow' as JSON strings in tEXt chunks
        prompt_data = None
        workflow_data = None

        if "prompt" in raw_info:
            try:
                prompt_data = json.loads(raw_info["prompt"])
            except (json.JSONDecodeError, TypeError):
                prompt_data = None

        if "workflow" in raw_info:
            try:
                workflow_data = json.loads(raw_info["workflow"])
            except (json.JSONDecodeError, TypeError):
                workflow_data = None

        # Extract prompt text and model from the parsed prompt graph
        prompt_text = ""
        model = ""
        seed = ""

        if prompt_data:
            # ComfyUI prompt is a dict of nodes: {"3": {"class_type": "...", "inputs": {...}}}
            for node_id, node in prompt_data.items():
                if not isinstance(node, dict):
                    continue
                class_type = node.get("class_type", "")
                inputs = node.get("inputs", {})

                # Capture positive prompt text
                if class_type in ("CLIPTextEncode", "CLIPTextEncodeSDXL"):
                    text = inputs.get("text", "")
                    if isinstance(text, str) and len(text) > len(prompt_text):
                        prompt_text = text

                # Capture model name
                if "model" in class_type.lower() or class_type in (
                    "CheckpointLoaderSimple", "UNETLoader", "LoraLoader"
                ):
                    if "ckpt_name" in inputs:
                        model = _as_str(inputs["ckpt_name"])
                    elif "unet_name" in inputs:
                        model = _as_str(inputs["unet_name"])
                    elif "lora_name" in inputs and not model:
                        model = _as_str(inputs["lora_name"])

                # Capture seed
                if "seed" in inputs:
                    seed = _as_str(inputs["seed"])
                elif class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"):
                    if "noise_seed" in inputs:
                        seed = _as_str(inputs["noise_seed"])

        info["prompt"] = prompt_text
        info["workflow"] = raw_info.get("workflow", "")
        info["model"] = model
        info["seed"] = seed

    except Exception as e:
        info["error"] = str(e)

    return info


# ─── Indexing ────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


_index_lock = threading.Lock()

def index_directory(vault_path, force=False):
    """Scan a directory and index all images.

    Serialized: only one indexer runs per process. Concurrent calls
    (Load Folder spam, reindex while bg-index runs) queue instead of
    colliding on the SQLite write lock.
    """
    with _index_lock:
        return _index_directory_locked(vault_path, force)

def _index_directory_locked(vault_path, force=False):
    """Actual scan — callers must hold _index_lock."""
    vault_path = Path(vault_path).resolve()
    conn = get_db()

    indexed = 0
    skipped = 0
    errors = 0

    try:
        for filepath in vault_path.rglob("*"):
            # Skip internal dot-dirs (dupe trash, tooling) so they never re-enter the vault
            if any(part.startswith(".") for part in filepath.relative_to(vault_path).parts):
                continue
            if filepath.suffix.lower() not in IMAGE_EXTS:
                continue

            rel_path = str(filepath)

            # Check if already indexed
            if not force:
                existing = conn.execute(
                    "SELECT id FROM images WHERE filepath = ?", (rel_path,)
                ).fetchone()
                if existing:
                    skipped += 1
                    continue

            stat = filepath.stat()
            meta = {"prompt": "", "workflow": "", "model": "", "seed": "", "width": None, "height": None}

            if filepath.suffix.lower() == ".png":
                meta = extract_png_metadata(filepath)

            # Skip analysis during initial index — run on-demand via /api/analyze
            # This keeps indexing fast (metadata only) and analysis separate

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO images
                    (filepath, filename, directory, filesize, width, height,
                     prompt, workflow, model, seed, file_created, file_modified,
                     rating, flag, rejected, tags, notes, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, '', '', ?)
                """, (
                    rel_path,
                    filepath.name,
                    str(filepath.parent),
                    stat.st_size,
                    meta.get("width"),
                    meta.get("height"),
                    meta.get("prompt", ""),
                    meta.get("workflow", "")[:50000],  # truncate huge workflows
                    meta.get("model", ""),
                    meta.get("seed", ""),
                    stat.st_ctime,
                    stat.st_mtime,
                    time.time(),
                ))
                indexed += 1
            except Exception as e:
                print(f"Error indexing {filepath}: {e}")
                errors += 1

            # Batch commit every 50 images
            if (indexed + errors) % 50 == 0:
                conn.commit()
    finally:
        conn.commit()
        conn.close()
    return {"indexed": indexed, "skipped": skipped, "errors": errors}


def generate_thumbnail(filepath, max_size=400):
    """Generate a thumbnail for an image, cached on disk."""
    filepath = Path(filepath)
    thumb_hash = hashlib.md5(str(filepath).encode()).hexdigest()[:12]
    thumb_name = f"{thumb_hash}_{filepath.name}.jpg"
    thumb_path = THUMB_DIR / thumb_name

    if thumb_path.exists():
        return thumb_path

    from PIL import Image, ImageOps
    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)  # Auto-rotate phone photos
    img.thumbnail((max_size, max_size), Image.LANCZOS)

    # Convert to RGB for JPEG (handles RGBA PNGs)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    img.save(thumb_path, "JPEG", quality=85)
    return thumb_path


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(_STATIC_DIR, "index.html")


@app.route("/api/stats")
def stats():
    with db() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM images").fetchone()["c"]
        rated = conn.execute("SELECT COUNT(*) as c FROM images WHERE rating > 0").fetchone()["c"]
        rejected = conn.execute("SELECT COUNT(*) as c FROM images WHERE rejected = 1").fetchone()["c"]
        flagged = conn.execute("SELECT COUNT(*) as c FROM images WHERE flag = 1").fetchone()["c"]
        unrated = conn.execute(
            "SELECT COUNT(*) as c FROM images WHERE rating = 0 AND rejected = 0"
        ).fetchone()["c"]
    return jsonify({
        "total": total, "rated": rated, "rejected": rejected,
        "flagged": flagged, "unrated": unrated
    })


@app.route("/api/images")
def list_images():
    """List images with optional filtering, sorting, and pagination."""
    with db() as conn:

        # Filters
        min_rating = request.args.get("min_rating", 0, type=int)
        max_rating = request.args.get("max_rating", 999, type=int)
        min_quality = request.args.get("min_quality", 0, type=float)
        max_quality = request.args.get("max_quality", 999, type=float)
        unrated_only = request.args.get("unrated_only", "false") == "true"
        include_rejected = request.args.get("include_rejected", "false") == "true"
        prompt_search = request.args.get("q", "")
        model_filter = request.args.get("model", "")
        directory_filter = request.args.get("dir", "")

        # Sort
        sort = request.args.get("sort", "file_modified")
        sort_dir = request.args.get("sort_dir", "desc").lower()
        if sort_dir not in ("asc", "desc"):
            sort_dir = "desc"
        valid_sorts = {"file_modified", "file_created", "filesize", "filename", "rating", "indexed_at", "quality_score", "sharpness", "saturation", "brightness", "contrast"}
        if sort not in valid_sorts:
            sort = "file_modified"

        # Pagination
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)
        limit = min(limit, 200)

        # Build WHERE clause separately so count doesn't need string manipulation
        where_clause = " WHERE 1=1"
        params = []
        if not include_rejected:
            where_clause += " AND rejected = 0"
        if unrated_only:
            where_clause += " AND rating = 0 AND rejected = 0"
        if min_rating > 0:
            where_clause += " AND rating >= ?"
            params.append(min_rating)
        if max_rating < 999:
            where_clause += " AND rating <= ?"
            params.append(max_rating)
        if min_quality > 0:
            where_clause += " AND quality_score >= ?"
            params.append(min_quality)
        if max_quality < 999:
            where_clause += " AND quality_score <= ?"
            params.append(max_quality)
        if prompt_search:
            where_clause += " AND prompt LIKE ?"
            params.append(f"%{prompt_search}%")
        if model_filter:
            where_clause += " AND model LIKE ?"
            params.append(f"%{model_filter}%")
        if directory_filter:
            where_clause += " AND directory LIKE ?"
            params.append(f"%{directory_filter}%")

        # Count total matching (before LIMIT/OFFSET)
        count_query = "SELECT COUNT(*) as c FROM images" + where_clause
        total = conn.execute(count_query, params).fetchone()["c"]

        query = "SELECT * FROM images" + where_clause + f" ORDER BY {sort} {sort_dir} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()

    return jsonify({"images": [dict(r) for r in rows], "total": total})


@app.route("/api/image/<int:image_id>")
def get_image(image_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if not row:
        abort(404)
    return jsonify(dict(row))


@app.route("/api/image/<int:image_id>/rate", methods=["POST"])
def rate_image(image_id):
    rating = request.json.get("rating", 0)
    rating = max(0, min(5, int(rating)))
    with db() as conn:
        conn.execute("UPDATE images SET rating = ? WHERE id = ?", (rating, image_id))
        conn.commit()
    return jsonify({"status": "ok", "rating": rating})


@app.route("/api/image/<int:image_id>/reject", methods=["POST"])
def reject_image(image_id):
    rejected = request.json.get("rejected", True)
    with db() as conn:
        conn.execute("UPDATE images SET rejected = ? WHERE id = ?", (1 if rejected else 0, image_id))
        conn.commit()
    return jsonify({"status": "ok", "rejected": rejected})


@app.route("/api/image/<int:image_id>/flag", methods=["POST"])
def flag_image(image_id):
    flag = request.json.get("flag", True)
    with db() as conn:
        conn.execute("UPDATE images SET flag = ? WHERE id = ?", (1 if flag else 0, image_id))
        conn.commit()
    return jsonify({"status": "ok", "flag": flag})


@app.route("/api/image/<int:image_id>/tags", methods=["POST"])
def tag_image(image_id):
    tags = request.json.get("tags", "")
    with db() as conn:
        conn.execute("UPDATE images SET tags = ? WHERE id = ?", (tags, image_id))
        conn.commit()
    return jsonify({"status": "ok", "tags": tags})


@app.route("/api/image/<int:image_id>/notes", methods=["POST"])
def note_image(image_id):
    notes = request.json.get("notes", "")
    with db() as conn:
        conn.execute("UPDATE images SET notes = ? WHERE id = ?", (notes, image_id))
        conn.commit()
    return jsonify({"status": "ok", "notes": notes})


@app.route("/api/thumbnail/<int:image_id>")
def thumbnail(image_id):
    with db() as conn:
        row = conn.execute("SELECT filepath FROM images WHERE id = ?", (image_id,)).fetchone()
    if not row:
        abort(404)

    filepath = row["filepath"]
    if not os.path.exists(filepath):
        abort(404)

    try:
        thumb_path = generate_thumbnail(filepath)
        return send_file(str(thumb_path), mimetype="image/jpeg")
    except Exception as e:
        abort(500, str(e))


@app.route("/api/fullimage/<int:image_id>")
def full_image(image_id):
    with db() as conn:
        row = conn.execute("SELECT filepath FROM images WHERE id = ?", (image_id,)).fetchone()
    if not row:
        abort(404)

    filepath = row["filepath"]
    if not os.path.exists(filepath):
        abort(404)

    return send_file(filepath)


@app.route("/api/models")
def list_models():
    with db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT model FROM images WHERE model != '' ORDER BY model"
        ).fetchall()
    return jsonify([r["model"] for r in rows])


@app.route("/api/directories")
def list_directories():
    with db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT directory FROM images ORDER BY directory"
        ).fetchall()
    return jsonify([r["directory"] for r in rows])


@app.route("/api/bulk/delete-rejected", methods=["POST"])
def bulk_delete_rejected():
    with db() as conn:
        rows = conn.execute("SELECT id, filepath FROM images WHERE rejected = 1").fetchall()
        deleted = 0
        errors = 0
        for row in rows:
            try:
                os.unlink(row["filepath"])
                # Clean up orphaned thumbnail
                thumb_hash = hashlib.md5(str(row["filepath"]).encode()).hexdigest()[:12]
                for f in THUMB_DIR.glob(f"{thumb_hash}_*"):
                    f.unlink(missing_ok=True)
                deleted += 1
            except Exception as e:
                print(f"Error deleting {row['filepath']}: {e}")
                errors += 1

        conn.execute("DELETE FROM images WHERE rejected = 1")
        conn.commit()
    return jsonify({"deleted": deleted, "errors": errors})


@app.route("/api/bulk/move-rated", methods=["POST"])
def bulk_move_rated():
    """Move rated images to a destination directory."""
    dest = request.json.get("destination")
    if not dest:
        abort(400, "destination required")

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    with db() as conn:
        rows = conn.execute(
            "SELECT id, filepath, filename, rating FROM images WHERE rating >= 1"
        ).fetchall()

        moved = 0
        errors = 0
        for row in rows:
            try:
                src = Path(row["filepath"])
                # Prefix with rating for sorting
                dst = dest_path / f"r{row['rating']}_{row['filename']}"
                shutil.move(str(src), str(dst))
                conn.execute(
                    "UPDATE images SET filepath = ? WHERE id = ?",
                    (str(dst), row["id"]),
                )
                moved += 1
            except Exception as e:
                print(f"Error moving {row['filepath']}: {e}")
                errors += 1

        conn.commit()
    return jsonify({"moved": moved, "errors": errors})


@app.route("/api/session/<key>", methods=["GET", "POST"])
def session_state(key):
    with db() as conn:
        if request.method == "POST":
            value = request.json.get("value", "")
            conn.execute(
                "INSERT OR REPLACE INTO session_state (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()
            return jsonify({"status": "ok"})
        else:
            row = conn.execute(
                "SELECT value FROM session_state WHERE key = ?", (key,)
            ).fetchone()
            return jsonify({"key": key, "value": row["value"] if row else ""})


@app.route("/api/reindex", methods=["POST"])
def reindex():
    global VAULT_ROOT
    if not VAULT_ROOT:
        return jsonify({"error": "No vault path configured"}), 400
    force = request.json.get("force", False)
    result = index_directory(VAULT_ROOT, force=force)
    return jsonify(result)


@app.route("/api/vault")
def current_vault():
    return jsonify({"vault": VAULT_ROOT})


@app.route("/api/load", methods=["POST"])
def load_vault():
    """Hot-switch the vault folder without restarting the server."""
    global VAULT_ROOT
    raw = (request.json or {}).get("path", "")
    # Windows "Copy as path" gives quoted paths; tolerate that
    path = raw.strip().strip('"').strip("'")
    if not path:
        return jsonify({"error": "No path given"}), 400
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        return jsonify({"error": f"Not a directory: {resolved}"}), 400

    VAULT_ROOT = str(resolved)
    init_db()  # per-vault DB (hash of path), so ratings never mix

    def _bg_index():
        try:
            result = index_directory(VAULT_ROOT, force=False)
            print(f"  [load-vault] {VAULT_ROOT}: {result['indexed']} indexed, "
                  f"{result['skipped']} skipped, {result['errors']} errors")
        except Exception as e:
            print(f"  [load-vault] indexing error: {e}")

    threading.Thread(target=_bg_index, daemon=True).start()
    return jsonify({"status": "ok", "vault": VAULT_ROOT})


_analysis_lock = threading.Lock()
_analysis_state = {"running": False, "total": 0, "done": 0, "errors": 0}

@app.route("/api/analyze", methods=["POST"])
def analyze_all():
    """Run quality analysis in background. Returns immediately."""
    if _analysis_state["running"]:
        return jsonify({"status": "already_running", **_analysis_state})

    with db() as conn:
        rows = conn.execute(
            "SELECT id, filepath FROM images WHERE analyzed_at IS NULL AND file_corrupt = 0"
        ).fetchall()

    if _analysis_state["running"]:
        return jsonify({"status": "already_running", **_analysis_state})

    with _analysis_lock:
        if _analysis_state["running"]:
            return jsonify({"status": "already_running", **_analysis_state})
        _analysis_state["total"] = len(rows)
        _analysis_state["done"] = 0
        _analysis_state["errors"] = 0
        _analysis_state["running"] = True

    def run_analysis(rows):
        conn = get_db()
        for row in rows:
            try:
                metrics = analyze_image(row["filepath"])
                qs = quality_score(metrics)
                conn.execute("""
                    UPDATE images SET
                        sharpness = ?, brightness = ?, contrast = ?, saturation = ?,
                        entropy = ?, quality_score = ?, is_grayscale = ?,
                        mean_color = ?, unique_colors = ?, file_corrupt = ?,
                        analyzed_at = ?
                    WHERE id = ?
                """, (
                    metrics.get("sharpness"),
                    metrics.get("brightness"),
                    metrics.get("contrast"),
                    metrics.get("saturation"),
                    metrics.get("entropy"),
                    qs,
                    1 if metrics.get("is_grayscale") else 0,
                    json.dumps(metrics.get("mean_color")) if metrics.get("mean_color") else None,
                    metrics.get("unique_colors"),
                    1 if metrics.get("file_corrupt") else 0,
                    time.time(),
                    row["id"],
                ))
                _analysis_state["done"] += 1
                if _analysis_state["done"] % 10 == 0:
                    conn.commit()
            except Exception as e:
                print(f"Error analyzing {row['filepath']}: {e}")
                _analysis_state["errors"] += 1
                _analysis_state["done"] += 1
        conn.commit()
        conn.close()
        with _analysis_lock:
            _analysis_state["running"] = False
            _analysis_state["total"] = 0

    t = threading.Thread(target=run_analysis, args=(rows,), daemon=True)
    t.start()
    return jsonify({"status": "started", "total": _analysis_state["total"]})


@app.route("/api/analyze/progress")
def analyze_progress():
    return jsonify(_analysis_state)


# ─── Dupe workflow (near-dupe review) ────────────────────────────────────────

_dupe_lock = threading.Lock()
_dupe_state = {
    "running": False, "phase": "", "total": 0, "done": 0, "errors": 0,
    "exact_dupes": [],  # list of {new_rel, matches: [rel, ...]}
    "stop_requested": False,
}


def _pool_dir():
    """The pool is the vault root itself; dupe tooling lives in dot-dirs."""
    return VAULT_ROOT


def _relpaths_for(abs_paths, pool):
    return [os.path.relpath(p, pool) for p in abs_paths]


@app.route("/api/dupes/status")
def dupes_status():
    pool = _pool_dir()
    if not pool:
        return jsonify({"error": "No vault path configured"}), 400
    manifest = dupe_pull.load_manifest(pool)
    hashes = dupefind.load_hashes(pool)
    edges = dupefind.load_edges(pool)
    files = dupefind.pool_files(pool)
    return jsonify({
        "pool": pool,
        "images_on_disk": len(files),
        "has_manifest": manifest is not None,
        "manifest_files": len(manifest["files"]) if manifest else 0,
        "hashed": len(hashes),
        "has_edges": edges is not None,
        "trash_dir": os.path.isdir(os.path.join(pool, dupefind.TRASH_DIR)),
        "running": _dupe_state["running"],
    })


@app.route("/api/dupes/pull", methods=["POST"])
def dupes_pull():
    pool = _pool_dir()
    if not pool:
        return jsonify({"error": "No vault path configured"}), 400
    data = request.json or {}
    source = data.get("source", "")
    if not source:
        return jsonify({"error": "source folder required"}), 400
    if not os.path.isdir(source):
        return jsonify({"error": f"folder not found: {source}"}), 400

    res = dupe_pull.pull_into_pool(pool, [source])
    # Index the newly moved files so they appear in the vault immediately
    idx = index_directory(pool, force=False)
    return jsonify({
        "moved": res["moved"],
        "already": res["already"],
        "exact_dupes": len(res["exact_dupes"]),
        "errors": res["errors"],
        "indexed": idx["indexed"],
        "pool": res["pool"],
    })


@app.route("/api/dupes/cluster", methods=["POST"])
def dupes_cluster():
    """Hash the pool + compute pairwise edges in the background."""
    pool = _pool_dir()
    if not pool:
        return jsonify({"error": "No vault path configured"}), 400
    if _dupe_state["running"]:
        return jsonify({"status": "already_running", **_dupe_state})

    with _dupe_lock:
        _dupe_state.update(running=True, phase="hashing", total=0, done=0, errors=0,
                           exact_dupes=[], stop_requested=False)

    def run():
        try:
            def prog(done, total):
                with _dupe_lock:
                    _dupe_state["done"] = done
                    _dupe_state["total"] = total

            def on_exact(new_rel, matches):
                with _dupe_lock:
                    _dupe_state["exact_dupes"].append({"new": new_rel, "matches": matches})

            def should_stop():
                with _dupe_lock:
                    return _dupe_state["stop_requested"]

            hashes, new_count, stopped = dupefind.compute_hashes(
                pool, progress=prog, on_exact=on_exact, should_stop=should_stop)
            if stopped:
                with _dupe_lock:
                    _dupe_state.update(running=False, phase="stopped",
                                       done=len(hashes), total=len(hashes), errors=0)
                return
            with _dupe_lock:
                _dupe_state["phase"] = "edges"
                _dupe_state["done"] = 0
                _dupe_state["total"] = len(hashes)
            a, b, d = dupefind.compute_edges(hashes, progress=prog, should_stop=should_stop)
            if a is None:
                with _dupe_lock:
                    _dupe_state.update(running=False, phase="stopped", errors=0)
                return
            dupefind.save_edges(pool, a, b, d)
            with _dupe_lock:
                _dupe_state.update(running=False, phase="done", done=len(d), total=len(d), errors=0)
        except Exception as e:
            print(f"Cluster error: {e}")
            with _dupe_lock:
                _dupe_state.update(running=False, phase="error", errors=1)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started", **_dupe_state})


@app.route("/api/dupes/cluster/stop", methods=["POST"])
def dupes_cluster_stop():
    """Gracefully stop a running scan at the next checkpoint. Progress is kept."""
    with _dupe_lock:
        if not _dupe_state["running"]:
            return jsonify({"status": "not_running"})
        _dupe_state["stop_requested"] = True
    return jsonify({"status": "stopping"})


@app.route("/api/dupes/cluster/progress")
def dupes_cluster_progress():
    return jsonify(_dupe_state)


@app.route("/api/dupes/families")
def dupes_families():
    """Families at a confidence level. Edges auto-recomputed when stale/missing."""
    pool = _pool_dir()
    if not pool:
        return jsonify({"error": "No vault path configured"}), 400
    confidence = request.args.get("confidence", 70, type=int)
    threshold = dupefind.threshold_for_confidence(confidence)

    hashes = dupefind.load_hashes(pool)
    if not hashes:
        return jsonify({"families": [], "threshold": threshold, "confidence": confidence,
                        "stats": {"families": 0, "images_in_families": 0, "isolated": 0, "max_family_size": 0}})

    # Recompute edges if missing or stale (hashes changed since edges were saved)
    edges = dupefind.load_edges(pool)
    hp = os.path.join(pool, dupefind.HASHES_FILE)
    ep = os.path.join(pool, dupefind.EDGES_FILE)
    if edges is None or (os.path.exists(hp) and os.path.exists(ep)
                         and os.path.getmtime(hp) > os.path.getmtime(ep)):
        a, b, d = dupefind.compute_edges(hashes)
        dupefind.save_edges(pool, a, b, d)
    else:
        a, b, d = edges

    fams = dupefind.families_at_threshold(a, b, d, threshold)
    rels = sorted(hashes.keys())

    # Pagination: families per page (never split a family across pages)
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    page = max(1, page)
    per_page = max(1, min(per_page, 50))
    total_families = len(fams)
    total_pages = max(1, (total_families + per_page - 1) // per_page)
    page = min(page, total_pages)
    page_fams = fams[(page - 1) * per_page: page * per_page]

    with db() as conn:
        families = []
        for fam in page_fams:
            dists = dupefind.member_min_dist(fam, a, b, d)
            members = []
            for idx in fam:
                rel = rels[idx]
                fp = os.path.join(pool, rel)
                row = conn.execute("SELECT id, quality_score FROM images WHERE filepath = ?", (fp,)).fetchone()
                members.append({
                    "id": row["id"] if row else None,
                    "rel": rel,
                    "filename": os.path.basename(rel),
                    "filepath": fp,
                    "min_dist": dists.get(idx),
                    "quality_score": row["quality_score"] if row else None,
                })
            families.append({"size": len(members), "members": members})

    stats = dupefind.family_stats(a, b, d, threshold, len(hashes))
    return jsonify({
        "families": families,
        "threshold": threshold,
        "confidence": confidence,
        "stats": stats,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_families": total_families,
            "total_pages": total_pages,
        },
    })


@app.route("/api/dupes/resolve", methods=["POST"])
def dupes_resolve():
    """Resolve a family: keepers -> curated dir, rejects -> trash (or delete)."""
    pool = _pool_dir()
    if not pool:
        return jsonify({"error": "No vault path configured"}), 400
    data = request.json or {}
    keep = data.get("keep", [])
    reject = data.get("reject", [])
    delete = bool(data.get("delete", False))
    curated = data.get("curated_dir") or os.path.join(os.path.dirname(pool.rstrip(os.sep)),
                                                      os.path.basename(pool.rstrip(os.sep)) + "_curated")
    if not keep and not reject:
        return jsonify({"error": "nothing to resolve"}), 400

    res = dupefind.resolve_family(pool, keep, reject, curated, delete=delete)

    # Update DB: keepers' filepaths changed; rejects are gone from the pool.
    with db() as conn:
        for old, new in zip(keep, res["kept"]):
            conn.execute(
                "UPDATE images SET filepath = ?, filename = ?, directory = ? WHERE filepath = ?",
                (new, os.path.basename(new), os.path.dirname(new), old),
            )
        for old in reject:
            row = conn.execute("SELECT id FROM images WHERE filepath = ?", (old,)).fetchone()
            if row:
                conn.execute("DELETE FROM images WHERE id = ?", (row["id"],))
                thumb_hash = hashlib.md5(old.encode()).hexdigest()[:12]
                for f in THUMB_DIR.glob(f"{thumb_hash}_*"):
                    f.unlink(missing_ok=True)
        conn.commit()

    # Drop resolved images from the hash cache so re-clustering is consistent.
    removed = _relpaths_for(keep + reject, pool)
    dupefind.prune_hashes(pool, removed)

    return jsonify({
        "kept": res["kept"],
        "trashed": res["trashed"],
        "errors": res["errors"],
        "curated_dir": curated,
    })


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global VAULT_ROOT

    if getattr(sys, 'frozen', False):
        # PyInstaller exe mode: vault path optional, auto-open browser
        parser = argparse.ArgumentParser(description="Vault Sifter")
        parser.add_argument("vault", nargs="?", default=None, help="Path to vault directory")
        parser.add_argument("--port", type=int, default=8844)
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--force", action="store_true")
        args = parser.parse_args()

        vault_path = Path(args.vault) if args.vault else None

        # If no vault given, try to use a file dialog
        if vault_path is None:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                chosen = filedialog.askdirectory(title="Select your image vault folder")
                root.destroy()
                if chosen:
                    vault_path = Path(chosen)
                else:
                    return
            except ImportError:
                print("No vault path given and tkinter not available.")
                print("Usage: VaultSifter.exe <vault_path>")
                return
    else:
        parser = argparse.ArgumentParser(description="Vault Sifter — review your image generation vault")
        parser.add_argument("vault", nargs="?", default=None, help="Path to your image vault directory (optional — can be loaded via the UI)")
        parser.add_argument("--port", type=int, default=8844, help="Port to serve on")
        parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
        parser.add_argument("--force", action="store_true", help="Force re-index all images")
        args = parser.parse_args()
        vault_path = Path(args.vault) if args.vault else None

    if vault_path is not None:
        vault_path = vault_path.resolve()
        if not vault_path.is_dir():
            print(f"Error: {vault_path} is not a directory")
            return
        VAULT_ROOT = str(vault_path)
        init_db()

    url = f"http://{args.host}:{args.port}"
    print(f"Vault Sifter")
    print(f"  Vault: {VAULT_ROOT if vault_path else '(none — use Load Folder in the UI)'}")
    if DB_PATH:
        print(f"  DB: {DB_PATH}")
    print(f"  Serving on {url}")
    print()

    if vault_path is not None:
        print("Indexing in background...")

        def _bg_index():
            result = index_directory(vault_path, force=args.force)
            print(f"  Indexing complete: {result['indexed']} indexed, {result['skipped']} skipped, {result['errors']} errors")

        import threading
        threading.Thread(target=_bg_index, daemon=True).start()

    # Auto-open browser in exe mode
    if getattr(sys, 'frozen', False):
        import threading, webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
