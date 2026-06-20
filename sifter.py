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

DB_PATH = Path.home() / ".vault-sifter" / "sifter.db"
THUMB_DIR = Path.home() / ".vault-sifter" / "thumbnails"
VAULT_ROOT = None


# ─── Database ────────────────────────────────────────────────────────────────

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
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
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
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
                        model = inputs["ckpt_name"]
                    elif "unet_name" in inputs:
                        model = inputs["unet_name"]
                    elif "lora_name" in inputs and not model:
                        model = inputs["lora_name"]

                # Capture seed
                if "seed" in inputs:
                    seed = str(inputs["seed"])
                elif class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"):
                    if "noise_seed" in inputs:
                        seed = str(inputs["noise_seed"])

        info["prompt"] = prompt_text
        info["workflow"] = raw_info.get("workflow", "")
        info["model"] = model
        info["seed"] = seed

    except Exception as e:
        info["error"] = str(e)

    return info


# ─── Indexing ────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def index_directory(vault_path, force=False):
    """Scan a directory and index all images."""
    vault_path = Path(vault_path).resolve()
    conn = get_db()

    indexed = 0
    skipped = 0
    errors = 0

    try:
        for filepath in vault_path.rglob("*"):
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
        parser.add_argument("vault", help="Path to your image vault directory")
        parser.add_argument("--port", type=int, default=8844, help="Port to serve on")
        parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
        parser.add_argument("--force", action="store_true", help="Force re-index all images")
        args = parser.parse_args()
        vault_path = Path(args.vault)

    vault_path = vault_path.resolve()
    if not vault_path.is_dir():
        print(f"Error: {vault_path} is not a directory")
        return

    VAULT_ROOT = str(vault_path)
    init_db()

    print(f"Vault Sifter")
    print(f"  Vault: {vault_path}")
    print(f"  DB: {DB_PATH}")
    print(f"  Thumbnails: {THUMB_DIR}")
    print()

    print("Indexing images...")
    result = index_directory(vault_path, force=args.force)
    print(f"  Indexed: {result['indexed']}")
    print(f"  Skipped: {result['skipped']}")
    print(f"  Errors: {result['errors']}")
    print()

    url = f"http://{args.host}:{args.port}"
    print(f"Serving on {url}")

    # Auto-open browser in exe mode
    if getattr(sys, 'frozen', False):
        import threading, webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
