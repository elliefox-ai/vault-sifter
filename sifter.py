#!/usr/bin/env python3
"""Vault Sifter — local image vault review tool."""

import argparse
import json
import os
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from io import BytesIO

from flask import Flask, jsonify, request, send_file, send_from_directory, abort

app = Flask(__name__, static_folder="static")

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


# ─── Metadata Extraction ──────────────────────────────────────────────────────

def extract_png_metadata(filepath):
    """Extract prompt/workflow metadata from PNG tEXt chunks (ComfyUI format)."""
    info = {}
    try:
        from PIL import Image
        img = Image.open(filepath)
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

    conn.commit()
    conn.close()
    return {"indexed": indexed, "skipped": skipped, "errors": errors}


def generate_thumbnail(filepath, max_size=400):
    """Generate a thumbnail for an image, cached on disk."""
    filepath = Path(filepath)
    thumb_name = f"{hash(str(filepath))}_{filepath.name}.jpg"
    thumb_path = THUMB_DIR / thumb_name

    if thumb_path.exists():
        return thumb_path

    from PIL import Image
    img = Image.open(filepath)
    img.thumbnail((max_size, max_size), Image.LANCZOS)

    # Convert to RGB for JPEG (handles RGBA PNGs)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    img.save(thumb_path, "JPEG", quality=85)
    return thumb_path


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/stats")
def stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) as c FROM images").fetchone()["c"]
    rated = conn.execute("SELECT COUNT(*) as c FROM images WHERE rating > 0").fetchone()["c"]
    rejected = conn.execute("SELECT COUNT(*) as c FROM images WHERE rejected = 1").fetchone()["c"]
    flagged = conn.execute("SELECT COUNT(*) as c FROM images WHERE flag = 1").fetchone()["c"]
    unrated = conn.execute(
        "SELECT COUNT(*) as c FROM images WHERE rating = 0 AND rejected = 0"
    ).fetchone()["c"]
    conn.close()
    return jsonify({
        "total": total, "rated": rated, "rejected": rejected,
        "flagged": flagged, "unrated": unrated
    })


@app.route("/api/images")
def list_images():
    """List images with optional filtering, sorting, and pagination."""
    conn = get_db()

    # Filters
    min_rating = request.args.get("min_rating", 0, type=int)
    max_rating = request.args.get("max_rating", 999, type=int)
    unrated_only = request.args.get("unrated_only", "false") == "true"
    include_rejected = request.args.get("include_rejected", "false") == "true"
    prompt_search = request.args.get("q", "")
    model_filter = request.args.get("model", "")
    directory_filter = request.args.get("dir", "")

    # Sort
    sort = request.args.get("sort", "file_modified")
    sort_dir = request.args.get("sort_dir", "desc")
    valid_sorts = {"file_modified", "file_created", "filesize", "filename", "rating", "indexed_at"}
    if sort not in valid_sorts:
        sort = "file_modified"

    # Pagination
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    limit = min(limit, 200)

    query = "SELECT * FROM images WHERE 1=1"
    params = []

    if not include_rejected:
        query += " AND rejected = 0"
    if unrated_only:
        query += " AND rating = 0 AND rejected = 0"
    if min_rating > 0:
        query += " AND rating >= ?"
        params.append(min_rating)
    if max_rating < 999:
        query += " AND rating <= ?"
        params.append(max_rating)
    if prompt_search:
        query += " AND prompt LIKE ?"
        params.append(f"%{prompt_search}%")
    if model_filter:
        query += " AND model LIKE ?"
        params.append(f"%{model_filter}%")
    if directory_filter:
        query += " AND directory LIKE ?"
        params.append(f"%{directory_filter}%")

    query += f" ORDER BY {sort} {sort_dir} LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return jsonify([dict(r) for r in rows])


@app.route("/api/image/<int:image_id>")
def get_image(image_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    conn.close()
    if not row:
        abort(404)
    return jsonify(dict(row))


@app.route("/api/image/<int:image_id>/rate", methods=["POST"])
def rate_image(image_id):
    rating = request.json.get("rating", 0)
    rating = max(0, min(5, int(rating)))
    conn = get_db()
    conn.execute("UPDATE images SET rating = ? WHERE id = ?", (rating, image_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "rating": rating})


@app.route("/api/image/<int:image_id>/reject", methods=["POST"])
def reject_image(image_id):
    rejected = request.json.get("rejected", True)
    conn = get_db()
    conn.execute("UPDATE images SET rejected = ? WHERE id = ?", (1 if rejected else 0, image_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "rejected": rejected})


@app.route("/api/image/<int:image_id>/flag", methods=["POST"])
def flag_image(image_id):
    flag = request.json.get("flag", True)
    conn = get_db()
    conn.execute("UPDATE images SET flag = ? WHERE id = ?", (1 if flag else 0, image_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "flag": flag})


@app.route("/api/image/<int:image_id>/tags", methods=["POST"])
def tag_image(image_id):
    tags = request.json.get("tags", "")
    conn = get_db()
    conn.execute("UPDATE images SET tags = ? WHERE id = ?", (tags, image_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "tags": tags})


@app.route("/api/image/<int:image_id>/notes", methods=["POST"])
def note_image(image_id):
    notes = request.json.get("notes", "")
    conn = get_db()
    conn.execute("UPDATE images SET notes = ? WHERE id = ?", (notes, image_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "notes": notes})


@app.route("/api/thumbnail/<int:image_id>")
def thumbnail(image_id):
    conn = get_db()
    row = conn.execute("SELECT filepath FROM images WHERE id = ?", (image_id,)).fetchone()
    conn.close()
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
    conn = get_db()
    row = conn.execute("SELECT filepath FROM images WHERE id = ?", (image_id,)).fetchone()
    conn.close()
    if not row:
        abort(404)

    filepath = row["filepath"]
    if not os.path.exists(filepath):
        abort(404)

    return send_file(filepath)


@app.route("/api/models")
def list_models():
    conn = get_db()
    rows = conn.execute(
        "SELECT DISTINCT model FROM images WHERE model != '' ORDER BY model"
    ).fetchall()
    conn.close()
    return jsonify([r["model"] for r in rows])


@app.route("/api/directories")
def list_directories():
    conn = get_db()
    rows = conn.execute(
        "SELECT DISTINCT directory FROM images ORDER BY directory"
    ).fetchall()
    conn.close()
    return jsonify([r["directory"] for r in rows])


@app.route("/api/bulk/delete-rejected", methods=["POST"])
def bulk_delete_rejected():
    conn = get_db()
    rows = conn.execute("SELECT id, filepath FROM images WHERE rejected = 1").fetchall()
    deleted = 0
    errors = 0
    for row in rows:
        try:
            os.unlink(row["filepath"])
            deleted += 1
        except Exception as e:
            print(f"Error deleting {row['filepath']}: {e}")
            errors += 1

    conn.execute("DELETE FROM images WHERE rejected = 1")
    conn.commit()
    conn.close()
    return jsonify({"deleted": deleted, "errors": errors})


@app.route("/api/bulk/move-rated", methods=["POST"])
def bulk_move_rated():
    """Move rated images to a destination directory."""
    dest = request.json.get("destination")
    if not dest:
        abort(400, "destination required")

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    conn = get_db()
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
    conn.close()
    return jsonify({"moved": moved, "errors": errors})


@app.route("/api/session/<key>", methods=["GET", "POST"])
def session_state(key):
    conn = get_db()
    if request.method == "POST":
        value = request.json.get("value", "")
        conn.execute(
            "INSERT OR REPLACE INTO session_state (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()
        conn.close()
        return jsonify({"status": "ok"})
    else:
        row = conn.execute(
            "SELECT value FROM session_state WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        return jsonify({"key": key, "value": row["value"] if row else ""})


@app.route("/api/reindex", methods=["POST"])
def reindex():
    global VAULT_ROOT
    force = request.json.get("force", False)
    result = index_directory(VAULT_ROOT, force=force)
    return jsonify(result)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global VAULT_ROOT
    parser = argparse.ArgumentParser(description="Vault Sifter — review your image generation vault")
    parser.add_argument("vault", help="Path to your image vault directory")
    parser.add_argument("--port", type=int, default=8844, help="Port to serve on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--force", action="store_true", help="Force re-index all images")
    args = parser.parse_args()

    vault_path = Path(args.vault).resolve()
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

    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
