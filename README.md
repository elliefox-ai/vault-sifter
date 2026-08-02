# Vault Sifter

A lightweight local web app for reviewing and culling large AI image generation vaults.

If you run ComfyUI, Stable Diffusion, or any local image generation pipeline, you probably have hundreds of images with names like `ComfyUI_temp_uxqpo_01220_.png` and no good way to sort the keepers from the rejects. Vault Sifter fixes that.

## Features

- **Metadata extraction** — Reads prompt/workflow metadata embedded in PNG files (ComfyUI `tEXt` chunks)
- **Keyboard-driven review** — `j/k` to navigate, `1-5` to rate, `x` to reject, `space` to flag
- **Side-by-side compare** — Select 2-4 images for comparison view
- **Smart filtering** — By date, model, prompt text, rating, unrated-only
- **Bulk actions** — Delete rejects, move keepers, export curated set
- **Session memory** — Remembers your position, so you can sift in chunks
- **Near-dupe families** — Pull a folder into a review pool (moves, preserves source structure), cluster by perceptual hash (pHash + dHash), review families side-by-side, mark keep/dump per member, resolve: keepers → curated folder, dumps → trash
- **Confidence slider** — Maps directly to the similarity threshold; nudge it down until the right images get sucked into families. Loose by default — you make the final call
- **100% local** — No cloud, no upload, no API keys

## Near-Dupe Workflow

1. **Pull** — paste a source folder in the Dupes panel; images are **moved** (not copied) into the pool, preserving their folder structure (`pool/<source>/<subdirs>/...`). Byte-identical twins are recorded as exact dupes and left in place. A `manifest.json` tracks everything.
2. **Cluster** — click *Find dupes*; images are hashed (checkpointed) and compared pairwise, then grouped into families of 2+.
3. **Review** — each family shows thumbnail strips; click members to toggle keep ✓ / dump 🗑, or open the family in the side-by-side compare view.
4. **Resolve** — keepers move to a curated folder (default: `<pool>_curated`, structure preserved), dumps go to `.dupe-trash/` inside the pool (recoverable; hard delete available). DB rows and hash caches update automatically.

Tune the confidence slider to re-cluster live. The families endpoint recomputes edges automatically after pulls and resolves.

> Escalation path: if re-renders with changed pose/composition slip through hashing, the spec documents CLIP/DINOv2 embeddings as the next tier — not built yet, by design (no gold-plating).

## Quick Start

```bash
git clone https://github.com/elliefox-ai/vault-sifter.git
cd vault-sifter
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python sifter.py /path/to/your/comfy-output
```

Then open `http://localhost:8844` in your browser.

## Tech

- **Backend:** Python + Flask
- **Frontend:** Vanilla JS, no build step
- **Storage:** SQLite (auto-created in `~/.vault-sifter/`)
- **Image processing:** Pillow for thumbnails + metadata extraction

## Requirements

- Python 3.10+
- Pillow
- Flask

## License

MIT
