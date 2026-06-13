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
- **100% local** — No cloud, no upload, no API keys

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
