# AI Image Detector — Production Ensemble v7.1

Detect **AI-generated images**, **deepfakes**, and **authentic photos** — including images re-compressed through **Telegram / WhatsApp** with metadata stripped.

## What changed in v7.1

- **Pretrained semantic detector** ([Dafilab/ai-image-detector](https://huggingface.co/Dafilab/ai-image-detector) EfficientNet-B4) — compression-robust; fixes false “real” on GPT-4o / DALL-E faces after messaging apps.
- **Warmup Caching & Package Stabilization** — resolves missing `transformers` library issues and cleans virtual environment locks automatically.
- **Dual detection** — parallel **AI-generated** vs **deepfake** scores with fused verdict.
- **Messenger-aware fusion** — HF + CLIP weighted up; PRNU/frequency down-weighted when EXIF is missing.
- **UI & Theme Toggle Fixes** — full, seamless Light and Dark mode transitions across all cards, gauges, borders, and buttons with local storage persistence.

## Quick start (Windows)

1. Double-click **`deepfake run.bat`**
2. Choose **`[4] Fix/Update Dependencies`** once (installs Python packages + downloads model weights).
3. Choose **`[1] Start Web App`**
4. Open **http://127.0.0.1:3000**

## Manual start

```bash
# Backend
cd backend
python -m venv ../.venv
../.venv/Scripts/activate   # Windows
pip install -r requirements.txt
python scripts/download_models.py
python main.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

## API

| Endpoint | Description |
|----------|-------------|
| `POST /api/v2/analyze` | Full 4-layer ensemble + `dual_detection` block |
| `POST /api/analyze` | Same engine, v1-shaped JSON |
| `POST /api/analyze/legacy` | Old heuristic-only path (comparison) |

### Example `dual_detection` response

```json
{
  "dual_detection": {
    "ai_generated": { "score": 87.2, "likely_fake": true },
    "deepfake": { "score": 14.1, "likely_fake": false, "face_count": 1 },
    "primary": "ai_generated",
    "layer_scores": { "semantic_hf": 82.1, "semantic_clip": 76.4 }
  }
}
```

## Architecture

```
Upload → Context (messenger / pristine / web)
      → Layer 1: EXIF, ELA, C2PA
      → Layer 2: HF classifier + CLIP + CNN + DCT + EfficientNet (faces)
      → Layer 3: Dual fusion (AI-generated ‖ Deepfake)
      → Layer 4: GradCAM + explanation
```

**Messenger mode:** semantic models (HF 42% + CLIP 33%) dominate; pixel forensics capped to avoid JPEG false positives.

## Requirements

- Python 3.10+
- Node.js 18+
- ~500MB disk for PyTorch + HF weights (first run downloads automatically)
- GPU optional (CUDA speeds up CLIP/HF)

## Documentation

- [CHANGES.md](CHANGES.md) — full changelog and ensemble weights
- [RESEARCH.md](RESEARCH.md) — papers and open-source references

## License

Use responsibly. Detection scores are probabilistic — not legal proof of authenticity.
