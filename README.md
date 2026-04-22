current2classic 

> Inspired by [@artbutmakeitsports](https://www.instagram.com/artbutmakeitsports/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/current2classic/blob/main/current2classicMAGICHOURBOUNTY.ipynb)

---

## How it works

current2classic uses a multi-dimensional similarity search across a WikiArt painting index to match your photo to the most visually similar classical artwork.

Three signals are combined into a weighted score:

| Signal | Weight | What it captures |
|--------|--------|-----------------|
| CLIP (ViT-L/14) embeddings | 60% | Semantic content, pose, overall vibe |
| HSV color histogram | 25% | Color palette and tone |
| Spatial edge composition | 15% | Layout and structural framing |

Candidates are retrieved via FAISS nearest-neighbor search on the CLIP index, then re-ranked by the combined score.

```
Photo → CLIP embed + color hist + edge grid
      → FAISS retrieves top candidates
      → weighted re-rank
      → top-10 painting matches + metadata
```

---

## Features

- **Image search** — upload any photo, get top-K classical painting matches
- **Text search** — describe a scene in plain English, CLIP finds the paintings
- **Tunable weights** — adjust semantic / color / composition balance with sliders
- **Gradio UI** — shareable web interface with `share=True`
- **Index caching** — embeddings saved to disk, skips recomputation on rerun

---

## Stack

- [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) — vision-language embeddings
- [`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart) — 81k painting dataset (3k indexed by default)
- [FAISS](https://github.com/facebookresearch/faiss) — fast vector similarity search
- [Gradio](https://gradio.app) — interactive web UI
- OpenCV — HSV histograms + Canny edge composition features

---

## Quickstart

Run in Google Colab — no local setup needed.

1. Open the notebook via the badge above
2. Set runtime to **GPU** (`Runtime → Change runtime type → T4`)
3. Run all cells top to bottom
4. Upload a photo when prompted — results appear in ~1–2 seconds

---

## Configuration

At the top of the notebook:

```python
DATASET_SIZE = 3000    # paintings to index — increase for better matches
TOP_K        = 10      # results to return
CLIP_MODEL_ID = 'openai/clip-vit-large-patch14'

W_CLIP  = 0.60         # semantic / visual weight
W_COLOR = 0.25         # color palette weight
W_COMP  = 0.15         # composition weight
```

Bump `DATASET_SIZE` to 10k–15k for noticeably better match quality. The index is cached after the first run so you only pay the embedding cost once.

---

## Hardware requirements

| Scale | System RAM | VRAM | Recommended GPU |
|-------|-----------|------|----------------|
| 3k paintings (default) | ~3 GB | ~2.5 GB | T4 (free) |
| 15k paintings | ~6 GB | ~3 GB | T4 (free) |
| 80k+ paintings | ~5 GB* | ~4–5 GB | A100 |

*at 80k+ scale, don't keep raw images in RAM — store paths and load on demand.

---

## Scaling up

To improve match quality, increase the dataset size and optionally merge additional sources:

```python
# Larger WikiArt slice
DATASET_SIZE = 15000

# Additional painting datasets worth merging:
# - biglam/european_art     (15k paintings, 25+ institutions, 12th–18th century)
# - metmuseum/openaccess    (400k+ works, filter to paintings + public domain)
# - Mitsua/art-museums-pd-440k  (293k CC0 images from major museums)
```

For 30k+ paintings, switch the CLIP index to `IndexIVFFlat` for faster search:

```python
nlist = 256
quantizer = faiss.IndexFlatIP(EMBED_DIM)
clip_index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
clip_index.train(clip_embs)
clip_index.nprobe = 20
```
