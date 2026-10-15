# AeroVision — vision-language perception for aerial inspection that agents can act on

[![ci](https://github.com/D-L-Narayana/aerovision/actions/workflows/ci.yml/badge.svg)](https://github.com/D-L-Narayana/aerovision/actions)
![pytorch](https://img.shields.io/badge/PyTorch-CLIP%20ViT--B%2F32-ee4c2c) ![hf](https://img.shields.io/badge/Hugging%20Face-transformers%20%2B%20transformers.js-ffcc4d) ![license](https://img.shields.io/badge/license-MIT-blue)

**Live demo (CLIP runs in your browser):** https://d-l-narayana.github.io/aerovision

AeroVision is the *perception layer* for an autonomous drone-ops agent. It answers two questions about a frame —
*what am I looking at?* (CLIP scene understanding with zero-shot and linear-probe heads) and *is anything wrong?*
(radiometric hotspot / hot-string detection graded by the inspection SOP) — and emits one schema-validated
`InspectionReport` that an agent tool loop (e.g. [AeroSentry](https://github.com/D-L-Narayana/aerosentry-agents)) can
act on, including flight-sensitivity flags (airport, people, industrial exclusion zones).

```
RGB frame ──▶ CLIP ViT-B/32 image embedding ──┬─▶ zero-shot head (4-template prompt ensemble)
                                               └─▶ linear probe head (frozen features, 50 img/class)  ─┐
thermal frame (°C) ──▶ per-string reference ──▶ hotspot / hot-string detector ──▶ SOP severity ─────────┼─▶ InspectionReport (Pydantic)
optional generative VLM narrative (OpenAI-compatible) ──▶ JSON-validated, confidence < 0.6 ⇒ unverified ─┘      └─▶ to_tool_result()
```

## Results (reproducible: `python evals/run_evals.py`)

**Aerial scene classification — UC Merced Land Use, 21 classes × 100 images (256 px), `openai/clip-vit-base-patch32`, PyTorch CPU**

| method | top-1 | notes |
|---|---|---|
| zero-shot, single prompt `"an aerial photo of {class}."` | 58.1 % | |
| zero-shot, 4-template prompt ensemble | **67.3 %** (top-3 87.6 %) | +9.2 pts from prompting alone, no training |
| **linear probe on frozen CLIP features** | **93.0 %** | 50 labelled images / class, stratified 50/50 split |

Few-shot probe (labelled images per class → test top-1, mean of 3 seeds): 1 → 66.7 % · 2 → 71.5 % · 5 → 81.8 % ·
10 → 86.9 % · 20 → 90.6 % · 50 → 93.0 %. Ten labelled frames per class already beat zero-shot by ~20 points — the
practical recipe for adapting a VLM to a new site. Hardest zero-shot classes (dense/medium residential, runway,
chaparral) go from 0–28 % to 66–100 % with the probe (`evals/results.json → per_class`).

**Thermal anomaly detection — 200 synthetic radiometric frames, 314 ground-truth hotspots**

| metric | value |
|---|---|
| precision / recall on reportable (≥ 10 °C) hotspots | **100 % / 93.6 %** |
| severity agreement (medium / high per SOP thresholds) | 95.5 % |
| latency | 0.6 ms / frame (NumPy, pure CPU) |

CLIP image encoding: 64 ms / image on 2 vCPU (PyTorch); ~100–200 ms in-browser (WASM, int8) in the demo.

## What it demonstrates

| Capability | Where |
|---|---|
| VLM for aerial imagery: CLIP via Hugging Face `transformers` + PyTorch, prompt ensembling, cosine-softmax zero-shot | `aerovision/clip_model.py`, `labels.py` |
| Cheap adaptation: linear probe on frozen embeddings (scikit-learn), few-shot curve, JSON export → identical head in the browser | `aerovision/probe.py`, `web/probe_head.json` |
| Sensor imagery analysis: per-string radiometric reference, 4-connected component hotspots, SOP severity grading | `aerovision/thermal.py` |
| Agent-ready output: Pydantic `InspectionReport` with range/enum validation, `recommended_action`, confidence gate (VLM < 0.6 ⇒ deterministic detector decides) | `aerovision/report.py`, `vlm.py` |
| Decision support for flight safety: scene → flight-sensitivity flags (airport, people, industrial exclusion) | `aerovision/labels.py::FLIGHT_SENSITIVITY` |
| Evaluation: per-class accuracy, few-shot curve, detector P/R vs ground truth, latency | `evals/run_evals.py`, `evals/results.md` |
| Deployment surface: FastAPI `/inspect`, CLI, Docker (model baked), GitHub Actions CI; static browser demo on Vercel | `aerovision/api.py`, `cli.py`, `Dockerfile`, `web/` |

## Quickstart

```bash
git clone https://github.com/D-L-Narayana/aerovision && cd aerovision
pip install -r requirements.txt                      # CPU torch is fine
python scripts/make_thermal.py                       # 200 synthetic radiometric frames + ground truth
python -m pytest -q                                  # 7 tests, no model download
python evals/run_evals.py                            # downloads UC Merced (416 MB) + CLIP, writes evals/results.{json,md}, web assets

python -m aerovision.cli --rgb web/samples/storagetanks.jpg --thermal data/thermal/frames.npz --frame 3
uvicorn aerovision.api:app --reload                  # POST /inspect  (multipart: rgb=image, thermal=.npy)
```

Generative narrative (optional): `AEROVISION_PROVIDER=openai OPENAI_API_KEY=... AEROVISION_MODEL=gpt-4o-mini`
(or any OpenAI-compatible vision endpoint — Qwen2-VL on vLLM, LLaVA on Ollama). The response must validate against
`VLMFinding`; invalid JSON or confidence < 0.6 is treated as unverified per the SOP and never raises severity on its own.

## Browser demo

`web/` is a static site: transformers.js loads `Xenova/clip-vit-base-patch32` (vision tower, int8), text embeddings for
the 21 classes and the probe head are precomputed by `evals/run_evals.py`, and the thermal detector is a line-for-line
JS port. Drop any aerial image, or click a UC Merced sample.

## Data

UC Merced Land Use (Yang & Newsam, ACM SIGSPATIAL 2010) via the Hugging Face hub (`blanchon/UC_Merced`). Thermal
frames are synthetic (`scripts/make_thermal.py`, seeded). No customer data.

## License

MIT
