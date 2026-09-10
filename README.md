# Drone 3D Reconstruction — Quality-Aware Adaptive System

> **SIH 2024** · Computer Vision & Robotics Track

A quality-aware adaptive 3D reconstruction pipeline that takes a drone video and produces a textured 3D point cloud, evaluates reconstruction quality per-region, identifies where it failed and why, and recommends specific additional drone imagery to fix it.

---

## Core Claim (what we measure)

> *Quality-aware frame selection achieves equal or better reconstruction completeness at 60–75% fewer frames, less compute, and — critically — tells you exactly where the model is unreliable and how to fix it.*

---

## Quick Start

### 1. Install system dependencies (macOS + Homebrew)

```bash
brew install colmap ffmpeg
```

### 2. Set up Python environment

```bash
cd 3d-reconstruction
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Set up frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Run

**Option A — Full web UI (two terminals):**

```bash
# Terminal 1: Backend
python run.py

# Terminal 2: Frontend dev server
cd frontend && npm run dev
# Open: http://localhost:5173
```

**Option B — CLI only:**

```bash
python run.py --video path/to/drone.mp4 --method both
```

---

## Pipeline

```
DRONE VIDEO
    ↓
Frame Extraction (FFmpeg)
    ↓
Frame Quality Scoring  ← OUR WORK
  sharpness · exposure · feature density · uniformity · viewpoint gain
    ↓
Keyframe Selection     ← OUR WORK
  greedy SSIM deduplication + temporal coverage
    ↓
COLMAP Sparse Reconstruction (CLI subprocess)
  feature_extractor → sequential_matcher → mapper → model_converter
    ↓
Quality Engine         ← OUR WORK
  registration rate · reprojection error · track length · density · coverage
    ↓
Confidence Map         ← OUR WORK
  per-voxel confidence (High / Medium / Low)  →  coloured PLY
    ↓
Failure Classifier     ← OUR WORK
  missing_viewpoint | low_overlap | motion_blur | textureless
    ↓
Recapture Recommender  ← OUR WORK
  actionable flight suggestions per diagnosed region
    ↓
3D Viewer (Three.js)   — interactive point cloud + confidence overlay
```

---

## Architecture

```
Frontend (React/Vite + Three.js)
    ↕  REST + WebSocket
Backend (FastAPI)
    ↓
Job Manager (async)
    ↓
Pipeline Runner
    ├── FrameExtractor      (FFmpeg)
    ├── FrameScorer         (OpenCV — OUR WORK)
    ├── KeyframeSelector    (OUR WORK)
    ├── ColmapReconstructor (COLMAP CLI)
    ├── QualityEngine       (OUR WORK)
    ├── ConfidenceMapGen    (OUR WORK)
    ├── FailureClassifier   (OUR WORK)
    └── RecaptureRecommender(OUR WORK)
```

---

## What is Open-Source vs. Our Contribution

| Component | Technology | Who |
|---|---|---|
| Video decoding | FFmpeg | Open-source |
| SfM + bundle adjustment | COLMAP | Open-source |
| Dense mesh | Open3D | Open-source |
| **Frame quality scorer** | OpenCV + NumPy | **US** |
| **Keyframe selector** | Our algorithm | **US** |
| **Quality engine** | Our metrics | **US** |
| **Confidence map generator** | Our code | **US** |
| **Failure classifier** | Our code | **US** |
| **Recapture recommender** | Our code | **US** |
| **Baseline vs. optimised comparator** | Our code | **US** |

---

## Frame Scoring Formula

```
Score(f) = 0.30 · sharpness(f)
         + 0.15 · exposure_quality(f)
         + 0.25 · feature_density(f)
         + 0.20 · feature_uniformity(f)    ← spatial spread across 4×4 grid
         + 0.10 · viewpoint_gain(f)        ← SSIM distance from prior selected frame
```

All weights configurable in `config/frame_scoring.yaml`.

---

## Quality Scoring (0–100)

| Component | Weight | Metric | Source |
|---|---|---|---|
| Camera registration | 30% | registered/total | MEASURED |
| Track length | 20% | mean imgs/point | MEASURED |
| Reprojection error | 20% | pixels | MEASURED |
| Point density | 15% | pts vs 50k cap | MEASURED |
| Spatial coverage | 15% | camera spread | ESTIMATED |

---

## Failure Classification

| Type | Diagnosis |
|---|---|
| `missing_viewpoint` | < 3 cameras observe region |
| `low_overlap` | < 2 pts/voxel despite cameras |
| `motion_blur` | reprojection error > 3 px |
| `textureless` | cameras + overlap but no features |

---

## Project Structure

```
3d-reconstruction/
├── run.py                    # single-command entry point
├── config/                   # all tunable parameters
├── backend/                  # FastAPI + job manager
├── frame_selection/          # extractor, scorer, selector  ← OUR WORK
├── reconstruction/           # COLMAP wrapper + interface
├── quality/                  # metrics, confidence, classifier, recommender  ← OUR WORK
├── frontend/                 # React/Vite + Three.js
└── tests/                    # pytest unit tests
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

All parameters are in `config/` — no code changes needed to tune the system:

- `config/default.yaml` — app settings, extraction fps, quality thresholds
- `config/frame_scoring.yaml` — scorer weights, selection targets
- `config/reconstruction.yaml` — COLMAP parameters

---

## Measurement Honesty

We never fabricate accuracy numbers. The comparison table shows:
- **MEASURED** — direct output from COLMAP (registration rate, reprojection error, point count, track length, processing time)
- **ESTIMATED** — inferred from model properties (spatial coverage)

Ground-truth accuracy (Chamfer distance, etc.) requires a reference scan — shown only when one is available.
