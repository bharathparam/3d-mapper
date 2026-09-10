"""
FastAPI application entry point.

Provides:
  REST API  — /api/jobs, /api/files
  WebSocket — /ws/jobs/{job_id}
  Static    — serves the React frontend build
"""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import jobs as jobs_api
from backend.api import ws as ws_api

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config loading ────────────────────────────────────────────────────────────

def _load_config(path: str = "config/default.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        logger.warning("Config not found at %s — using defaults.", path)
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


CONFIG = _load_config()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Drone 3D Reconstruction API",
    description="Quality-aware adaptive 3D reconstruction from drone video.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise API modules
jobs_api.init(CONFIG)

# Routes
app.include_router(jobs_api.router)
app.include_router(ws_api.router)

# Serve React frontend (if built)
frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    logger.info("Serving React frontend from %s", frontend_dist)
else:
    @app.get("/")
    def root():
        return {
            "message": "Drone 3D Reconstruction API",
            "docs": "/docs",
            "frontend": "Run: cd frontend && npm run dev",
        }


@app.on_event("startup")
async def startup() -> None:
    # Ensure projects directory exists
    projects_dir = Path(CONFIG.get("app", {}).get("projects_dir", "projects"))
    projects_dir.mkdir(exist_ok=True)
    logger.info("Projects directory: %s", projects_dir.resolve())
    logger.info("API ready. Docs: http://localhost:8000/docs")
