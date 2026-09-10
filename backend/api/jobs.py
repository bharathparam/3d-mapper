"""REST endpoints for job lifecycle management."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.core.job_manager import job_manager
from backend.core.pipeline_runner import PipelineRunner
from backend.models.job import CreateJobRequest, Job, ReconstructionMethod

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# Set at startup by main.py
_app_config: dict = {}
_runner: PipelineRunner | None = None


def init(app_config: dict) -> None:
    global _app_config, _runner
    _app_config = app_config
    _runner = PipelineRunner(app_config, job_manager)


# ── POST /api/jobs ────────────────────────────────────────────────────────────

@router.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    method: str = Form("optimized"),
):
    """Upload a video and start a reconstruction job."""
    # Validate method
    try:
        recon_method = ReconstructionMethod(method)
    except ValueError:
        raise HTTPException(400, f"Invalid method '{method}'. Use: baseline | optimized | both")

    # Validate file type
    if video.content_type not in ("video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "application/octet-stream"):
        if not (video.filename or "").lower().endswith((".mp4", ".mov", ".avi", ".webm")):
            raise HTTPException(400, "Unsupported file type. Upload MP4, MOV, AVI, or WebM.")

    # Create project directory
    job = Job(method=recon_method, video_filename=video.filename or "video.mp4")
    projects_dir = Path(_app_config.get("app", {}).get("projects_dir", "projects"))
    project_dir = projects_dir / job.job_id
    input_dir = project_dir / "input" / "videos"
    input_dir.mkdir(parents=True, exist_ok=True)

    video_path = input_dir / (video.filename or "video.mp4")

    # Save uploaded video
    try:
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
    except Exception as exc:
        shutil.rmtree(project_dir, ignore_errors=True)
        raise HTTPException(500, f"Failed to save video: {exc}")

    job.project_dir = str(project_dir)
    await job_manager.create(job)

    # Kick off pipeline in background
    background_tasks.add_task(_run_pipeline, job, video_path)

    return {"job_id": job.job_id, "status": job.status, "method": job.method}


# ── GET /api/jobs/{job_id} ────────────────────────────────────────────────────

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = await job_manager.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return job.model_dump()


# ── GET /api/jobs ─────────────────────────────────────────────────────────────

@router.get("/jobs")
async def list_jobs():
    jobs = await job_manager.list_all()
    return [j.model_dump(include={"job_id", "status", "method", "video_filename", "progress", "created_at"}) for j in jobs]


# ── GET /api/jobs/{job_id}/results ────────────────────────────────────────────

@router.get("/jobs/{job_id}/results")
async def get_results(job_id: str):
    job = await job_manager.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    if job.status != "completed":
        raise HTTPException(202, f"Job is not complete yet (status: {job.status}).")

    result_path = Path(job.project_dir) / "output" / "result.json"
    if not result_path.exists():
        raise HTTPException(500, "Result file not found — pipeline may have errored.")

    import json
    return json.loads(result_path.read_text())


# ── GET /api/files/{job_id}/{...path} ────────────────────────────────────────

@router.get("/files/{job_id}/{file_path:path}")
async def serve_file(job_id: str, file_path: str):
    """Serve reconstruction output files (PLY, JSON, etc.)."""
    job = await job_manager.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    full_path = Path(job.project_dir) / file_path
    if not full_path.exists():
        raise HTTPException(404, f"File not found: {file_path}")
    if not full_path.is_relative_to(Path(job.project_dir)):
        raise HTTPException(403, "Access denied.")

    return FileResponse(str(full_path))


# ── Background task ───────────────────────────────────────────────────────────

async def _run_pipeline(job: Job, video_path: Path) -> None:
    if _runner is None:
        logger.error("PipelineRunner not initialised")
        return
    try:
        await _runner.run(job, video_path)
    except Exception as exc:
        logger.exception("Pipeline error for job %s: %s", job.job_id, exc)
