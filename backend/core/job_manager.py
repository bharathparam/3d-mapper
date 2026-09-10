"""
In-memory job registry with WebSocket broadcast support.

Jobs are also serialised to {project_dir}/job.json so the state
survives a server restart (best-effort; not a database).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from backend.models.job import Job, JobStatus, ProgressEvent, Stage, StageStatus

logger = logging.getLogger(__name__)


class JobManager:
    """Thread-safe in-process job store with WebSocket pub-sub."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._subscribers: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    # ── CRUD ────────────────────────────────────────────────────────────────

    async def create(self, job: Job) -> Job:
        async with self._lock:
            self._jobs[job.job_id] = job
            self._subscribers[job.job_id] = []
        self._persist(job)
        return job

    async def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def list_all(self) -> list[Job]:
        return list(self._jobs.values())

    # ── Stage helpers ────────────────────────────────────────────────────────

    async def start_stage(
        self,
        job_id: str,
        stage_name: str,
        message: str = "",
        progress: int | None = None,
    ) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        stage = job.get_stage(stage_name)
        if stage:
            stage.status = StageStatus.RUNNING
            stage.message = message
            stage.started_at = datetime.utcnow()
        job.current_stage = stage_name
        job.status = JobStatus.RUNNING
        if progress is not None:
            job.progress = progress
        job.updated_at = datetime.utcnow()
        self._persist(job)
        await self._broadcast(
            job_id,
            ProgressEvent(
                event="stage_started",
                stage=stage_name,
                message=message,
                progress=job.progress,
            ),
        )

    async def complete_stage(
        self,
        job_id: str,
        stage_name: str,
        message: str = "",
        data: dict[str, Any] | None = None,
        progress: int | None = None,
    ) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        stage = job.get_stage(stage_name)
        if stage:
            stage.status = StageStatus.COMPLETED
            stage.message = message
            stage.completed_at = datetime.utcnow()
            if data:
                stage.data = data
        if progress is not None:
            job.progress = progress
        job.updated_at = datetime.utcnow()
        self._persist(job)
        await self._broadcast(
            job_id,
            ProgressEvent(
                event="stage_completed",
                stage=stage_name,
                message=message,
                progress=job.progress,
                data=data or {},
            ),
        )

    async def fail_stage(
        self,
        job_id: str,
        stage_name: str,
        message: str,
    ) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        stage = job.get_stage(stage_name)
        if stage:
            stage.status = StageStatus.FAILED
            stage.message = message
            stage.completed_at = datetime.utcnow()
        job.status = JobStatus.FAILED
        job.error_message = message
        job.updated_at = datetime.utcnow()
        self._persist(job)
        await self._broadcast(
            job_id,
            ProgressEvent(event="error", stage=stage_name, message=message),
        )

    async def complete_job(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.updated_at = datetime.utcnow()
        self._persist(job)
        await self._broadcast(
            job_id,
            ProgressEvent(event="done", progress=100, message="Reconstruction complete."),
        )

    async def fail_job(self, job_id: str, message: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.status = JobStatus.FAILED
        job.error_message = message
        job.updated_at = datetime.utcnow()
        self._persist(job)
        await self._broadcast(
            job_id,
            ProgressEvent(event="error", message=message),
        )

    # ── WebSocket pub-sub ────────────────────────────────────────────────────

    async def subscribe(self, job_id: str, ws: WebSocket) -> None:
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []
        self._subscribers[job_id].append(ws)

    async def unsubscribe(self, job_id: str, ws: WebSocket) -> None:
        if job_id in self._subscribers:
            self._subscribers[job_id] = [
                s for s in self._subscribers[job_id] if s is not ws
            ]

    async def _broadcast(self, job_id: str, event: ProgressEvent) -> None:
        subs = self._subscribers.get(job_id, [])
        dead: list[WebSocket] = []
        payload = event.model_dump_json()
        for ws in subs:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.unsubscribe(job_id, ws)

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist(self, job: Job) -> None:
        """Write job state to disk (best-effort)."""
        if not job.project_dir:
            return
        try:
            path = Path(job.project_dir) / "job.json"
            path.write_text(job.model_dump_json(indent=2))
        except Exception as exc:
            logger.warning("Could not persist job %s: %s", job.job_id, exc)


# Singleton — shared across the FastAPI app
job_manager = JobManager()
