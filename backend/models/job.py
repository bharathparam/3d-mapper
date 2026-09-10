"""Pydantic models for job state and progress events."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReconstructionMethod(str, Enum):
    BASELINE = "baseline"       # all extracted frames → COLMAP
    OPTIMIZED = "optimized"     # quality-aware selection → COLMAP
    BOTH = "both"               # run both and compare


class Stage(BaseModel):
    name: str
    label: str
    status: StageStatus = StageStatus.PENDING
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    method: ReconstructionMethod = ReconstructionMethod.OPTIMIZED
    video_filename: str = ""
    status: JobStatus = JobStatus.PENDING
    progress: int = 0           # 0–100
    current_stage: str = ""
    error_message: str | None = None
    stages: list[Stage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    project_dir: str = ""

    def get_stage(self, name: str) -> Stage | None:
        return next((s for s in self.stages if s.name == name), None)


class CreateJobRequest(BaseModel):
    method: ReconstructionMethod = ReconstructionMethod.OPTIMIZED


class ProgressEvent(BaseModel):
    """WebSocket broadcast payload."""
    event: str                      # stage_started | stage_completed | progress | error | done
    stage: str = ""
    message: str = ""
    progress: int = 0
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
