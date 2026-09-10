"""Abstract interface for all reconstruction backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class ReconstructionInput:
    image_dir: Path                         # directory containing input images
    output_dir: Path                        # where to write all outputs
    camera_priors: dict | None = None       # optional GPS/metadata priors (phase 7)
    progress_cb: Callable[[str, int], None] | None = None  # (message, pct)


@dataclass
class ReconstructionOutput:
    success: bool
    sparse_ply_path: Path | None = None     # exported point cloud
    confidence_ply_path: Path | None = None # per-point confidence colours
    camera_poses_json: Path | None = None   # camera trajectory
    model_dir: Path | None = None           # COLMAP model directory (txt format)
    registered_images: int = 0
    total_images: int = 0
    sparse_point_count: int = 0
    mean_reprojection_error: float = 0.0
    mean_track_length: float = 0.0
    processing_time_sec: float = 0.0
    error_message: str = ""
    log_lines: list[str] = field(default_factory=list)


class BaseReconstructor(ABC):
    """
    Abstract base for reconstruction backends.

    Implementations: ColmapReconstructor, OptimizedReconstructor
    Future:          DUSt3RReconstructor, MASt3RReconstructor
    """

    @abstractmethod
    def reconstruct(self, inp: ReconstructionInput) -> ReconstructionOutput:
        """Run the full reconstruction pipeline and return results."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name (e.g. 'COLMAP Baseline')."""
