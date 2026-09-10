"""
Reconstruction quality engine — our contribution.

Reads the COLMAP sparse model (TXT format) and computes a set of
measured and estimated quality metrics, then produces a 0–100 score.

Every metric is clearly labelled as MEASURED or ESTIMATED.
We never fabricate accuracy values.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from backend.models.results import ColmapStats, QualityScore

logger = logging.getLogger(__name__)

MetricType = Literal["measured", "estimated"]


@dataclass
class QualityMetric:
    name: str
    value: float
    unit: str
    metric_type: MetricType
    description: str = ""


@dataclass
class QualityReport:
    metrics: list[QualityMetric] = field(default_factory=list)
    colmap_stats: ColmapStats = field(default_factory=ColmapStats)
    quality_score: QualityScore = field(default_factory=QualityScore)
    warnings: list[str] = field(default_factory=list)


class ReconstructionQualityEngine:
    """
    Analyses a COLMAP sparse model and produces quality metrics.

    Input:  Path to exported TXT model (cameras.txt, images.txt, points3D.txt)
    Output: QualityReport with scored metrics
    """

    def __init__(self, config: dict) -> None:
        q = config.get("quality", {})
        weights = q.get("score_weights", {})
        self.w_registration = float(weights.get("camera_registration", 0.30))
        self.w_track = float(weights.get("track_length", 0.20))
        self.w_error = float(weights.get("reprojection_error", 0.20))
        self.w_density = float(weights.get("point_density", 0.15))
        self.w_coverage = float(weights.get("spatial_coverage", 0.15))

    def analyse(
        self,
        model_dir: Path,
        total_input_images: int,
        registered_images: int | None = None,
    ) -> QualityReport:
        report = QualityReport()
        model_dir = Path(model_dir)

        if not model_dir.exists():
            report.warnings.append(f"Model directory not found: {model_dir}")
            return report

        # ── Parse images.txt ────────────────────────────────────────────────
        images_data = self._parse_images(model_dir / "images.txt")
        n_registered = len(images_data)
        if registered_images is not None:
            n_registered = registered_images
        n_total = total_input_images or max(n_registered, 1)

        registration_rate = n_registered / max(n_total, 1)

        # ── Parse points3D.txt ───────────────────────────────────────────────
        points_data = self._parse_points3d(model_dir / "points3D.txt")
        n_points = len(points_data["errors"])
        mean_error = float(np.mean(points_data["errors"])) if points_data["errors"] else 0.0
        mean_track = float(np.mean(points_data["track_lengths"])) if points_data["track_lengths"] else 0.0

        # ── Spatial coverage (estimated) ─────────────────────────────────────
        spatial_coverage = self._estimate_spatial_coverage(images_data)

        # ── Build metrics list ───────────────────────────────────────────────
        report.metrics = [
            QualityMetric(
                "camera_registration_rate",
                round(registration_rate * 100, 1),
                "%",
                "measured",
                f"{n_registered}/{n_total} images registered",
            ),
            QualityMetric(
                "mean_reprojection_error",
                round(mean_error, 3),
                "px",
                "measured",
                "Mean re-projection error from COLMAP bundle adjustment",
            ),
            QualityMetric(
                "sparse_point_count",
                float(n_points),
                "pts",
                "measured",
                "Number of 3D points in sparse model",
            ),
            QualityMetric(
                "mean_track_length",
                round(mean_track, 2),
                "imgs/pt",
                "measured",
                "Average number of images observing each 3D point",
            ),
            QualityMetric(
                "spatial_coverage",
                round(spatial_coverage * 100, 1),
                "%",
                "estimated",
                "Fraction of camera-viewable volume with reconstructed points",
            ),
        ]

        # ── Colmap stats ─────────────────────────────────────────────────────
        report.colmap_stats = ColmapStats(
            total_images=n_total,
            registered_images=n_registered,
            registration_rate=registration_rate,
            sparse_point_count=n_points,
            mean_track_length=mean_track,
            mean_reprojection_error=mean_error,
        )

        # ── Compute overall score 0–100 ──────────────────────────────────────
        score = self._compute_score(
            registration_rate=registration_rate,
            mean_track=mean_track,
            mean_error=mean_error,
            n_points=n_points,
            spatial_coverage=spatial_coverage,
        )
        report.quality_score = score

        # ── Warnings ─────────────────────────────────────────────────────────
        if registration_rate < 0.50:
            report.warnings.append(
                f"Low registration rate ({registration_rate:.0%}). "
                "Consider recapturing with slower drone speed and more overlap."
            )
        if mean_error > 2.0:
            report.warnings.append(
                f"High reprojection error ({mean_error:.2f} px). "
                "Possible motion blur or rapid camera rotation."
            )
        if mean_track < 3.0 and n_points > 0:
            report.warnings.append(
                f"Short feature tracks (mean {mean_track:.1f} images/point). "
                "Scene may have low texture or insufficient overlap."
            )

        logger.info(
            "Quality: score=%.1f, reg=%.1f%%, err=%.2f px, pts=%d",
            score.overall, registration_rate * 100, mean_error, n_points,
        )
        return report

    # ── Score computation ────────────────────────────────────────────────────

    def _compute_score(
        self,
        registration_rate: float,
        mean_track: float,
        mean_error: float,
        n_points: int,
        spatial_coverage: float,
    ) -> QualityScore:
        # Registration: 0–100, direct
        reg_score = registration_rate * 100

        # Track length: 3 → 50%, 8 → 100%, cap at 100
        track_score = min(mean_track / 8.0, 1.0) * 100

        # Reprojection error: 0 px → 100, 5 px → 0 (linear)
        error_score = max(0.0, (1.0 - mean_error / 5.0)) * 100

        # Point density: >50k pts → 100%, linear, cap
        density_score = min(n_points / 50_000, 1.0) * 100

        # Coverage: 0–100, direct
        coverage_score = spatial_coverage * 100

        overall = (
            self.w_registration * reg_score
            + self.w_track * track_score
            + self.w_error * error_score
            + self.w_density * density_score
            + self.w_coverage * coverage_score
        )
        overall = max(0.0, min(100.0, overall))

        label = (
            "Excellent" if overall >= 85
            else "Good" if overall >= 70
            else "Fair" if overall >= 50
            else "Poor"
        )

        return QualityScore(
            overall=round(overall, 1),
            camera_registration=round(reg_score, 1),
            feature_coverage=round(track_score, 1),
            point_density=round(density_score, 1),
            spatial_coverage=round(coverage_score, 1),
            label=label,
        )

    # ── Parsers ──────────────────────────────────────────────────────────────

    def _parse_images(self, path: Path) -> list[dict]:
        """Return list of image metadata dicts from images.txt."""
        if not path.exists():
            return []
        images = []
        with open(path) as f:
            lines = [l for l in f if not l.startswith("#") and l.strip()]
        # Every 2 lines = one image
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) < 9:
                continue
            try:
                images.append({
                    "id": int(parts[0]),
                    "qw": float(parts[1]), "qx": float(parts[2]),
                    "qy": float(parts[3]), "qz": float(parts[4]),
                    "tx": float(parts[5]), "ty": float(parts[6]), "tz": float(parts[7]),
                    "camera_id": int(parts[8]),
                    "name": parts[9] if len(parts) > 9 else "",
                })
            except (ValueError, IndexError):
                continue
        return images

    def _parse_points3d(self, path: Path) -> dict:
        """Return errors and track lengths from points3D.txt."""
        result: dict = {"errors": [], "track_lengths": [], "xyz": []}
        if not path.exists():
            return result
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                try:
                    result["xyz"].append([float(parts[1]), float(parts[2]), float(parts[3])])
                    result["errors"].append(float(parts[7]))
                    track_len = max((len(parts) - 8) // 2, 1)
                    result["track_lengths"].append(track_len)
                except (ValueError, IndexError):
                    continue
        return result

    def _estimate_spatial_coverage(self, images: list[dict]) -> float:
        """
        Rough spatial coverage estimate based on camera position diversity.
        ESTIMATED metric — not ground-truth accuracy.
        """
        if len(images) < 2:
            return 0.0

        # Use camera translations as a proxy for coverage
        # (proper: project frustums into scene bounding box)
        positions = np.array([[img["tx"], img["ty"], img["tz"]] for img in images])
        if positions.shape[0] < 2:
            return 0.0

        # Normalised spread: std of camera positions
        spread = float(np.mean(np.std(positions, axis=0)))
        # Map to [0, 1] with soft cap
        coverage = min(spread / 10.0, 1.0)
        return coverage
