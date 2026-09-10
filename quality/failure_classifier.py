"""
Failure mode classifier for low-confidence reconstruction regions.

For each bad region, determines WHY reconstruction failed and classifies it
into one of four typed failure modes. This diagnosis drives the recapture
recommendations.

Failure modes
-------------
missing_viewpoint   Camera rays are too few or too collinear → need new angle
low_overlap         Point density is very low → images didn't overlap enough
motion_blur         Reprojection error is high with sufficient cameras → blur
textureless         Few features matched → scene lacks visual texture
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from quality.confidence_map import ConfidenceRegion

logger = logging.getLogger(__name__)

FAILURE_MISSING_VIEWPOINT = "missing_viewpoint"
FAILURE_LOW_OVERLAP       = "low_overlap"
FAILURE_MOTION_BLUR       = "motion_blur"
FAILURE_TEXTURELESS       = "textureless"


@dataclass
class DiagnosedRegion:
    region: ConfidenceRegion
    failure_type: str
    confidence: float
    diagnosis: str          # human-readable explanation
    severity: str           # "high" | "medium" | "low"


class FailureClassifier:
    """
    Classifies low-confidence regions by failure mode.

    Decision logic (in priority order):
    1. camera_count < threshold                → missing_viewpoint
    2. point_count < density_threshold         → low_overlap
    3. reprojection_error > blur_threshold     → motion_blur
    4. default                                 → textureless
    """

    def __init__(self, config: dict | None = None) -> None:
        c = (config or {}).get("quality", {})
        self.min_cameras = int(c.get("min_cameras_per_region", 3))
        self.density_threshold = int(c.get("medium_confidence_min_points_per_voxel", 2))
        self.blur_error_threshold = float(c.get("medium_confidence_max_reprojection_error", 3.0))

    def classify(self, regions: list[ConfidenceRegion]) -> list[DiagnosedRegion]:
        """Classify a list of low-confidence regions."""
        diagnosed: list[DiagnosedRegion] = []
        for region in regions:
            failure_type, diagnosis = self._classify_one(region)
            severity = self._severity(region.confidence_score)
            diagnosed.append(DiagnosedRegion(
                region=region,
                failure_type=failure_type,
                confidence=region.confidence_score,
                diagnosis=diagnosis,
                severity=severity,
            ))
        # Sort by severity (high first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        diagnosed.sort(key=lambda d: severity_order.get(d.severity, 2))
        logger.info(
            "Classified %d low-confidence regions: %s",
            len(diagnosed),
            {FAILURE_MISSING_VIEWPOINT: 0, FAILURE_LOW_OVERLAP: 0,
             FAILURE_MOTION_BLUR: 0, FAILURE_TEXTURELESS: 0}
            | {d.failure_type: sum(1 for x in diagnosed if x.failure_type == d.failure_type)
               for d in diagnosed},
        )
        return diagnosed

    def _classify_one(self, r: ConfidenceRegion) -> tuple[str, str]:
        if r.camera_count < self.min_cameras:
            return (
                FAILURE_MISSING_VIEWPOINT,
                f"Only {r.camera_count} camera(s) observe this region "
                f"(minimum required: {self.min_cameras}). "
                "The area was not captured from enough different angles.",
            )

        if r.point_count < self.density_threshold:
            return (
                FAILURE_LOW_OVERLAP,
                f"Very sparse point density ({r.point_count} pts/voxel). "
                "Adjacent drone passes may not have overlapped sufficiently.",
            )

        if r.mean_reprojection_error > self.blur_error_threshold:
            return (
                FAILURE_MOTION_BLUR,
                f"High reprojection error ({r.mean_reprojection_error:.2f} px) "
                "despite adequate camera coverage. "
                "Likely cause: motion blur or rapid camera rotation during capture.",
            )

        return (
            FAILURE_TEXTURELESS,
            "Reconstruction failed despite cameras and overlap — likely a "
            "low-texture surface (uniform colour, water, glass) that produces "
            "too few distinctive feature matches.",
        )

    @staticmethod
    def _severity(confidence_score: float) -> str:
        if confidence_score < 0.20:
            return "high"
        if confidence_score < 0.35:
            return "medium"
        return "low"
