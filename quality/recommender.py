"""
Adaptive recapture recommendation engine.

Translates typed failure modes into actionable flight suggestions.
Does not control a real drone — generates text + optional spatial hints.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from backend.models.results import RecaptureRecommendation
from quality.failure_classifier import (
    FAILURE_LOW_OVERLAP,
    FAILURE_MISSING_VIEWPOINT,
    FAILURE_MOTION_BLUR,
    FAILURE_TEXTURELESS,
    DiagnosedRegion,
)

logger = logging.getLogger(__name__)


class RecaptureRecommender:
    """Generates actionable recapture recommendations from diagnosed regions."""

    def generate(
        self,
        diagnosed_regions: list[DiagnosedRegion],
    ) -> list[RecaptureRecommendation]:
        recs: list[RecaptureRecommendation] = []
        for i, dr in enumerate(diagnosed_regions):
            rec = self._recommend_one(i, dr)
            recs.append(rec)
        return recs

    def _recommend_one(self, idx: int, dr: DiagnosedRegion) -> RecaptureRecommendation:
        r = dr.region
        ft = dr.failure_type

        if ft == FAILURE_MISSING_VIEWPOINT:
            # Suggest a perpendicular pass over the region
            direction = self._suggest_perpendicular_direction(r.center)
            return RecaptureRecommendation(
                region_id=idx,
                priority=dr.severity,
                failure_type=ft,
                description=(
                    f"Low-confidence region at ({r.center[0]:.1f}, {r.center[1]:.1f}, {r.center[2]:.1f}).\n"
                    f"Problem: Only {r.camera_count} camera(s) viewed this area — insufficient angular coverage.\n"
                    "Recommendation: Fly an additional pass over this region from a different direction.\n"
                    "Suggested approach: 20–35° different viewing angle relative to existing passes.\n"
                    "This provides the triangulation baseline needed to reconstruct this area accurately."
                ),
                suggested_direction=direction.tolist(),
                suggested_altitude_delta=None,
                suggested_overlap_increase=None,
            )

        if ft == FAILURE_LOW_OVERLAP:
            return RecaptureRecommendation(
                region_id=idx,
                priority=dr.severity,
                failure_type=ft,
                description=(
                    f"Low-confidence region at ({r.center[0]:.1f}, {r.center[1]:.1f}, {r.center[2]:.1f}).\n"
                    f"Problem: Only {r.point_count} reconstructed points in this voxel — insufficient overlap.\n"
                    "Recommendation: Increase image overlap over this area.\n"
                    "Options:\n"
                    "  • Reduce drone speed by ~30% when flying over this zone\n"
                    "  • Increase front-lap from the current rate to ≥80%\n"
                    "  • Fly at a lower altitude to increase GSD and feature density"
                ),
                suggested_direction=None,
                suggested_altitude_delta=-5.0,    # suggest flying 5 m lower
                suggested_overlap_increase=0.20,   # suggest +20% overlap
            )

        if ft == FAILURE_MOTION_BLUR:
            return RecaptureRecommendation(
                region_id=idx,
                priority=dr.severity,
                failure_type=ft,
                description=(
                    f"Low-confidence region at ({r.center[0]:.1f}, {r.center[1]:.1f}, {r.center[2]:.1f}).\n"
                    f"Problem: High reprojection error ({r.mean_reprojection_error:.2f} px) — likely motion blur.\n"
                    "Recommendation: Recapture this area with reduced drone speed.\n"
                    "Options:\n"
                    "  • Reduce maximum speed over this zone to ≤3 m/s\n"
                    "  • Use a faster shutter speed (≥1/1000 s for moving platforms)\n"
                    "  • Enable electronic image stabilisation if available\n"
                    "  • Hover briefly at key positions rather than continuous flight"
                ),
                suggested_direction=None,
                suggested_altitude_delta=None,
                suggested_overlap_increase=None,
            )

        # textureless
        return RecaptureRecommendation(
            region_id=idx,
            priority=dr.severity,
            failure_type=ft,
            description=(
                f"Low-confidence region at ({r.center[0]:.1f}, {r.center[1]:.1f}, {r.center[2]:.1f}).\n"
                "Problem: Textureless or reflective surface — insufficient features for matching.\n"
                "Options:\n"
                "  • Capture under different lighting conditions (e.g. overcast vs. direct sun)\n"
                "  • Use a sensor with polarisation filter to reduce specular reflections\n"
                "  • Use multi-spectral imagery if the surface is vegetation or water\n"
                "Note: Some surfaces (water, glass, uniform concrete) are fundamentally\n"
                "      difficult for feature-based photogrammetry regardless of capture strategy."
            ),
            suggested_direction=None,
            suggested_altitude_delta=None,
            suggested_overlap_increase=None,
        )

    @staticmethod
    def _suggest_perpendicular_direction(center: np.ndarray) -> np.ndarray:
        """
        Suggest a unit direction vector 25° offset from nadir
        pointing toward the region center.
        """
        nadir = np.array([0.0, 0.0, -1.0])  # straight down
        if np.linalg.norm(center) < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        horizontal = center[:2]
        if np.linalg.norm(horizontal) < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        h_dir = horizontal / np.linalg.norm(horizontal)
        angle = math.radians(25)
        direction = np.array([
            h_dir[0] * math.sin(angle),
            h_dir[1] * math.sin(angle),
            -math.cos(angle),
        ])
        return direction / np.linalg.norm(direction)
