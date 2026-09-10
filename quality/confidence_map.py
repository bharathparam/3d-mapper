"""
Per-region confidence map generator.

Reads sparse 3D points from the COLMAP model, voxelises the scene,
and computes a confidence level for each voxel based on:
  - Point density (measured)
  - Mean reprojection error (measured)
  - Number of observing cameras (measured)

Exports a coloured PLY where each point's RGB encodes its confidence:
  Green  = High confidence
  Yellow = Medium confidence
  Red    = Low confidence
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HIGH_COLOR   = (34,  197, 94)   # emerald green
MEDIUM_COLOR = (234, 179, 8)    # amber
LOW_COLOR    = (239, 68,  68)   # red


@dataclass
class ConfidenceRegion:
    """A spatial region with a confidence classification."""
    center: np.ndarray              # 3D centroid
    point_count: int
    camera_count: int               # approximate number of observing cameras
    mean_reprojection_error: float
    confidence_level: str           # "high" | "medium" | "low"
    confidence_score: float         # 0–1
    radius: float                   # approximate region radius (scene units)
    point_indices: list[int] = field(default_factory=list)


class ConfidenceMapGenerator:
    """Generates per-point and per-region confidence data from a COLMAP model."""

    def __init__(self, config: dict) -> None:
        q = config.get("quality", {})
        self.voxel_size: float = float(q.get("voxel_size", 0.5))
        self.high_min_pts: int = int(q.get("high_confidence_min_points_per_voxel", 5))
        self.medium_min_pts: int = int(q.get("medium_confidence_min_points_per_voxel", 2))
        self.high_max_err: float = float(q.get("high_confidence_max_reprojection_error", 1.5))
        self.medium_max_err: float = float(q.get("medium_confidence_max_reprojection_error", 3.0))

    def generate(
        self,
        model_dir: Path,
        output_dir: Path,
    ) -> tuple[Path | None, list[ConfidenceRegion], dict]:
        """
        Generate confidence PLY and return low-confidence regions.

        Returns
        -------
        (confidence_ply_path, low_confidence_regions, summary_dict)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        points = self._load_points3d(model_dir / "points3D.txt")
        if not points["xyz"]:
            logger.warning("No 3D points found in %s", model_dir)
            return None, [], {}

        xyz = np.array(points["xyz"])           # (N, 3)
        errors = np.array(points["errors"])     # (N,)
        track_lens = np.array(points["track_lengths"])  # (N,) ~ camera count
        rgb = np.array(points["rgb"])           # (N, 3) original colours

        # ── Per-point confidence ─────────────────────────────────────────────
        confidence_scores = self._per_point_confidence(errors, track_lens)
        labels = self._classify(confidence_scores)

        # ── Export coloured PLY ──────────────────────────────────────────────
        conf_colors = np.zeros((len(xyz), 3), dtype=np.uint8)
        for i, label in enumerate(labels):
            if label == "high":
                conf_colors[i] = HIGH_COLOR
            elif label == "medium":
                conf_colors[i] = MEDIUM_COLOR
            else:
                conf_colors[i] = LOW_COLOR

        ply_path = output_dir / "confidence.ply"
        self._write_ply(ply_path, xyz, conf_colors)

        # Also write a PLY with original colours for the standard view
        orig_ply_path = output_dir / "sparse_colored.ply"
        self._write_ply(orig_ply_path, xyz, rgb)

        # ── Voxel-level region analysis ──────────────────────────────────────
        regions = self._analyse_voxels(xyz, errors, track_lens, confidence_scores, labels)
        low_regions = [r for r in regions if r.confidence_level == "low"]

        # ── Summary ──────────────────────────────────────────────────────────
        summary = {
            "total_points": int(len(xyz)),
            "high_count": int(np.sum(labels == "high")),
            "medium_count": int(np.sum(labels == "medium")),
            "low_count": int(np.sum(labels == "low")),
            "high_ratio": round(float(np.mean(labels == "high")), 3),
            "medium_ratio": round(float(np.mean(labels == "medium")), 3),
            "low_ratio": round(float(np.mean(labels == "low")), 3),
            "low_confidence_region_count": len(low_regions),
        }

        # Save summary JSON
        (output_dir / "confidence_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        logger.info(
            "Confidence map: %d pts — high=%.0f%% medium=%.0f%% low=%.0f%%",
            summary["total_points"],
            summary["high_ratio"] * 100,
            summary["medium_ratio"] * 100,
            summary["low_ratio"] * 100,
        )

        return ply_path, low_regions, summary

    # ── Private ──────────────────────────────────────────────────────────────

    def _per_point_confidence(
        self,
        errors: np.ndarray,
        track_lens: np.ndarray,
    ) -> np.ndarray:
        """Combine reprojection error and track length into a 0–1 confidence score."""
        # Error component: 0 px → 1.0, 5 px → 0.0
        error_score = np.clip(1.0 - errors / 5.0, 0.0, 1.0)

        # Track length component: 1 → 0.0, 8+ → 1.0
        track_score = np.clip((track_lens - 1) / 7.0, 0.0, 1.0)

        return 0.6 * error_score + 0.4 * track_score

    def _classify(self, scores: np.ndarray) -> np.ndarray:
        labels = np.full(len(scores), "medium", dtype=object)
        labels[scores >= 0.65] = "high"
        labels[scores < 0.35] = "low"
        return labels

    def _analyse_voxels(
        self,
        xyz: np.ndarray,
        errors: np.ndarray,
        track_lens: np.ndarray,
        conf_scores: np.ndarray,
        labels: np.ndarray,
    ) -> list[ConfidenceRegion]:
        """Group points into voxels and produce region-level confidence."""
        if len(xyz) == 0:
            return []

        # Voxel grid indices
        min_pt = xyz.min(axis=0)
        voxel_indices = np.floor((xyz - min_pt) / self.voxel_size).astype(int)
        voxel_keys = [tuple(v) for v in voxel_indices]

        # Group by voxel
        from collections import defaultdict
        voxel_pts: dict = defaultdict(list)
        for i, key in enumerate(voxel_keys):
            voxel_pts[key].append(i)

        regions: list[ConfidenceRegion] = []
        for key, indices in voxel_pts.items():
            arr = np.array(indices)
            center = xyz[arr].mean(axis=0)
            mean_err = float(errors[arr].mean())
            mean_track = float(track_lens[arr].mean())
            mean_conf = float(conf_scores[arr].mean())

            # Classify voxel
            if len(indices) >= self.high_min_pts and mean_err <= self.high_max_err:
                level = "high"
            elif len(indices) >= self.medium_min_pts and mean_err <= self.medium_max_err:
                level = "medium"
            else:
                level = "low"

            regions.append(ConfidenceRegion(
                center=center,
                point_count=len(indices),
                camera_count=int(mean_track),
                mean_reprojection_error=mean_err,
                confidence_level=level,
                confidence_score=mean_conf,
                radius=self.voxel_size * 0.86,  # half-diagonal of voxel
                point_indices=arr.tolist(),
            ))

        return regions

    def _load_points3d(self, path: Path) -> dict:
        result: dict = {"xyz": [], "errors": [], "track_lengths": [], "rgb": []}
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
                    result["rgb"].append([int(parts[4]), int(parts[5]), int(parts[6])])
                    result["errors"].append(float(parts[7]))
                    result["track_lengths"].append(max((len(parts) - 8) // 2, 1))
                except (ValueError, IndexError):
                    continue
        return result

    @staticmethod
    def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
        """Write a binary PLY file with positions and colours."""
        n = len(xyz)
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        with open(path, "w") as f:
            f.write(header)
            for i in range(n):
                x, y, z = xyz[i]
                r, g, b = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        logger.debug("Wrote %d-point PLY to %s", n, path)
