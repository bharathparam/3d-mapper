"""
COLMAP reconstruction backend — called via CLI subprocess.

This is the BASELINE method: takes a directory of images and runs the
full COLMAP pipeline without any frame selection intelligence.

Pipeline
--------
1. feature_extractor   → SIFT features per image
2. sequential_matcher  → match adjacent image pairs
3. mapper              → SfM + bundle adjustment → sparse model
4. model_converter     → export 3D points as PLY
"""
from __future__ import annotations

import json
import re
import logging
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

from reconstruction.base_reconstructor import (
    BaseReconstructor,
    ReconstructionInput,
    ReconstructionOutput,
)

logger = logging.getLogger(__name__)


class ColmapReconstructor(BaseReconstructor):
    """Wraps the COLMAP CLI for sparse 3D reconstruction."""

    def __init__(self, config: dict, colmap_bin: str | None = None) -> None:
        self.config = config.get("feature_extraction", config)
        self.full_cfg = config
        self.colmap_bin = colmap_bin or shutil.which("colmap") or "colmap"

    @property
    def name(self) -> str:
        return "COLMAP Baseline"

    # ── Public API ───────────────────────────────────────────────────────────

    def reconstruct(self, inp: ReconstructionInput) -> ReconstructionOutput:
        t_start = time.perf_counter()
        out = ReconstructionOutput(success=False)
        log: list[str] = []

        try:
            self._verify_colmap()
            self._verify_images(inp.image_dir)

            db_path = inp.output_dir / "database.db"
            sparse_dir = inp.output_dir / "sparse"
            txt_dir = inp.output_dir / "sparse_txt"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            txt_dir.mkdir(parents=True, exist_ok=True)

            total_images = len(list(inp.image_dir.glob("*.jpg")) + list(inp.image_dir.glob("*.png")))

            # 1. Feature extraction
            self._progress(inp, "Extracting SIFT features…", 10)
            self._run_feature_extraction(db_path, inp.image_dir, log)

            # 2. Feature matching
            self._progress(inp, "Matching feature pairs…", 30)
            self._run_feature_matching(db_path, log)

            # 3. Sparse reconstruction
            self._progress(inp, "Running SfM / bundle adjustment…", 50)
            self._run_mapper(db_path, inp.image_dir, sparse_dir, log)

            # 4. Find best model component
            model_path = self._best_model(sparse_dir)
            if model_path is None:
                out.error_message = (
                    "COLMAP mapper produced no valid reconstruction.\n"
                    "Possible causes: insufficient overlap, excessive blur, or low texture.\n"
                    "Try: slower drone speed, more overlap, or sharper footage."
                )
                out.log_lines = log
                return out

            # 5. Export TXT model (for quality analysis)
            self._progress(inp, "Exporting model…", 80)
            self._run_model_converter(model_path, txt_dir, "TXT", log)

            # 6. Export PLY
            ply_path = inp.output_dir / "sparse.ply"
            self._run_model_converter(model_path, ply_path, "PLY", log)

            # 7. Parse model statistics
            self._progress(inp, "Parsing model statistics…", 90)
            stats = self._parse_txt_model(txt_dir)

            reg_rate = stats["registered"] / max(total_images, 1)
            thresholds = self.full_cfg.get("thresholds", {})

            if stats["registered"] < int(thresholds.get("min_registered_images", 5)):
                out.error_message = (
                    f"Only {stats['registered']}/{total_images} images registered "
                    f"({reg_rate:.0%}).\n"
                    "Likely causes: insufficient overlap or excessive motion blur.\n"
                    "Recommendation: fly slower with more overlapping passes."
                )
                out.log_lines = log
                return out

            # 8. Export camera poses JSON
            poses_json = inp.output_dir / "camera_poses.json"
            self._export_camera_poses(txt_dir, poses_json)

            out.success = True
            out.sparse_ply_path = ply_path
            out.camera_poses_json = poses_json
            out.model_dir = txt_dir
            out.registered_images = stats["registered"]
            out.total_images = total_images
            out.sparse_point_count = stats["point_count"]
            out.mean_reprojection_error = stats["mean_error"]
            out.mean_track_length = stats["mean_track_length"]
            out.processing_time_sec = time.perf_counter() - t_start
            out.log_lines = log

            logger.info(
                "COLMAP done: %d/%d images, %d pts, err=%.2f px, %.1f s",
                out.registered_images, out.total_images,
                out.sparse_point_count, out.mean_reprojection_error,
                out.processing_time_sec,
            )

        except Exception as exc:
            out.error_message = str(exc)
            out.log_lines = log
            logger.exception("COLMAP reconstruction failed")

        return out

    # ── COLMAP commands ──────────────────────────────────────────────────────

    def _get_colmap_version(self) -> tuple[int, int]:
        """Detect COLMAP version tuple e.g. (4, 2) or (3, 8)."""
        try:
            res = subprocess.run([self.colmap_bin, "-h"], capture_output=True, text=True)
            text = res.stdout + res.stderr
            m = re.search(r"COLMAP\s+(\d+)\.(\d+)", text)
            if m:
                return int(m.group(1)), int(m.group(2))
        except Exception:
            pass
        return (3, 8)

    def _run_feature_extraction(
        self, db: Path, image_dir: Path, log: list[str]
    ) -> None:
        fe = self.full_cfg.get("feature_extraction", {})
        ver = self._get_colmap_version()
        cmd = [
            self.colmap_bin, "feature_extractor",
            "--database_path", str(db),
            "--image_path", str(image_dir),
            "--ImageReader.camera_model", fe.get("camera_model", "SIMPLE_RADIAL"),
        ]
        if ver >= (3, 9):
            cmd += [
                "--FeatureExtraction.use_gpu", "0",
                "--FeatureExtraction.max_image_size", str(fe.get("max_image_size", 3200)),
                "--SiftExtraction.max_num_features", str(fe.get("max_num_features", 8192)),
            ]
        else:
            cmd += [
                "--SiftExtraction.use_gpu", "0",
                "--SiftExtraction.max_image_size", str(fe.get("max_image_size", 3200)),
                "--SiftExtraction.max_num_features", str(fe.get("max_num_features", 8192)),
            ]
        self._run(cmd, log, "feature_extractor")

    def _run_feature_matching(self, db: Path, log: list[str]) -> None:
        fm = self.full_cfg.get("feature_matching", {})
        matcher = fm.get("matcher", "sequential")
        ver = self._get_colmap_version()
        cmd = [
            self.colmap_bin, f"{matcher}_matcher",
            "--database_path", str(db),
        ]
        if ver >= (3, 9):
            cmd += ["--FeatureMatching.use_gpu", "0"]
        else:
            cmd += ["--SiftMatching.use_gpu", "0"]

        if matcher == "sequential":
            cmd += ["--SequentialMatching.overlap", str(fm.get("overlap", 10))]
        self._run(cmd, log, f"{matcher}_matcher")

    def _run_mapper(
        self, db: Path, image_dir: Path, output_dir: Path, log: list[str]
    ) -> None:
        mp = self.full_cfg.get("mapper", {})
        cmd = [
            self.colmap_bin, "mapper",
            "--database_path", str(db),
            "--image_path", str(image_dir),
            "--output_path", str(output_dir),
            "--Mapper.min_num_matches", str(mp.get("min_num_matches", 15)),
            "--Mapper.init_min_num_inliers", str(mp.get("init_min_num_inliers", 100)),
            "--Mapper.abs_pose_min_num_inliers", str(mp.get("abs_pose_min_num_inliers", 30)),
            "--Mapper.abs_pose_min_inlier_ratio", str(mp.get("abs_pose_min_inlier_ratio", 0.25)),
            "--Mapper.max_reg_trials", str(mp.get("max_reg_trials", 3)),
            "--Mapper.num_threads", str(mp.get("num_threads", -1)),
        ]
        self._run(cmd, log, "mapper")

    def _run_model_converter(
        self, input_path: Path, output_path: Path, output_type: str, log: list[str]
    ) -> None:
        cmd = [
            self.colmap_bin, "model_converter",
            "--input_path", str(input_path),
            "--output_path", str(output_path),
            "--output_type", output_type,
        ]
        self._run(cmd, log, f"model_converter ({output_type})")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _run(self, cmd: list[str], log: list[str], step: str) -> None:
        logger.debug("COLMAP %s: %s", step, " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            log.extend(result.stdout.splitlines()[-20:])   # last 20 lines
        if result.returncode != 0:
            log.append(f"STDERR: {result.stderr[-500:]}")
            raise RuntimeError(
                f"COLMAP {step} failed (exit {result.returncode}).\n"
                f"{result.stderr[-500:]}"
            )

    def _verify_colmap(self) -> None:
        try:
            subprocess.run([self.colmap_bin, "--help"], capture_output=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"COLMAP not found at '{self.colmap_bin}'.\n"
                "Install with: brew install colmap"
            ) from exc

    def _verify_images(self, image_dir: Path) -> None:
        images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        if not images:
            raise RuntimeError(f"No images found in {image_dir}")
        logger.info("Found %d images in %s", len(images), image_dir)

    def _best_model(self, sparse_dir: Path) -> Path | None:
        """Find the model component with the most images."""
        model_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            return None
        best = max(
            model_dirs,
            key=lambda d: len(list(d.glob("images.*"))),
            default=None,
        )
        return best

    def _parse_txt_model(self, txt_dir: Path) -> dict:
        """Parse COLMAP TXT model to extract statistics."""
        result = {
            "registered": 0,
            "point_count": 0,
            "mean_error": 0.0,
            "mean_track_length": 0.0,
        }

        # Count registered images
        images_txt = txt_dir / "images.txt"
        if images_txt.exists():
            with open(images_txt) as f:
                lines = [l for l in f if not l.startswith("#") and l.strip()]
            # Each image = 2 lines
            result["registered"] = len(lines) // 2

        # Parse points3D.txt
        points_txt = txt_dir / "points3D.txt"
        if points_txt.exists():
            errors: list[float] = []
            track_lengths: list[int] = []
            with open(points_txt) as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) < 8:
                        continue
                    try:
                        errors.append(float(parts[7]))
                        # TRACK[] starts at index 8, each entry = 2 values
                        track_len = (len(parts) - 8) // 2
                        track_lengths.append(track_len)
                    except (ValueError, IndexError):
                        continue

            result["point_count"] = len(errors)
            result["mean_error"] = float(np.mean(errors)) if errors else 0.0
            result["mean_track_length"] = float(np.mean(track_lengths)) if track_lengths else 0.0

        return result

    def _export_camera_poses(self, txt_dir: Path, out_json: Path) -> None:
        """Export camera centres to JSON for the 3D viewer trajectory."""
        images_txt = txt_dir / "images.txt"
        if not images_txt.exists():
            return

        poses = []
        with open(images_txt) as f:
            lines = [l.rstrip() for l in f if not l.startswith("#") and l.strip()]

        # Every 2 lines = one image (header + point observations)
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) < 9:
                continue
            try:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                name = parts[9]

                # Camera centre C = -R^T * T
                R = self._quat_to_rotation(qw, qx, qy, qz)
                C = (-R.T @ np.array([tx, ty, tz])).tolist()

                poses.append({"image": name, "center": C, "quat": [qw, qx, qy, qz]})
            except (ValueError, IndexError):
                continue

        out_json.write_text(json.dumps({"cameras": poses}, indent=2))
        logger.info("Exported %d camera poses to %s", len(poses), out_json)

    @staticmethod
    def _quat_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        """Convert quaternion to 3×3 rotation matrix."""
        n = qw*qw + qx*qx + qy*qy + qz*qz
        if n < 1e-10:
            return np.eye(3)
        s = 2.0 / n
        return np.array([
            [1 - s*(qy**2 + qz**2),  s*(qx*qy - qz*qw),    s*(qx*qz + qy*qw)],
            [s*(qx*qy + qz*qw),      1 - s*(qx**2 + qz**2), s*(qy*qz - qx*qw)],
            [s*(qx*qz - qy*qw),      s*(qy*qz + qx*qw),    1 - s*(qx**2 + qy**2)],
        ])
    
    def _progress(self, inp: ReconstructionInput, msg: str, pct: int) -> None:
        if inp.progress_cb:
            inp.progress_cb(msg, pct)
