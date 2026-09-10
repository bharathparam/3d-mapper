"""
Our optimized reconstruction method.

Applies quality-aware frame selection before running COLMAP.
This is the method whose results we compare against the baseline.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from frame_selection.extractor import FrameExtractor, FrameInfo
from frame_selection.scorer import FrameScorer
from frame_selection.selector import KeyframeSelector, SelectionResult
from reconstruction.base_reconstructor import (
    BaseReconstructor,
    ReconstructionInput,
    ReconstructionOutput,
)
from reconstruction.colmap_reconstructor import ColmapReconstructor

logger = logging.getLogger(__name__)


class OptimizedReconstructor(BaseReconstructor):
    """
    Quality-aware frame selection → COLMAP reconstruction.

    The frame selection is our contribution; COLMAP is the geometric backend.
    """

    def __init__(self, config: dict, colmap_bin: str | None = None) -> None:
        self.config = config
        self._scorer = FrameScorer(config.get("frame_scoring_config", {}))
        self._selector = KeyframeSelector(config.get("frame_scoring_config", {}))
        self._colmap = ColmapReconstructor(
            config.get("reconstruction_config", {}),
            colmap_bin=colmap_bin,
        )

    @property
    def name(self) -> str:
        return "Quality-Aware Adaptive (Our Method)"

    def reconstruct(self, inp: ReconstructionInput) -> ReconstructionOutput:
        t_start = time.perf_counter()

        # Expect pre-extracted frames in inp.image_dir
        frame_paths = sorted(
            list(inp.image_dir.glob("*.jpg")) + list(inp.image_dir.glob("*.png"))
        )
        if not frame_paths:
            out = ReconstructionOutput(success=False)
            out.error_message = f"No images found in {inp.image_dir}"
            return out

        # Build FrameInfo list
        frame_infos = [
            FrameInfo(frame_index=i, path=p, timestamp_sec=float(i))
            for i, p in enumerate(frame_paths)
        ]

        # ── Score all frames ─────────────────────────────────────────────────
        self._progress(inp, "Scoring frame quality…", 5)
        scores = self._scorer.score_batch(frame_infos)

        # ── Select keyframes ─────────────────────────────────────────────────
        self._progress(inp, "Selecting optimal keyframes…", 15)
        selection = self._selector.select(scores)

        logger.info(
            "Frame selection: %d → %d frames (%.1f%%)",
            selection.total_input,
            len(selection.selected),
            selection.selection_ratio * 100,
        )

        for line in selection.method_log:
            logger.info("  %s", line)

        # ── Copy keyframes to a separate dir ─────────────────────────────────
        keyframe_dir = inp.output_dir.parent / "keyframes"
        self._selector.copy_keyframes(selection, keyframe_dir)

        # ── Run COLMAP on selected keyframes ─────────────────────────────────
        self._progress(inp, "Running COLMAP on selected keyframes…", 25)
        colmap_inp = ReconstructionInput(
            image_dir=keyframe_dir,
            output_dir=inp.output_dir,
            camera_priors=inp.camera_priors,
            progress_cb=lambda msg, pct: self._progress(inp, msg, 25 + pct // 2),
        )
        colmap_out = self._colmap.reconstruct(colmap_inp)

        # Attach frame selection metadata
        colmap_out.total_images = selection.total_input
        if colmap_out.success:
            colmap_out.processing_time_sec = time.perf_counter() - t_start

        # Persist frame scores JSON for analysis
        scores_json = inp.output_dir / "frame_scores.json"
        self._save_scores(scores, selection, scores_json)

        return colmap_out

    def _progress(self, inp: ReconstructionInput, msg: str, pct: int) -> None:
        if inp.progress_cb:
            inp.progress_cb(msg, pct)

    @staticmethod
    def _save_scores(scores, selection: SelectionResult, path: Path) -> None:
        import json

        selected_indices = {s.frame_index for s in selection.selected}
        data = {
            "total_frames": len(scores),
            "selected_frames": len(selection.selected),
            "selection_ratio": selection.selection_ratio,
            "method_log": selection.method_log,
            "scores": [
                {
                    "frame_index": s.frame_index,
                    "timestamp_sec": s.timestamp_sec,
                    "total_score": round(s.total_score, 4),
                    "sharpness": round(s.sharpness, 4),
                    "exposure": round(s.exposure, 4),
                    "feature_density": round(s.feature_density, 4),
                    "feature_uniformity": round(s.feature_uniformity, 4),
                    "viewpoint_gain": round(s.viewpoint_gain, 4),
                    "feature_count": s.feature_count,
                    "is_blurry": s.is_blurry,
                    "is_selected": s.frame_index in selected_indices,
                }
                for s in scores
            ],
        }
        path.write_text(json.dumps(data, indent=2))
