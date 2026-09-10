"""
Intelligent keyframe selector — our primary technical contribution.

Strategy
--------
1. Filter frames below a minimum quality threshold.
2. Divide the video into N temporal windows.
3. Within each window, rank frames by total_score.
4. Apply greedy cross-window deduplication via SSIM.
5. Guarantee minimum frames per window (temporal coverage).
6. Clamp total selected count within [min, max].

The viewpoint_gain component of each frame score is estimated here
using SSIM as a cheap proxy for frame-to-frame camera movement.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from frame_selection.scorer import FrameScore

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    selected: list[FrameScore]      # chosen keyframes, sorted by frame_index
    rejected: list[FrameScore]      # filtered-out frames
    total_input: int
    selection_ratio: float
    method_log: list[str]           # human-readable explanation of decisions


class KeyframeSelector:
    """
    Selects the most informative subset of frames for COLMAP reconstruction.

    The selection goal: maximum scene coverage with minimum redundancy.
    """

    def __init__(self, config: dict) -> None:
        sc = config.get("frame_scoring", config)
        sel = sc.get("selection", {})

        self.min_quality: float = float(sel.get("min_quality_threshold", 0.20))
        self.target_ratio: float = float(sel.get("target_selection_ratio", 0.30))
        self.min_frames: int = int(sel.get("min_selected_frames", 20))
        self.max_frames: int = int(sel.get("max_selected_frames", 600))
        self.temporal_windows: int = int(sel.get("temporal_windows", 10))
        self.min_per_window: int = int(sel.get("min_per_window", 2))
        self.ssim_threshold: float = float(sel.get("ssim_redundancy_threshold", 0.92))

    # ── Public API ───────────────────────────────────────────────────────────

    def select(
        self,
        scores: list[FrameScore],
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> SelectionResult:
        log: list[str] = []
        total = len(scores)
        if total == 0:
            return SelectionResult([], [], 0, 0.0, ["No frames to select from."])

        # ── Step 1: Quality filter ──────────────────────────────────────────
        quality_ok = [s for s in scores if s.total_score >= self.min_quality]
        quality_bad = [s for s in scores if s.total_score < self.min_quality]
        log.append(
            f"Quality filter: {len(quality_ok)}/{total} frames passed "
            f"(threshold={self.min_quality:.2f})"
        )

        if len(quality_ok) < self.min_frames:
            # Fallback: take the best frames regardless of threshold
            quality_ok = sorted(scores, key=lambda s: s.total_score, reverse=True)[
                : max(self.min_frames, total // 3)
            ]
            quality_bad = [s for s in scores if s not in quality_ok]
            log.append(
                f"Quality filter fallback: using top {len(quality_ok)} frames "
                "regardless of threshold"
            )

        # ── Step 2: Temporal windowing ──────────────────────────────────────
        max_idx = max(s.frame_index for s in quality_ok)
        window_size = max(max_idx // self.temporal_windows, 1)
        windows: list[list[FrameScore]] = [[] for _ in range(self.temporal_windows)]

        for s in quality_ok:
            w_idx = min(s.frame_index // window_size, self.temporal_windows - 1)
            windows[w_idx].append(s)

        # Sort each window by score desc
        for w in windows:
            w.sort(key=lambda s: s.total_score, reverse=True)

        log.append(
            f"Temporal windows: {self.temporal_windows}, "
            f"~{window_size} frames each"
        )

        # ── Step 3: Target count per window ────────────────────────────────
        target_total = int(total * self.target_ratio)
        target_total = max(self.min_frames, min(self.max_frames, target_total))
        per_window_target = max(self.min_per_window, target_total // self.temporal_windows)

        # ── Step 4: Greedy SSIM deduplication ─────────────────────────────
        selected: list[FrameScore] = []
        selected_images: list[np.ndarray] = []  # cached small versions for SSIM

        total_windows = len(windows)
        for wi, window in enumerate(windows):
            if progress_cb:
                progress_cb(wi + 1, total_windows)

            window_selected = 0
            for candidate in window:
                if window_selected >= per_window_target:
                    break
                if len(selected) >= self.max_frames:
                    break

                # Load image for SSIM check
                img = self._load_small(candidate.path)
                if img is None:
                    continue

                if not selected or not self._is_redundant(img, selected_images):
                    # Update viewpoint_gain: SSIM distance from last selected
                    if selected_images:
                        ssim_val = self._ssim(img, selected_images[-1])
                        candidate.viewpoint_gain = float(1.0 - ssim_val)
                    else:
                        candidate.viewpoint_gain = 1.0  # first frame gets max gain

                    # Recompute total_score with viewpoint_gain now set
                    # (avoid circular import — do it inline)
                    candidate.total_score = (
                        candidate.total_score + 0.10 * candidate.viewpoint_gain
                    )

                    selected.append(candidate)
                    selected_images.append(img)
                    window_selected += 1

            # Guarantee minimum per window
            if window_selected < self.min_per_window and window:
                for fallback in window:
                    if fallback not in selected:
                        img = self._load_small(fallback.path)
                        if img is not None:
                            selected.append(fallback)
                            selected_images.append(img)
                            window_selected += 1
                    if window_selected >= self.min_per_window:
                        break

        # Sort by frame_index for chronological ordering
        selected.sort(key=lambda s: s.frame_index)

        # Build rejected set
        selected_indices = {s.frame_index for s in selected}
        rejected = [s for s in scores if s.frame_index not in selected_indices]

        ratio = len(selected) / total if total else 0.0
        log.append(
            f"Selected {len(selected)}/{total} frames "
            f"({ratio:.1%} of total, target was {self.target_ratio:.1%})"
        )
        log.append(f"SSIM redundancy threshold: {self.ssim_threshold:.2f}")

        logger.info(
            "Keyframe selection: %d → %d frames (%.1f%%)",
            total, len(selected), ratio * 100,
        )

        return SelectionResult(
            selected=selected,
            rejected=rejected,
            total_input=total,
            selection_ratio=ratio,
            method_log=log,
        )

    def copy_keyframes(
        self,
        result: SelectionResult,
        destination: Path,
    ) -> list[Path]:
        """Copy selected keyframe images to *destination* directory."""
        destination.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for score in result.selected:
            dst = destination / score.path.name
            shutil.copy2(score.path, dst)
            copied.append(dst)
        logger.info("Copied %d keyframes to %s", len(copied), destination)
        return copied

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_small(self, path: Path, width: int = 320) -> np.ndarray | None:
        """Load a downscaled grayscale image for fast SSIM comparison."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        h, w = img.shape
        scale = width / w
        return cv2.resize(img, (width, int(h * scale)))

    def _is_redundant(
        self,
        candidate: np.ndarray,
        selected_images: list[np.ndarray],
    ) -> bool:
        """Return True if candidate is too similar to any already-selected frame."""
        for sel_img in selected_images[-5:]:   # check only recent N for speed
            if self._ssim(candidate, sel_img) >= self.ssim_threshold:
                return True
        return False

    @staticmethod
    def _ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Fast Structural Similarity Index between two grayscale images."""
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))

        C1, C2 = 6.5025, 58.5225   # (0.01*255)^2, (0.03*255)^2
        a_f = a.astype(np.float64)
        b_f = b.astype(np.float64)

        mu1 = cv2.GaussianBlur(a_f, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(b_f, (11, 11), 1.5)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(a_f * a_f, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(b_f * b_f, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(a_f * b_f, (11, 11), 1.5) - mu1_mu2

        ssim_map = (
            (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return float(ssim_map.mean())
