"""
Per-frame quality scorer — our primary technical contribution.

Score(f) = w1·sharpness + w2·exposure + w3·feature_density
         + w4·feature_uniformity + w5·viewpoint_gain

All components are normalised to [0, 1].
Weights are loaded from config/frame_scoring.yaml.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameScore:
    frame_index: int
    path: Path
    timestamp_sec: float

    # Raw component scores [0, 1]
    sharpness: float = 0.0
    exposure: float = 0.0
    feature_density: float = 0.0
    feature_uniformity: float = 0.0
    viewpoint_gain: float = 0.0      # computed relative to already-selected set

    # Final weighted total
    total_score: float = 0.0

    # Diagnostics
    laplacian_var: float = 0.0
    feature_count: int = 0
    is_blurry: bool = False
    is_overexposed: bool = False
    is_underexposed: bool = False


class FrameScorer:
    """
    Computes a quality score for each frame.

    Blurriness:
        Laplacian variance of the grayscale image.
        Higher variance = sharper image.

    Exposure:
        Histogram analysis. Ideal frames have most pixels in [40, 215].

    Feature density:
        ORB keypoint count normalised by a reference count.

    Feature uniformity:
        Shannon entropy of feature distribution on a 4×4 grid.
        Higher entropy = features spread across the whole frame (better for SfM).

    Viewpoint gain:
        Estimated from optical-flow magnitude relative to preceding selected frame.
        Large flow = camera has moved enough to add new geometry.
        (Computed in selector, passed back here as the final score component.)
    """

    BLUR_THRESHOLD = 50.0           # Laplacian var below this → blurry flag

    def __init__(self, config: dict) -> None:
        sc = config.get("frame_scoring", config)   # accept full or sub-dict
        weights_raw = sc.get("weights", {})
        self.weights = {
            "sharpness":         float(weights_raw.get("sharpness", 0.30)),
            "exposure":          float(weights_raw.get("exposure", 0.15)),
            "feature_density":   float(weights_raw.get("feature_density", 0.25)),
            "feature_uniformity":float(weights_raw.get("feature_uniformity", 0.20)),
            "viewpoint_gain":    float(weights_raw.get("viewpoint_gain", 0.10)),
        }
        # Normalise weights
        total_w = sum(self.weights.values()) or 1.0
        self.weights = {k: v / total_w for k, v in self.weights.items()}

        sharp_cfg = sc.get("sharpness", {})
        self.sharp_cap: float = float(sharp_cfg.get("normalization_cap", 500.0))

        exp_cfg = sc.get("exposure", {})
        self.exp_low: int = int(exp_cfg.get("good_range_low", 40))
        self.exp_high: int = int(exp_cfg.get("good_range_high", 215))
        self.overexpose_pen: float = float(exp_cfg.get("overexpose_penalty", 0.5))
        self.underexpose_pen: float = float(exp_cfg.get("underexpose_penalty", 0.5))

        feat_cfg = sc.get("feature_detection", {})
        detector_name = feat_cfg.get("detector", "ORB").upper()
        max_features = int(feat_cfg.get("max_features", 2000))
        self.normalization_count: int = int(feat_cfg.get("normalization_count", 1500))
        self.grid_size: int = int(feat_cfg.get("grid_size", 4))

        if detector_name == "SIFT":
            self._detector = cv2.SIFT_create(nfeatures=max_features)
        else:
            self._detector = cv2.ORB_create(nfeatures=max_features)

    # ── Public API ───────────────────────────────────────────────────────────

    def score_frame(self, frame_index: int, path: Path, timestamp_sec: float) -> FrameScore:
        """Load image from *path* and compute all component scores."""
        bgr = cv2.imread(str(path))
        if bgr is None:
            logger.warning("Could not read frame: %s", path)
            return FrameScore(frame_index=frame_index, path=path, timestamp_sec=timestamp_sec)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        sharpness, lap_var = self._sharpness(gray)
        exposure, over, under = self._exposure(gray)
        density, uniformity, kp_count, keypoints = self._features(gray)

        # viewpoint_gain starts at 0; selector fills it in later
        score = FrameScore(
            frame_index=frame_index,
            path=path,
            timestamp_sec=timestamp_sec,
            sharpness=sharpness,
            exposure=exposure,
            feature_density=density,
            feature_uniformity=uniformity,
            viewpoint_gain=0.0,
            laplacian_var=lap_var,
            feature_count=kp_count,
            is_blurry=lap_var < self.BLUR_THRESHOLD,
            is_overexposed=over,
            is_underexposed=under,
        )
        score.total_score = self._combine(score)
        return score

    def score_batch(
        self,
        frame_infos: list,          # list[FrameInfo]
        progress_cb=None,           # optional callback(done, total)
    ) -> list[FrameScore]:
        """Score a list of FrameInfo objects."""
        results: list[FrameScore] = []
        total = len(frame_infos)
        t0 = time.perf_counter()

        for i, fi in enumerate(frame_infos):
            s = self.score_frame(fi.frame_index, fi.path, fi.timestamp_sec)
            results.append(s)
            if progress_cb and (i % max(1, total // 20) == 0 or i == total - 1):
                progress_cb(i + 1, total)

        elapsed = time.perf_counter() - t0
        logger.info("Scored %d frames in %.1f s (%.0f fps)", total, elapsed, total / max(elapsed, 1e-9))
        return results

    # ── Score components ─────────────────────────────────────────────────────

    def _sharpness(self, gray: np.ndarray) -> tuple[float, float]:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        normalised = min(lap_var / self.sharp_cap, 1.0)
        return normalised, lap_var

    def _exposure(self, gray: np.ndarray) -> tuple[float, bool, bool]:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / (hist.sum() + 1e-9)

        good_mass = float(hist_norm[self.exp_low : self.exp_high + 1].sum())
        over_mass = float(hist_norm[240:].sum())
        under_mass = float(hist_norm[:16].sum())

        raw = good_mass - self.overexpose_pen * over_mass - self.underexpose_pen * under_mass
        score = float(np.clip(raw, 0.0, 1.0))
        return score, over_mass > 0.2, under_mass > 0.2

    def _features(self, gray: np.ndarray) -> tuple[float, float, int, list]:
        keypoints = self._detector.detect(gray, None)

        density = min(len(keypoints) / self.normalization_count, 1.0)

        if not keypoints:
            return density, 0.0, 0, []

        # Grid-based Shannon entropy
        h, w = gray.shape
        gs = self.grid_size
        cell_h = max(h // gs, 1)
        cell_w = max(w // gs, 1)
        grid = np.zeros((gs, gs), dtype=float)

        for kp in keypoints:
            ci = min(int(kp.pt[1]) // cell_h, gs - 1)
            cj = min(int(kp.pt[0]) // cell_w, gs - 1)
            grid[ci, cj] += 1.0

        flat = grid.flatten() + 1e-9
        prob = flat / flat.sum()
        entropy = float(-np.sum(prob * np.log(prob)))
        max_entropy = float(np.log(gs * gs))
        uniformity = entropy / max_entropy if max_entropy > 0 else 0.0

        return density, uniformity, len(keypoints), keypoints

    def _combine(self, s: FrameScore) -> float:
        w = self.weights
        return (
            w["sharpness"] * s.sharpness
            + w["exposure"] * s.exposure
            + w["feature_density"] * s.feature_density
            + w["feature_uniformity"] * s.feature_uniformity
            + w["viewpoint_gain"] * s.viewpoint_gain
        )
