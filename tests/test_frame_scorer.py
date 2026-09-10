"""Unit tests for the frame quality scorer."""
import numpy as np
import pytest
import cv2
import tempfile
from pathlib import Path
from frame_selection.scorer import FrameScorer


SCORER_CONFIG = {
    "weights": {"sharpness": 0.30, "exposure": 0.15, "feature_density": 0.25,
                "feature_uniformity": 0.20, "viewpoint_gain": 0.10},
    "sharpness": {"normalization_cap": 500.0},
    "exposure": {"good_range_low": 40, "good_range_high": 215,
                 "overexpose_penalty": 0.5, "underexpose_penalty": 0.5},
    "feature_detection": {"detector": "ORB", "max_features": 500,
                          "normalization_count": 500, "grid_size": 4},
}


def make_synthetic_frame(mode: str = "normal") -> np.ndarray:
    """Create a synthetic test frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    if mode == "normal":
        # Textured frame with good exposure
        np.random.seed(42)
        noise = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        # Add edges for ORB to detect
        for i in range(0, 480, 30):
            cv2.line(frame, (0, i), (640, i), (200, 200, 200), 1)
        for j in range(0, 640, 30):
            cv2.line(frame, (j, 0), (j, 480), (200, 200, 200), 1)
        frame = cv2.addWeighted(frame, 0.3, noise, 0.7, 0)
    elif mode == "blurry":
        frame[:] = 128
        cv2.GaussianBlur(frame, (51, 51), 0, dst=frame)
    elif mode == "overexposed":
        frame[:] = 245
    elif mode == "underexposed":
        frame[:] = 5
    return frame


def save_frame(frame: np.ndarray) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, frame)
    return Path(tmp.name)


class TestFrameScorer:
    def setup_method(self):
        self.scorer = FrameScorer(SCORER_CONFIG)

    def test_sharpness_sharp_vs_blurry(self):
        sharp_frame = make_synthetic_frame("normal")
        blurry_frame = make_synthetic_frame("blurry")
        sharp_gray = cv2.cvtColor(sharp_frame, cv2.COLOR_BGR2GRAY)
        blurry_gray = cv2.cvtColor(blurry_frame, cv2.COLOR_BGR2GRAY)
        s_sharp, _ = self.scorer._sharpness(sharp_gray)
        s_blurry, _ = self.scorer._sharpness(blurry_gray)
        assert s_sharp > s_blurry, "Sharp frame should have higher sharpness score"

    def test_exposure_good_vs_overexposed(self):
        good_frame = cv2.cvtColor(make_synthetic_frame("normal"), cv2.COLOR_BGR2GRAY)
        over_frame = cv2.cvtColor(make_synthetic_frame("overexposed"), cv2.COLOR_BGR2GRAY)
        s_good, _, _ = self.scorer._exposure(good_frame)
        s_over, _, _ = self.scorer._exposure(over_frame)
        assert s_good > s_over, "Well-exposed frame should outscore overexposed"

    def test_exposure_underexposed(self):
        under_frame = cv2.cvtColor(make_synthetic_frame("underexposed"), cv2.COLOR_BGR2GRAY)
        score, _, is_under = self.scorer._exposure(under_frame)
        assert is_under, "Underexposed frame should be flagged"

    def test_score_range(self):
        """All score components must be in [0, 1]."""
        path = save_frame(make_synthetic_frame("normal"))
        try:
            s = self.scorer.score_frame(0, path, 0.0)
            assert 0.0 <= s.sharpness <= 1.0
            assert 0.0 <= s.exposure <= 1.0
            assert 0.0 <= s.feature_density <= 1.0
            assert 0.0 <= s.feature_uniformity <= 1.0
            assert 0.0 <= s.total_score <= 1.0
        finally:
            path.unlink(missing_ok=True)

    def test_blurry_flag(self):
        path = save_frame(make_synthetic_frame("blurry"))
        try:
            s = self.scorer.score_frame(0, path, 0.0)
            assert s.is_blurry, "Blurry frame should be flagged"
        finally:
            path.unlink(missing_ok=True)

    def test_batch_scoring(self):
        from frame_selection.extractor import FrameInfo
        paths = [save_frame(make_synthetic_frame()) for _ in range(5)]
        try:
            infos = [FrameInfo(i, p, float(i)) for i, p in enumerate(paths)]
            scores = self.scorer.score_batch(infos)
            assert len(scores) == 5
            assert all(0 <= s.total_score <= 1 for s in scores)
        finally:
            for p in paths:
                p.unlink(missing_ok=True)
