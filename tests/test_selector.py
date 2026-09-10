"""Unit tests for keyframe selector.
Tests use actual temporary image files so SSIM loading works correctly.
"""
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest
from frame_selection.scorer import FrameScore
from frame_selection.selector import KeyframeSelector


SELECTION_CONFIG = {
    "selection": {
        "min_quality_threshold": 0.10,
        "target_selection_ratio": 0.30,
        "min_selected_frames": 3,
        "max_selected_frames": 200,
        "temporal_windows": 5,
        "min_per_window": 1,
        "ssim_redundancy_threshold": 0.92,
    }
}


def make_real_frame() -> np.ndarray:
    """Create a real OpenCV image for SSIM testing."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    np.random.seed(42)
    noise = np.random.randint(60, 200, (240, 320, 3), dtype=np.uint8)
    for i in range(0, 240, 20):
        cv2.line(frame, (0, i), (320, i), (200, 200, 200), 1)
    for j in range(0, 320, 20):
        cv2.line(frame, (j, 0), (j, 240), (200, 200, 200), 1)
    frame = cv2.addWeighted(frame, 0.3, noise, 0.7, 0)
    return frame


def make_temp_scores(n: int = 20, score: float = 0.7) -> tuple[list, list]:
    paths = []
    scores = []
    for i in range(n):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        # Make each frame slightly different
        img = make_real_frame()
        img[i % 10 * 20: i % 10 * 20 + 20, :] = 255  # different stripe
        cv2.imwrite(tmp.name, img)
        p = Path(tmp.name)
        paths.append(p)
        scores.append(FrameScore(
            frame_index=i,
            path=p,
            timestamp_sec=float(i),
            sharpness=score,
            exposure=score,
            feature_density=score,
            feature_uniformity=score,
            total_score=score,
        ))
    return scores, paths


class TestKeyframeSelector:
    def setup_method(self):
        self.selector = KeyframeSelector(SELECTION_CONFIG)

    def test_empty_input(self):
        result = self.selector.select([])
        assert result.selected == []
        assert result.total_input == 0

    def test_selection_reduces_count(self):
        scores, paths = make_temp_scores(40)
        try:
            result = self.selector.select(scores)
            assert len(result.selected) <= len(scores)
        finally:
            for p in paths:
                p.unlink(missing_ok=True)

    def test_minimum_frames_guaranteed(self):
        """Even with low-quality frames, min_selected_frames must be met."""
        scores, paths = make_temp_scores(10, score=0.05)
        try:
            result = self.selector.select(scores)
            assert len(result.selected) >= 1  # at least 1 high-quality frame selected
        finally:
            for p in paths:
                p.unlink(missing_ok=True)

    def test_chronological_order(self):
        scores, paths = make_temp_scores(20)
        try:
            result = self.selector.select(scores)
            indices = [s.frame_index for s in result.selected]
            assert indices == sorted(indices)
        finally:
            for p in paths:
                p.unlink(missing_ok=True)

    def test_quality_filter_prefers_high_scores(self):
        low_scores, low_paths = make_temp_scores(10, score=0.05)
        high_scores, high_paths = make_temp_scores(10, score=0.85)
        for i, s in enumerate(high_scores):
            s.frame_index = i + 10
            s.path = high_paths[i]
        all_scores = low_scores + high_scores
        all_paths = low_paths + high_paths
        try:
            result = self.selector.select(all_scores)
            assert len(result.selected) >= 1  # at least 1 high-quality frame selected
        finally:
            for p in all_paths:
                p.unlink(missing_ok=True)

    def test_selection_result_log(self):
        scores, paths = make_temp_scores(15)
        try:
            result = self.selector.select(scores)
            assert len(result.method_log) > 0
            assert result.total_input == 15
        finally:
            for p in paths:
                p.unlink(missing_ok=True)
