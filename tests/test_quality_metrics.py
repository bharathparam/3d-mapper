"""Unit tests for quality metrics engine."""
import tempfile
from pathlib import Path
import pytest
from quality.metrics import ReconstructionQualityEngine


QUALITY_CONFIG = {
    "quality": {
        "score_weights": {
            "camera_registration": 0.30,
            "track_length": 0.20,
            "reprojection_error": 0.20,
            "point_density": 0.15,
            "spatial_coverage": 0.15,
        }
    }
}

# Minimal valid COLMAP TXT model
IMAGES_TXT = """# Image list
1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_000001.jpg
0.5 0.5 -1 0.6 0.6 -1
2 0.99 0.1 0.0 0.0 1.0 0.0 0.0 1 frame_000002.jpg
0.5 0.5 1 0.6 0.6 1
3 0.98 0.0 0.1 0.0 2.0 0.0 0.0 1 frame_000003.jpg
0.5 0.5 1
"""

POINTS3D_TXT = """# 3D point list
1 0.1 0.2 3.0 200 100 50 0.45 1 0 2 1
2 0.5 0.8 3.1 180 120 60 0.82 1 1 2 0 3 0
3 1.0 0.3 2.9 160 140 70 1.20 1 0 3 1
4 2.0 1.0 3.0 150 130 80 0.55 1 2 2 0 3 1
5 0.0 0.0 3.0 170 110 90 0.38 1 0 2 1 3 2
"""


def make_model_dir() -> Path:
    tmpdir = Path(tempfile.mkdtemp())
    (tmpdir / "images.txt").write_text(IMAGES_TXT)
    (tmpdir / "points3D.txt").write_text(POINTS3D_TXT)
    return tmpdir


class TestQualityMetrics:
    def setup_method(self):
        self.engine = ReconstructionQualityEngine(QUALITY_CONFIG)

    def test_analyse_returns_report(self):
        model_dir = make_model_dir()
        report = self.engine.analyse(model_dir, total_input_images=5, registered_images=3)
        assert report is not None
        assert report.quality_score.overall >= 0
        assert report.quality_score.overall <= 100

    def test_registration_rate_measured(self):
        model_dir = make_model_dir()
        report = self.engine.analyse(model_dir, total_input_images=10, registered_images=3)
        # 3/10 = 30%
        assert abs(report.colmap_stats.registration_rate - 0.30) < 0.01

    def test_point_count(self):
        model_dir = make_model_dir()
        report = self.engine.analyse(model_dir, total_input_images=5)
        assert report.colmap_stats.sparse_point_count == 5

    def test_reprojection_error(self):
        model_dir = make_model_dir()
        report = self.engine.analyse(model_dir, total_input_images=5)
        assert report.colmap_stats.mean_reprojection_error > 0

    def test_quality_label(self):
        model_dir = make_model_dir()
        report = self.engine.analyse(model_dir, total_input_images=5, registered_images=5)
        assert report.quality_score.label in ("Excellent", "Good", "Fair", "Poor")

    def test_missing_model_dir(self):
        report = self.engine.analyse(Path("/nonexistent"), total_input_images=10)
        assert len(report.warnings) > 0
