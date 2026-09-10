"""Pydantic models for reconstruction results and quality metrics."""
from __future__ import annotations

from pydantic import BaseModel


class FrameStats(BaseModel):
    total_extracted: int = 0
    total_scored: int = 0
    selected_for_reconstruction: int = 0
    selection_ratio: float = 0.0
    mean_sharpness: float = 0.0
    mean_exposure: float = 0.0
    mean_feature_density: float = 0.0


class ColmapStats(BaseModel):
    total_images: int = 0
    registered_images: int = 0
    registration_rate: float = 0.0
    sparse_point_count: int = 0
    mean_track_length: float = 0.0
    mean_reprojection_error: float = 0.0
    processing_time_seconds: float = 0.0


class QualityScore(BaseModel):
    overall: float = 0.0            # 0–100
    camera_registration: float = 0.0
    feature_coverage: float = 0.0
    point_density: float = 0.0
    spatial_coverage: float = 0.0
    label: str = ""                 # Excellent / Good / Fair / Poor


class ConfidenceBreakdown(BaseModel):
    high_ratio: float = 0.0
    medium_ratio: float = 0.0
    low_ratio: float = 0.0
    low_confidence_region_count: int = 0


class FailureRegion(BaseModel):
    region_id: int
    center: list[float]             # [x, y, z] world coords
    radius: float                   # approximate radius in scene units
    failure_type: str               # missing_viewpoint | motion_blur | low_overlap | textureless
    confidence_score: float         # 0–1 (lower = worse)
    point_count: int
    camera_count: int
    mean_reprojection_error: float | None = None
    recommendation: str = ""


class RecaptureRecommendation(BaseModel):
    region_id: int
    priority: str                   # high | medium | low
    failure_type: str
    description: str
    suggested_direction: list[float] | None = None    # unit vector
    suggested_altitude_delta: float | None = None     # meters (relative)
    suggested_overlap_increase: float | None = None   # fraction


class ComparisonRow(BaseModel):
    metric: str
    baseline_value: str
    optimized_value: str
    delta: str
    unit: str = ""
    is_measured: bool = True        # False = estimated


class ReconstructionResult(BaseModel):
    job_id: str
    method: str
    artifacts: dict[str, str] = {}          # label → URL path
    frame_stats: FrameStats = FrameStats()
    colmap_stats: ColmapStats = ColmapStats()
    quality_score: QualityScore = QualityScore()
    confidence: ConfidenceBreakdown = ConfidenceBreakdown()
    failure_regions: list[FailureRegion] = []
    recommendations: list[RecaptureRecommendation] = []
    comparison: list[ComparisonRow] = []     # populated when method == BOTH
