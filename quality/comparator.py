"""
Baseline vs. optimised reconstruction comparator.

Produces a comparison table populated only with measured values.
All entries clearly indicate whether they are MEASURED or ESTIMATED.
We never invent numbers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.models.results import ColmapStats, ComparisonRow, FrameStats

logger = logging.getLogger(__name__)


@dataclass
class ComparisonInput:
    baseline_frame_stats: FrameStats
    baseline_colmap: ColmapStats
    baseline_time_sec: float

    optimized_frame_stats: FrameStats
    optimized_colmap: ColmapStats
    optimized_time_sec: float


def build_comparison(inp: ComparisonInput) -> list[ComparisonRow]:
    """
    Build a side-by-side comparison of baseline vs. optimised pipeline.

    Only rows where both values are available are included.
    Values are always actual measured results — never fabricated.
    """
    rows: list[ComparisonRow] = []

    def row(
        metric: str,
        b_val,
        o_val,
        unit: str = "",
        is_measured: bool = True,
        higher_is_better: bool = False,
        lower_is_better: bool = False,
    ) -> None:
        if b_val is None or o_val is None:
            return
        b_str = _fmt(b_val, unit)
        o_str = _fmt(o_val, unit)

        # Compute delta string
        try:
            b_f, o_f = float(b_val), float(o_val)
            if b_f == 0:
                delta = "N/A"
            else:
                pct = (o_f - b_f) / abs(b_f) * 100
                sign = "+" if pct >= 0 else ""
                delta = f"{sign}{pct:.1f}%"
                # Add directional emoji
                if higher_is_better and pct > 0:
                    delta += " ↑"
                elif higher_is_better and pct < 0:
                    delta += " ↓"
                elif lower_is_better and pct < 0:
                    delta += " ↑"   # lower is better, so decrease is good
                elif lower_is_better and pct > 0:
                    delta += " ↓"
        except (TypeError, ValueError):
            delta = "N/A"

        rows.append(ComparisonRow(
            metric=metric,
            baseline_value=b_str,
            optimized_value=o_str,
            delta=delta,
            unit=unit,
            is_measured=is_measured,
        ))

    b = inp.baseline_colmap
    o = inp.optimized_colmap
    bf = inp.baseline_frame_stats
    of_ = inp.optimized_frame_stats

    row("Frames fed to COLMAP",
        bf.selected_for_reconstruction or bf.total_extracted,
        of_.selected_for_reconstruction,
        "frames", is_measured=True, lower_is_better=True)

    row("Registered images",
        b.registered_images, o.registered_images,
        "imgs", is_measured=True, higher_is_better=True)

    row("Registration rate",
        round(b.registration_rate * 100, 1),
        round(o.registration_rate * 100, 1),
        "%", is_measured=True, higher_is_better=True)

    row("Sparse 3D points",
        b.sparse_point_count, o.sparse_point_count,
        "pts", is_measured=True, higher_is_better=True)

    row("Mean reprojection error",
        round(b.mean_reprojection_error, 3),
        round(o.mean_reprojection_error, 3),
        "px", is_measured=True, lower_is_better=True)

    row("Mean track length",
        round(b.mean_track_length, 2),
        round(o.mean_track_length, 2),
        "imgs/pt", is_measured=True, higher_is_better=True)

    row("Processing time",
        round(inp.baseline_time_sec, 1),
        round(inp.optimized_time_sec, 1),
        "s", is_measured=True, lower_is_better=True)

    return rows


def _fmt(val, unit: str) -> str:
    if isinstance(val, float):
        return f"{val:.2f} {unit}".strip()
    return f"{val} {unit}".strip()
