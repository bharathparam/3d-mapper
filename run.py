#!/usr/bin/env python3
"""
Single-command entry point.

Usage
-----
  # Start the full web application (API + open browser)
  python run.py

  # Run reconstruction from CLI without the web UI
  python run.py --video path/to/drone.mp4 [--method optimized|baseline]

  # Start API server only (React dev server handles frontend)
  python run.py --server-only
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
import webbrowser
from pathlib import Path

import yaml
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config/default.yaml") -> dict:
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def check_dependencies() -> list[str]:
    """Return list of missing required tools."""
    import shutil
    missing = []
    if not shutil.which("colmap"):
        missing.append("colmap  →  brew install colmap")
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg  →  brew install ffmpeg")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Drone 3D Reconstruction System")
    parser.add_argument("--video", help="Path to drone video for CLI-mode reconstruction")
    parser.add_argument("--method", choices=["optimized", "baseline", "both"], default="optimized")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--server-only", action="store_true", help="Start API server without opening browser")
    parser.add_argument("--skip-dep-check", action="store_true", help="Start server even if colmap or ffmpeg are missing")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Dependency check ──────────────────────────────────────────────────────
    missing = check_dependencies()
    if missing:
        if args.skip_dep_check:
            logger.warning("Missing external dependencies (colmap/ffmpeg):")
            for m in missing:
                logger.warning("  • %s", m)
            logger.warning("Running anyway with --skip-dep-check. Reconstruction tasks will fail until installed.")
        else:
            logger.error("Missing required dependencies:")
            for m in missing:
                logger.error("  • %s", m)
            logger.error("Install them with: brew install colmap ffmpeg")
            logger.error("(Or run with --skip-dep-check to preview the UI/API without running reconstructions)")
            sys.exit(1)

    # ── CLI mode (no web UI) ──────────────────────────────────────────────────
    if args.video:
        _run_cli(args.video, args.method, config)
        return

    # ── Server mode ───────────────────────────────────────────────────────────
    host = config.get("app", {}).get("host", args.host)
    port = config.get("app", {}).get("port", args.port)

    logger.info("Starting Drone 3D Reconstruction System")
    logger.info("  API:      http://%s:%d/docs", host, port)
    logger.info("  Frontend: cd frontend && npm run dev  (http://localhost:5173)")

    if not args.server_only:
        import threading
        def _open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:5173")
        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=config.get("app", {}).get("debug", False),
        log_level="info",
    )


def _run_cli(video_path: str, method: str, config: dict) -> None:
    """Run reconstruction from CLI and print results."""
    from pathlib import Path
    import time, json
    from frame_selection.extractor import FrameExtractor
    from frame_selection.scorer import FrameScorer
    from frame_selection.selector import KeyframeSelector
    from reconstruction.colmap_reconstructor import ColmapReconstructor
    from reconstruction.base_reconstructor import ReconstructionInput
    from quality.metrics import ReconstructionQualityEngine
    from quality.confidence_map import ConfidenceMapGenerator
    from quality.failure_classifier import FailureClassifier
    from quality.recommender import RecaptureRecommender
    import uuid

    video = Path(video_path)
    if not video.exists():
        logger.error("Video not found: %s", video)
        sys.exit(1)

    job_id = str(uuid.uuid4())[:8]
    projects_dir = Path(config.get("app", {}).get("projects_dir", "projects"))
    project_dir = projects_dir / job_id
    frames_dir = project_dir / "processing" / "frames"
    sparse_out = project_dir / "processing" / "sparse"
    output_dir = project_dir / "output"
    for d in [frames_dir, sparse_out, output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Drone 3D Reconstruction — CLI Mode")
    logger.info("  Video:  %s", video)
    logger.info("  Method: %s", method)
    logger.info("  Job:    %s", job_id)
    logger.info("=" * 60)

    # Extract
    logger.info("[1/6] Extracting frames…")
    extractor = FrameExtractor(config)
    extraction = extractor.extract(video, frames_dir)
    logger.info("      → %d frames extracted", len(extraction.frames))

    # Score + select (optimized only)
    if method in ("optimized", "both"):
        scoring_cfg_path = config.get("frame_selection", {}).get("config_file", "config/frame_scoring.yaml")
        scoring_cfg = yaml.safe_load(Path(scoring_cfg_path).read_text()) if Path(scoring_cfg_path).exists() else {}

        logger.info("[2/6] Scoring frames…")
        scorer = FrameScorer(scoring_cfg)
        scores = scorer.score_batch(extraction.frames)

        logger.info("[3/6] Selecting keyframes…")
        selector = KeyframeSelector(scoring_cfg)
        selection = selector.select(scores)
        kf_dir = project_dir / "processing" / "keyframes"
        selector.copy_keyframes(selection, kf_dir)
        image_dir = kf_dir
        logger.info("      → %d/%d keyframes selected (%.0f%%)",
                    len(selection.selected), len(scores), selection.selection_ratio * 100)
    else:
        image_dir = frames_dir

    # COLMAP
    recon_cfg_path = config.get("reconstruction", {}).get("config_file", "config/reconstruction.yaml")
    recon_cfg = yaml.safe_load(Path(recon_cfg_path).read_text()) if Path(recon_cfg_path).exists() else {}

    logger.info("[4/6] Running COLMAP reconstruction…")
    t0 = time.perf_counter()
    colmap = ColmapReconstructor(recon_cfg)
    colmap_out = colmap.reconstruct(ReconstructionInput(
        image_dir=image_dir, output_dir=sparse_out,
        progress_cb=lambda m, p: logger.info("      [%d%%] %s", p, m),
    ))

    if not colmap_out.success:
        logger.error("Reconstruction failed: %s", colmap_out.error_message)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    logger.info("      → %d/%d images registered, %d pts, err=%.2f px, %.0f s",
                colmap_out.registered_images, colmap_out.total_images,
                colmap_out.sparse_point_count, colmap_out.mean_reprojection_error, elapsed)

    # Copy outputs
    if colmap_out.sparse_ply_path:
        import shutil
        shutil.copy2(colmap_out.sparse_ply_path, output_dir / "sparse.ply")

    # Quality
    logger.info("[5/6] Computing quality metrics…")
    qe = ReconstructionQualityEngine(config)
    report = qe.analyse(colmap_out.model_dir, colmap_out.total_images, colmap_out.registered_images)

    # Confidence
    logger.info("[6/6] Generating confidence map…")
    cmg = ConfidenceMapGenerator(config)
    conf_ply, low_regions, conf_summary = cmg.generate(colmap_out.model_dir, output_dir)

    classifier = FailureClassifier(config)
    diagnosed = classifier.classify(low_regions)
    recommender = RecaptureRecommender()
    recs = recommender.generate(diagnosed)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RECONSTRUCTION QUALITY")
    print("=" * 60)
    qs = report.quality_score
    print(f"  Overall Score:       {qs.overall:.0f}/100 ({qs.label})")
    print(f"  Camera Registration: {qs.camera_registration:.1f}%")
    print(f"  Feature Coverage:    {qs.feature_coverage:.1f}%")
    print(f"  Point Density:       {qs.point_density:.1f}%")
    print(f"  Spatial Coverage:    {qs.spatial_coverage:.1f}%  [ESTIMATED]")
    print(f"  Reprojection Error:  {report.colmap_stats.mean_reprojection_error:.3f} px  [MEASURED]")
    print(f"  Sparse Points:       {report.colmap_stats.sparse_point_count:,}  [MEASURED]")
    print(f"  Processing Time:     {elapsed:.0f} s")
    print()
    print(f"  Low-confidence regions: {conf_summary.get('low_confidence_region_count', 0)}")
    for i, rec in enumerate(recs):
        print(f"\n  [Region {i+1} — {rec.failure_type} — {rec.priority} priority]")
        for line in rec.description.split("\n")[:4]:
            print(f"    {line}")

    print()
    print(f"  Output:  {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
