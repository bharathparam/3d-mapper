"""
Orchestrates the full reconstruction pipeline for a given job.

Calls each stage in order, updates the job manager with progress,
and handles errors gracefully with user-friendly messages.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path

import yaml

from backend.core.job_manager import JobManager
from backend.models.job import Job, ReconstructionMethod, Stage, StageStatus
from backend.models.results import (
    ConfidenceBreakdown,
    FailureRegion,
    FrameStats,
    ReconstructionResult,
    RecaptureRecommendation,
)
from frame_selection.extractor import FrameExtractor
from frame_selection.scorer import FrameScorer
from frame_selection.selector import KeyframeSelector
from quality.comparator import ComparisonInput, build_comparison
from quality.confidence_map import ConfidenceMapGenerator
from quality.failure_classifier import FailureClassifier
from quality.metrics import ReconstructionQualityEngine
from quality.recommender import RecaptureRecommender
from reconstruction.colmap_reconstructor import ColmapReconstructor
from reconstruction.surface_mesher import SurfaceMesher
from reconstruction.ml_completer import MLPointCompleter
from reconstruction.texture_mapper import TextureMapper
from reconstruction.ai_depth_estimator import AIDepthEstimator
from reconstruction.gaussian_splatter import GaussianSplatter
from reconstruction.optimized_reconstructor import OptimizedReconstructor

logger = logging.getLogger(__name__)

# Stage definitions (name, label, progress_pct at completion)
STAGES_OPTIMIZED = [
    ("frame_extraction",       "Extract Frames",           15),
    ("frame_scoring",          "Score Frame Quality",       25),
    ("keyframe_selection",     "Select Keyframes",          30),
    ("feature_extraction",     "COLMAP: Feature Extraction", 50),
    ("feature_matching",       "COLMAP: Feature Matching",  65),
    ("sparse_reconstruction",  "Sparse Reconstruction",     70),
    ("surface_meshing",        "3D Mesh & Hole Filling",    75),
    ("ml_completion",          "ML Point Cloud Completion", 80),
    ("ai_depth_estimation",    "AI Monocular Depth Field",  85),
    ("texture_mapping",        "Photorealistic 3D Texturing", 90),
    ("gaussian_splatting",     "3D Gaussian Splatting (3DGS)", 93),
    ("quality_analysis",       "Quality Analysis",          90),
    ("confidence_map",         "Confidence Map",            95),
    ("recommendations",        "Generate Recommendations",  100),
]

STAGES_BASELINE = [
    ("frame_extraction",       "Extract Frames",           20),
    ("feature_extraction",     "COLMAP: Feature Extraction", 45),
    ("feature_matching",       "COLMAP: Feature Matching",  65),
    ("sparse_reconstruction",  "Sparse Reconstruction",     80),
    ("quality_analysis",       "Quality Analysis",          92),
    ("confidence_map",         "Confidence Map",            100),
]


class PipelineRunner:
    """Runs the full reconstruction pipeline for one job."""

    def __init__(self, app_config: dict, job_manager: JobManager) -> None:
        self.cfg = app_config
        self.jm = job_manager
        self.projects_dir = Path(app_config.get("app", {}).get("projects_dir", "projects"))

        tools = app_config.get("tools", {})
        self.colmap_bin: str = tools.get("colmap") or shutil.which("colmap") or "colmap"
        self.ffmpeg_bin: str = tools.get("ffmpeg") or shutil.which("ffmpeg") or "ffmpeg"

    # ── Public entry point ───────────────────────────────────────────────────

    async def run(self, job: Job, video_path: Path) -> ReconstructionResult:
        """Run pipeline asynchronously (called in a background task)."""
        project_dir = Path(job.project_dir)
        stages = (
            STAGES_OPTIMIZED
            if job.method in (ReconstructionMethod.OPTIMIZED, ReconstructionMethod.BOTH)
            else STAGES_BASELINE
        )

        # Initialise stage list on the job
        job.stages = [
            Stage(name=name, label=label)
            for name, label, _ in stages
        ]
        await self.jm.create(job)   # persist initial state

        main_loop = asyncio.get_running_loop()
        try:
            result = await main_loop.run_in_executor(
                None,
                self._run_sync,
                job,
                video_path,
                project_dir,
                stages,
                main_loop,
            )
            await self.jm.complete_job(job.job_id)
            return result

        except Exception as exc:
            logger.exception("Pipeline failed for job %s", job.job_id)
            await self.jm.fail_job(job.job_id, str(exc))
            raise

    def _run_sync(
        self,
        job: Job,
        video_path: Path,
        project_dir: Path,
        stages: list,
        main_loop: asyncio.AbstractEventLoop,
    ) -> ReconstructionResult:
        """Synchronous pipeline — runs in a thread pool executor."""
        def notify(stage_name: str, action: str, **kwargs):
            """Stage event dispatched safely to the main asyncio loop."""
            if action == "start":
                coro = self.jm.start_stage(job.job_id, stage_name, **kwargs)
            elif action == "done":
                coro = self.jm.complete_stage(job.job_id, stage_name, **kwargs)
            else:
                coro = self.jm.fail_stage(job.job_id, stage_name, **kwargs)
            try:
                future = asyncio.run_coroutine_threadsafe(coro, main_loop)
                future.result(timeout=10)
            except Exception as e:
                logger.warning(f"Stage notification error ({stage_name} {action}): {e}")

        # ── Load stage-specific configs ──────────────────────────────────────
        scoring_cfg = self._load_yaml(
            self.cfg.get("frame_selection", {}).get("config_file", "config/frame_scoring.yaml")
        )
        recon_cfg = self._load_yaml(
            self.cfg.get("reconstruction", {}).get("config_file", "config/reconstruction.yaml")
        )

        # ── Directory layout ─────────────────────────────────────────────────
        frames_dir = project_dir / "processing" / "frames"
        kf_dir = project_dir / "processing" / "keyframes"
        sparse_out = project_dir / "processing" / "sparse"
        output_dir = project_dir / "output"

        frames_dir.mkdir(parents=True, exist_ok=True)
        kf_dir.mkdir(parents=True, exist_ok=True)
        sparse_out.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Shared result accumulators
        frame_stats = FrameStats()
        baseline_result = None
        optimized_result = None
        quality_report = None

        # ═══════════════════════════════════════════════════════════════════
        # PHASE A: Frame extraction (same for both methods)
        # ═══════════════════════════════════════════════════════════════════
        notify("frame_extraction", "start", message="Extracting frames via FFmpeg…", progress=5)
        extractor = FrameExtractor(self.cfg)
        try:
            extraction = extractor.extract(video_path, frames_dir)
        except RuntimeError as exc:
            notify("frame_extraction", "fail", message=str(exc))
            raise

        frame_stats.total_extracted = len(extraction.frames)
        notify("frame_extraction", "done",
               message=f"Extracted {len(extraction.frames)} frames ({extraction.video_duration_sec:.0f}s video)",
               data={"frame_count": len(extraction.frames)}, progress=15)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE B: Frame scoring + selection (optimized only)
        # ═══════════════════════════════════════════════════════════════════
        if job.method in (ReconstructionMethod.OPTIMIZED, ReconstructionMethod.BOTH):
            notify("frame_scoring", "start", message="Scoring frame quality…", progress=16)
            scorer = FrameScorer(scoring_cfg)
            scores = scorer.score_batch(extraction.frames)
            frame_stats.total_scored = len(scores)
            frame_stats.mean_sharpness = round(sum(s.sharpness for s in scores) / max(len(scores), 1), 3)
            frame_stats.mean_exposure = round(sum(s.exposure for s in scores) / max(len(scores), 1), 3)
            frame_stats.mean_feature_density = round(sum(s.feature_density for s in scores) / max(len(scores), 1), 3)
            notify("frame_scoring", "done",
                   message=f"Scored {len(scores)} frames. Mean sharpness: {frame_stats.mean_sharpness:.2f}",
                   data={"mean_sharpness": frame_stats.mean_sharpness}, progress=25)

            notify("keyframe_selection", "start", message="Selecting optimal keyframes…", progress=26)
            selector = KeyframeSelector(scoring_cfg)
            selection = selector.select(scores)
            selector.copy_keyframes(selection, kf_dir)
            frame_stats.selected_for_reconstruction = len(selection.selected)
            frame_stats.selection_ratio = round(selection.selection_ratio, 3)
            notify("keyframe_selection", "done",
                   message=f"Selected {len(selection.selected)}/{len(scores)} keyframes ({selection.selection_ratio:.0%})",
                   data={"selected": len(selection.selected), "ratio": selection.selection_ratio},
                   progress=30)

            image_dir_for_recon = kf_dir
        else:
            # Baseline: use all extracted frames
            frame_stats.selected_for_reconstruction = len(extraction.frames)
            frame_stats.selection_ratio = 1.0
            image_dir_for_recon = frames_dir

        # ═══════════════════════════════════════════════════════════════════
        # PHASE C: COLMAP reconstruction
        # ═══════════════════════════════════════════════════════════════════
        def colmap_progress_cb(msg: str, pct: int) -> None:
            # Map COLMAP sub-steps to named stages
            try:
                if "feature" in msg.lower() and "extract" in msg.lower():
                    future = asyncio.run_coroutine_threadsafe(
                        self.jm.start_stage(job.job_id, "feature_extraction", message=msg, progress=30 + pct // 5),
                        main_loop,
                    )
                    future.result(timeout=5)
                elif "match" in msg.lower():
                    future = asyncio.run_coroutine_threadsafe(
                        self.jm.start_stage(job.job_id, "feature_matching", message=msg, progress=50),
                        main_loop,
                    )
                    future.result(timeout=5)
                elif "sfm" in msg.lower() or "bundle" in msg.lower() or "mapper" in msg.lower():
                    future = asyncio.run_coroutine_threadsafe(
                        self.jm.start_stage(job.job_id, "sparse_reconstruction", message=msg, progress=65),
                        main_loop,
                    )
                    future.result(timeout=5)
            except Exception as e:
                logger.warning(f"COLMAP progress callback notification error: {e}")

        notify("feature_extraction", "start", message="Running COLMAP feature extraction…", progress=31)
        notify("feature_matching", "start", message="Waiting for feature extraction…", progress=31)
        notify("sparse_reconstruction", "start", message="Waiting for matching…", progress=31)

        from reconstruction.base_reconstructor import ReconstructionInput
        recon_input = ReconstructionInput(
            image_dir=image_dir_for_recon,
            output_dir=sparse_out,
            progress_cb=colmap_progress_cb,
        )

        colmap = ColmapReconstructor(recon_cfg, colmap_bin=self.colmap_bin)
        t0 = time.perf_counter()
        colmap_out = colmap.reconstruct(recon_input)
        elapsed = time.perf_counter() - t0

        if not colmap_out.success:
            notify("sparse_reconstruction", "fail", message=colmap_out.error_message)
            raise RuntimeError(colmap_out.error_message)

        notify("feature_extraction", "done",
               message="Feature extraction complete", progress=50)
        notify("feature_matching", "done",
               message="Feature matching complete", progress=65)
        notify("sparse_reconstruction", "done",
               message=f"Registered {colmap_out.registered_images}/{colmap_out.total_images} images, "
                       f"{colmap_out.sparse_point_count:,} 3D points",
               data={
                   "registered": colmap_out.registered_images,
                   "total": frame_stats.selected_for_reconstruction,
                   "points": colmap_out.sparse_point_count,
                   "error": round(colmap_out.mean_reprojection_error, 3),
               },
               progress=80)

        # Copy outputs to project output dir
        if colmap_out.sparse_ply_path:
            shutil.copy2(colmap_out.sparse_ply_path, output_dir / "sparse.ply")
        if colmap_out.camera_poses_json:
            shutil.copy2(colmap_out.camera_poses_json, output_dir / "camera_poses.json")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE D: Quality analysis
        # ═══════════════════════════════════════════════════════════════════
        notify("quality_analysis", "start", message="Analysing reconstruction quality…", progress=81)
        qe = ReconstructionQualityEngine(self.cfg)
        quality_report = qe.analyse(
            model_dir=colmap_out.model_dir,
            total_input_images=frame_stats.selected_for_reconstruction,
            registered_images=colmap_out.registered_images,
        )
        quality_report.colmap_stats.processing_time_seconds = elapsed

        notify("quality_analysis", "done",
               message=f"Quality score: {quality_report.quality_score.overall:.0f}/100 ({quality_report.quality_score.label})",
               data={"score": quality_report.quality_score.overall},
               progress=90)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE E: Confidence map
        # ═══════════════════════════════════════════════════════════════════
        notify("confidence_map", "start", message="Generating confidence map…", progress=91)
        cmg = ConfidenceMapGenerator(self.cfg)
        conf_ply, low_regions, conf_summary = cmg.generate(
            model_dir=colmap_out.model_dir,
            output_dir=output_dir,
        )
        notify("confidence_map", "done",
               message=f"{conf_summary.get('low_confidence_region_count', 0)} low-confidence region(s) detected",
               data=conf_summary, progress=95)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE F: Failure classification + recommendations
        # ═══════════════════════════════════════════════════════════════════
        notify("recommendations", "start", message="Classifying failures and generating recommendations…", progress=96)
        classifier = FailureClassifier(self.cfg)
        diagnosed = classifier.classify(low_regions)

        recommender = RecaptureRecommender()
        recs = recommender.generate(diagnosed)
        notify("recommendations", "done",
               message=f"{len(recs)} recapture recommendation(s) generated",
               progress=100)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE G: AI 3D Infilling, Texture Mapping & Gaussian Splatting
        # ═══════════════════════════════════════════════════════════════════
        dense_dir = job_dir / "processing" / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)
        sparse_txt_dir = sparse_out / "sparse_txt" if (sparse_out / "sparse_txt").exists() else colmap_out.model_dir

        try:
            # 1. Surface Meshing & Peak Spatial Density Centering
            mesher = SurfaceMesher()
            mesher.process(
                input_ply=output_dir / "sparse.ply",
                output_mesh_ply=dense_dir / "mesh.ply",
                output_mesh_obj=dense_dir / "mesh.obj",
                output_dense_ply=dense_dir / "dense.ply",
                output_primary_ply=dense_dir / "primary_cloud.ply",
                output_centered_sparse_ply=output_dir / "sparse.ply",
            )
        except Exception as e:
            logger.warning("Meshing stage warning: %s", e)

        try:
            # 2. ML Point Cloud Completion & Void Infilling
            completer = MLPointCompleter(target_points=80000)
            completer.complete_point_cloud(
                input_ply=dense_dir / "primary_cloud.ply" if (dense_dir / "primary_cloud.ply").exists() else output_dir / "sparse.ply",
                output_completed_ply=dense_dir / "ml_completed_dense.ply",
                output_completed_mesh=dense_dir / "ml_completed_mesh.ply",
            )
        except Exception as e:
            logger.warning("ML Point completion warning: %s", e)

        try:
            # 3. Multi-View Real-Photo Texture Mapping -> GLB
            target_mesh = dense_dir / "ml_completed_mesh.ply" if (dense_dir / "ml_completed_mesh.ply").exists() else dense_dir / "mesh.ply"
            if target_mesh.exists():
                tex_mapper = TextureMapper(atlas_size=2048)
                tex_mapper.generate_textured_mesh(
                    mesh_ply_path=target_mesh,
                    keyframes_dir=kf_dir,
                    sparse_txt_dir=sparse_txt_dir,
                    output_glb_path=dense_dir / "textured_model.glb",
                    output_obj_path=dense_dir / "textured_model.obj",
                    output_texture_png=dense_dir / "texture_atlas.png",
                )
        except Exception as e:
            logger.warning("Texture mapping warning: %s", e)

        try:
            # 4. AI Monocular Metric Depth Unprojection
            depth_est = AIDepthEstimator(target_cloud_points=120000, stride=8)
            depth_est.estimate_dense_depth_cloud(
                keyframes_dir=kf_dir,
                sparse_txt_dir=sparse_txt_dir,
                output_dense_ply=dense_dir / "ai_depth_dense.ply",
            )
        except Exception as e:
            logger.warning("AI Depth unprojection warning: %s", e)

        try:
            # 5. 3D Gaussian Splatting Radiance Field Export
            splat_source = dense_dir / "ml_completed_dense.ply" if (dense_dir / "ml_completed_dense.ply").exists() else output_dir / "sparse.ply"
            splatter = GaussianSplatter()
            splatter.generate_splats_from_point_cloud(
                input_ply=splat_source,
                output_splat_path=dense_dir / "point_cloud.splat",
                output_ply_path=dense_dir / "point_cloud_3dgs.ply",
            )
        except Exception as e:
            logger.warning("Gaussian splatting warning: %s", e)

        # Copy all processed 3D assets to output directory
        for f in dense_dir.glob("*"):
            if f.is_file():
                shutil.copy2(f, output_dir / f.name)

        # ── Build final result ────────────────────────────────────────────
        artifacts: dict[str, str] = {}
        for fname in [
            "sparse.ply", "dense.ply", "mesh.ply", "mesh.obj", "confidence.ply",
            "camera_poses.json", "ml_completed_dense.ply", "ml_completed_mesh.ply",
            "textured_model.glb", "textured_model.obj", "texture_atlas.png",
            "ai_depth_dense.ply", "point_cloud.splat", "point_cloud_3dgs.ply",
            "primary_cloud.ply",
        ]:
            if (output_dir / fname).exists():
                key = fname.replace(".", "_")
                artifacts[key] = f"/api/files/{job.job_id}/output/{fname}"

        failure_region_models = [
            FailureRegion(
                region_id=i,
                center=dr.region.center.tolist(),
                radius=dr.region.radius,
                failure_type=dr.failure_type,
                confidence_score=round(dr.confidence, 3),
                point_count=dr.region.point_count,
                camera_count=dr.region.camera_count,
                mean_reprojection_error=round(dr.region.mean_reprojection_error, 3),
                recommendation=recs[i].description if i < len(recs) else "",
            )
            for i, dr in enumerate(diagnosed)
        ]

        result = ReconstructionResult(
            job_id=job.job_id,
            method=job.method.value,
            artifacts=artifacts,
            frame_stats=frame_stats,
            colmap_stats=quality_report.colmap_stats,
            quality_score=quality_report.quality_score,
            confidence=ConfidenceBreakdown(
                high_ratio=conf_summary.get("high_ratio", 0),
                medium_ratio=conf_summary.get("medium_ratio", 0),
                low_ratio=conf_summary.get("low_ratio", 0),
                low_confidence_region_count=len(low_regions),
            ),
            failure_regions=failure_region_models,
            recommendations=recs,
        )

        # Persist result JSON
        import json
        (output_dir / "result.json").write_text(result.model_dump_json(indent=2))
        return result

    @staticmethod
    def _load_yaml(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            logger.warning("Config file not found: %s", path)
            return {}
        with open(p) as f:
            return yaml.safe_load(f) or {}
