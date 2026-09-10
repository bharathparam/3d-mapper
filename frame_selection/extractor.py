"""
Video frame extractor using FFmpeg subprocess.

Extracts frames at a configurable rate and returns metadata for each frame.
Does NOT load frames into memory — returns paths only.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    frame_index: int            # 0-based extraction order
    path: Path                  # absolute path to the saved frame image
    timestamp_sec: float        # position in the original video (seconds)
    filename: str = field(init=False)

    def __post_init__(self) -> None:
        self.filename = self.path.name


@dataclass
class ExtractionResult:
    frames: list[FrameInfo]
    video_duration_sec: float
    video_fps: float
    total_video_frames: int
    extraction_fps: float
    output_dir: Path
    elapsed_sec: float


class FrameExtractor:
    """Extracts frames from a video file using FFmpeg."""

    def __init__(self, config: dict) -> None:
        self.fps: float = config.get("extraction", {}).get("fps", 2.0)
        self.max_frames: int = config.get("extraction", {}).get("max_frames", 2000)
        self.min_frames: int = config.get("extraction", {}).get("min_frames", 10)
        self.quality: int = config.get("extraction", {}).get("quality", 95)
        self.resize_width: int | None = config.get("extraction", {}).get("resize_width")

        # Resolve tool paths
        tools = config.get("tools", {})
        self.ffmpeg_bin: str = tools.get("ffmpeg") or shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe_bin: str = tools.get("ffprobe") or shutil.which("ffprobe") or "ffprobe"

    # ── Public API ───────────────────────────────────────────────────────────

    def extract(self, video_path: Path, output_dir: Path) -> ExtractionResult:
        """
        Extract frames from *video_path* into *output_dir*.
        Returns an ExtractionResult with metadata for each frame.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self._verify_ffmpeg()

        duration, video_fps, n_total = self._probe_video(video_path)
        logger.info(
            "Video: %.1f s, %.2f fps, ~%d total frames",
            duration, video_fps, n_total,
        )

        # Build FFmpeg command
        vf_parts: list[str] = [f"fps={self.fps}"]
        if self.resize_width:
            vf_parts.append(f"scale={self.resize_width}:-2")
        vf = ",".join(vf_parts)

        output_pattern = str(output_dir / "frame_%06d.jpg")

        cmd = [
            self.ffmpeg_bin,
            "-i", str(video_path),
            "-vf", vf,
            "-q:v", str(max(1, min(31, 31 - int(self.quality * 0.3)))),
            "-frames:v", str(self.max_frames),
            output_pattern,
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
        ]

        logger.info("Running FFmpeg: %s", " ".join(cmd))
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (exit {result.returncode}):\n{result.stderr}"
            )

        # Collect extracted frames
        frame_paths = sorted(output_dir.glob("frame_*.jpg"))

        if len(frame_paths) < self.min_frames:
            raise RuntimeError(
                f"Only {len(frame_paths)} frames extracted (minimum {self.min_frames}).\n"
                "Possible causes: video too short, corrupted file, or FFmpeg error.\n"
                f"FFmpeg stderr: {result.stderr[-500:]}"
            )

        frames = [
            FrameInfo(
                frame_index=i,
                path=p,
                timestamp_sec=i / self.fps,
            )
            for i, p in enumerate(frame_paths)
        ]

        logger.info(
            "Extracted %d frames in %.1f s (%.1f fps extraction rate)",
            len(frames), elapsed, self.fps,
        )

        return ExtractionResult(
            frames=frames,
            video_duration_sec=duration,
            video_fps=video_fps,
            total_video_frames=n_total,
            extraction_fps=self.fps,
            output_dir=output_dir,
            elapsed_sec=elapsed,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _verify_ffmpeg(self) -> None:
        try:
            subprocess.run(
                [self.ffmpeg_bin, "-version"],
                capture_output=True, check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"FFmpeg not found at '{self.ffmpeg_bin}'.\n"
                "Install with: brew install ffmpeg"
            ) from exc

    def _probe_video(self, video_path: Path) -> tuple[float, float, int]:
        """Return (duration_sec, fps, approx_total_frames)."""
        cmd = [
            self.ffprobe_bin,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("ffprobe failed; estimating video properties.")
            return 0.0, 30.0, 0

        import json
        data = json.loads(result.stdout)
        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if not video_stream:
            return 0.0, 30.0, 0

        # Parse FPS (can be "30000/1001" format)
        fps_str = video_stream.get("avg_frame_rate", "30/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            fps = 30.0

        duration = float(video_stream.get("duration", 0))
        n_frames = int(video_stream.get("nb_frames", 0)) or int(duration * fps)

        return duration, fps, n_frames
