"""
Video analysis module — download, extract frames/audio, transcribe, analyze.
"""
import asyncio
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from config import Config
from transcribers.base import BaseTranscriber, TranscriptionResult, Segment
from vision.base import BaseVision, FrameAnalysis

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}


@dataclass
class VideoResult:
    """Combined result of video analysis."""
    transcription: Optional[TranscriptionResult]
    frames: list[FrameAnalysis]
    source: str
    duration: float = 0.0
    metadata: dict = field(default_factory=dict)


class VideoAnalyzer:
    """Analyze video files or URLs: extract frames, transcribe audio, analyze visuals."""

    def __init__(
        self,
        transcriber: BaseTranscriber,
        vision: Optional[BaseVision] = None,
        config: Optional[Config] = None,
    ):
        self.transcriber = transcriber
        self.vision = vision
        self.config = config or Config.load()
        self._temp_dirs: list[Path] = []

    async def analyze(
        self,
        source: str,
        audio_only: bool = False,
        time_from: Optional[float] = None,
        time_to: Optional[float] = None,
        fps: Optional[float] = None,
        scene_detect: Optional[bool] = None,
    ) -> VideoResult:
        """Analyze a video file or URL.

        Args:
            source: Path to video file or URL (YouTube etc.)
            audio_only: Skip frame extraction/analysis.
            time_from: Start time in seconds.
            time_to: End time in seconds.
            fps: Frames per second for extraction (overrides config).
            scene_detect: Use scene detection (overrides config).
        """
        work_dir = Path(tempfile.mkdtemp(prefix="vvam_video_"))
        self._temp_dirs.append(work_dir)

        try:
            # Resolve source to local file
            video_path = await self._resolve_source(source, work_dir)
            duration = await self._get_duration(video_path)

            # Apply time range
            effective_from = time_from or 0.0
            effective_to = time_to or duration
            effective_duration = effective_to - effective_from

            metadata = {
                "source": source,
                "duration": duration,
                "time_from": effective_from,
                "time_to": effective_to,
            }

            # Determine if we need chunking
            chunk_minutes = self.config.video_chunk_minutes
            max_direct = self.config.video_max_direct_minutes

            if effective_duration > max_direct * 60:
                return await self._analyze_chunked(
                    video_path, work_dir, effective_from, effective_to,
                    chunk_minutes, audio_only, fps, scene_detect, metadata,
                )

            # Single-pass analysis
            transcription = await self._transcribe_segment(
                video_path, work_dir, effective_from, effective_to,
            )

            frames: list[FrameAnalysis] = []
            if not audio_only and self.vision:
                frames = await self._analyze_frames_segment(
                    video_path, work_dir, effective_from, effective_to,
                    fps, scene_detect,
                )

            return VideoResult(
                transcription=transcription,
                frames=frames,
                source=source,
                duration=duration,
                metadata=metadata,
            )
        finally:
            self._cleanup(work_dir)

    # --- Source resolution ---

    async def _resolve_source(self, source: str, work_dir: Path) -> Path:
        """Download URL or validate local file path."""
        path = Path(source)
        if path.exists() and path.suffix.lower() in SUPPORTED_VIDEO_EXTS:
            return path

        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            return await self._download_url(source, work_dir)

        raise FileNotFoundError(f"Video not found: {source}")

    async def _download_url(self, url: str, work_dir: Path) -> Path:
        """Download video via yt-dlp."""
        output_template = str(work_dir / "video.%(ext)s")
        cmd = [
            "yt-dlp",
            "-f", self.config.ytdlp_format,
            "-o", output_template,
            "--no-playlist",
            url,
        ]
        logger.info("Downloading: %s", url)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {stderr.decode()}")

        # Find downloaded file
        for f in work_dir.iterdir():
            if f.stem == "video" and f.suffix.lower() in SUPPORTED_VIDEO_EXTS:
                return f
        # Fallback: any video file
        for f in work_dir.iterdir():
            if f.suffix.lower() in SUPPORTED_VIDEO_EXTS:
                return f
        raise RuntimeError("yt-dlp produced no video file")

    # --- Duration ---

    async def _get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds via ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        try:
            return float(stdout.decode().strip())
        except ValueError:
            return 0.0

    # --- Audio extraction & transcription ---

    async def _extract_audio(
        self, video_path: Path, work_dir: Path,
        start: float, end: float,
    ) -> Path:
        """Extract audio segment as WAV 16kHz mono."""
        audio_path = work_dir / "audio.wav"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(audio_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed: {stderr.decode()}")
        return audio_path

    async def _transcribe_segment(
        self, video_path: Path, work_dir: Path,
        start: float, end: float,
    ) -> TranscriptionResult:
        """Extract audio and transcribe a segment."""
        audio_path = await self._extract_audio(video_path, work_dir, start, end)
        result = await self.transcriber.transcribe(audio_path)
        # Offset timestamps by start time
        if start > 0:
            for seg in result.segments:
                seg.start += start
                seg.end += start
        return result

    # --- Frame extraction & analysis ---

    async def _extract_frames(
        self, video_path: Path, work_dir: Path,
        start: float, end: float,
        fps: Optional[float] = None,
        scene_detect: Optional[bool] = None,
    ) -> list[tuple[float, Path]]:
        """Extract frames from video segment. Returns list of (timestamp, path)."""
        frames_dir = work_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        use_fps = fps if fps is not None else self.config.video_fps
        use_scene = scene_detect if scene_detect is not None else self.config.video_scene_detect

        if use_scene:
            vf = f"select='gt(scene,{self.config.video_scene_threshold})',showinfo"
        else:
            vf = f"fps={use_fps},showinfo"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(video_path),
            "-vf", vf,
            "-vsync", "vfr",
            str(frames_dir / "frame_%04d.png"),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        # Parse timestamps from showinfo output
        frames: list[tuple[float, Path]] = []
        stderr_text = stderr.decode(errors="replace")
        frame_files = sorted(frames_dir.glob("frame_*.png"))

        pts_times: list[float] = []
        for line in stderr_text.split("\n"):
            if "pts_time:" in line:
                for part in line.split():
                    if part.startswith("pts_time:"):
                        try:
                            pts_times.append(float(part.split(":")[1]))
                        except (ValueError, IndexError):
                            pass

        for i, frame_path in enumerate(frame_files):
            if i < len(pts_times):
                ts = pts_times[i] + start
            else:
                # Fallback: estimate from fps
                ts = start + i / use_fps
            frames.append((ts, frame_path))

        return frames

    async def _analyze_frames_segment(
        self, video_path: Path, work_dir: Path,
        start: float, end: float,
        fps: Optional[float] = None,
        scene_detect: Optional[bool] = None,
    ) -> list[FrameAnalysis]:
        """Extract and analyze frames for a segment."""
        if not self.vision:
            return []
        frames = await self._extract_frames(
            video_path, work_dir, start, end, fps, scene_detect,
        )
        if not frames:
            return []
        return await self.vision.analyze_frames(frames)

    # --- Chunked processing ---

    async def _analyze_chunked(
        self, video_path: Path, work_dir: Path,
        start: float, end: float,
        chunk_minutes: int,
        audio_only: bool,
        fps: Optional[float],
        scene_detect: Optional[bool],
        metadata: dict,
    ) -> VideoResult:
        """Process long video in chunks."""
        chunk_duration = chunk_minutes * 60
        all_segments: list[Segment] = []
        all_frames: list[FrameAnalysis] = []
        language = ""

        t = start
        chunk_idx = 0
        while t < end:
            chunk_end = min(t + chunk_duration, end)
            chunk_dir = work_dir / f"chunk_{chunk_idx}"
            chunk_dir.mkdir(exist_ok=True)

            logger.info("Processing chunk %d: %.1f - %.1f", chunk_idx, t, chunk_end)

            transcription = await self._transcribe_segment(
                video_path, chunk_dir, t, chunk_end,
            )
            all_segments.extend(transcription.segments)
            if transcription.language and not language:
                language = transcription.language

            if not audio_only and self.vision:
                frames = await self._analyze_frames_segment(
                    video_path, chunk_dir, t, chunk_end, fps, scene_detect,
                )
                all_frames.extend(frames)

            t = chunk_end
            chunk_idx += 1

        combined_transcription = TranscriptionResult(
            segments=all_segments,
            language=language,
        )

        return VideoResult(
            transcription=combined_transcription,
            frames=all_frames,
            source=metadata.get("source", ""),
            duration=metadata.get("duration", 0.0),
            metadata=metadata,
        )

    # --- Cleanup ---

    def _cleanup(self, work_dir: Path):
        """Remove temporary directory."""
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

    async def close(self):
        """Cleanup all temporary directories."""
        for d in self._temp_dirs:
            self._cleanup(d)
        self._temp_dirs.clear()
