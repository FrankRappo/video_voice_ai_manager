"""
Screenshot extraction — extract frames from video at specific timecodes.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


class ScreenshotExtractor:
    """Extract screenshots (frames) from video files."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()

    async def extract_at(
        self,
        video_path: str,
        timestamp: float,
        output_dir: str = ".",
    ) -> Path:
        """Extract a single frame at a given timecode.

        Args:
            video_path: Path to video file.
            timestamp: Time in seconds.
            output_dir: Directory to save the screenshot.

        Returns:
            Path to the extracted screenshot.
        """
        video = Path(video_path)
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = self._timecode_filename(timestamp)
        output_path = out_dir / filename

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", str(video),
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg screenshot failed: {stderr.decode()}")

        return output_path

    async def extract_range(
        self,
        video_path: str,
        start: float,
        end: float,
        fps: Optional[float] = None,
        output_dir: str = ".",
    ) -> list[Path]:
        """Extract frames from a time range at given fps.

        Args:
            video_path: Path to video file.
            start: Start time in seconds.
            end: End time in seconds.
            fps: Frames per second (default from config).
            output_dir: Directory to save screenshots.

        Returns:
            List of paths to extracted screenshots.
        """
        video = Path(video_path)
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        use_fps = fps if fps is not None else self.config.video_fps
        duration = end - start

        # Extract frames via ffmpeg
        # Use a temp pattern, then rename with timecodes
        temp_pattern = out_dir / "vvam_tmp_%06d.png"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(video),
            "-vf", f"fps={use_fps}",
            "-q:v", "2",
            str(temp_pattern),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg frame extraction failed: {stderr.decode()}")

        # Rename temp files to timecoded names
        temp_files = sorted(out_dir.glob("vvam_tmp_*.png"))
        result_paths: list[Path] = []

        for i, tmp_file in enumerate(temp_files):
            timestamp = start + i / use_fps
            if timestamp > end:
                tmp_file.unlink(missing_ok=True)
                continue
            new_name = self._timecode_filename(timestamp)
            new_path = out_dir / new_name
            tmp_file.rename(new_path)
            result_paths.append(new_path)

        return result_paths

    @staticmethod
    def _timecode_filename(seconds: float) -> str:
        """Generate filename from timecode: 00m_05s.png, 01m_23s.png."""
        total = int(seconds)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h:02d}h_{m:02d}m_{s:02d}s.png"
        return f"{m:02d}m_{s:02d}s.png"
