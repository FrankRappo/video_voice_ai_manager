"""
Base vision interface.
All vision backends must implement this.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrameAnalysis:
    """Analysis of a single video frame."""
    timestamp: float   # seconds
    description: str
    frame_path: str
    structured_data: dict | None = None  # JSON-parsed structured analysis (for client-feedback mode)

    @property
    def timestamp_str(self) -> str:
        h = int(self.timestamp // 3600)
        m = int((self.timestamp % 3600) // 60)
        s = int(self.timestamp % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"


class BaseVision(ABC):
    """Interface for all vision backends."""

    @abstractmethod
    async def analyze_frame(self, image_path: Path, prompt: str = "") -> str:
        """Analyze a single image/frame, return description."""
        ...

    @abstractmethod
    async def analyze_frames(self, frames: list[tuple[float, Path]], prompt: str = "") -> list[FrameAnalysis]:
        """Analyze multiple frames with timestamps."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...
