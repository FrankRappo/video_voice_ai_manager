"""
Base transcriber interface.
All transcriber backends must implement this.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    """A timestamped piece of transcription."""
    start: float   # seconds
    end: float     # seconds
    text: str

    @property
    def start_ts(self) -> str:
        return _format_time(self.start)

    @property
    def end_ts(self) -> str:
        return _format_time(self.end)


@dataclass
class TranscriptionResult:
    """Full transcription result."""
    segments: list[Segment]
    language: str = ""
    full_text: str = ""

    def __post_init__(self):
        if not self.full_text and self.segments:
            self.full_text = " ".join(s.text for s in self.segments)


class BaseTranscriber(ABC):
    """Interface for all transcription backends."""

    @abstractmethod
    async def transcribe(self, audio_path: Path, language: str = "") -> TranscriptionResult:
        """Transcribe audio file, return timestamped segments."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Backend name for display."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is ready to use."""
        ...


def _format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS or MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
