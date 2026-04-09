"""
SRT subtitle formatter.
Generates standard SubRip (.srt) subtitle files.
"""
from __future__ import annotations

from transcribers.base import TranscriptionResult


def _srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_srt(transcript: TranscriptionResult | None) -> str:
    """
    Generate SRT subtitle content from a transcription.

    Each segment becomes one subtitle entry with sequential numbering,
    SRT-format timestamps, and the segment text.
    """
    if not transcript:
        return ""
    blocks: list[str] = []
    for i, seg in enumerate(transcript.segments, 1):
        start = _srt_timestamp(seg.start)
        end = _srt_timestamp(seg.end)
        blocks.append(f"{i}\n{start} --> {end}\n{seg.text}")
    return "\n\n".join(blocks) + "\n" if blocks else ""
