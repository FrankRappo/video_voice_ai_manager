"""
JSON output formatter.
Produces a structured JSON document suitable for API consumption.
"""
from __future__ import annotations

import json
from typing import Any

from transcribers.base import TranscriptionResult
from vision.base import FrameAnalysis


def format_json(
    transcript: TranscriptionResult | None,
    frame_analyses: list[FrameAnalysis] | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Generate a JSON report.

    Returns a pretty-printed JSON string with metadata, segments,
    frames, and full_text fields.
    """
    metadata = metadata or {}
    result: dict[str, Any] = {"metadata": metadata}

    if transcript:
        result["segments"] = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in transcript.segments
        ]
        result["language"] = transcript.language
        result["full_text"] = transcript.full_text
    else:
        result["segments"] = []
        result["language"] = ""
        result["full_text"] = ""

    if frame_analyses:
        result["frames"] = [
            {
                "timestamp": fa.timestamp,
                "description": fa.description,
                "frame_path": fa.frame_path,
            }
            for fa in frame_analyses
        ]
    else:
        result["frames"] = []

    return json.dumps(result, ensure_ascii=False, indent=2)
