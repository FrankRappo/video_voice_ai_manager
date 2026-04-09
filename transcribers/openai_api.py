"""
OpenAI Whisper API transcriber.
"""
import asyncio
from pathlib import Path

from .base import BaseTranscriber, TranscriptionResult, Segment


class OpenAITranscriber(BaseTranscriber):

    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.api_key = api_key
        self.model_name = model

    def name(self) -> str:
        return f"OpenAI Whisper ({self.model_name})"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def transcribe(self, audio_path: Path, language: str = "") -> TranscriptionResult:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        client = OpenAI(api_key=self.api_key)

        def _run():
            kwargs = {
                "model": self.model_name,
                "file": open(audio_path, "rb"),
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
            }
            if language:
                kwargs["language"] = language

            return client.audio.transcriptions.create(**kwargs)

        response = await asyncio.to_thread(_run)

        segments = []
        for s in getattr(response, "segments", []):
            segments.append(Segment(
                start=s.get("start", 0.0) if isinstance(s, dict) else s.start,
                end=s.get("end", 0.0) if isinstance(s, dict) else s.end,
                text=(s.get("text", "") if isinstance(s, dict) else s.text).strip(),
            ))

        if not segments and hasattr(response, "text"):
            segments = [Segment(start=0.0, end=0.0, text=response.text)]

        return TranscriptionResult(
            segments=segments,
            language=getattr(response, "language", language),
        )
