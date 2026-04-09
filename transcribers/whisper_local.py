"""
Local Whisper transcriber (offline, no API needed).
Supports both openai-whisper and faster-whisper.
"""
import asyncio
from pathlib import Path

from .base import BaseTranscriber, TranscriptionResult, Segment


class WhisperLocalTranscriber(BaseTranscriber):

    def __init__(self, model: str = "medium", language: str = ""):
        self.model_name = model
        self.language = language
        self._model = None
        self._use_faster = None

    def name(self) -> str:
        variant = "faster-whisper" if self._use_faster else "whisper"
        return f"{variant} ({self.model_name})"

    def is_available(self) -> bool:
        try:
            import faster_whisper
            self._use_faster = True
            return True
        except ImportError:
            pass
        try:
            import whisper
            self._use_faster = False
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is not None:
            return

        if self._use_faster is None:
            self.is_available()

        if self._use_faster:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, compute_type="int8")
        else:
            import whisper
            self._model = whisper.load_model(self.model_name)

    async def transcribe(self, audio_path: Path, language: str = "") -> TranscriptionResult:
        lang = language or self.language

        def _run():
            self._load_model()

            if self._use_faster:
                return self._transcribe_faster(audio_path, lang)
            else:
                return self._transcribe_whisper(audio_path, lang)

        return await asyncio.to_thread(_run)

    def _transcribe_faster(self, audio_path: Path, language: str) -> TranscriptionResult:
        kwargs = {}
        if language:
            kwargs["language"] = language

        segs, info = self._model.transcribe(str(audio_path), **kwargs)

        segments = []
        for s in segs:
            segments.append(Segment(start=s.start, end=s.end, text=s.text.strip()))

        return TranscriptionResult(
            segments=segments,
            language=info.language if hasattr(info, "language") else language,
        )

    def _transcribe_whisper(self, audio_path: Path, language: str) -> TranscriptionResult:
        kwargs = {}
        if language:
            kwargs["language"] = language

        result = self._model.transcribe(str(audio_path), **kwargs)

        segments = []
        for s in result.get("segments", []):
            segments.append(Segment(
                start=s["start"],
                end=s["end"],
                text=s["text"].strip(),
            ))

        return TranscriptionResult(
            segments=segments,
            language=result.get("language", language),
        )
