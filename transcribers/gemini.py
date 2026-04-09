"""
Gemini Native Audio transcriber.
Uses Gemini's native audio understanding — no separate STT needed.
"""
import json
import asyncio
from pathlib import Path

from .base import BaseTranscriber, TranscriptionResult, Segment


TRANSCRIBE_PROMPT = """Transcribe this audio with precise timestamps.

Return ONLY valid JSON in this exact format:
{
  "language": "detected language code",
  "segments": [
    {"start": 0.0, "end": 5.2, "text": "transcribed text here"},
    {"start": 5.2, "end": 10.1, "text": "more text here"}
  ]
}

Rules:
- Timestamps in seconds (float)
- Keep original language, do not translate
- Split by natural pauses/sentences
- Be precise with timestamps"""


class GeminiTranscriber(BaseTranscriber):

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-native-audio-latest"):
        self.api_key = api_key
        self.model = model

    def name(self) -> str:
        return f"Gemini ({self.model})"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def transcribe(self, audio_path: Path, language: str = "") -> TranscriptionResult:
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")

        client = genai.Client(api_key=self.api_key)

        # Upload audio file
        upload = await asyncio.to_thread(
            client.files.upload, file=str(audio_path)
        )

        prompt = TRANSCRIBE_PROMPT
        if language:
            prompt += f"\nExpected language: {language}"

        # Generate transcription
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self.model,
            contents=[upload, prompt],
        )

        # Parse response
        text = response.text.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: treat entire response as plain text
            return TranscriptionResult(
                segments=[Segment(start=0.0, end=0.0, text=response.text)],
                language=language,
            )

        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"])
            for s in data.get("segments", [])
        ]

        return TranscriptionResult(
            segments=segments,
            language=data.get("language", language),
        )
