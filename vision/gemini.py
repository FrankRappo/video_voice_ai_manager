"""
Gemini Vision backend.
"""
import asyncio
from pathlib import Path

from .base import BaseVision, FrameAnalysis

DEFAULT_PROMPT = """Describe what you see in this video frame in detail.
Include: visible text, UI elements, people, actions, objects.
Be concise but thorough. Answer in the same language as any visible text, or English if no text."""


class GeminiVision(BaseVision):

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    def name(self) -> str:
        return f"Gemini Vision ({self.model})"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def analyze_frame(self, image_path: Path, prompt: str = "") -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")

        client = genai.Client(api_key=self.api_key)
        prompt = prompt or DEFAULT_PROMPT

        img_bytes = image_path.read_bytes()
        suffix = image_path.suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
            suffix.lstrip("."), "image/png"
        )

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self.model,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type=mime),
                prompt,
            ],
        )
        return response.text

    async def analyze_frames(
        self, frames: list[tuple[float, Path]], prompt: str = ""
    ) -> list[FrameAnalysis]:
        results = []
        for timestamp, frame_path in frames:
            desc = await self.analyze_frame(frame_path, prompt)
            results.append(FrameAnalysis(
                timestamp=timestamp,
                description=desc,
                frame_path=str(frame_path),
            ))
        return results
