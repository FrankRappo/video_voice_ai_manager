"""
OpenAI Vision backend (GPT-4o).
"""
import asyncio
import base64
from pathlib import Path

from .base import BaseVision, FrameAnalysis

DEFAULT_PROMPT = """Describe what you see in this video frame in detail.
Include: visible text, UI elements, people, actions, objects.
Be concise but thorough."""


class OpenAIVision(BaseVision):

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

    def name(self) -> str:
        return f"OpenAI Vision ({self.model})"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def analyze_frame(self, image_path: Path, prompt: str = "") -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        client = OpenAI(api_key=self.api_key)
        prompt = prompt or DEFAULT_PROMPT

        img_b64 = base64.b64encode(image_path.read_bytes()).decode()
        suffix = image_path.suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")

        def _run():
            return client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    ],
                }],
                max_tokens=1000,
            )

        response = await asyncio.to_thread(_run)
        return response.choices[0].message.content

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
