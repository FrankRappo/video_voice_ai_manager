"""
Ollama Vision backend (local, offline).
"""
import asyncio
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

from .base import BaseVision, FrameAnalysis

DEFAULT_PROMPT = """Describe what you see in this video frame in detail.
Include: visible text, UI elements, people, actions, objects.
Be concise but thorough."""


class OllamaVision(BaseVision):

    def __init__(self, url: str = "http://localhost:11434", model: str = "llava"):
        self.url = url.rstrip("/")
        self.model = model

    def name(self) -> str:
        return f"Ollama ({self.model})"

    def is_available(self) -> bool:
        try:
            req = Request(f"{self.url}/api/tags", method="GET")
            urlopen(req, timeout=3)
            return True
        except Exception:
            return False

    async def analyze_frame(self, image_path: Path, prompt: str = "") -> str:
        prompt = prompt or DEFAULT_PROMPT
        img_b64 = base64.b64encode(image_path.read_bytes()).decode()

        def _run():
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
            }).encode()

            req = Request(
                f"{self.url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urlopen(req, timeout=120)
            data = json.loads(resp.read())
            return data.get("response", "")

        return await asyncio.to_thread(_run)

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
