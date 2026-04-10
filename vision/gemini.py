"""
Gemini Vision backend.
"""
import asyncio
import json
import logging
from pathlib import Path

from .base import BaseVision, FrameAnalysis

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """Describe what you see in this video frame in detail.
Include: visible text, UI elements, people, actions, objects.
Be concise but thorough. Answer in the same language as any visible text, or English if no text."""

CLIENT_FEEDBACK_PROMPT = """Analyze this video frame as a UI screenshot. Return a JSON object with these fields:

{
  "screen_description": "Brief description of what's shown on screen",
  "ui_elements": [
    {"type": "button|dropdown|input|table|modal|form|label|tab|checkbox|other", "label": "visible label text", "value": "current value if any", "state": "open|closed|selected|disabled|active|filled|empty"}
  ],
  "numeric_values": [
    {"value": "the number or percentage as string", "context": "what this number refers to", "unit": "currency|percent|count|other"}
  ],
  "visible_text": ["all readable text strings on screen"],
  "screen_state": "what the user is doing: browsing|filling_form|viewing_results|modal_open|error|loading|other"
}

Rules:
- Extract ALL numeric values visible on screen (prices, percentages, counts, budgets)
- For dropdowns: note if open/closed, list visible options and current selection
- For tables: note column headers and row data with numbers
- For modals/dialogs: note title, content, and buttons
- Be precise with numbers — write exact values as shown
- Answer in the language of visible text"""


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

        # Retry with exponential backoff for transient errors
        for attempt in range(5):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type=mime),
                        prompt,
                    ],
                )
                return response.text
            except Exception as e:
                if attempt < 4 and ("503" in str(e) or "429" in str(e) or "UNAVAILABLE" in str(e) or "overloaded" in str(e).lower()):
                    wait = 2 ** attempt
                    logger.warning("Retrying frame %s (attempt %d, wait %ds): %s", image_path.name, attempt + 1, wait, e)
                    await asyncio.sleep(wait)
                else:
                    raise

    async def analyze_frames(
        self, frames: list[tuple[float, Path]], prompt: str = ""
    ) -> list[FrameAnalysis]:
        results = []
        structured = prompt == CLIENT_FEEDBACK_PROMPT
        for timestamp, frame_path in frames:
            desc = await self.analyze_frame(frame_path, prompt)
            structured_data = None
            if structured:
                structured_data = self._try_parse_json(desc)
            results.append(FrameAnalysis(
                timestamp=timestamp,
                description=desc,
                frame_path=str(frame_path),
                structured_data=structured_data,
            ))
        return results

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        """Try to parse JSON from model response, stripping markdown fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n", 1)
            text = lines[1] if len(lines) > 1 else ""
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
