"""
Correlator — cross-references transcription with frame analysis
to produce structured client feedback items.

Uses Gemini 2.5 Flash for LLM-based correlation and classification.
Supports two modes:
1. Pre-analyzed: takes FrameAnalysis objects with text descriptions
2. Direct images: takes raw frame images + transcription in a single API call
   (much more efficient for rate-limited APIs)
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transcribers.base import TranscriptionResult, Segment
from vision.base import FrameAnalysis

logger = logging.getLogger(__name__)


@dataclass
class FeedbackItem:
    """A single piece of structured client feedback."""
    id: str                          # e.g. "BUG-1", "WISH-3"
    category: str                    # BUG | WISH | POSITIVE | QUESTION
    title: str                       # short summary
    description: str                 # detailed description
    priority: str                    # P0 | P1 | P2
    quotes: list[dict] = field(default_factory=list)       # [{time: "01:45", text: "..."}]
    frame_refs: list[dict] = field(default_factory=list)   # [{frame: "frame_023.png", time: "01:50", description: "..."}]
    numeric_conflicts: list[dict] = field(default_factory=list)  # [{speech_value, screen_value, entity}]
    action_needed: str = ""          # what to do about it


@dataclass
class CorrelationResult:
    """Full correlation output."""
    feedback_items: list[FeedbackItem]
    positives: list[dict] = field(default_factory=list)  # [{element, frame, quote}]
    summary_table: list[dict] = field(default_factory=list)


class Correlator:
    """Cross-reference transcription with frame analysis to produce structured feedback."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    async def correlate(
        self,
        transcription: TranscriptionResult,
        frames: list[FrameAnalysis],
    ) -> CorrelationResult:
        """Main correlation pipeline — uses pre-analyzed frame descriptions."""
        transcript_text = self._format_transcript(transcription)
        frames_text = self._format_frames(frames)
        prompt = self._build_correlation_prompt(transcript_text, frames_text)
        raw_result = await self._call_gemini(prompt)
        return self._parse_result(raw_result)

    async def correlate_with_images(
        self,
        transcription: TranscriptionResult,
        frame_paths: list[tuple[float, Path]],
        max_frames: int = 20,
    ) -> CorrelationResult:
        """Single-call correlation: sends images + transcription together.

        Much more efficient — uses 1 API call instead of N+1.
        Selects up to max_frames evenly distributed frames.
        """
        # Select subset of frames evenly distributed
        if len(frame_paths) > max_frames:
            step = len(frame_paths) / max_frames
            selected = [frame_paths[int(i * step)] for i in range(max_frames)]
        else:
            selected = frame_paths

        transcript_text = self._format_transcript(transcription)
        prompt = self._build_direct_correlation_prompt(transcript_text, selected)
        raw_result = await self._call_gemini_multimodal(prompt, selected)
        return self._parse_result(raw_result)

    def _format_transcript(self, transcription: TranscriptionResult) -> str:
        """Format transcription segments for the correlation prompt."""
        lines = []
        for seg in transcription.segments:
            lines.append(f"[{seg.start_ts} - {seg.end_ts}] {seg.text}")
        return "\n".join(lines)

    def _format_frames(self, frames: list[FrameAnalysis]) -> str:
        """Format frame analyses for the correlation prompt."""
        lines = []
        for fa in frames:
            frame_name = fa.frame_path.split("/")[-1] if "/" in fa.frame_path else fa.frame_path
            if fa.structured_data:
                lines.append(f"[{fa.timestamp_str}] {frame_name}:")
                lines.append(json.dumps(fa.structured_data, ensure_ascii=False, indent=2))
            else:
                lines.append(f"[{fa.timestamp_str}] {frame_name}: {fa.description}")
            lines.append("")
        return "\n".join(lines)

    def _build_direct_correlation_prompt(self, transcript: str, frames: list[tuple[float, Path]]) -> str:
        """Build prompt for direct image correlation (images sent separately as parts)."""
        frame_list = "\n".join(
            f"- Image {i+1}: {fp.name} at timestamp {self._ts(ts)}"
            for i, (ts, fp) in enumerate(frames)
        )
        return f"""You are analyzing a client feedback video where a user reviews a software product.
You are given:
1. Timestamped speech transcription (below)
2. Screenshots from the video at specific timestamps (attached as images)

Your task: cross-reference what the user SAYS with what's SHOWN ON SCREEN at that time.

## KEY RULES:

**CONFLICT DETECTION (most important):**
- If the user mentions a number/value for some entity, AND the screen shows a DIFFERENT number for the same entity → this is a WISH (feature request to change from screen value to spoken value)
- Example: user says "I'd set it to 50%" but screen shows max 15% → WISH to expand from 15% to 50%

**BUG detection:**
- User expresses confusion/frustration about unexpected behavior ("странно", "непонятно", "что-то не то")
- Screen shows data that contradicts expected behavior (e.g., low prices for high budget)

**POSITIVE detection:**
- User expresses approval ("офигенно", "классно", "нравится", "красиво", "охеренно")
- Screen shows the feature being praised — these are things NOT to change

**QUESTION detection:**
- User asks a question or expresses uncertainty ("надо ли оставлять", "узнаю")

## ATTACHED IMAGES (in order):
{frame_list}

## TRANSCRIPTION:
{transcript}

## OUTPUT FORMAT:
Return ONLY valid JSON:
{{
  "feedback_items": [
    {{
      "id": "BUG-1",
      "category": "BUG",
      "title": "Short descriptive title in Russian",
      "description": "Detailed description of the issue",
      "priority": "P0",
      "quotes": [
        {{"time": "01:45", "text": "exact quote from transcription"}}
      ],
      "frame_refs": [
        {{"frame": "frame_NNN.png", "time": "01:50", "description": "what the frame shows"}}
      ],
      "numeric_conflicts": [
        {{"speech_value": "50%", "screen_value": "15%", "entity": "допуск сверх бюджета"}}
      ],
      "action_needed": "What needs to be done"
    }}
  ],
  "positives": [
    {{"element": "UI element name", "frame": "frame_NNN.png", "quote": "user's positive comment", "time": "01:10"}}
  ]
}}

IMPORTANT:
- category must be one of: BUG, WISH, QUESTION
- For BUGs use P0, for WISHes use P1, for QUESTIONs use P2
- Include ALL significant feedback points (expect 5-8 items minimum)
- For each item include at least one quote with timestamp and one frame reference
- numeric_conflicts: when speech mentions different numbers than screen shows — this is the KEY feature
- Keep quotes in original language (Russian)
- IDs: BUG-1, BUG-2, WISH-1, WISH-2, etc.
- Pay special attention to:
  * Budget filtering issues (wrong prices for given budgets)
  * Login/auth modals blocking workflow
  * Feature requests about Quickvo integration
  * Budget range (from-to) request
  * Date range / season request
  * Budget tolerance (допуск) percentage conflict between speech and screen"""

    @staticmethod
    def _ts(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _build_correlation_prompt(self, transcript: str, frames: str) -> str:
        return f"""You are analyzing a client feedback video where a user reviews a software product.
You have two data sources:
1. Timestamped speech transcription
2. Timestamped screen frame descriptions (with UI elements and numeric values)

Your task: cross-reference speech with screen state to produce structured feedback.

## KEY RULES FOR CLASSIFICATION:

**CONFLICT DETECTION (most important):**
- If the user mentions a number/value for some entity, AND the screen shows a DIFFERENT number for the same entity → this is a WISH (feature request to change from screen value to spoken value)
- Example: user says "I'd set it to 50%" but screen shows max 15% → WISH to expand from 15% to 50%

**BUG detection:**
- User expresses confusion/frustration about unexpected behavior ("strange", "weird", "something wrong", "why does it...")
- Screen shows data that contradicts expected behavior

**POSITIVE detection:**
- User expresses approval ("great", "love it", "awesome", "cool", "nice")
- Screen shows the feature being praised
- These are things NOT to change

**QUESTION detection:**
- User asks a question or expresses uncertainty about whether something should stay or go

## TRANSCRIPTION:
{transcript}

## FRAME DESCRIPTIONS:
{frames}

## OUTPUT FORMAT:
Return ONLY valid JSON:
{{
  "feedback_items": [
    {{
      "id": "BUG-1",
      "category": "BUG",
      "title": "Short title",
      "description": "Detailed description of the issue",
      "priority": "P0",
      "quotes": [
        {{"time": "01:45", "text": "exact quote from transcription"}}
      ],
      "frame_refs": [
        {{"frame": "frame_023.png", "time": "01:50", "description": "what the frame shows"}}
      ],
      "numeric_conflicts": [
        {{"speech_value": "50%", "screen_value": "15%", "entity": "допуск сверх бюджета"}}
      ],
      "action_needed": "What needs to be done"
    }}
  ],
  "positives": [
    {{"element": "UI element name", "frame": "frame_015.png", "quote": "user's positive comment", "time": "01:10"}}
  ]
}}

IMPORTANT:
- category must be one of: BUG, WISH, QUESTION
- For BUGs use P0, for WISHes use P1, for QUESTIONs use P2
- Include ALL significant feedback points — bugs, feature requests, questions
- For each item include at least one quote with timestamp and one frame reference
- numeric_conflicts should be populated when speech mentions different numbers than what's on screen
- Keep quotes in original language (Russian)
- IDs should be sequential: BUG-1, BUG-2, WISH-1, WISH-2, etc.
- Look carefully for numeric conflicts between what user SAYS and what screen SHOWS"""

    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API for correlation with retry."""
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")

        client = genai.Client(api_key=self.api_key)

        for attempt in range(5):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=self.model,
                    contents=[prompt],
                )
                return response.text
            except Exception as e:
                if attempt < 4 and ("503" in str(e) or "429" in str(e) or "UNAVAILABLE" in str(e)):
                    wait = (2 ** attempt) * 5
                    logger.warning("Correlator retry (attempt %d, wait %ds): %s", attempt + 1, wait, e)
                    await asyncio.sleep(wait)
                else:
                    raise

    async def _call_gemini_multimodal(self, prompt: str, frames: list[tuple[float, Path]]) -> str:
        """Call Gemini API with images + text in a single request."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")

        client = genai.Client(api_key=self.api_key)

        # Build contents: images first, then prompt
        contents = []
        for ts, fp in frames:
            img_bytes = fp.read_bytes()
            suffix = fp.suffix.lower().lstrip(".")
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
        contents.append(prompt)

        for attempt in range(5):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=self.model,
                    contents=contents,
                )
                return response.text
            except Exception as e:
                if attempt < 4 and ("503" in str(e) or "429" in str(e) or "UNAVAILABLE" in str(e)):
                    wait = (2 ** attempt) * 10
                    logger.warning("Correlator multimodal retry (attempt %d, wait %ds): %s", attempt + 1, wait, e)
                    await asyncio.sleep(wait)
                else:
                    raise

    def _parse_result(self, raw: str) -> CorrelationResult:
        """Parse LLM JSON response into CorrelationResult."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n", 1)
            raw = lines[1] if len(lines) > 1 else ""
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Failed to parse correlator response as JSON")
            return CorrelationResult(feedback_items=[], positives=[])

        items = []
        for item_data in data.get("feedback_items", []):
            items.append(FeedbackItem(
                id=item_data.get("id", ""),
                category=item_data.get("category", "WISH"),
                title=item_data.get("title", ""),
                description=item_data.get("description", ""),
                priority=item_data.get("priority", "P1"),
                quotes=item_data.get("quotes", []),
                frame_refs=item_data.get("frame_refs", []),
                numeric_conflicts=item_data.get("numeric_conflicts", []),
                action_needed=item_data.get("action_needed", ""),
            ))

        positives = data.get("positives", [])

        return CorrelationResult(
            feedback_items=items,
            positives=positives,
        )
