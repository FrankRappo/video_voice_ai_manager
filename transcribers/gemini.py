"""
Gemini Native Audio transcriber.
Uses Gemini Live API (bidiGenerateContent) with input_audio_transcription
for native audio models, or standard generateContent for regular models.
"""
import json
import asyncio
import wave
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

# Models that only support bidiGenerateContent (Live API)
LIVE_ONLY_MODELS = {
    "gemini-2.5-flash-native-audio-latest",
    "gemini-2.5-flash-native-audio-preview-09-2025",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-3.1-flash-live-preview",
}


def _strip_overlap(prev: str, cur: str, window_words: int = 30) -> str:
    """Strip from `cur` the prefix that duplicates the tail of `prev`.

    Chunks are sent with a few seconds of audio overlap so words straddling a
    seam aren't lost; that overlap shows up as duplicated text at the boundary.
    The model transcribes the same overlap slightly differently in each window
    ("Авито" vs "Авита", "сделать и чтобы" vs "будете делать и чтобы"), so a
    char-exact match misses most cases. Fuzzy word-level matching via
    difflib handles the discrepancies.
    """
    import re
    from difflib import SequenceMatcher

    def tokens(s: str):
        return [(m.start(), m.end(), m.group().lower())
                for m in re.finditer(r"\w+", s)]

    prev_toks = tokens(prev)
    cur_toks = tokens(cur)
    if not prev_toks or not cur_toks:
        return cur.lstrip()

    prev_tail = prev_toks[-window_words:]
    cur_head = cur_toks[:window_words]
    prev_words = [t[2] for t in prev_tail]
    cur_words = [t[2] for t in cur_head]

    matcher = SequenceMatcher(None, prev_words, cur_words, autojunk=False)
    match = matcher.find_longest_match(0, len(prev_words), 0, len(cur_words))

    # Need at least 2 matching words, anchored near prev's tail and cur's head.
    if match.size < 2:
        return cur.lstrip()
    anchored_in_prev = match.a + match.size >= len(prev_words) - 2
    anchored_in_cur = match.b <= 3
    if not (anchored_in_prev and anchored_in_cur):
        return cur.lstrip()

    cut_word_idx = match.b + match.size
    if cut_word_idx >= len(cur_head):
        return ""
    cut_char = cur_head[cut_word_idx][0]
    return cur[cut_char:].lstrip(" ,.!?-—")


class GeminiTranscriber(BaseTranscriber):

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-native-audio-latest"):
        self.api_key = api_key
        self.model = model

    def name(self) -> str:
        return f"Gemini ({self.model})"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _is_live_model(self) -> bool:
        return self.model in LIVE_ONLY_MODELS

    async def transcribe(self, audio_path: Path, language: str = "") -> TranscriptionResult:
        if self._is_live_model():
            return await self._transcribe_live(audio_path, language)
        else:
            return await self._transcribe_standard(audio_path, language)

    async def _transcribe_live(self, audio_path: Path, language: str = "") -> TranscriptionResult:
        """Transcribe using Live API with explicit VAD and dual transcription."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")

        client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1alpha"})

        # Convert audio to WAV 16kHz mono PCM if needed
        wav_path = await self._ensure_wav(audio_path)
        pcm_data = self._read_wav_pcm(wav_path)

        bytes_per_sec = 32000  # 16kHz * 2 bytes per sample

        # 30-sec windows with 4-sec overlap so boundary words are captured by
        # at least one chunk (the model occasionally drops words straddling a
        # chunk seam — e.g. "Яндекс|СДЭК" → "Яндекс is, дек"). Window kept at
        # 30s because longer windows at realtime send rate trip the Live API
        # websocket keepalive.
        segment_sec = 30
        overlap_sec = 4
        step_bytes = bytes_per_sec * (segment_sec - overlap_sec)
        max_segment_bytes = bytes_per_sec * segment_sec
        audio_segments = []
        for i in range(0, len(pcm_data), step_bytes):
            seg_data = pcm_data[i:i + max_segment_bytes]
            seg_start = i / float(bytes_per_sec)
            audio_segments.append((seg_start, seg_data))
            if i + max_segment_bytes >= len(pcm_data):
                break

        all_segments = []

        for seg_start, seg_pcm in audio_segments:
            seg_text = await self._transcribe_live_chunk(client, seg_pcm)
            if seg_text:
                seg_duration = len(seg_pcm) / float(bytes_per_sec)
                if all_segments:
                    seg_text = _strip_overlap(all_segments[-1].text, seg_text)
                if seg_text:
                    all_segments.append(Segment(
                        start=seg_start,
                        end=seg_start + seg_duration,
                        text=seg_text,
                    ))

        # Clean up temp wav
        if wav_path != audio_path and wav_path.exists():
            wav_path.unlink()

        full_text = " ".join(s.text for s in all_segments).strip()

        return TranscriptionResult(
            segments=all_segments,
            language=language or "auto",
            full_text=full_text,
        )

    async def _transcribe_live_chunk(self, client, pcm_data: bytes) -> str:
        """Transcribe a single chunk of PCM audio via Live API session.

        Uses explicit VAD signals to prevent the model from interrupting
        audio processing, and collects both input and output transcriptions
        to maximize coverage.
        """
        from google.genai import types

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True,
                ),
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=(
                    "Listen to all audio completely. When the user finishes, "
                    "repeat EVERY single word you heard, verbatim, in the "
                    "original language. Do not skip, summarize, or translate."
                ))]
            ),
        )

        input_parts = []
        output_parts = []

        async with client.aio.live.connect(model=self.model, config=config) as session:
            # Signal speech activity start (prevents model from interrupting)
            await session.send_realtime_input(activity_start=types.ActivityStart())

            # Send audio in 0.5-sec chunks at 2x real-time speed.
            # 1x trips the Live API websocket keepalive; the model handles 2x
            # fine as long as we leave a grace period before closing activity.
            chunk_size = 16000  # 0.5 sec of 16kHz 16-bit mono
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i + chunk_size]
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                )
                await asyncio.sleep(0.25)

            # Grace period so model finishes transcribing the tail before we
            # close activity — otherwise last 1–2 seconds get truncated.
            await asyncio.sleep(1.5)

            # Signal speech activity end and stream end
            await session.send_realtime_input(activity_end=types.ActivityEnd())
            await session.send_realtime_input(audio_stream_end=True)

            # Collect both input and output transcriptions
            async for msg in session.receive():
                if msg.server_content:
                    sc = msg.server_content
                    if hasattr(sc, "input_transcription") and sc.input_transcription:
                        if sc.input_transcription.text:
                            input_parts.append(sc.input_transcription.text)
                    if hasattr(sc, "output_transcription") and sc.output_transcription:
                        if sc.output_transcription.text:
                            output_parts.append(sc.output_transcription.text)
                    if sc.turn_complete:
                        break

        # input_transcription is the verbatim STT of user audio.
        # output_transcription is the model's response (paraphrase/repetition)
        # and is unreliable — it can summarize or hallucinate. Always trust
        # input_transcription; fall back to output only if input is empty.
        input_text = "".join(input_parts).strip()
        output_text = "".join(output_parts).strip()
        return input_text or output_text

    async def _transcribe_standard(self, audio_path: Path, language: str = "") -> TranscriptionResult:
        """Transcribe using standard generateContent API."""
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")

        client = genai.Client(api_key=self.api_key)

        upload = await asyncio.to_thread(
            client.files.upload, file=str(audio_path)
        )

        prompt = TRANSCRIBE_PROMPT
        if language:
            prompt += f"\nExpected language: {language}"

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self.model,
            contents=[upload, prompt],
        )

        return self._parse_response(response.text, language)

    async def _ensure_wav(self, audio_path: Path) -> Path:
        """Convert audio to 16kHz mono WAV if not already."""
        if audio_path.suffix.lower() == ".wav":
            try:
                with wave.open(str(audio_path), "rb") as wf:
                    if wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getsampwidth() == 2:
                        return audio_path
            except Exception:
                pass

        wav_path = audio_path.parent / f"{audio_path.stem}_16k.wav"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            str(wav_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return wav_path

    def _read_wav_pcm(self, wav_path: Path) -> bytes:
        """Read raw PCM bytes from WAV file."""
        with wave.open(str(wav_path), "rb") as wf:
            return wf.readframes(wf.getnframes())

    def _parse_response(self, text: str, language: str) -> TranscriptionResult:
        """Parse JSON response from Gemini into TranscriptionResult."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n", 1)
            text = lines[1] if len(lines) > 1 else ""
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return TranscriptionResult(
                segments=[Segment(start=0.0, end=0.0, text=text)],
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
