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

        # Split into 30-sec segments (optimal for transcription completeness)
        max_segment_bytes = bytes_per_sec * 30
        audio_segments = []
        for i in range(0, len(pcm_data), max_segment_bytes):
            seg_data = pcm_data[i:i + max_segment_bytes]
            seg_start = i / float(bytes_per_sec)
            audio_segments.append((seg_start, seg_data))

        all_segments = []

        for seg_start, seg_pcm in audio_segments:
            seg_text = await self._transcribe_live_chunk(client, seg_pcm)
            if seg_text:
                seg_duration = len(seg_pcm) / float(bytes_per_sec)
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

            # Send audio in 0.5-sec chunks at 2x real-time speed
            chunk_size = 16000  # 0.5 sec of 16kHz 16-bit mono
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i + chunk_size]
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                )
                await asyncio.sleep(0.25)  # 0.5s audio / 0.25s = 2x speed

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

        input_text = "".join(input_parts).strip()
        output_text = "".join(output_parts).strip()

        # Return whichever transcription is more complete
        return input_text if len(input_text) >= len(output_text) else output_text

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
