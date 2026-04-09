"""
Dictation module — transcribe audio to clean text, pipe-friendly output.
"""
import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

from config import Config
from transcribers.base import BaseTranscriber, TranscriptionResult

logger = logging.getLogger(__name__)


class Dictator:
    """Transcribe audio (file or microphone) to clean text.

    Designed for pipe-friendly output:
        vvam dictate recording.ogg | aider --message-file -
    """

    def __init__(
        self,
        transcriber: BaseTranscriber,
        config: Optional[Config] = None,
    ):
        self.transcriber = transcriber
        self.config = config or Config.load()

    async def dictate(self, audio_path: str) -> str:
        """Transcribe audio file and return clean text.

        Args:
            audio_path: Path to audio file.

        Returns:
            Transcribed text as a plain string.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Convert to WAV if needed
        wav_path, tmp_path = await self._ensure_wav(path)

        try:
            result = await self.transcriber.transcribe(wav_path)
            return result.full_text.strip()
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    async def dictate_to_stdout(self, audio_path: str) -> None:
        """Transcribe and write directly to stdout (pipe-friendly)."""
        text = await self.dictate(audio_path)
        sys.stdout.write(text)
        sys.stdout.write("\n")
        sys.stdout.flush()

    async def record_and_dictate(self, duration: float = 0.0) -> str:
        """Record from microphone and transcribe.

        Args:
            duration: Recording duration in seconds. 0 = until stopped (Ctrl+C).

        Returns:
            Transcribed text.

        Raises:
            ImportError: If sounddevice is not installed.
        """
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise ImportError(
                "Microphone recording requires 'sounddevice' and 'numpy'. "
                "Install with: pip install sounddevice numpy"
            )

        sample_rate = 16000
        channels = 1

        if duration > 0:
            logger.info("Recording for %.1f seconds...", duration)
            frames = int(duration * sample_rate)
            audio_data = sd.rec(
                frames, samplerate=sample_rate, channels=channels,
                dtype="int16",
            )
            sd.wait()
        else:
            # Record until Ctrl+C
            logger.info("Recording... Press Ctrl+C to stop.")
            chunks: list = []
            chunk_duration = 0.5  # seconds
            chunk_frames = int(chunk_duration * sample_rate)
            try:
                while True:
                    chunk = sd.rec(
                        chunk_frames, samplerate=sample_rate,
                        channels=channels, dtype="int16",
                    )
                    sd.wait()
                    chunks.append(chunk)
            except KeyboardInterrupt:
                pass
            if not chunks:
                return ""
            audio_data = np.concatenate(chunks, axis=0)

        # Save to temp WAV
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="vvam_mic_")
        tmp_path = Path(tmp_name)

        try:
            import wave
            with wave.open(str(tmp_path), "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())

            result = await self.transcriber.transcribe(tmp_path)
            return result.full_text.strip()
        finally:
            tmp_path.unlink(missing_ok=True)

    async def _ensure_wav(self, path: Path) -> tuple[Path, Optional[Path]]:
        """Convert to WAV 16kHz mono if not already WAV."""
        if path.suffix.lower() == ".wav":
            return path, None

        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="vvam_dict_")
        tmp_path = Path(tmp_name)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(path),
            "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(tmp_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg conversion failed: {stderr.decode()}")

        return tmp_path, tmp_path
