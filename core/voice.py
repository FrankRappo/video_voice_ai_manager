"""
Voice message analysis — transcribe voice messages from Telegram, WhatsApp, etc.
"""
import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import Config
from transcribers.base import BaseTranscriber, TranscriptionResult

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".wma"}
NATIVE_WAV_FORMATS = {".wav"}

# Messenger detection heuristics
TELEGRAM_EXTS = {".ogg"}
WHATSAPP_EXTS = {".opus"}


@dataclass
class VoiceResult:
    """Result of a single voice message analysis."""
    file: str
    transcription: TranscriptionResult
    messenger: str  # "telegram", "whatsapp", or "unknown"
    original_format: str


class VoiceAnalyzer:
    """Transcribe voice messages (ogg, opus, mp3, wav, etc.)."""

    def __init__(
        self,
        transcriber: BaseTranscriber,
        config: Optional[Config] = None,
    ):
        self.transcriber = transcriber
        self.config = config or Config.load()

    async def analyze(self, path: str) -> list[VoiceResult]:
        """Analyze a single file or all voice files in a directory.

        Args:
            path: Path to a voice file or directory with voice files.

        Returns:
            List of VoiceResult (one per file).
        """
        p = Path(path)
        if p.is_dir():
            return await self._analyze_directory(p)
        if p.is_file():
            result = await self._analyze_file(p)
            return [result]
        raise FileNotFoundError(f"Path not found: {path}")

    async def _analyze_directory(self, directory: Path) -> list[VoiceResult]:
        """Batch-process all voice files in a directory."""
        files = sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
        )
        if not files:
            logger.warning("No voice files found in %s", directory)
            return []

        results = []
        for f in files:
            try:
                result = await self._analyze_file(f)
                results.append(result)
            except Exception as e:
                logger.error("Failed to process %s: %s", f.name, e)
        return results

    async def _analyze_file(self, file_path: Path) -> VoiceResult:
        """Analyze a single voice file."""
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}")

        messenger = self._detect_messenger(file_path)

        # Convert to WAV if needed
        if ext in NATIVE_WAV_FORMATS:
            audio_path = file_path
            tmp_path = None
        else:
            audio_path, tmp_path = await self._convert_to_wav(file_path)

        try:
            transcription = await self.transcriber.transcribe(audio_path)
            return VoiceResult(
                file=str(file_path),
                transcription=transcription,
                messenger=messenger,
                original_format=ext.lstrip("."),
            )
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    async def _convert_to_wav(self, file_path: Path) -> tuple[Path, Path]:
        """Convert audio file to WAV 16kHz mono via ffmpeg.

        Returns:
            Tuple of (wav_path, tmp_path_to_cleanup).
        """
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="vvam_voice_")
        tmp_path = Path(tmp_name)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(file_path),
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
            raise RuntimeError(f"ffmpeg conversion failed for {file_path.name}: {stderr.decode()}")

        return tmp_path, tmp_path

    @staticmethod
    def _detect_messenger(file_path: Path) -> str:
        """Auto-detect messenger from file extension and naming patterns."""
        ext = file_path.suffix.lower()
        name = file_path.name.lower()

        # Telegram voice messages are typically .ogg (Opus in OGG container)
        if ext in TELEGRAM_EXTS:
            return "telegram"

        # WhatsApp voice messages are typically .opus
        if ext in WHATSAPP_EXTS:
            return "whatsapp"

        # Additional heuristics based on filename patterns
        if "ptt" in name or "whatsapp" in name:
            return "whatsapp"
        if "voice" in name and ext == ".ogg":
            return "telegram"

        return "unknown"
