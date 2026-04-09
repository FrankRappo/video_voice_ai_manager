"""Transcriber factory."""
from config import Config
from transcribers.base import BaseTranscriber


def get_transcriber(cfg: Config) -> BaseTranscriber:
    """Instantiate the transcriber backend from config."""
    if cfg.transcriber == "gemini":
        from transcribers.gemini import GeminiTranscriber
        return GeminiTranscriber(api_key=cfg.gemini_api_key, model=cfg.gemini_audio_model)
    elif cfg.transcriber == "openai":
        from transcribers.openai_api import OpenAITranscriber
        return OpenAITranscriber(api_key=cfg.openai_api_key, model=cfg.openai_whisper_model)
    elif cfg.transcriber == "whisper-local":
        from transcribers.whisper_local import WhisperLocalTranscriber
        return WhisperLocalTranscriber(model=cfg.whisper_model, language=cfg.whisper_language)
    raise ValueError(f"Unknown transcriber: {cfg.transcriber}")
