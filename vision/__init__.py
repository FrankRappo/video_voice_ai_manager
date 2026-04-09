"""Vision factory."""
from config import Config
from vision.base import BaseVision


def get_vision(cfg: Config) -> BaseVision:
    """Instantiate the vision backend from config."""
    if cfg.vision == "gemini":
        from vision.gemini import GeminiVision
        return GeminiVision(api_key=cfg.gemini_api_key, model=cfg.gemini_model)
    elif cfg.vision == "openai":
        from vision.openai_api import OpenAIVision
        return OpenAIVision(api_key=cfg.openai_api_key, model=cfg.openai_model)
    elif cfg.vision == "ollama":
        from vision.ollama import OllamaVision
        return OllamaVision(url=cfg.ollama_url, model=cfg.ollama_model)
    raise ValueError(f"Unknown vision backend: {cfg.vision}")
