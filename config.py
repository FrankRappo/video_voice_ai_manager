"""
VVAM Configuration — API keys, backend selection, paths.
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

CONFIG_FILE = "vvam.json"
ENV_PREFIX = "VVAM_"


@dataclass
class Config:
    # API Keys
    gemini_api_key: str = ""
    openai_api_key: str = ""

    # Backend selection
    transcriber: str = "gemini"  # gemini | whisper-local | openai
    vision: str = "gemini"       # gemini | openai | ollama

    # Gemini settings
    gemini_model: str = "gemini-2.5-flash"
    gemini_audio_model: str = "gemini-2.5-flash-native-audio-latest"

    # OpenAI settings
    openai_model: str = "gpt-4o"
    openai_whisper_model: str = "whisper-1"

    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llava"

    # Whisper local settings
    whisper_model: str = "medium"
    whisper_language: str = "ru"

    # Video processing
    video_fps: float = 0.2          # 1 frame per 5 seconds
    video_scene_detect: bool = True  # use scene detection
    video_scene_threshold: float = 0.3
    video_chunk_minutes: int = 10
    video_max_direct_minutes: int = 10

    # Output
    output_format: str = "markdown"  # markdown | json | srt
    output_dir: str = "./vvam_output"

    # Web server
    web_host: str = "0.0.0.0"
    web_port: int = 8000

    # yt-dlp
    ytdlp_format: str = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load config from file, env vars, with fallbacks."""
        config = cls()

        # 1. Load from config file
        path = Path(config_path) if config_path else Path(CONFIG_FILE)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # 2. Load from ~/.vvam/config.json (global)
        global_config = Path.home() / ".vvam" / "config.json"
        if global_config.exists():
            with open(global_config) as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(config, key) and not getattr(config, key):
                    setattr(config, key, value)

        # 3. Override from env vars (VVAM_GEMINI_API_KEY, etc.)
        for key in vars(config):
            env_key = ENV_PREFIX + key.upper()
            env_val = os.environ.get(env_key)
            if env_val:
                current = getattr(config, key)
                if isinstance(current, bool):
                    setattr(config, key, env_val.lower() in ("true", "1", "yes"))
                elif isinstance(current, int):
                    setattr(config, key, int(env_val))
                elif isinstance(current, float):
                    setattr(config, key, float(env_val))
                else:
                    setattr(config, key, env_val)

        # 4. Legacy env vars
        if not config.gemini_api_key:
            config.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        if not config.openai_api_key:
            config.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

        return config

    def save(self, config_path: Optional[str] = None):
        """Save config to file."""
        path = Path(config_path) if config_path else Path(CONFIG_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(vars(self), f, indent=2, ensure_ascii=False)

    def validate(self) -> list[str]:
        """Return list of issues with current config."""
        issues = []
        if self.transcriber == "gemini" and not self.gemini_api_key:
            issues.append("Gemini API key required (set VVAM_GEMINI_API_KEY or GEMINI_API_KEY)")
        if self.transcriber == "openai" and not self.openai_api_key:
            issues.append("OpenAI API key required (set VVAM_OPENAI_API_KEY or OPENAI_API_KEY)")
        if self.vision == "gemini" and not self.gemini_api_key:
            issues.append("Gemini API key required for vision")
        if self.vision == "openai" and not self.openai_api_key:
            issues.append("OpenAI API key required for vision")
        return issues
