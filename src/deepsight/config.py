"""Runtime configuration for the deepsight server.

Settings are read from environment variables (``DEEPSIGHT_*`` prefix) and
an optional ``.env`` file, so the server deploys without any code changes:
``DEEPSIGHT_REASONING_MODEL``, ``DEEPSIGHT_VISION_BASE_URL``, etc.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server + backend configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEEPSIGHT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "info"

    # --- reasoning backend (the text-only LLM being given vision) ---
    reasoning_base_url: str = "https://api.deepseek.com/v1"
    reasoning_api_key: str = ""
    reasoning_model: str = "deepseek-v4-flash"
    reasoning_temperature: float = 0.2

    # --- vision backend (the eyes; Ollama minicpm-v by default) ---
    vision_base_url: str = "http://127.0.0.1:11434"
    vision_api_key: str = ""
    vision_model: str = "minicpm-v:latest"
    vision_temperature: float = 0.0

    # --- vision-session loop ---
    max_look_rounds: int = 5
    sketch_enabled: bool = True

    # --- perception cache ---
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    @property
    def reasoning_key(self) -> str | None:
        """API key for the reasoning backend, or None when unset."""
        return self.reasoning_api_key or None

    @property
    def vision_key(self) -> str | None:
        """API key for the vision backend, or None when unset."""
        return self.vision_api_key or None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
