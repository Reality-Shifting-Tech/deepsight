"""Runtime configuration for deepsight.

Settings are read from environment variables (``DEEPSIGHT_*`` prefix) and
an optional ``.env`` file. Vision is device-native; the only required piece
is the path to the compiled Apple Vision binary.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Device-native vision + optional reasoning-loop configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEEPSIGHT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- native eyes (Apple Vision framework binary) ---
    vision_bin: str = "vision_eyes"

    # --- reasoning backend (the text-only LLM being given vision) ---
    reasoning_base_url: str = "https://api.deepseek.com/v1"
    reasoning_api_key: str = ""
    reasoning_model: str = "deepseek-v4-flash"
    reasoning_temperature: float = 0.2
    reasoning_max_tokens: int = 1024
    reasoning_tool_round_max_tokens: int = 1024

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


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
