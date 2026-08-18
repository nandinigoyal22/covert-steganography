"""
config/settings.py
──────────────────
Application settings loaded from environment variables (or a .env file).

Uses pydantic-settings so that every setting is validated at startup:
    - Missing OPENAI_API_KEY → immediate, clear error.
    - Wrong types (e.g. OPENAI_TIMEOUT="abc") → ValidationError.

Usage:
    from member1.config.settings import Settings
    settings = Settings()           # reads from env / .env
    settings = Settings(_env_file=".env")  # explicit .env path
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configurable knobs for Member 1, in one place."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── OpenAI-compatible API ────────────────────────────────────────
    OPENAI_API_KEY: str                           # required — no default
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TIMEOUT: int = 30                      # seconds

    # ── General ──────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
