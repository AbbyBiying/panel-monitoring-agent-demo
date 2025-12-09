# panel_monitoring/app/config.py
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # --- Core ---
    env: Literal["dev", "stage", "prod"] = Field(default="dev", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- LangSmith ---
    langsmith_project: str = Field(
        default="panel-monitoring-agent", alias="LANGSMITH_PROJECT"
    )
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")

    # --- OpenAI ---
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # --- Vertex AI / Gen AI ---
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    google_cloud_project: Optional[str] = Field(
        default=None, alias="GOOGLE_CLOUD_PROJECT"
    )
    google_cloud_location: Optional[str] = Field(
        default=None, alias="GOOGLE_CLOUD_LOCATION"
    )

    # pydantic-settings config
    model_config = SettingsConfigDict(
        env_file=".env",  # Look for variables in .env
        env_file_encoding="utf-8",  # Encoding for the .env file
        case_sensitive=False,  # Environment variables are case-insensitive
        extra="ignore",  # Ignore extra environment variables not defined in the model
    )

    @field_validator("log_level")
    @classmethod
    def normalize_level(cls, v: str, _: ValidationInfo) -> str:
        v2 = (v or "").upper()
        return v2 if v2 in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"} else "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton Settings (cached)."""
    return Settings()
