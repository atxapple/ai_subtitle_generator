from functools import lru_cache

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(
        default="whisper-1",
        alias="OPENAI_MODEL",
        description="OpenAI model used for speech-to-text. Defaults to whisper-1.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:  # pragma: no cover - configuration errors are fatal
        missing = ", ".join(error["loc"][0] for error in exc.errors())
        raise RuntimeError(
            "Missing required environment variables: "
            f"{missing}. Set them in the environment or .env file."
        ) from exc
