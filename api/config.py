from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    cors_origins: list[str] = ["*"]

    default_sample: int = 100
    default_pool: int = 500
    max_sample: int = 250
    max_pool: int = 1000


settings = Settings()
