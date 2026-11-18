from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rolefit:rolefit@localhost:5432/rolefit"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    extract_model: str = "gpt-4o-mini"
    rerank_model: str = "gpt-4o-mini"

    allowed_origins: str = "http://localhost:8000"

    max_resume_chars: int = 20000
    max_query_chars: int = 500
    rate_limit_search: str = "20/minute"
    rate_limit_match: str = "6/minute"

    ingest_concurrency: int = 16

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
