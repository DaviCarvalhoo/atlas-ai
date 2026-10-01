"""Typed application settings, loaded from environment variables / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: Literal["offline", "openai", "anthropic", "azure", "xai"] = Field(
        "offline", alias="ATLAS_LLM_PROVIDER"
    )
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="ATLAS_OPENAI_MODEL")
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("claude-opus-5-5", alias="ATLAS_ANTHROPIC_MODEL")
    azure_api_key: str | None = Field(None, alias="AZURE_OPENAI_API_KEY")
    azure_endpoint: str | None = Field(None, alias="AZURE_OPENAI_ENDPOINT")
    azure_api_version: str = Field("2024-10-21", alias="AZURE_OPENAI_API_VERSION")
    azure_deployment: str | None = Field(None, alias="ATLAS_AZURE_DEPLOYMENT")
    # xAI (Grok) exposes an OpenAI-compatible API
    xai_api_key: str | None = Field(None, alias="XAI_API_KEY")
    xai_model: str = Field("grok-4", alias="ATLAS_XAI_MODEL")
    xai_base_url: str = Field("https://api.x.ai/v1", alias="ATLAS_XAI_BASE_URL")

    # Embeddings
    embedding_provider: Literal["lsa", "sentence-transformers", "openai"] = Field(
        "lsa", alias="ATLAS_EMBEDDING_PROVIDER"
    )
    st_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", alias="ATLAS_ST_MODEL"
    )
    openai_embedding_model: str = Field(
        "text-embedding-3-small", alias="ATLAS_OPENAI_EMBEDDING_MODEL"
    )

    # Storage
    database_url: str = Field("sqlite:///data/atlas.db", alias="ATLAS_DATABASE_URL")
    data_dir: Path = Field(Path("data"), alias="ATLAS_DATA_DIR")
    artifacts_dir: Path = Field(Path("artifacts"), alias="ATLAS_ARTIFACTS_DIR")
    knowledge_base_dir: Path = Field(Path("knowledge_base"), alias="ATLAS_KB_DIR")
    sql_dir: Path = Field(Path("sql"), alias="ATLAS_SQL_DIR")

    # MLOps
    mlflow_tracking_uri: str = Field("sqlite:///mlflow.db", alias="MLFLOW_TRACKING_URI")

    # API
    api_key: str = Field("change-me", alias="ATLAS_API_KEY")
    rate_limit_per_minute: int = Field(120, alias="ATLAS_RATE_LIMIT_PER_MINUTE")

    # Business rules
    auto_route_threshold: float = Field(0.6, alias="ATLAS_AUTO_ROUTE_THRESHOLD")

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def vector_dir(self) -> Path:
        return self.artifacts_dir / "chroma"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()
