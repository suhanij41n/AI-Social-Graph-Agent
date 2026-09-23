"""Central configuration loaded from .env and config/ranking_weights.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
RANKING_WEIGHTS_PATH = REPO_ROOT / "config" / "ranking_weights.yaml"
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_GENERATED_DIR = REPO_ROOT / "data" / "generated"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "socialgraph123"

    qdrant_mode: str = "server"  # "server" or "embedded"
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str = "./data/qdrant_local"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    random_seed: int = 42
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_ranking_weights() -> dict:
    with open(RANKING_WEIGHTS_PATH) as f:
        return yaml.safe_load(f)
