from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Paths
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"

    # Embeddings (local, via sentence-transformers)
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Generation (local, via Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "medgemma:4b"


settings = Settings()
