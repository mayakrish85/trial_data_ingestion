from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    raw_dir: str = Field(default="data/raw")
    processed_dir: str = Field(default="data/processed")
    embeddings_dir: str = Field(default="data/embeddings")

    max_tokens: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    embedding_model: str = Field(default="intfloat/e5-base-v2")
    batch_size: int = Field(default=32)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    return Settings()
