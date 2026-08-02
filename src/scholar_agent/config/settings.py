"""Environment-backed application settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings required to compose the local application."""

    llm_provider_type: str = Field(default="ollama", min_length=1)
    model_name: str = Field(default="qwen3:1.7b", min_length=1)
    scratch_gpt_checkpoint_path: Path = Path("data/scholar_gpt.pt")
    ollama_url: str = Field(default="http://localhost:11434", min_length=1)
    vector_db_path: Path = Path("data/vector_store")
    document_library_path: Path = Path("data/documents")
    catalog_db_path: Path = Path("data/catalog.sqlite3")
    learner_profile_db_path: Path = Path("data/learner_profiles.sqlite3")
    embedding_model_name: str = Field(default="BAAI/bge-m3", min_length=1)
    embedding_device: str = Field(default="cpu", min_length=1)
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    retrieval_top_k: int = Field(default=4, ge=1)
    max_upload_mb: int = Field(default=50, ge=1)
    llm_context_length: int = Field(default=4096, ge=256)
    llm_max_tokens: int = Field(default=1024, ge=32)
    ollama_num_parallel: int = Field(default=1, ge=1)
    ollama_max_loaded_models: int = Field(default=1, ge=1)
    agent_max_actions_per_turn: int = Field(default=4, ge=1)
    agent_max_actions_per_session: int = Field(default=64, ge=1)
    agent_max_objectives: int = Field(default=6, ge=1, le=6)
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def to_container_config(self) -> dict[str, object]:
        """Return the configuration values required by infrastructure providers."""
        return {
            "llm_provider_type": self.llm_provider_type,
            "model_name": self.model_name,
            "scratch_gpt_checkpoint_path": self.scratch_gpt_checkpoint_path,
            "ollama_url": self.ollama_url,
            "vector_db_path": self.vector_db_path,
            "document_library_path": self.document_library_path,
            "catalog_db_path": self.catalog_db_path,
            "learner_profile_db_path": self.learner_profile_db_path,
            "embedding_model_name": self.embedding_model_name,
            "embedding_device": self.embedding_device,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_top_k": self.retrieval_top_k,
            "maximum_upload_bytes": self.max_upload_mb * 1024 * 1024,
            "llm_context_length": self.llm_context_length,
            "llm_max_tokens": self.llm_max_tokens,
            "ollama_num_parallel": self.ollama_num_parallel,
            "ollama_max_loaded_models": self.ollama_max_loaded_models,
            "agent_max_actions_per_turn": self.agent_max_actions_per_turn,
            "agent_max_actions_per_session": self.agent_max_actions_per_session,
            "agent_max_objectives": self.agent_max_objectives,
        }
