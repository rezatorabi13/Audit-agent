"""Typed application settings, resolved from the environment or a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["anthropic", "ollama"]


class Settings(BaseSettings):
    """Runtime configuration for the audit agent.

    Every field can be overridden with an ``AUDIT_``-prefixed environment
    variable, e.g. ``AUDIT_TOP_K=8``. The Anthropic key keeps its conventional
    unprefixed name so it interoperates with other tooling.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AUDIT_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM -------------------------------------------------------------
    llm_provider: LLMProviderName = "anthropic"
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-sonnet-5"
    ollama_model: str = "llama3.2:3b"
    max_tokens: int = 1024
    temperature: float = 0.0

    # --- Retrieval -------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "audit_documents"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    # --- Engagement ------------------------------------------------------
    client_name: str = "Northwind Retail Group Inc."

    # --- Paths -----------------------------------------------------------
    data_dir: Path = Path("data")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def docs_dir(self) -> Path:
        """Source documents (contracts, invoices, policies) for retrieval."""
        return self.data_dir / "docs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tables_dir(self) -> Path:
        """Structured inputs: the general ledger and related-party register."""
        return self.data_dir / "tables"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chroma_dir(self) -> Path:
        """Persistent vector index location."""
        return self.data_dir / "chroma"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def output_dir(self) -> Path:
        """Where generated memos are written."""
        return self.data_dir / "output"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def models_dir(self) -> Path:
        """Fine-tuned adapters produced by the optional finetuning module."""
        return self.data_dir / "models"

    @property
    def ledger_path(self) -> Path:
        return self.tables_dir / "general_ledger_extract.csv"

    @property
    def related_party_path(self) -> Path:
        return self.tables_dir / "related_party_list.csv"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
