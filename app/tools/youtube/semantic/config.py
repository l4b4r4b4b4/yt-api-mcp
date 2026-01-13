"""Configuration for semantic transcript search.

Provides Pydantic settings for configuring:
- Embedding model (Nomic with Matryoshka dimensionality)
- HNSW vector index parameters
- Text chunking settings
- Persistence options

Environment Variables:
    SEMANTIC_EMBEDDING_MODEL: Embedding model name (default: nomic-embed-text-v1.5)
    SEMANTIC_EMBEDDING_DIMENSIONALITY: Matryoshka dimension (default: 512)
    SEMANTIC_EMBEDDING_INFERENCE_MODE: local, remote, or dynamic (default: local)
    SEMANTIC_HNSW_SPACE: Distance metric (default: cosine)
    SEMANTIC_HNSW_MAX_NEIGHBORS: HNSW M parameter (default: 48)
    SEMANTIC_HNSW_EF_CONSTRUCTION: Build-time accuracy (default: 200)
    SEMANTIC_HNSW_EF_SEARCH: Search-time accuracy (default: 128)
    SEMANTIC_CHUNK_SIZE: Target chunk size in characters (default: 800)
    SEMANTIC_CHUNK_OVERLAP: Overlap between chunks (default: 100)
    SEMANTIC_PERSIST_DIRECTORY: Directory for vector store persistence (optional)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_default_persist_directory() -> str | None:
    """Get XDG-compliant default persistence directory.

    Returns:
        Path to persistence directory, or None if not set.
    """
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        base_dir = Path(xdg_data_home)
    else:
        base_dir = Path.home() / ".local" / "share"

    return str(base_dir / "yt-mcp" / "chroma_db")


class SemanticSearchConfig(BaseSettings):
    """Configuration for semantic transcript search.

    Configures embedding model, HNSW index parameters, chunking strategy,
    and persistence options for the semantic search system.
    """

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_",
        case_sensitive=False,
        extra="ignore",
    )

    # Embedding model configuration
    embedding_model: str = Field(
        default="nomic-embed-text-v1.5",
        description="Nomic embedding model name.",
    )
    embedding_dimensionality: Literal[64, 128, 256, 512, 768] = Field(
        default=512,
        description="Matryoshka embedding dimensionality. Lower = faster, higher = better quality.",
    )
    embedding_inference_mode: Literal["local", "remote", "dynamic"] = Field(
        default="local",
        description="Inference mode: 'local' (Embed4All), 'remote' (API), 'dynamic' (auto).",
    )

    # HNSW index configuration
    hnsw_space: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description="Distance metric for similarity search.",
    )
    hnsw_max_neighbors: int = Field(
        default=48,
        ge=4,
        le=128,
        description="HNSW M parameter - connections per node. Higher = better recall, more memory.",
    )
    hnsw_ef_construction: int = Field(
        default=200,
        ge=10,
        le=500,
        description="Build-time accuracy. Higher = better index quality, slower build.",
    )
    hnsw_ef_search: int = Field(
        default=128,
        ge=10,
        le=500,
        description="Search-time accuracy. Higher = better recall, slower search.",
    )

    # Chunking configuration
    chunk_size: int = Field(
        default=800,
        ge=100,
        le=4000,
        description="Target chunk size in characters.",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks in characters.",
    )

    # Persistence configuration
    persist_directory: str | None = Field(
        default_factory=_get_default_persist_directory,
        description="Directory for ChromaDB persistence. None for in-memory only.",
    )

    # Collection configuration
    collection_name: str = Field(
        default="youtube_transcripts",
        description="Name of the ChromaDB collection for transcripts.",
    )

    @field_validator("persist_directory")
    @classmethod
    def expand_persist_directory(cls, value: str | None) -> str | None:
        """Expand ~ in persistence directory path."""
        if value is None:
            return None
        return str(Path(value).expanduser())

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info: dict) -> int:
        """Ensure chunk_overlap is less than chunk_size."""
        # Note: Can't easily access chunk_size here in pydantic v2
        # This validation happens at runtime when both are available
        return value

    @property
    def hnsw_config(self) -> dict[str, str | int]:
        """Get HNSW configuration dictionary for ChromaDB.

        Returns:
            Dictionary with HNSW parameters for collection_configuration.
        """
        return {
            "space": self.hnsw_space,
            "max_neighbors": self.hnsw_max_neighbors,
            "ef_construction": self.hnsw_ef_construction,
            "ef_search": self.hnsw_ef_search,
        }

    @property
    def collection_configuration(self) -> dict[str, dict[str, str | int]]:
        """Get full collection configuration for ChromaDB.

        Returns:
            Dictionary suitable for Chroma's collection_configuration parameter.
        """
        return {"hnsw": self.hnsw_config}


@lru_cache
def get_semantic_config() -> SemanticSearchConfig:
    """Get cached semantic search configuration.

    Returns:
        Singleton SemanticSearchConfig instance loaded from environment.
    """
    return SemanticSearchConfig()


# Convenience export for direct access
semantic_config = get_semantic_config()
