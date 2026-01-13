"""Vector store initialization for semantic transcript search.

Provides factory functions for creating and managing ChromaDB vector stores
with optimized HNSW configuration for transcript similarity search.

The store uses ChromaDB's collection_configuration API (v1.3.5+) for
configuring HNSW parameters at collection creation time.

Example:
    >>> from app.tools.youtube.semantic.store import get_vector_store
    >>> store = get_vector_store()
    >>> results = store.similarity_search("nix garbage collection", k=5)
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_chroma import Chroma

from app.tools.youtube.semantic.config import SemanticSearchConfig, get_semantic_config
from app.tools.youtube.semantic.embeddings import create_embeddings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def create_vector_store(
    config: SemanticSearchConfig,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Create a Chroma vector store with the given configuration.

    Args:
        config: Semantic search configuration with HNSW and persistence settings.
        embeddings: Optional embeddings instance. If None, creates from config.

    Returns:
        Configured Chroma vector store instance.

    Example:
        >>> from app.tools.youtube.semantic.config import SemanticSearchConfig
        >>> config = SemanticSearchConfig(hnsw_max_neighbors=64)
        >>> store = create_vector_store(config)
    """
    if embeddings is None:
        embeddings = create_embeddings(config)

    return Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        collection_configuration=config.collection_configuration,
        persist_directory=config.persist_directory,
    )


@lru_cache
def get_vector_store() -> Chroma:
    """Get cached vector store instance using default configuration.

    Uses the global semantic search configuration loaded from environment
    variables. The instance is cached for reuse across calls.

    Returns:
        Singleton Chroma vector store instance.

    Example:
        >>> store = get_vector_store()
        >>> store.add_texts(["transcript chunk 1", "transcript chunk 2"])
    """
    config = get_semantic_config()
    return create_vector_store(config)
