"""Embedding model configuration for semantic transcript search.

Provides factory functions for creating Nomic embeddings with Matryoshka
dimensionality support. Uses langchain-nomic for native integration.

The Nomic Embed Text v1.5 model supports Matryoshka Representation Learning,
allowing embeddings to be truncated to smaller dimensions (64, 128, 256, 512, 768)
while maintaining quality proportional to the dimension.

Example:
    >>> from app.tools.youtube.semantic.embeddings import get_embeddings
    >>> embeddings = get_embeddings()
    >>> vector = embeddings.embed_query("example text")
    >>> len(vector)
    512
"""

from __future__ import annotations

from functools import lru_cache

from langchain_nomic import NomicEmbeddings

from app.tools.youtube.semantic.config import SemanticSearchConfig, get_semantic_config


def create_embeddings(config: SemanticSearchConfig) -> NomicEmbeddings:
    """Create a NomicEmbeddings instance with the given configuration.

    Args:
        config: Semantic search configuration with embedding settings.

    Returns:
        Configured NomicEmbeddings instance with Matryoshka dimensionality.

    Example:
        >>> from app.tools.youtube.semantic.config import SemanticSearchConfig
        >>> config = SemanticSearchConfig(embedding_dimensionality=256)
        >>> embeddings = create_embeddings(config)
    """
    return NomicEmbeddings(
        model=config.embedding_model,
        dimensionality=config.embedding_dimensionality,
        inference_mode=config.embedding_inference_mode,
    )


@lru_cache
def get_embeddings() -> NomicEmbeddings:
    """Get cached embeddings instance using default configuration.

    Uses the global semantic search configuration loaded from environment
    variables. The instance is cached for reuse across calls.

    Returns:
        Singleton NomicEmbeddings instance.

    Example:
        >>> embeddings = get_embeddings()
        >>> vectors = embeddings.embed_documents(["text 1", "text 2"])
    """
    config = get_semantic_config()
    return create_embeddings(config)
