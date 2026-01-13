"""Semantic transcript search module.

Provides semantic search capabilities over YouTube video transcripts
using LangChain, ChromaDB, and Nomic Matryoshka embeddings.

This module enables:
- Indexing video transcripts with timestamp-aware chunking
- Semantic similarity search with natural language queries
- Metadata filtering by channel, video, language, and time range
- Timestamped playback URLs for search results

Components:
    SemanticSearchConfig: Pydantic settings for embeddings, HNSW, and chunking.
    get_embeddings: Factory for NomicEmbeddings with Matryoshka dimensionality.
    get_vector_store: Factory for ChromaDB vector store with HNSW config.
    TranscriptChunker: Transcript-aware text splitter preserving timestamps.
    TranscriptIndexer: Batch indexing logic for channel/video transcripts.
    IndexingResult: Dataclass for indexing operation results.

Example:
    >>> from app.tools.youtube.semantic import get_vector_store, SemanticSearchConfig
    >>> store = get_vector_store()
    >>> results = store.similarity_search("nix garbage collection", k=5)
"""

from app.tools.youtube.semantic.chunker import TranscriptChunker
from app.tools.youtube.semantic.config import (
    SemanticSearchConfig,
    get_semantic_config,
    semantic_config,
)
from app.tools.youtube.semantic.embeddings import create_embeddings, get_embeddings
from app.tools.youtube.semantic.indexer import IndexingResult, TranscriptIndexer
from app.tools.youtube.semantic.store import create_vector_store, get_vector_store
from app.tools.youtube.semantic.tools import (
    index_channel_transcripts,
    semantic_search_transcripts,
)

__all__ = [
    "IndexingResult",
    "SemanticSearchConfig",
    "TranscriptChunker",
    "TranscriptIndexer",
    "create_embeddings",
    "create_vector_store",
    "get_embeddings",
    "get_semantic_config",
    "get_vector_store",
    "index_channel_transcripts",
    "semantic_config",
    "semantic_search_transcripts",
]
