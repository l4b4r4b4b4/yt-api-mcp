"""MCP tools for semantic transcript search.

Provides MCP-compatible tools for indexing and searching YouTube video
transcripts using semantic similarity. These tools integrate with the
mcp-refcache caching system for efficient handling of large result sets.

Tools:
    warmup_semantic_search: Pre-load embedding model and vector store.
    index_channel_transcripts: Index all transcripts from a YouTube channel.
    semantic_search_transcripts: Search indexed transcripts with natural language.

Example:
    >>> # These tools are registered with the MCP server automatically
    >>> # and can be called by LLM agents:
    >>>
    >>> # Index a channel's videos
    >>> result = await index_channel_transcripts(
    ...     channel_id="UCuAXFkgsw1L7xaCfnd5JJOw",
    ...     max_videos=50,
    ... )
    >>>
    >>> # Search for specific content
    >>> results = await semantic_search_transcripts(
    ...     query="how to configure Nix garbage collection",
    ...     channel_id="UCuAXFkgsw1L7xaCfnd5JJOw",
    ...     k=10,
    ... )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.tools.youtube.semantic.indexer import TranscriptIndexer

logger = logging.getLogger(__name__)


async def warmup_semantic_search() -> dict[str, Any]:
    """Pre-load the embedding model and vector store for semantic search.

    This tool downloads and initializes the Nomic embedding model (~270MB)
    and creates the vector store connection. Call this before your first
    semantic search to avoid timeout on the first query.

    The warmup process:
    1. Downloads the embedding model (if not cached)
    2. Loads the model into memory
    3. Runs a test embedding to warm up inference
    4. Initializes the vector store connection

    Returns:
        Dictionary with warmup status:
            - status: "ready" if successful
            - model: Name of the embedding model loaded
            - dimensionality: Embedding dimensions configured
            - inference_mode: How embeddings are computed (local/remote)
            - test_embedding_size: Size of test embedding (confirms model works)

    Example:
        >>> result = await warmup_semantic_search()
        >>> print(result["status"])
        "ready"
    """
    import time

    from app.tools.youtube.semantic.config import get_semantic_config
    from app.tools.youtube.semantic.embeddings import get_embeddings
    from app.tools.youtube.semantic.store import get_vector_store

    logger.info("Starting semantic search warmup...")
    start_time = time.time()

    # Load configuration
    config = get_semantic_config()
    logger.info(
        f"Config loaded: model={config.embedding_model}, dims={config.embedding_dimensionality}"
    )

    # Load embeddings (downloads model if needed)
    logger.info("Loading embedding model (may download ~270MB on first run)...")
    embeddings = get_embeddings()

    # Run a test embedding to warm up inference
    logger.info("Running test embedding...")
    test_vector = embeddings.embed_query("warmup test query")
    logger.info(f"Test embedding complete: {len(test_vector)} dimensions")

    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = get_vector_store()
    collection_name = (
        vector_store._collection.name
        if hasattr(vector_store, "_collection")
        else "unknown"
    )
    logger.info(f"Vector store ready: collection={collection_name}")

    elapsed_time = time.time() - start_time
    logger.info(f"Semantic search warmup complete in {elapsed_time:.2f}s")

    return {
        "status": "ready",
        "model": config.embedding_model,
        "dimensionality": config.embedding_dimensionality,
        "inference_mode": config.embedding_inference_mode,
        "test_embedding_size": len(test_vector),
        "warmup_time_seconds": round(elapsed_time, 2),
        "collection_name": collection_name,
    }


def get_indexer() -> TranscriptIndexer:
    """Get configured TranscriptIndexer instance.

    Creates a TranscriptIndexer with the default vector store and chunker
    configuration.

    Returns:
        Configured TranscriptIndexer instance.

    Example:
        >>> indexer = get_indexer()
        >>> result = await indexer.index_video("dQw4w9WgXcQ")
    """
    from app.tools.youtube.semantic.chunker import TranscriptChunker
    from app.tools.youtube.semantic.config import get_semantic_config
    from app.tools.youtube.semantic.indexer import TranscriptIndexer
    from app.tools.youtube.semantic.store import get_vector_store

    config = get_semantic_config()
    return TranscriptIndexer(
        vector_store=get_vector_store(),
        chunker=TranscriptChunker(config),
    )


async def index_channel_transcripts(
    channel_id: str,
    max_videos: int = 50,
    language: str = "en",
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Index all video transcripts from a YouTube channel for semantic search.

    Fetches videos from the specified channel, retrieves their transcripts,
    chunks them with timestamp preservation, and indexes them in the vector store.

    Args:
        channel_id: YouTube channel ID (e.g., "UCuAXFkgsw1L7xaCfnd5JJOw").
        max_videos: Maximum number of videos to index (default: 50).
        language: Preferred transcript language code (default: "en").
        force_reindex: If True, re-index videos even if already indexed.

    Returns:
        Dictionary with indexing results:
            - indexed_count: Number of videos successfully indexed
            - chunk_count: Total chunks created
            - skipped_count: Videos skipped (already indexed or no transcript)
            - error_count: Number of failed videos
            - errors: List of error messages
            - video_ids: List of indexed video IDs

    Example:
        >>> result = await index_channel_transcripts(
        ...     channel_id="UCuAXFkgsw1L7xaCfnd5JJOw",
        ...     max_videos=10,
        ... )
        >>> print(f"Indexed {result['indexed_count']} videos")
    """
    logger.info(
        f"index_channel_transcripts called: channel_id={channel_id}, "
        f"max_videos={max_videos}, language={language}, force_reindex={force_reindex}"
    )

    indexer = get_indexer()
    result = await indexer.index_channel(
        channel_id=channel_id,
        max_videos=max_videos,
        language=language,
        force_reindex=force_reindex,
    )

    return result.to_dict()


async def index_video_transcript(
    video_id: str,
    language: str = "en",
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Index a single video's transcript for semantic search.

    Retrieves the video's transcript, chunks it with timestamp preservation,
    and indexes it in the vector store.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ").
        language: Preferred transcript language code (default: "en").
        force_reindex: If True, re-index even if already indexed.

    Returns:
        Dictionary with indexing results:
            - indexed_count: 1 if successful, 0 otherwise
            - chunk_count: Number of chunks created
            - skipped_count: 1 if already indexed (and not force_reindex)
            - error_count: 1 if failed
            - errors: List of error messages (if any)
            - video_ids: List containing the video ID if successful

    Example:
        >>> result = await index_video_transcript("dQw4w9WgXcQ")
        >>> print(f"Created {result['chunk_count']} chunks")
    """
    logger.info(
        f"index_video_transcript called: video_id={video_id}, "
        f"language={language}, force_reindex={force_reindex}"
    )

    indexer = get_indexer()
    result = await indexer.index_video(
        video_id=video_id,
        language=language,
        force_reindex=force_reindex,
    )

    return result.to_dict()


async def semantic_search_transcripts(
    query: str,
    channel_ids: list[str] | None = None,
    video_ids: list[str] | None = None,
    k: int = 10,
    language: str = "en",
    max_videos_per_channel: int = 50,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Search transcripts using natural language with automatic indexing.

    Performs semantic similarity search over video transcripts. Automatically
    indexes any missing transcripts before searching, providing a seamless
    experience without requiring explicit indexing calls.

    Args:
        query: Natural language search query (e.g., "Nix garbage collection generations").
        channel_ids: Optional list of YouTube channel IDs to scope the search.
            Videos from these channels will be auto-indexed if not already indexed.
        video_ids: Optional list of specific video IDs to scope the search.
            These videos will be auto-indexed if not already indexed.
        k: Number of results to return (default: 10).
        language: Preferred transcript language code (default: "en").
        max_videos_per_channel: Maximum videos to fetch per channel (default: 50).
        min_score: Optional minimum similarity score threshold (0-1, lower is better
            for cosine distance).

    Returns:
        Dictionary with search results:
            - query: The original search query
            - results: List of matches, each with:
                - video_id: YouTube video ID
                - video_title: Video title
                - video_url: Direct link to video
                - text: Matched transcript segment
                - start_time: Start time in seconds
                - end_time: End time in seconds
                - timestamp_url: URL with timestamp for direct playback
                - score: Similarity score (lower is better for cosine distance)
                - channel_id: YouTube channel ID
                - channel_title: Channel name
            - total_results: Number of results returned
            - indexing_stats: Statistics about auto-indexing performed:
                - videos_checked: Number of videos checked for indexing
                - videos_indexed: Number of videos newly indexed
                - videos_already_indexed: Number of videos already in index
                - videos_failed: Number of videos that failed to index
            - scope: Description of search scope applied

    Example:
        >>> # Search a single channel
        >>> results = await semantic_search_transcripts(
        ...     query="how to configure garbage collection",
        ...     channel_ids=["UCuAXFkgsw1L7xaCfnd5JJOw"],
        ... )
        >>> print(results["results"][0]["text"])

        >>> # Search specific videos
        >>> results = await semantic_search_transcripts(
        ...     query="nix flakes tutorial",
        ...     video_ids=["dQw4w9WgXcQ", "abc123xyz"],
        ... )

        >>> # Search multiple channels
        >>> results = await semantic_search_transcripts(
        ...     query="declarative configuration",
        ...     channel_ids=["UCuAXFkgsw1L7xaCfnd5JJOw", "UC-another-channel"],
        ...     k=20,
        ... )

    Note:
        - First search on new content will be slower due to indexing (~1-2 min for 50 videos)
        - Subsequent searches are fast (already indexed)
        - If neither channel_ids nor video_ids provided, searches all indexed content
        - Indexing uses ~1 API quota unit per video (transcripts are free)
    """
    from app.tools.youtube.search import get_channel_videos

    logger.info(
        f"semantic_search_transcripts called: query={query!r}, "
        f"channel_ids={channel_ids}, video_ids={video_ids}, k={k}, language={language}"
    )

    indexer = get_indexer()
    scoped_video_ids: set[str] = set()
    indexing_stats = {
        "videos_checked": 0,
        "videos_indexed": 0,
        "videos_already_indexed": 0,
        "videos_failed": 0,
    }

    # Step 1: Determine scope - collect all video IDs to search over
    if video_ids:
        scoped_video_ids.update(video_ids)
        logger.debug(f"Added {len(video_ids)} explicit video IDs to scope")

    if channel_ids:
        for channel_id in channel_ids:
            try:
                videos = await get_channel_videos(
                    channel_id, max_results=max_videos_per_channel
                )
                channel_video_ids = [v["video_id"] for v in videos]
                scoped_video_ids.update(channel_video_ids)
                logger.debug(
                    f"Added {len(channel_video_ids)} videos from channel {channel_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch videos from channel {channel_id}: {e}")

    # Step 2: Auto-index missing videos (only if scope is defined)
    if scoped_video_ids:
        logger.info(f"Checking {len(scoped_video_ids)} videos for indexing")
        for video_id in scoped_video_ids:
            indexing_stats["videos_checked"] += 1

            if indexer.is_video_indexed(video_id):
                indexing_stats["videos_already_indexed"] += 1
                logger.debug(f"Video {video_id} already indexed, skipping")
                continue

            # Index the missing video
            logger.debug(f"Indexing video {video_id}")
            result = await indexer.index_video(video_id, language=language)

            if result.indexed_count > 0:
                indexing_stats["videos_indexed"] += 1
                logger.debug(f"Indexed video {video_id}: {result.chunk_count} chunks")
            elif result.error_count > 0:
                indexing_stats["videos_failed"] += 1
                logger.warning(f"Failed to index video {video_id}: {result.errors}")
            else:
                # Skipped (e.g., no transcript available)
                indexing_stats["videos_failed"] += 1
                logger.debug(f"Skipped video {video_id} (no transcript or other issue)")

        logger.info(
            f"Indexing complete: {indexing_stats['videos_indexed']} indexed, "
            f"{indexing_stats['videos_already_indexed']} already indexed, "
            f"{indexing_stats['videos_failed']} failed"
        )

    # Step 3: Build filter and perform search
    filter_dict: dict[str, Any] | None = None
    if scoped_video_ids:
        # Filter to only search within scoped videos
        filter_dict = {"video_id": {"$in": list(scoped_video_ids)}}

    try:
        search_results = indexer.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict,
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "indexing_stats": indexing_stats,
            "scope": _describe_scope(channel_ids, video_ids),
            "error": str(e),
        }

    # Step 4: Format results
    formatted_results = []
    for doc, score in search_results:
        # Apply min_score filter if provided
        # Note: For cosine distance, lower scores are better (0 = identical)
        if min_score is not None and score > min_score:
            continue

        formatted_results.append(
            {
                "video_id": doc.metadata.get("video_id"),
                "video_title": doc.metadata.get("video_title"),
                "video_url": doc.metadata.get("video_url"),
                "text": doc.page_content,
                "start_time": doc.metadata.get("start_time"),
                "end_time": doc.metadata.get("end_time"),
                "timestamp_url": doc.metadata.get("timestamp_url"),
                "score": float(score),
                "channel_id": doc.metadata.get("channel_id"),
                "channel_title": doc.metadata.get("channel_title"),
                "language": doc.metadata.get("language"),
                "chunk_index": doc.metadata.get("chunk_index"),
            }
        )

    logger.info(
        f"Search complete: {len(formatted_results)} results for query {query!r}"
    )

    return {
        "query": query,
        "results": formatted_results,
        "total_results": len(formatted_results),
        "indexing_stats": indexing_stats,
        "scope": _describe_scope(channel_ids, video_ids),
    }


def _describe_scope(
    channel_ids: list[str] | None,
    video_ids: list[str] | None,
) -> str:
    """Generate a human-readable description of the search scope."""
    parts = []
    if channel_ids:
        parts.append(f"{len(channel_ids)} channel(s)")
    if video_ids:
        parts.append(f"{len(video_ids)} specific video(s)")
    if not parts:
        return "all indexed content"
    return " + ".join(parts)


async def get_indexed_videos(
    channel_id: str | None = None,
) -> dict[str, Any]:
    """Get list of videos that have been indexed for semantic search.

    Args:
        channel_id: Optional filter by YouTube channel ID.

    Returns:
        Dictionary with indexed video information:
            - videos: List of indexed videos with metadata
            - total_count: Total number of indexed videos
            - channel_filter: Channel ID filter if applied

    Raises:
        NotImplementedError: This is a placeholder for future implementation.
    """
    # TODO: Implement as a utility tool
    raise NotImplementedError("get_indexed_videos will be implemented in a future task")


async def delete_indexed_video(
    video_id: str,
) -> dict[str, Any]:
    """Delete a video's transcript chunks from the semantic search index.

    Args:
        video_id: YouTube video ID to remove from the index.

    Returns:
        Dictionary with deletion results:
            - video_id: The deleted video ID
            - chunks_deleted: Number of chunks removed
            - success: Whether the deletion was successful

    Raises:
        NotImplementedError: This is a placeholder for future implementation.
    """
    # TODO: Implement as a utility tool
    raise NotImplementedError(
        "delete_indexed_video will be implemented in a future task"
    )
