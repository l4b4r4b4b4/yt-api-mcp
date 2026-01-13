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

    Creates a TranscriptIndexer with the default vector store, transcript
    chunker, and comment chunker configuration.

    Returns:
        Configured TranscriptIndexer instance.

    Example:
        >>> indexer = get_indexer()
        >>> result = await indexer.index_video("dQw4w9WgXcQ")
    """
    from app.tools.youtube.semantic.chunker import TranscriptChunker
    from app.tools.youtube.semantic.comment_chunker import CommentChunker
    from app.tools.youtube.semantic.config import get_semantic_config
    from app.tools.youtube.semantic.indexer import TranscriptIndexer
    from app.tools.youtube.semantic.store import get_vector_store

    config = get_semantic_config()
    return TranscriptIndexer(
        vector_store=get_vector_store(),
        chunker=TranscriptChunker(config),
        comment_chunker=CommentChunker(),
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


async def semantic_search_comments(
    query: str,
    channel_ids: list[str] | None = None,
    video_ids: list[str] | None = None,
    k: int = 10,
    max_comments_per_video: int = 100,
    max_videos_per_channel: int = 50,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Search comments using natural language with automatic indexing.

    Performs semantic similarity search over video comments. Automatically
    indexes any missing comments before searching, providing a seamless
    experience without requiring explicit indexing calls.

    Args:
        query: Natural language search query (e.g., "questions about flakes").
        channel_ids: Optional list of YouTube channel IDs to scope the search.
            Videos from these channels will be auto-indexed if not already indexed.
        video_ids: Optional list of specific video IDs to scope the search.
            These videos will be auto-indexed if not already indexed.
        k: Number of results to return (default: 10).
        max_comments_per_video: Maximum comments to index per video (default: 100).
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
                - text: Matched comment text
                - author: Comment author name
                - like_count: Number of likes on the comment
                - reply_count: Number of replies
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
        >>> # Search a single channel's comments
        >>> results = await semantic_search_comments(
        ...     query="questions about garbage collection",
        ...     channel_ids=["UCuAXFkgsw1L7xaCfnd5JJOw"],
        ... )
        >>> print(results["results"][0]["text"])

        >>> # Search specific video comments
        >>> results = await semantic_search_comments(
        ...     query="thanks for the tutorial",
        ...     video_ids=["dQw4w9WgXcQ"],
        ... )

    Note:
        - First search on new content will be slower due to indexing
        - Subsequent searches are fast (already indexed)
        - If neither channel_ids nor video_ids provided, searches all indexed comments
        - Indexing uses ~1 API quota unit per video
    """
    from app.tools.youtube.search import get_channel_videos

    logger.info(
        f"semantic_search_comments called: query={query!r}, "
        f"channel_ids={channel_ids}, video_ids={video_ids}, k={k}"
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

    # Step 2: Auto-index missing video comments (only if scope is defined)
    if scoped_video_ids:
        logger.info(f"Checking {len(scoped_video_ids)} videos for comment indexing")
        for video_id in scoped_video_ids:
            indexing_stats["videos_checked"] += 1

            if indexer.is_video_comments_indexed(video_id):
                indexing_stats["videos_already_indexed"] += 1
                logger.debug(f"Video {video_id} comments already indexed, skipping")
                continue

            # Index the missing video's comments
            logger.debug(f"Indexing comments for video {video_id}")
            result = await indexer.index_video_comments(
                video_id, max_comments=max_comments_per_video
            )

            if result.indexed_count > 0:
                indexing_stats["videos_indexed"] += 1
                logger.debug(
                    f"Indexed video {video_id} comments: {result.chunk_count} chunks"
                )
            elif result.error_count > 0:
                indexing_stats["videos_failed"] += 1
                logger.warning(
                    f"Failed to index video {video_id} comments: {result.errors}"
                )
            else:
                # Skipped (e.g., no comments available)
                indexing_stats["videos_failed"] += 1
                logger.debug(f"Skipped video {video_id} (no comments or other issue)")

        logger.info(
            f"Comment indexing complete: {indexing_stats['videos_indexed']} indexed, "
            f"{indexing_stats['videos_already_indexed']} already indexed, "
            f"{indexing_stats['videos_failed']} failed"
        )

    # Step 3: Build filter and perform search
    # Filter to only search comments (not transcripts)
    filter_dict: dict[str, Any] = {"content_type": "comment"}

    if scoped_video_ids:
        # Combine content_type filter with video scope
        filter_dict = {
            "$and": [
                {"content_type": "comment"},
                {"video_id": {"$in": list(scoped_video_ids)}},
            ]
        }

    try:
        search_results = indexer.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict,
        )
    except Exception as e:
        logger.error(f"Comment search failed: {e}")
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
        if min_score is not None and score > min_score:
            continue

        formatted_results.append(
            {
                "video_id": doc.metadata.get("video_id"),
                "video_title": doc.metadata.get("video_title"),
                "video_url": doc.metadata.get("video_url"),
                "text": doc.page_content,
                "author": doc.metadata.get("author"),
                "like_count": doc.metadata.get("like_count"),
                "reply_count": doc.metadata.get("reply_count"),
                "published_at": doc.metadata.get("published_at"),
                "score": float(score),
                "channel_id": doc.metadata.get("channel_id"),
                "channel_title": doc.metadata.get("channel_title"),
                "chunk_index": doc.metadata.get("chunk_index"),
            }
        )

    logger.info(
        f"Comment search complete: {len(formatted_results)} results for query {query!r}"
    )

    return {
        "query": query,
        "results": formatted_results,
        "total_results": len(formatted_results),
        "indexing_stats": indexing_stats,
        "scope": _describe_scope(channel_ids, video_ids),
    }


async def semantic_search_all(
    query: str,
    content_types: list[str] | None = None,
    channel_ids: list[str] | None = None,
    video_ids: list[str] | None = None,
    k: int = 10,
    language: str = "en",
    max_comments_per_video: int = 100,
    max_videos_per_channel: int = 50,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Search across all content types (transcripts and comments).

    Performs unified semantic search over both video transcripts and comments.
    Automatically indexes any missing content before searching, providing a
    seamless experience without requiring explicit indexing calls.

    Args:
        query: Natural language search query (e.g., "Nix garbage collection").
        content_types: List of content types to search. Options: ["transcript", "comment"].
            If None or empty, searches all types.
        channel_ids: Optional list of YouTube channel IDs to scope the search.
            Videos from these channels will be auto-indexed if not already indexed.
        video_ids: Optional list of specific video IDs to scope the search.
            These videos will be auto-indexed if not already indexed.
        k: Number of results to return (default: 10).
        language: Preferred transcript language code (default: "en").
        max_comments_per_video: Maximum comments to index per video (default: 100).
        max_videos_per_channel: Maximum videos to fetch per channel (default: 50).
        min_score: Optional minimum similarity score threshold (0-1, lower is better
            for cosine distance).

    Returns:
        Dictionary with search results:
            - query: The original search query
            - results: List of matches, each with:
                - content_type: "transcript" or "comment"
                - video_id: YouTube video ID
                - video_title: Video title
                - video_url: Direct link to video
                - text: Matched content text
                - score: Similarity score (lower is better for cosine distance)
                - channel_id: YouTube channel ID
                - channel_title: Channel name
                - For transcripts: start_time, end_time, timestamp_url, language
                - For comments: author, like_count, reply_count, published_at
            - total_results: Number of results returned
            - indexing_stats: Statistics about auto-indexing performed:
                - transcripts: {videos_checked, videos_indexed, videos_already_indexed, videos_failed}
                - comments: {videos_checked, videos_indexed, videos_already_indexed, videos_failed}
            - scope: Description of search scope applied
            - content_types_searched: List of content types that were searched

    Example:
        >>> # Search all content types
        >>> results = await semantic_search_all(
        ...     query="garbage collection generations",
        ...     channel_ids=["UCuAXFkgsw1L7xaCfnd5JJOw"],
        ... )
        >>> for hit in results["results"]:
        ...     print(f"[{hit['content_type']}] {hit['text'][:50]}...")

        >>> # Search only transcripts
        >>> results = await semantic_search_all(
        ...     query="nix flakes tutorial",
        ...     content_types=["transcript"],
        ...     video_ids=["dQw4w9WgXcQ"],
        ... )

    Note:
        - First search on new content will be slower due to indexing
        - Subsequent searches are fast (already indexed)
        - If neither channel_ids nor video_ids provided, searches all indexed content
        - Results are sorted by relevance score across all content types
    """
    from app.tools.youtube.search import get_channel_videos

    # Validate and normalize content_types
    valid_types = {"transcript", "comment"}
    if content_types:
        search_types = [t for t in content_types if t in valid_types]
        if not search_types:
            search_types = list(valid_types)
    else:
        search_types = list(valid_types)

    logger.info(
        f"semantic_search_all called: query={query!r}, "
        f"content_types={search_types}, channel_ids={channel_ids}, "
        f"video_ids={video_ids}, k={k}"
    )

    indexer = get_indexer()
    scoped_video_ids: set[str] = set()

    # Initialize indexing stats for both content types
    indexing_stats = {
        "transcripts": {
            "videos_checked": 0,
            "videos_indexed": 0,
            "videos_already_indexed": 0,
            "videos_failed": 0,
        },
        "comments": {
            "videos_checked": 0,
            "videos_indexed": 0,
            "videos_already_indexed": 0,
            "videos_failed": 0,
        },
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

    # Step 2: Auto-index missing content (only if scope is defined)
    if scoped_video_ids:
        logger.info(f"Checking {len(scoped_video_ids)} videos for indexing")

        for video_id in scoped_video_ids:
            # Index transcripts if searching transcripts
            if "transcript" in search_types:
                indexing_stats["transcripts"]["videos_checked"] += 1

                if indexer.is_video_indexed(video_id):
                    indexing_stats["transcripts"]["videos_already_indexed"] += 1
                else:
                    result = await indexer.index_video(video_id, language=language)
                    if result.indexed_count > 0:
                        indexing_stats["transcripts"]["videos_indexed"] += 1
                    elif result.error_count > 0:
                        indexing_stats["transcripts"]["videos_failed"] += 1
                    else:
                        indexing_stats["transcripts"]["videos_failed"] += 1

            # Index comments if searching comments
            if "comment" in search_types:
                indexing_stats["comments"]["videos_checked"] += 1

                if indexer.is_video_comments_indexed(video_id):
                    indexing_stats["comments"]["videos_already_indexed"] += 1
                else:
                    result = await indexer.index_video_comments(
                        video_id, max_comments=max_comments_per_video
                    )
                    if result.indexed_count > 0:
                        indexing_stats["comments"]["videos_indexed"] += 1
                    elif result.error_count > 0:
                        indexing_stats["comments"]["videos_failed"] += 1
                    else:
                        indexing_stats["comments"]["videos_failed"] += 1

        logger.info(
            f"Indexing complete - Transcripts: {indexing_stats['transcripts']['videos_indexed']} indexed, "
            f"Comments: {indexing_stats['comments']['videos_indexed']} indexed"
        )

    # Step 3: Build filter and perform search
    filter_dict: dict[str, Any] | None = None

    # Build content type filter
    if len(search_types) == 1:
        content_filter = {"content_type": search_types[0]}
    else:
        content_filter = {"content_type": {"$in": search_types}}

    if scoped_video_ids:
        # Combine content_type filter with video scope
        filter_dict = {
            "$and": [
                content_filter,
                {"video_id": {"$in": list(scoped_video_ids)}},
            ]
        }
    else:
        # Just filter by content type (or None if searching all)
        if len(search_types) < len(valid_types):
            filter_dict = content_filter
        # If searching all types with no scope, no filter needed

    try:
        search_results = indexer.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict,
        )
    except Exception as e:
        logger.error(f"Unified search failed: {e}")
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "indexing_stats": indexing_stats,
            "scope": _describe_scope(channel_ids, video_ids),
            "content_types_searched": search_types,
            "error": str(e),
        }

    # Step 4: Format results based on content type
    formatted_results = []
    for doc, score in search_results:
        # Apply min_score filter if provided
        if min_score is not None and score > min_score:
            continue

        content_type = doc.metadata.get("content_type", "transcript")

        # Base result fields (common to all types)
        result_item = {
            "content_type": content_type,
            "video_id": doc.metadata.get("video_id"),
            "video_title": doc.metadata.get("video_title"),
            "video_url": doc.metadata.get("video_url"),
            "text": doc.page_content,
            "score": float(score),
            "channel_id": doc.metadata.get("channel_id"),
            "channel_title": doc.metadata.get("channel_title"),
            "chunk_index": doc.metadata.get("chunk_index"),
        }

        # Add type-specific fields
        if content_type == "transcript":
            result_item.update(
                {
                    "start_time": doc.metadata.get("start_time"),
                    "end_time": doc.metadata.get("end_time"),
                    "timestamp_url": doc.metadata.get("timestamp_url"),
                    "language": doc.metadata.get("language"),
                }
            )
        elif content_type == "comment":
            result_item.update(
                {
                    "author": doc.metadata.get("author"),
                    "like_count": doc.metadata.get("like_count"),
                    "reply_count": doc.metadata.get("reply_count"),
                    "published_at": doc.metadata.get("published_at"),
                }
            )

        formatted_results.append(result_item)

    logger.info(
        f"Unified search complete: {len(formatted_results)} results for query {query!r}"
    )

    return {
        "query": query,
        "results": formatted_results,
        "total_results": len(formatted_results),
        "indexing_stats": indexing_stats,
        "scope": _describe_scope(channel_ids, video_ids),
        "content_types_searched": search_types,
    }


async def get_indexed_videos(
    channel_id: str | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    """Get list of videos that have been indexed for semantic search.

    Retrieves all indexed video IDs with optional filtering by channel
    and/or content type. Useful for understanding what content is available
    for semantic search.

    Args:
        channel_id: Optional filter by YouTube channel ID.
        content_type: Optional filter by content type ("transcript" or "comment").
            If None, returns videos indexed for any content type.

    Returns:
        Dictionary with indexed video information:
            - video_ids: List of indexed video IDs
            - total_count: Total number of indexed videos
            - channel_filter: Channel ID filter if applied
            - content_type_filter: Content type filter if applied

    Example:
        >>> # Get all indexed videos
        >>> result = await get_indexed_videos()
        >>> print(f"Total indexed: {result['total_count']}")

        >>> # Get videos with indexed comments only
        >>> result = await get_indexed_videos(content_type="comment")
    """
    logger.info(
        f"get_indexed_videos called: channel_id={channel_id}, content_type={content_type}"
    )

    indexer = get_indexer()

    try:
        video_ids = await indexer.get_indexed_video_ids_by_content_type(
            content_type=content_type,
            channel_id=channel_id,
        )

        logger.info(f"Found {len(video_ids)} indexed videos")

        return {
            "video_ids": video_ids,
            "total_count": len(video_ids),
            "channel_filter": channel_id,
            "content_type_filter": content_type,
        }
    except Exception as e:
        logger.error(f"Failed to get indexed videos: {e}")
        return {
            "video_ids": [],
            "total_count": 0,
            "channel_filter": channel_id,
            "content_type_filter": content_type,
            "error": str(e),
        }


async def delete_indexed_video(
    video_id: str,
    content_type: str | None = None,
) -> dict[str, Any]:
    """Delete a video's content from the semantic search index.

    Removes indexed content (transcripts, comments, or both) for a specific
    video. Useful for re-indexing or cleaning up the index.

    Args:
        video_id: YouTube video ID to remove from the index.
        content_type: Optional content type to delete ("transcript" or "comment").
            If None, deletes all content types for the video.

    Returns:
        Dictionary with deletion results:
            - video_id: The deleted video ID
            - transcripts_deleted: Number of transcript chunks removed
            - comments_deleted: Number of comment chunks removed
            - total_deleted: Total chunks removed
            - success: Whether the deletion was successful

    Example:
        >>> # Delete all content for a video
        >>> result = await delete_indexed_video("dQw4w9WgXcQ")
        >>> print(f"Deleted {result['total_deleted']} chunks")

        >>> # Delete only comments
        >>> result = await delete_indexed_video("dQw4w9WgXcQ", content_type="comment")
    """
    logger.info(
        f"delete_indexed_video called: video_id={video_id}, content_type={content_type}"
    )

    indexer = get_indexer()
    transcripts_deleted = 0
    comments_deleted = 0

    try:
        # Delete transcripts if requested or if deleting all
        if content_type is None or content_type == "transcript":
            transcripts_deleted = await indexer.delete_video(video_id)
            logger.debug(f"Deleted {transcripts_deleted} transcript chunks")

        # Delete comments if requested or if deleting all
        if content_type is None or content_type == "comment":
            comments_deleted = await indexer.delete_video_comments(video_id)
            logger.debug(f"Deleted {comments_deleted} comment chunks")

        total_deleted = transcripts_deleted + comments_deleted
        logger.info(f"Deleted {total_deleted} total chunks for video {video_id}")

        return {
            "video_id": video_id,
            "transcripts_deleted": transcripts_deleted,
            "comments_deleted": comments_deleted,
            "total_deleted": total_deleted,
            "content_type_filter": content_type,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Failed to delete video {video_id}: {e}")
        return {
            "video_id": video_id,
            "transcripts_deleted": transcripts_deleted,
            "comments_deleted": comments_deleted,
            "total_deleted": transcripts_deleted + comments_deleted,
            "content_type_filter": content_type,
            "success": False,
            "error": str(e),
        }
