"""Cache utilities for YouTube MCP ELT pipeline.

This module provides the core caching functionality for the ELT pipeline,
enabling the "Load" layer's primary function: storing and retrieving data
from RefCache efficiently.

Key functions:
- extract_or_cache: The workhorse - fetches from cache or extracts fresh data
- load_to_cache: Explicitly store data in cache
- get_cached: Retrieve data from cache (returns None if not found)
- invalidate_cache: Remove specific cache entries
- build_cache_key: Construct consistent cache keys

Design Principles:
1. Raw data is cached after extraction (saves quota)
2. Cache keys are deterministic and consistent
3. TTLs are determined by data type/namespace
4. All operations are async-friendly
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from app.tools.youtube.load.namespaces import (
    RAW_CHANNEL_INFO,
    RAW_CHANNEL_INFO_BATCH,
    RAW_CHANNEL_VIDEOS,
    RAW_COMMENTS,
    RAW_SEARCH_CHANNELS,
    RAW_SEARCH_VIDEOS,
    RAW_TRANSCRIPTS,
    RAW_TRENDING,
    RAW_VIDEO_DETAILS,
    RAW_VIDEO_DETAILS_BATCH,
)
from app.tools.youtube.load.ttl import get_ttl_seconds

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")


def build_cache_key(namespace: str, *args: str, **kwargs: str) -> str:
    """Build a consistent cache key from namespace and parameters.

    Creates a deterministic cache key by combining the namespace with
    positional and keyword arguments. Keys are normalized for consistency.

    Args:
        namespace: The cache namespace (e.g., "youtube.raw.search_videos")
        *args: Positional key components (e.g., query string)
        **kwargs: Keyword key components (e.g., region="US")

    Returns:
        A consistent cache key string.

    Example:
        >>> build_cache_key("youtube.raw.search_videos", "kubernetes")
        "youtube.raw.search_videos:kubernetes"

        >>> build_cache_key("youtube.raw.trending", region="US", category="28")
        "youtube.raw.trending:category=28:region=US"

        >>> build_cache_key("youtube.raw.video_details_batch", "id1", "id2", "id3")
        "youtube.raw.video_details_batch:id1:id2:id3"
    """
    parts = [namespace]

    # Add positional args
    for arg in args:
        # Normalize: lowercase, strip whitespace
        normalized = str(arg).strip().lower()
        parts.append(normalized)

    # Add sorted keyword args for determinism
    for key in sorted(kwargs.keys()):
        value = str(kwargs[key]).strip().lower()
        parts.append(f"{key}={value}")

    return ":".join(parts)


def build_batch_key(namespace: str, ids: list[str]) -> str:
    """Build a cache key for batch operations.

    For batch operations (e.g., fetching multiple video details),
    creates a deterministic key by sorting and joining IDs.

    Args:
        namespace: The cache namespace for batch operations.
        ids: List of IDs to include in the key.

    Returns:
        A consistent cache key for the batch.

    Example:
        >>> build_batch_key("youtube.raw.video_details_batch", ["c", "a", "b"])
        "youtube.raw.video_details_batch:a:b:c"
    """
    # Sort IDs for deterministic keys
    sorted_ids = sorted(ids)
    return build_cache_key(namespace, *sorted_ids)


def hash_key(key: str, max_length: int = 128) -> str:
    """Hash a cache key if it exceeds max length.

    Some cache backends have key length limits. This function hashes
    long keys while preserving short ones for readability.

    Args:
        key: The cache key to potentially hash.
        max_length: Maximum key length before hashing (default: 128).

    Returns:
        The original key if short enough, or a hashed version.

    Example:
        >>> hash_key("short:key")
        "short:key"

        >>> hash_key("a" * 200)  # Returns hash
        "youtube:sha256:abc123..."
    """
    if len(key) <= max_length:
        return key

    # Hash long keys, preserve namespace prefix for debugging
    namespace = key.split(":")[0] if ":" in key else "unknown"
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:32]
    return f"{namespace}:sha256:{key_hash}"


async def extract_or_cache[T](
    namespace: str,
    key: str,
    extractor: Callable[[], Awaitable[T]],
    cache: Any | None = None,
) -> T:
    """Extract data from API or return cached version.

    This is the primary function for the ELT pipeline's Extract+Load phases.
    It checks the cache first, and only calls the extractor if no cached
    data is found (or if cached data is expired).

    Args:
        namespace: The cache namespace (determines TTL).
        key: The cache key (use build_cache_key to construct).
        extractor: Async function that fetches fresh data from API.
        cache: RefCache instance (optional, for dependency injection in tests).

    Returns:
        The data (either from cache or freshly extracted).

    Raises:
        Any exceptions from the extractor function.

    Example:
        >>> async def fetch_videos():
        ...     return await search_videos("kubernetes", max_results=50)
        ...
        >>> key = build_cache_key(RAW_SEARCH_VIDEOS, "kubernetes")
        >>> videos = await extract_or_cache(RAW_SEARCH_VIDEOS, key, fetch_videos)

    Note:
        When cache is None, this function always calls the extractor.
        In production, pass the RefCache instance from the MCP server.
    """
    full_key = hash_key(key)

    # Try cache first (if cache is available)
    if cache is not None:
        try:
            cached_value = await get_cached(full_key, cache)
            if cached_value is not None:
                logger.debug(f"Cache hit: {full_key}")
                return cached_value
            logger.debug(f"Cache miss: {full_key}")
        except Exception as e:
            # Log but don't fail - extract fresh data instead
            logger.warning(f"Cache lookup failed for {full_key}: {e}")

    # Extract fresh data
    logger.info(f"Extracting fresh data for: {full_key}")
    data = await extractor()

    # Store in cache (if cache is available)
    if cache is not None:
        try:
            ttl_seconds = get_ttl_seconds(namespace)
            await load_to_cache(full_key, data, ttl_seconds, cache)
            logger.debug(f"Cached data for {full_key} (TTL: {ttl_seconds}s)")
        except Exception as e:
            # Log but don't fail - data was extracted successfully
            logger.warning(f"Cache store failed for {full_key}: {e}")

    return data


async def load_to_cache(
    key: str,
    data: Any,
    ttl_seconds: int,
    cache: Any,
) -> None:
    """Store data in the cache.

    Explicitly stores data in RefCache with the specified TTL.
    Used internally by extract_or_cache and can be called directly
    for pre-populating cache.

    Args:
        key: The cache key.
        data: The data to cache.
        ttl_seconds: Time-to-live in seconds.
        cache: RefCache instance.

    Example:
        >>> await load_to_cache(
        ...     "youtube.raw.video_details:abc123",
        ...     {"title": "My Video", ...},
        ...     86400,  # 24 hours
        ...     cache
        ... )
    """
    # RefCache API integration point
    # The actual implementation depends on RefCache's interface
    # This is a placeholder that will be integrated with RefCache
    if hasattr(cache, "set"):
        await cache.set(key, data, ttl=ttl_seconds)
    elif hasattr(cache, "store"):
        await cache.store(key, data, ttl=ttl_seconds)
    else:
        logger.warning(f"Cache does not have set/store method: {type(cache)}")


async def get_cached(key: str, cache: Any) -> Any | None:
    """Retrieve data from the cache.

    Returns None if the key is not found or expired.

    Args:
        key: The cache key to look up.
        cache: RefCache instance.

    Returns:
        The cached data, or None if not found/expired.

    Example:
        >>> data = await get_cached("youtube.raw.video_details:abc123", cache)
        >>> if data is None:
        ...     print("Not in cache")
    """
    # RefCache API integration point
    if hasattr(cache, "get"):
        result = await cache.get(key)
        # Handle RefCache's "not found" return value
        if result is None or (isinstance(result, dict) and result.get("_not_found")):
            return None
        return result
    elif hasattr(cache, "retrieve"):
        return await cache.retrieve(key)
    else:
        logger.warning(f"Cache does not have get/retrieve method: {type(cache)}")
        return None


async def invalidate_cache(key: str, cache: Any) -> bool:
    """Remove a specific entry from the cache.

    Args:
        key: The cache key to invalidate.
        cache: RefCache instance.

    Returns:
        True if the key was found and removed, False otherwise.

    Example:
        >>> await invalidate_cache("youtube.raw.video_details:abc123", cache)
        True
    """
    if hasattr(cache, "delete"):
        return await cache.delete(key)
    elif hasattr(cache, "invalidate"):
        return await cache.invalidate(key)
    else:
        logger.warning(f"Cache does not have delete/invalidate method: {type(cache)}")
        return False


async def invalidate_namespace(namespace: str, cache: Any) -> int:
    """Remove all entries in a namespace from the cache.

    Useful for bulk cache invalidation, e.g., when you know
    a category of data is stale.

    Args:
        namespace: The namespace prefix to invalidate.
        cache: RefCache instance.

    Returns:
        Number of keys invalidated.

    Example:
        >>> await invalidate_namespace("youtube.raw.search_videos", cache)
        15  # 15 cached searches were invalidated
    """
    if hasattr(cache, "clear_namespace"):
        return await cache.clear_namespace(namespace)
    elif hasattr(cache, "delete_pattern"):
        return await cache.delete_pattern(f"{namespace}:*")
    else:
        logger.warning(f"Cache does not support namespace invalidation: {type(cache)}")
        return 0


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON OPERATIONS
# =============================================================================


def video_search_key(query: str, max_results: int = 50) -> str:
    """Build cache key for video search results."""
    return build_cache_key(RAW_SEARCH_VIDEOS, query, max_results=str(max_results))


def channel_search_key(query: str, max_results: int = 50) -> str:
    """Build cache key for channel search results."""
    return build_cache_key(RAW_SEARCH_CHANNELS, query, max_results=str(max_results))


def video_details_key(video_id: str) -> str:
    """Build cache key for single video details."""
    return build_cache_key(RAW_VIDEO_DETAILS, video_id)


def video_details_batch_key(video_ids: list[str]) -> str:
    """Build cache key for batch video details."""
    return build_batch_key(RAW_VIDEO_DETAILS_BATCH, video_ids)


def channel_info_key(channel_id: str) -> str:
    """Build cache key for single channel info."""
    return build_cache_key(RAW_CHANNEL_INFO, channel_id)


def channel_info_batch_key(channel_ids: list[str]) -> str:
    """Build cache key for batch channel info."""
    return build_batch_key(RAW_CHANNEL_INFO_BATCH, channel_ids)


def channel_videos_key(channel_id: str, max_results: int = 50) -> str:
    """Build cache key for channel videos list."""
    return build_cache_key(RAW_CHANNEL_VIDEOS, channel_id, max_results=str(max_results))


def comments_key(video_id: str, max_results: int = 100) -> str:
    """Build cache key for video comments."""
    return build_cache_key(RAW_COMMENTS, video_id, max_results=str(max_results))


def trending_key(region: str = "US", category: str | None = None) -> str:
    """Build cache key for trending videos."""
    if category:
        return build_cache_key(RAW_TRENDING, region=region, category=category)
    return build_cache_key(RAW_TRENDING, region=region)


def transcript_key(video_id: str, language: str = "en") -> str:
    """Build cache key for video transcript."""
    return build_cache_key(RAW_TRANSCRIPTS, video_id, language=language)


__all__ = [
    "build_batch_key",
    "build_cache_key",
    "channel_info_batch_key",
    "channel_info_key",
    "channel_search_key",
    "channel_videos_key",
    "comments_key",
    "extract_or_cache",
    "get_cached",
    "hash_key",
    "invalidate_cache",
    "invalidate_namespace",
    "load_to_cache",
    "transcript_key",
    "trending_key",
    "video_details_batch_key",
    "video_details_key",
    "video_search_key",
]
