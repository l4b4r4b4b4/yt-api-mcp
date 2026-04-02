"""Video extraction functions for YouTube MCP ELT pipeline.

This module provides raw data extraction from the YouTube API for video-related
data. These functions wrap existing YouTube tools and return unprocessed data
suitable for caching in the Load layer.

Extract layer principles:
1. Return raw data exactly as received from API
2. No transformation or analysis
3. Designed for caching - consistent output format
4. Quota-consuming operations that benefit from caching
"""

from __future__ import annotations

import logging
from typing import Any

from app.tools.youtube.load.cache import (
    extract_or_cache,
    video_details_key,
    video_search_key,
)
from app.tools.youtube.load.namespaces import (
    RAW_CHANNEL_VIDEOS,
    RAW_SEARCH_VIDEOS,
    RAW_VIDEO_DETAILS,
)
from app.tools.youtube.metadata import get_video_details
from app.tools.youtube.search import get_channel_videos, search_videos

logger = logging.getLogger(__name__)


async def extract_videos_raw(
    query: str,
    max_results: int = 50,
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract raw video search results from YouTube API.

    Wraps the search_videos function for the ELT pipeline.
    Returns raw search results without any transformation.

    Args:
        query: Search query string (e.g., "kubernetes tutorial").
        max_results: Maximum number of results (1-50, default: 50).
        cache: Optional RefCache instance for caching results.

    Returns:
        List of raw video search result dictionaries containing:
        - video_id: YouTube video ID
        - title: Video title
        - description: Video description snippet
        - url: Full YouTube watch URL
        - thumbnail: Thumbnail image URL
        - channel_title: Name of the channel
        - published_at: ISO 8601 publication timestamp

    Raises:
        YouTubeQuotaExceededError: If API quota is exceeded.
        YouTubeAPIError: For other API errors.

    Example:
        >>> videos = await extract_videos_raw("kubernetes", max_results=20)
        >>> print(len(videos))
        20
        >>> print(videos[0]["title"])
        "Kubernetes Tutorial for Beginners"

    Note:
        This function costs 100 quota units per request.
        Results are cached when cache parameter is provided.
    """
    logger.info(f"Extracting videos for query: {query!r}, max_results={max_results}")

    # If no cache, extract directly
    if cache is None:
        return await search_videos(query, max_results=max_results)

    # Build cache key
    key = video_search_key(query, max_results)

    # Extract or use cached
    async def extractor() -> list[dict[str, Any]]:
        return await search_videos(query, max_results=max_results)

    return await extract_or_cache(RAW_SEARCH_VIDEOS, key, extractor, cache)


async def extract_channel_videos_raw(
    channel_id: str,
    max_results: int = 50,
    order: str = "date",
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract raw video list from a specific channel.

    Wraps the get_channel_videos function for the ELT pipeline.
    Returns raw video list without any transformation.

    Args:
        channel_id: YouTube channel ID (e.g., "UCuAXFkgsw1L7xaCfnd5JJOw").
        max_results: Maximum number of videos (1-50, default: 50).
        order: Sort order ("date", "rating", "viewCount", "title").
        cache: Optional RefCache instance for caching results.

    Returns:
        List of raw video dictionaries from the channel.

    Raises:
        ValueError: If channel_id is invalid format.
        YouTubeAPIError: For API errors.

    Example:
        >>> videos = await extract_channel_videos_raw("UCuAXFkgsw1L7xaCfnd5JJOw")
        >>> print(len(videos))
        50
    """
    logger.info(f"Extracting channel videos: {channel_id}, max_results={max_results}")

    # If no cache, extract directly
    if cache is None:
        return await get_channel_videos(
            channel_id, max_results=max_results, order=order
        )

    # Build cache key (include order in key for uniqueness)
    from app.tools.youtube.load.cache import build_cache_key

    key = build_cache_key(
        RAW_CHANNEL_VIDEOS,
        channel_id,
        max_results=str(max_results),
        order=order,
    )

    # Extract or use cached
    async def extractor() -> list[dict[str, Any]]:
        return await get_channel_videos(
            channel_id, max_results=max_results, order=order
        )

    return await extract_or_cache(RAW_CHANNEL_VIDEOS, key, extractor, cache)


async def extract_video_details_single(
    video_id: str,
    cache: Any | None = None,
) -> dict[str, Any]:
    """Extract detailed metadata for a single video.

    Wraps the get_video_details function for the ELT pipeline.
    Returns raw video details without any transformation.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ").
        cache: Optional RefCache instance for caching results.

    Returns:
        Dictionary with full video metadata including:
        - title, description, video_id, url, thumbnail
        - view_count, like_count, comment_count
        - duration (ISO 8601 format), tags
        - channel_title, published_at

    Raises:
        ValueError: If video_id is invalid.
        YouTubeAPIError: If video not found or API error.

    Example:
        >>> details = await extract_video_details_single("dQw4w9WgXcQ")
        >>> print(details["view_count"])
        "1000000000"
    """
    logger.info(f"Extracting video details: {video_id}")

    # If no cache, extract directly
    if cache is None:
        return await get_video_details(video_id)

    # Build cache key
    key = video_details_key(video_id)

    # Extract or use cached
    async def extractor() -> dict[str, Any]:
        return await get_video_details(video_id)

    return await extract_or_cache(RAW_VIDEO_DETAILS, key, extractor, cache)


async def extract_video_details_batch(
    video_ids: list[str],
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract detailed metadata for multiple videos.

    Fetches details for multiple videos, with optional caching per video.
    More quota-efficient than calling extract_video_details_single repeatedly.

    Args:
        video_ids: List of YouTube video IDs (max ~50 recommended).
        cache: Optional RefCache instance for caching results.

    Returns:
        List of video detail dictionaries (same format as single).
        Order matches input video_ids. Missing videos are omitted.

    Raises:
        YouTubeAPIError: For API errors.

    Example:
        >>> details = await extract_video_details_batch(["id1", "id2", "id3"])
        >>> print(len(details))
        3

    Note:
        Each video is cached individually for maximum reusability.
        This allows single video lookups to benefit from batch fetches.
    """
    if not video_ids:
        return []

    logger.info(f"Extracting batch video details: {len(video_ids)} videos")

    # Fetch each video individually (allows per-video caching)
    # This is better than batch caching because:
    # 1. Single video requests can hit cache
    # 2. Cache granularity is per-video, not per-batch
    results = []
    for video_id in video_ids:
        try:
            details = await extract_video_details_single(video_id, cache=cache)
            results.append(details)
        except Exception as e:
            # Log but continue - partial results are better than none
            logger.warning(f"Failed to get details for video {video_id}: {e}")

    logger.info(f"Successfully extracted {len(results)}/{len(video_ids)} video details")
    return results


__all__ = [
    "extract_channel_videos_raw",
    "extract_video_details_batch",
    "extract_video_details_single",
    "extract_videos_raw",
]
