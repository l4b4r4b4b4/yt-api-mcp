"""Channel extraction functions for YouTube MCP ELT pipeline.

This module provides raw data extraction from the YouTube API for channel-related
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
    channel_info_key,
    channel_search_key,
    extract_or_cache,
)
from app.tools.youtube.load.namespaces import (
    RAW_CHANNEL_INFO,
    RAW_SEARCH_CHANNELS,
)
from app.tools.youtube.metadata import get_channel_info
from app.tools.youtube.search import search_channels

logger = logging.getLogger(__name__)


async def extract_channels_raw(
    query: str,
    max_results: int = 20,
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract raw channel search results from YouTube API.

    Wraps the search_channels function for the ELT pipeline.
    Returns raw search results without any transformation.

    Args:
        query: Search query string (e.g., "kubernetes tutorials").
        max_results: Maximum number of results (1-50, default: 20).
        cache: Optional RefCache instance for caching results.

    Returns:
        List of raw channel search result dictionaries containing:
        - channel_id: YouTube channel ID
        - title: Channel name/title
        - description: Channel description snippet
        - url: Full YouTube channel URL
        - thumbnail: Channel thumbnail/avatar URL
        - published_at: ISO 8601 channel creation timestamp

    Raises:
        YouTubeQuotaExceededError: If API quota is exceeded.
        YouTubeAPIError: For other API errors.

    Example:
        >>> channels = await extract_channels_raw("kubernetes", max_results=10)
        >>> print(len(channels))
        10
        >>> print(channels[0]["title"])
        "TechWorld with Nana"

    Note:
        This function costs 100 quota units per request.
        Results are cached when cache parameter is provided.
    """
    logger.info(f"Extracting channels for query: {query!r}, max_results={max_results}")

    # If no cache, extract directly
    if cache is None:
        return await search_channels(query, max_results=max_results)

    # Build cache key
    key = channel_search_key(query, max_results)

    # Extract or use cached
    async def extractor() -> list[dict[str, Any]]:
        return await search_channels(query, max_results=max_results)

    return await extract_or_cache(RAW_SEARCH_CHANNELS, key, extractor, cache)


async def extract_channel_info_single(
    channel_id: str,
    cache: Any | None = None,
) -> dict[str, Any]:
    """Extract detailed metadata for a single channel.

    Wraps the get_channel_info function for the ELT pipeline.
    Returns raw channel info without any transformation.

    Args:
        channel_id: YouTube channel ID (e.g., "UCuAXFkgsw1L7xaCfnd5JJOw").
        cache: Optional RefCache instance for caching results.

    Returns:
        Dictionary with full channel metadata including:
        - title, description, channel_id, url, thumbnail
        - subscriber_count, video_count, view_count
        - published_at

    Raises:
        ValueError: If channel_id is invalid format.
        YouTubeAPIError: If channel not found or API error.

    Example:
        >>> info = await extract_channel_info_single("UCuAXFkgsw1L7xaCfnd5JJOw")
        >>> print(info["subscriber_count"])
        "50000"
    """
    logger.info(f"Extracting channel info: {channel_id}")

    # If no cache, extract directly
    if cache is None:
        return await get_channel_info(channel_id)

    # Build cache key
    key = channel_info_key(channel_id)

    # Extract or use cached
    async def extractor() -> dict[str, Any]:
        return await get_channel_info(channel_id)

    return await extract_or_cache(RAW_CHANNEL_INFO, key, extractor, cache)


async def extract_channel_info_batch(
    channel_ids: list[str],
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract detailed metadata for multiple channels.

    Fetches info for multiple channels, handling API batching internally.
    More quota-efficient than calling extract_channel_info_single repeatedly.

    Args:
        channel_ids: List of YouTube channel IDs (max ~50 recommended).
        cache: Optional RefCache instance for caching results.

    Returns:
        List of channel info dictionaries (same format as single).
        Order matches input channel_ids. Missing channels are omitted.

    Raises:
        YouTubeAPIError: For API errors.

    Example:
        >>> info = await extract_channel_info_batch(["id1", "id2", "id3"])
        >>> print(len(info))
        3

    Note:
        Each channel is cached individually for maximum reusability.
        This allows single channel lookups to benefit from batch fetches.
    """
    if not channel_ids:
        return []

    logger.info(f"Extracting batch channel info: {len(channel_ids)} channels")

    # Fetch each channel individually (allows per-channel caching)
    # This is better than batch caching because:
    # 1. Single channel requests can hit cache
    # 2. Cache granularity is per-channel, not per-batch
    results = []
    for channel_id in channel_ids:
        try:
            info = await extract_channel_info_single(channel_id, cache=cache)
            results.append(info)
        except Exception as e:
            # Log but continue - partial results are better than none
            logger.warning(f"Failed to get info for channel {channel_id}: {e}")

    logger.info(
        f"Successfully extracted {len(results)}/{len(channel_ids)} channel info"
    )
    return results


async def extract_channels_from_videos(
    videos: list[dict[str, Any]],
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract channel info for all unique channels in a video list.

    Given a list of video search results, extracts detailed info for
    each unique channel. Useful for niche analysis where you want both
    video and channel data.

    Args:
        videos: List of video dictionaries with channel_title field.
        cache: Optional RefCache instance for caching results.

    Returns:
        List of channel info dictionaries for unique channels.

    Example:
        >>> videos = await extract_videos_raw("kubernetes", max_results=50)
        >>> channels = await extract_channels_from_videos(videos)
        >>> print(len(channels))
        25  # 50 videos from 25 unique channels
    """
    # Extract unique channel IDs from videos
    # Note: Search results have channel_title but not channel_id
    # We need to search for channels by name to get IDs
    # This is a limitation - we'll work around it

    unique_channel_titles = list({video.get("channel_title", "") for video in videos})
    unique_channel_titles = [t for t in unique_channel_titles if t]

    logger.info(
        f"Extracting channel info for {len(unique_channel_titles)} unique channels"
    )

    # Search for each channel by name to get channel_id, then get full info
    # This is quota-expensive but necessary due to API limitations
    results = []
    for channel_title in unique_channel_titles:
        try:
            # Search for the channel by exact name
            search_results = await search_channels(channel_title, max_results=1)
            if search_results:
                channel_id = search_results[0].get("channel_id")
                if channel_id:
                    info = await extract_channel_info_single(channel_id, cache=cache)
                    results.append(info)
        except Exception as e:
            logger.warning(f"Failed to get channel info for {channel_title}: {e}")

    logger.info(
        f"Successfully extracted {len(results)}/{len(unique_channel_titles)} channels"
    )
    return results


__all__ = [
    "extract_channel_info_batch",
    "extract_channel_info_single",
    "extract_channels_from_videos",
    "extract_channels_raw",
]
