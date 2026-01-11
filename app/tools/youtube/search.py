"""YouTube search functionality for videos and channels.

This module provides search tools for discovering YouTube content via the Data API v3.
All functions are designed to work with mcp-refcache for intelligent result caching.
"""

from __future__ import annotations

import logging
from typing import Any

from googleapiclient.errors import HttpError

from app.tools.youtube.client import get_youtube_service, handle_youtube_api_error
from app.tools.youtube.models import ChannelSearchResult, VideoSearchResult

logger = logging.getLogger(__name__)


async def search_videos(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search for YouTube videos by query string.

    Executes a search via the YouTube Data API v3 and returns video results
    with metadata including title, description, channel, thumbnails, and URLs.

    Args:
        query: Search query string (e.g., "NixOS tutorials", "vimjoyer nix").
        max_results: Maximum number of results to return (clamped to 1-50).

    Returns:
        List of video search results as dictionaries. Each result contains:
        - video_id: YouTube video ID
        - title: Video title
        - description: Video description snippet
        - url: Full YouTube watch URL
        - thumbnail: Thumbnail image URL
        - channel_title: Name of the channel
        - published_at: ISO 8601 publication timestamp

    Raises:
        YouTubeQuotaExceededError: If API quota is exceeded.
        YouTubeAuthError: If authentication fails.
        YouTubeNotFoundError: If resource not found (rare for search).
        YouTubeAPIError: For other API errors.

    Example:
        >>> results = await search_videos("vimjoyer nix", max_results=5)
        >>> print(results[0]["title"])
        'NixOS Garbage Collection Tutorial'
        >>> print(results[0]["url"])
        'https://www.youtube.com/watch?v=...'

    Note:
        This function is designed to be decorated with @cache.cached in server.py.
        Search costs 100 quota units per request. With 10,000 daily quota,
        you can perform ~100 searches per day. Caching significantly reduces usage.
    """
    # Clamp max_results to valid range (YouTube API enforces 1-50)
    max_results = max(1, min(50, max_results))

    logger.info(f"Searching videos: query={query!r}, max_results={max_results}")

    try:
        # Get authenticated YouTube service
        youtube = get_youtube_service()

        # Execute search request
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results,
        )
        response = request.execute()

        # Parse results into Pydantic models, then convert to dicts
        results = []
        for item in response.get("items", []):
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]

            # Extract thumbnail (prefer high quality, fallback to default)
            thumbnails = snippet.get("thumbnails", {})
            thumbnail_url = (
                thumbnails.get("high", {}).get("url")
                or thumbnails.get("medium", {}).get("url")
                or thumbnails.get("default", {}).get("url", "")
            )

            # Build VideoSearchResult model
            video_result = VideoSearchResult(
                title=snippet.get("title", ""),
                description=snippet.get("description", ""),
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                thumbnail=thumbnail_url,
                channel_title=snippet.get("channelTitle", ""),
                published_at=snippet.get("publishedAt", ""),
            )

            results.append(video_result.model_dump())

        logger.info(f"Found {len(results)} video results for query: {query!r}")
        return results

    except HttpError as e:
        logger.error(f"YouTube API error during video search: {e}")
        handle_youtube_api_error(e)
        # handle_youtube_api_error always raises, but type checker needs this
        raise


async def search_channels(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search for YouTube channels by query string.

    Executes a search via the YouTube Data API v3 and returns channel results
    with metadata including title, description, thumbnails, and URLs.

    Args:
        query: Search query string (e.g., "Vimjoyer", "NixOS channels").
        max_results: Maximum number of results to return (clamped to 1-50).

    Returns:
        List of channel search results as dictionaries. Each result contains:
        - channel_id: YouTube channel ID
        - title: Channel name/title
        - description: Channel description snippet
        - url: Full YouTube channel URL
        - thumbnail: Channel thumbnail/avatar URL
        - published_at: ISO 8601 channel creation timestamp

    Raises:
        YouTubeQuotaExceededError: If API quota is exceeded.
        YouTubeAuthError: If authentication fails.
        YouTubeNotFoundError: If resource not found (rare for search).
        YouTubeAPIError: For other API errors.

    Example:
        >>> results = await search_channels("vimjoyer", max_results=5)
        >>> print(results[0]["title"])
        'Vimjoyer'
        >>> print(results[0]["url"])
        'https://www.youtube.com/channel/UCxyz...'

    Note:
        This function is designed to be decorated with @cache.cached in server.py.
        Search costs 100 quota units per request. With 10,000 daily quota,
        you can perform ~100 searches per day. Caching significantly reduces usage.
    """
    # Clamp max_results to valid range (YouTube API enforces 1-50)
    max_results = max(1, min(50, max_results))

    logger.info(f"Searching channels: query={query!r}, max_results={max_results}")

    try:
        # Get authenticated YouTube service
        youtube = get_youtube_service()

        # Execute search request
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="channel",
            maxResults=max_results,
        )
        response = request.execute()

        # Parse results into Pydantic models, then convert to dicts
        results = []
        for item in response.get("items", []):
            channel_id = item["id"]["channelId"]
            snippet = item["snippet"]

            # Extract thumbnail (prefer high quality, fallback to default)
            thumbnails = snippet.get("thumbnails", {})
            thumbnail_url = (
                thumbnails.get("high", {}).get("url")
                or thumbnails.get("medium", {}).get("url")
                or thumbnails.get("default", {}).get("url", "")
            )

            # Build ChannelSearchResult model
            channel_result = ChannelSearchResult(
                title=snippet.get("title", ""),
                description=snippet.get("description", ""),
                channel_id=channel_id,
                url=f"https://www.youtube.com/channel/{channel_id}",
                thumbnail=thumbnail_url,
                published_at=snippet.get("publishedAt", ""),
            )

            results.append(channel_result.model_dump())

        logger.info(f"Found {len(results)} channel results for query: {query!r}")
        return results

    except HttpError as e:
        logger.error(f"YouTube API error during channel search: {e}")
        handle_youtube_api_error(e)
        # handle_youtube_api_error always raises, but type checker needs this
        raise


async def search_live_videos(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search for currently live YouTube videos.

    Searches for videos that are currently streaming live, filtering results
    to only active broadcasts. This is a specialized version of search_videos
    with the eventType filter set to "live".

    Args:
        query: Search query string (e.g., "gaming live", "news live now").
        max_results: Maximum number of results to return (clamped to 1-50).

    Returns:
        List of live video search results as dictionaries. Each result contains:
        - video_id: YouTube video ID
        - title: Video title
        - description: Video description snippet
        - url: Full YouTube watch URL
        - thumbnail: Thumbnail image URL
        - channel_title: Name of the channel
        - published_at: ISO 8601 publication timestamp

    Raises:
        YouTubeQuotaExceededError: If API quota is exceeded.
        YouTubeAuthError: If authentication fails.
        YouTubeNotFoundError: If resource not found (rare for search).
        YouTubeAPIError: For other API errors.

    Example:
        >>> results = await search_live_videos("gaming", max_results=10)
        >>> print(results[0]["title"])
        'Live Gaming Stream - Fortnite'
        >>> # Results only include currently active live streams

    Note:
        This function is designed to be decorated with @cache.cached in server.py.
        Search costs 100 quota units per request. Results are cached for 6 hours
        (same as regular search) since live stream search results change slowly.

        To check if a specific video is currently live, use the is_live() tool.
        To get live chat messages, use get_live_chat_messages() tool.
    """
    # Clamp max_results to valid range (YouTube API enforces 1-50)
    max_results = max(1, min(50, max_results))

    logger.info(f"Searching live videos: query={query!r}, max_results={max_results}")

    try:
        # Get authenticated YouTube service
        youtube = get_youtube_service()

        # Execute search request with eventType="live" filter
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            eventType="live",  # Only currently live streams
            maxResults=max_results,
        )
        response = request.execute()

        # Parse results into Pydantic models, then convert to dicts
        results = []
        for item in response.get("items", []):
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]

            # Extract thumbnail (prefer high quality, fallback to default)
            thumbnails = snippet.get("thumbnails", {})
            thumbnail_url = (
                thumbnails.get("high", {}).get("url")
                or thumbnails.get("medium", {}).get("url")
                or thumbnails.get("default", {}).get("url", "")
            )

            # Build VideoSearchResult model
            video_result = VideoSearchResult(
                title=snippet.get("title", ""),
                description=snippet.get("description", ""),
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                thumbnail=thumbnail_url,
                channel_title=snippet.get("channelTitle", ""),
                published_at=snippet.get("publishedAt", ""),
            )

            results.append(video_result.model_dump())

        logger.info(f"Found {len(results)} live video results for query: {query!r}")
        return results

    except HttpError as e:
        logger.error(f"YouTube API error during live video search: {e}")
        handle_youtube_api_error(e)
        # handle_youtube_api_error always raises, but type checker needs this
        raise


__all__ = [
    "search_channels",
    "search_live_videos",
    "search_videos",
]
