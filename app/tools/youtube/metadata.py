"""YouTube metadata tools - video and channel details.

This module provides tools for retrieving detailed metadata about YouTube videos
and channels, including statistics, descriptions, and other information not
available in search results.
"""

from __future__ import annotations

from typing import Any

from app.tools.youtube.client import YouTubeAPIError, get_youtube_service
from app.tools.youtube.models import ChannelInfo, VideoDetails


async def get_video_details(video_id: str) -> dict[str, Any]:
    """Get detailed information about a YouTube video.

    Retrieves comprehensive video metadata including title, description,
    statistics (views, likes, comments), duration, tags, and channel info.
    Results should be cached for 24 hours via the youtube.api namespace.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")

    Returns:
        VideoDetails dictionary with all available metadata including:
        - title, description, video_id, url, thumbnail
        - view_count, like_count, comment_count
        - duration (ISO 8601 format), tags
        - channel_title, published_at

    Raises:
        ValueError: If video_id is empty or invalid format
        YouTubeAPIError: If video not found or API error occurs

    Example:
        >>> details = await get_video_details("dQw4w9WgXcQ")
        >>> print(details["title"])
        "Never Gonna Give You Up"
        >>> print(details["view_count"])
        "1000000000"
    """
    if not video_id or not isinstance(video_id, str):
        raise ValueError("video_id must be a non-empty string")

    # Basic validation of video ID format (11 characters, alphanumeric + - and _)
    if len(video_id) != 11:
        raise ValueError(
            f"Invalid video ID format: {video_id}. "
            "YouTube video IDs are 11 characters long."
        )

    youtube = get_youtube_service()

    try:
        # Request video details with all relevant parts
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails", id=video_id
        )
        response = request.execute()

        # Check if video was found
        if not response.get("items"):
            raise YouTubeAPIError(
                f"Video not found: {video_id}. "
                "The video may be private, deleted, or the ID is incorrect."
            )

        video_data = response["items"][0]
        snippet = video_data.get("snippet", {})
        statistics = video_data.get("statistics", {})
        content_details = video_data.get("contentDetails", {})

        # Extract thumbnail (prefer high quality)
        thumbnails = snippet.get("thumbnails", {})
        thumbnail_url = (
            thumbnails.get("high", {}).get("url")
            or thumbnails.get("medium", {}).get("url")
            or thumbnails.get("default", {}).get("url")
            or "https://i.ytimg.com/vi/default.jpg"
        )

        # Build the video details dict
        video_details = VideoDetails(
            title=snippet.get("title", ""),
            description=snippet.get("description", ""),
            video_id=video_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
            thumbnail=thumbnail_url,
            channel_title=snippet.get("channelTitle", ""),
            published_at=snippet.get("publishedAt", ""),
            view_count=statistics.get("viewCount", "0"),
            like_count=statistics.get("likeCount", "0"),
            comment_count=statistics.get("commentCount", "0"),
            duration=content_details.get("duration", "PT0S"),
            tags=snippet.get("tags", []),
        )

        return video_details.model_dump()

    except YouTubeAPIError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        # Wrap other exceptions with context
        raise YouTubeAPIError(
            f"Failed to retrieve video details for {video_id}: {e!s}"
        ) from e


async def get_channel_info(channel_id: str) -> dict[str, Any]:
    """Get detailed information about a YouTube channel.

    Retrieves channel metadata including title, description, statistics
    (subscribers, videos, total views), and branding information.
    Results should be cached for 24 hours via the youtube.api namespace.

    Args:
        channel_id: YouTube channel ID (e.g., "UCuAXFkgsw1L7xaCfnd5JJOw")

    Returns:
        ChannelInfo dictionary with all available metadata including:
        - title, description, channel_id, url, thumbnail
        - subscriber_count, video_count, view_count
        - published_at

    Raises:
        ValueError: If channel_id is empty or invalid format
        YouTubeAPIError: If channel not found or API error occurs

    Example:
        >>> info = await get_channel_info("UCuAXFkgsw1L7xaCfnd5JJOw")
        >>> print(info["title"])
        "Vimjoyer"
        >>> print(info["subscriber_count"])
        "50000"
    """
    if not channel_id or not isinstance(channel_id, str):
        raise ValueError("channel_id must be a non-empty string")

    # Basic validation of channel ID format (starts with UC, 24 chars total)
    if not channel_id.startswith("UC") or len(channel_id) != 24:
        raise ValueError(
            f"Invalid channel ID format: {channel_id}. "
            "YouTube channel IDs start with 'UC' and are 24 characters long."
        )

    youtube = get_youtube_service()

    try:
        # Request channel details with all relevant parts
        request = youtube.channels().list(
            part="snippet,statistics,brandingSettings", id=channel_id
        )
        response = request.execute()

        # Check if channel was found
        if not response.get("items"):
            raise YouTubeAPIError(
                f"Channel not found: {channel_id}. "
                "The channel may not exist or the ID is incorrect."
            )

        channel_data = response["items"][0]
        snippet = channel_data.get("snippet", {})
        statistics = channel_data.get("statistics", {})

        # Extract thumbnail (prefer high quality)
        thumbnails = snippet.get("thumbnails", {})
        thumbnail_url = (
            thumbnails.get("high", {}).get("url")
            or thumbnails.get("medium", {}).get("url")
            or thumbnails.get("default", {}).get("url")
            or "https://yt3.ggpht.com/default_channel.jpg"
        )

        # Handle hidden subscriber count
        subscriber_count = statistics.get("subscriberCount", "0")
        # If hiddenSubscriberCount is true, subscriberCount might not be present
        if statistics.get("hiddenSubscriberCount", False):
            subscriber_count = "0"  # Return 0 if hidden

        # Build the channel info dict
        channel_info = ChannelInfo(
            title=snippet.get("title", ""),
            description=snippet.get("description", ""),
            channel_id=channel_id,
            url=f"https://www.youtube.com/channel/{channel_id}",
            thumbnail=thumbnail_url,
            subscriber_count=subscriber_count,
            video_count=statistics.get("videoCount", "0"),
            view_count=statistics.get("viewCount", "0"),
            published_at=snippet.get("publishedAt", ""),
        )

        return channel_info.model_dump()

    except YouTubeAPIError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        # Wrap other exceptions with context
        raise YouTubeAPIError(
            f"Failed to retrieve channel info for {channel_id}: {e!s}"
        ) from e


__all__ = ["get_channel_info", "get_video_details"]
