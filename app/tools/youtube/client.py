"""YouTube API client factory and utilities.

This module provides a factory function for creating authenticated YouTube API clients
and utilities for handling API errors, quota limits, and rate limiting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError  # noqa: TC002

from app.config import get_settings

if TYPE_CHECKING:
    from googleapiclient.discovery import Resource

logger = logging.getLogger(__name__)


class YouTubeAPIError(Exception):
    """Base exception for YouTube API errors."""

    pass


class YouTubeQuotaExceededError(YouTubeAPIError):
    """Raised when YouTube API quota is exceeded."""

    pass


class YouTubeNotFoundError(YouTubeAPIError):
    """Raised when a requested resource is not found."""

    pass


class YouTubeAuthError(YouTubeAPIError):
    """Raised when authentication fails."""

    pass


def get_youtube_service() -> Resource:
    """Create and return an authenticated YouTube API service client.

    Returns:
        Authenticated YouTube API v3 service client.

    Raises:
        YouTubeAuthError: If API key is not configured or authentication fails.
        YouTubeAPIError: If client creation fails for other reasons.

    Example:
        >>> youtube = get_youtube_service()
        >>> response = youtube.videos().list(part="snippet", id="dQw4w9WgXcQ").execute()
    """
    settings = get_settings()

    if not settings.youtube_api_key:
        logger.error("YouTube API key not configured")
        raise YouTubeAuthError(
            "YouTube API key not configured. Set YOUTUBE_API_KEY environment variable."
        )

    try:
        logger.debug("Building YouTube API service client")
        service = build("youtube", "v3", developerKey=settings.youtube_api_key)
        logger.debug("YouTube API service client created successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to build YouTube service: {e!s}", exc_info=True)
        raise YouTubeAPIError(f"Failed to build YouTube service: {e!s}") from e


def handle_youtube_api_error(error: HttpError) -> None:
    """Handle YouTube API HTTP errors and raise appropriate exceptions.

    Args:
        error: The HttpError from googleapiclient.

    Raises:
        YouTubeQuotaExceededError: If quota limit is exceeded.
        YouTubeAuthError: If authentication fails.
        YouTubeNotFoundError: If resource is not found.
        YouTubeAPIError: For other API errors.
    """
    status_code = error.resp.status
    error_content = error.content.decode("utf-8") if error.content else str(error)

    logger.error(f"YouTube API error (status {status_code}): {error_content}")

    # Quota exceeded (403 with specific reason)
    if status_code == 403 and "quota" in error_content.lower():
        logger.error("YouTube API quota exceeded")
        raise YouTubeQuotaExceededError(
            "YouTube API quota exceeded. Quota resets at midnight Pacific Time. "
            "Consider enabling billing in Google Cloud Console for higher limits."
        ) from error

    # Authentication/authorization errors
    if status_code in (401, 403):
        logger.error("YouTube API authentication error")
        raise YouTubeAuthError(
            f"YouTube API authentication failed: {error_content}"
        ) from error

    # Resource not found
    if status_code == 404:
        logger.warning("YouTube resource not found")
        raise YouTubeNotFoundError(
            "Requested YouTube resource not found (video, channel, or playlist may not exist)"
        ) from error

    # Generic API error
    raise YouTubeAPIError(
        f"YouTube API error (status {status_code}): {error_content}"
    ) from error


def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from various YouTube URL formats or return ID as-is.

    Supports:
    - Direct video IDs (e.g., "dQw4w9WgXcQ")
    - Watch URLs (e.g., "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    - Short URLs (e.g., "https://youtu.be/dQw4w9WgXcQ")
    - Embed URLs (e.g., "https://www.youtube.com/embed/dQw4w9WgXcQ")

    Args:
        url_or_id: YouTube URL or video ID.

    Returns:
        Extracted video ID.

    Raises:
        ValueError: If video ID cannot be extracted.

    Example:
        >>> extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    url_or_id = url_or_id.strip()

    # Already a video ID (11 characters, alphanumeric + - and _)
    if len(url_or_id) == 11 and all(c.isalnum() or c in "-_" for c in url_or_id):
        return url_or_id

    # Extract from various URL formats
    try:
        # Watch URL: https://www.youtube.com/watch?v=VIDEO_ID
        if "youtube.com/watch" in url_or_id:
            return url_or_id.split("v=")[1].split("&")[0]

        # Short URL: https://youtu.be/VIDEO_ID
        if "youtu.be/" in url_or_id:
            return url_or_id.split("youtu.be/")[1].split("?")[0]

        # Embed URL: https://www.youtube.com/embed/VIDEO_ID
        if "youtube.com/embed/" in url_or_id:
            return url_or_id.split("embed/")[1].split("?")[0]

    except (IndexError, AttributeError) as e:
        logger.error(f"Failed to extract video ID from: {url_or_id}")
        raise ValueError(f"Could not extract video ID from: {url_or_id}") from e

    # Fallback: invalid format
    raise ValueError(f"Invalid YouTube URL or video ID format: {url_or_id}")


def extract_channel_id(url_or_id: str) -> str:
    """Extract channel ID from various YouTube URL formats or return ID as-is.

    Supports:
    - Direct channel IDs (e.g., "UCxyz...")
    - Channel URLs (e.g., "https://www.youtube.com/channel/UCxyz...")
    - Custom URLs (e.g., "@username") - Note: requires API lookup, returns as-is

    Args:
        url_or_id: YouTube URL or channel ID.

    Returns:
        Extracted channel ID or custom handle.

    Raises:
        ValueError: If channel ID cannot be extracted.

    Example:
        >>> extract_channel_id("https://www.youtube.com/channel/UCxyz")
        'UCxyz'
        >>> extract_channel_id("UCxyz")
        'UCxyz'
    """
    url_or_id = url_or_id.strip()

    # Already a channel ID (starts with UC, typically 24 chars)
    if url_or_id.startswith("UC") and len(url_or_id) >= 20:
        return url_or_id

    # Custom handle (@username)
    if url_or_id.startswith("@"):
        return url_or_id

    # Extract from URL formats
    try:
        # Channel URL: https://www.youtube.com/channel/CHANNEL_ID
        if "youtube.com/channel/" in url_or_id:
            return url_or_id.split("channel/")[1].split("?")[0].split("/")[0]

        # User URL (legacy): https://www.youtube.com/user/USERNAME
        # Note: This returns the username, which needs API lookup to get channel ID
        if "youtube.com/user/" in url_or_id:
            return url_or_id.split("user/")[1].split("?")[0].split("/")[0]

        # Custom URL: https://www.youtube.com/@username
        if "youtube.com/@" in url_or_id:
            return "@" + url_or_id.split("@")[1].split("?")[0].split("/")[0]

    except (IndexError, AttributeError) as e:
        logger.error(f"Failed to extract channel ID from: {url_or_id}")
        raise ValueError(f"Could not extract channel ID from: {url_or_id}") from e

    # Fallback: invalid format
    raise ValueError(f"Invalid YouTube channel URL or ID format: {url_or_id}")


__all__ = [
    "YouTubeAPIError",
    "YouTubeAuthError",
    "YouTubeNotFoundError",
    "YouTubeQuotaExceededError",
    "extract_channel_id",
    "extract_video_id",
    "get_youtube_service",
    "handle_youtube_api_error",
]
