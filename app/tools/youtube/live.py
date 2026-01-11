"""YouTube live streaming functionality for broadcasts and live chat.

This module provides tools for monitoring live streams, checking broadcast status,
and retrieving live chat messages. All functions are designed to work with
mcp-refcache for intelligent result caching with very short TTLs for real-time data.
"""

from __future__ import annotations

import logging
from typing import Any

from googleapiclient.errors import HttpError

from app.tools.youtube.client import get_youtube_service, handle_youtube_api_error
from app.tools.youtube.models import LiveChatMessage, LiveChatResponse, LiveStatus

logger = logging.getLogger(__name__)


async def is_live(video_id: str) -> dict[str, Any]:
    """Check if a YouTube video is currently live.

    Queries the YouTube Data API v3 to determine if a video is currently
    broadcasting live. Returns live status with viewer count and timing information.

    Args:
        video_id: YouTube video ID to check (e.g., "dQw4w9WgXcQ").

    Returns:
        Dictionary containing live status information:
        - video_id: YouTube video ID
        - is_live: Boolean indicating if video is currently live
        - viewer_count: Current concurrent viewers (None if not live)
        - scheduled_start_time: ISO 8601 scheduled start time (None if not scheduled)
        - actual_start_time: ISO 8601 actual start time (None if not started)
        - active_live_chat_id: Live chat ID (None if no chat or not live)

    Raises:
        YouTubeQuotaExceededError: If API quota is exceeded.
        YouTubeAuthError: If authentication fails.
        YouTubeNotFoundError: If video not found.
        YouTubeAPIError: For other API errors.

    Example:
        >>> status = await is_live("dQw4w9WgXcQ")
        >>> if status["is_live"]:
        ...     print(f"Live now with {status['viewer_count']} viewers!")
        >>> else:
        ...     print("Not currently live")

    Note:
        This function is designed to be decorated with @cache.cached in server.py
        with a very short TTL (30 seconds) since live status changes quickly.
        Costs 1 quota unit per request.
    """
    logger.info(f"Checking live status for video: {video_id}")

    try:
        # Get authenticated YouTube service
        youtube = get_youtube_service()

        # Request liveStreamingDetails part to check broadcast status
        request = youtube.videos().list(
            part="liveStreamingDetails",
            id=video_id,
        )
        response = request.execute()

        # Check if video exists
        items = response.get("items", [])
        if not items:
            logger.warning(f"Video not found: {video_id}")
            # Return not live status (graceful handling)
            live_status = LiveStatus(
                video_id=video_id,
                is_live=False,
                viewer_count=None,
                scheduled_start_time=None,
                actual_start_time=None,
                active_live_chat_id=None,
            )
            return live_status.model_dump()

        video = items[0]

        # Check if video has liveStreamingDetails (indicates it's a broadcast)
        live_details = video.get("liveStreamingDetails")
        if not live_details:
            logger.info(f"Video {video_id} is not a live broadcast")
            # Return not live status
            live_status = LiveStatus(
                video_id=video_id,
                is_live=False,
                viewer_count=None,
                scheduled_start_time=None,
                actual_start_time=None,
                active_live_chat_id=None,
            )
            return live_status.model_dump()

        # Extract live streaming information
        active_chat_id = live_details.get("activeLiveChatId")
        is_currently_live = active_chat_id is not None

        # Parse viewer count (only present when live)
        viewer_count = None
        if is_currently_live:
            concurrent_viewers = live_details.get("concurrentViewers")
            if concurrent_viewers is not None:
                viewer_count = int(concurrent_viewers)

        # Build LiveStatus model
        live_status = LiveStatus(
            video_id=video_id,
            is_live=is_currently_live,
            viewer_count=viewer_count,
            scheduled_start_time=live_details.get("scheduledStartTime"),
            actual_start_time=live_details.get("actualStartTime"),
            active_live_chat_id=active_chat_id,
        )

        logger.info(
            f"Live status for {video_id}: is_live={is_currently_live}, "
            f"viewers={viewer_count}"
        )
        return live_status.model_dump()

    except HttpError as e:
        logger.error(f"YouTube API error checking live status: {e}")
        handle_youtube_api_error(e)
        # handle_youtube_api_error always raises, but type checker needs this
        raise


async def get_live_chat_id(video_id: str) -> dict[str, Any]:
    """Get the live chat ID for a currently streaming video.

    Retrieves the active live chat ID required for fetching chat messages.
    This ID remains constant throughout the stream's duration.

    Args:
        video_id: YouTube video ID of the live stream.

    Returns:
        Dictionary containing:
        - video_id: YouTube video ID
        - live_chat_id: Active live chat ID
        - is_live: Boolean confirming video is live

    Raises:
        YouTubeQuotaExceededError: If API quota is exceeded.
        YouTubeAuthError: If authentication fails.
        YouTubeNotFoundError: If video not found.
        YouTubeAPIError: If video is not live or chat not available.

    Example:
        >>> result = await get_live_chat_id("dQw4w9WgXcQ")
        >>> chat_id = result["live_chat_id"]
        >>> # Use chat_id with get_live_chat_messages()

    Note:
        This function is designed to be decorated with @cache.cached in server.py
        with moderate TTL (5 minutes) since chat ID doesn't change during stream.
        Costs 1 quota unit per request.

        Raises an error if video is not currently live or chat is disabled.
        Use is_live() first to check if video is broadcasting.
    """
    logger.info(f"Getting live chat ID for video: {video_id}")

    try:
        # Get authenticated YouTube service
        youtube = get_youtube_service()

        # Request liveStreamingDetails part
        request = youtube.videos().list(
            part="liveStreamingDetails",
            id=video_id,
        )
        response = request.execute()

        # Check if video exists
        items = response.get("items", [])
        if not items:
            logger.error(f"Video not found: {video_id}")
            from app.tools.youtube.client import YouTubeNotFoundError

            raise YouTubeNotFoundError(f"Video not found: {video_id}")

        video = items[0]

        # Check if video has liveStreamingDetails
        live_details = video.get("liveStreamingDetails")
        if not live_details:
            logger.error(f"Video {video_id} is not a live broadcast")
            from app.tools.youtube.client import YouTubeAPIError

            raise YouTubeAPIError(f"Video {video_id} is not a live broadcast")

        # Extract active live chat ID
        chat_id = live_details.get("activeLiveChatId")
        if not chat_id:
            logger.error(f"Video {video_id} is not currently live or chat disabled")
            from app.tools.youtube.client import YouTubeAPIError

            raise YouTubeAPIError(
                f"Video {video_id} is not currently live or live chat is disabled"
            )

        logger.info(f"Found live chat ID for {video_id}: {chat_id}")
        return {
            "video_id": video_id,
            "live_chat_id": chat_id,
            "is_live": True,
        }

    except HttpError as e:
        logger.error(f"YouTube API error getting live chat ID: {e}")
        handle_youtube_api_error(e)
        # handle_youtube_api_error always raises, but type checker needs this
        raise


async def get_live_chat_messages(
    video_id: str,
    max_results: int = 200,
    page_token: str | None = None,
) -> dict[str, Any]:
    """Get recent live chat messages from a streaming video.

    Fetches live chat messages with pagination support for efficient polling.
    Use the returned next_page_token in subsequent calls to get only new messages.

    Args:
        video_id: YouTube video ID of the live stream.
        max_results: Maximum messages to return (clamped to 1-2000, default 200).
        page_token: Pagination token from previous call (None for first call).

    Returns:
        Dictionary containing:
        - video_id: YouTube video ID
        - messages: List of LiveChatMessage dictionaries with:
            - author: Message author display name
            - text: Message text content
            - published_at: ISO 8601 timestamp
            - author_channel_id: Author's channel ID
        - total_returned: Number of messages in this response
        - next_page_token: Token for next page (None if no more)
        - polling_interval_millis: YouTube's recommended polling interval

    Raises:
        YouTubeQuotaExceededError: If API quota is exceeded.
        YouTubeAuthError: If authentication fails.
        YouTubeNotFoundError: If video not found.
        YouTubeAPIError: If video not live or chat disabled.

    Example:
        >>> # First call - get latest messages
        >>> result = await get_live_chat_messages("dQw4w9WgXcQ", max_results=50)
        >>> print(f"Got {result['total_returned']} messages")
        >>>
        >>> # Second call - get only new messages since first call
        >>> result2 = await get_live_chat_messages(
        ...     "dQw4w9WgXcQ",
        ...     max_results=50,
        ...     page_token=result["next_page_token"]
        ... )

    Note:
        This function is designed to be decorated with @cache.cached in server.py
        with very short TTL (30 seconds) for near real-time chat monitoring.
        Costs 1 quota unit per request.

        Polling Pattern:
        1. First call: No page_token → Get latest messages + next_page_token
        2. Store next_page_token
        3. Subsequent calls: Pass page_token → Get only NEW messages
        4. Repeat step 3 every 30-60 seconds for continuous monitoring

        MCP Limitation: MCP is request/response only (not streaming).
        Agent must manually call this tool repeatedly to see new messages.
        30 second cache prevents excessive API usage during polling.
    """
    # Clamp max_results to valid range (YouTube API enforces 200-2000, but we allow 1+)
    max_results = max(1, min(2000, max_results))

    logger.info(
        f"Getting live chat messages for video: {video_id}, "
        f"max_results={max_results}, page_token={'present' if page_token else 'None'}"
    )

    try:
        # First, get the live chat ID
        chat_result = await get_live_chat_id(video_id)
        chat_id = chat_result["live_chat_id"]

        # Get authenticated YouTube service
        youtube = get_youtube_service()

        # Build request parameters
        request_params: dict[str, Any] = {
            "liveChatId": chat_id,
            "part": "snippet,authorDetails",
            "maxResults": max_results,
        }
        if page_token:
            request_params["pageToken"] = page_token

        # Execute chat messages request
        request = youtube.liveChatMessages().list(**request_params)
        response = request.execute()

        # Parse messages into LiveChatMessage models
        messages = []
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            author_details = item.get("authorDetails", {})

            # Extract message text (handle superChatDetails, etc.)
            message_text = ""
            if "textMessageDetails" in snippet:
                message_text = snippet["textMessageDetails"].get("messageText", "")
            elif "superChatDetails" in snippet:
                # Super chat messages
                message_text = snippet["superChatDetails"].get("userComment", "")

            # Build LiveChatMessage model
            chat_message = LiveChatMessage(
                author=author_details.get("displayName", "Unknown"),
                text=message_text,
                published_at=snippet.get("publishedAt", ""),
                author_channel_id=author_details.get("channelId", ""),
            )
            messages.append(chat_message.model_dump())

        # Extract pagination and polling info
        next_page_token = response.get("nextPageToken")
        polling_interval = response.get("pollingIntervalMillis", 5000)

        # Build response
        chat_response = LiveChatResponse(
            video_id=video_id,
            messages=messages,
            total_returned=len(messages),
            next_page_token=next_page_token,
            polling_interval_millis=polling_interval,
        )

        logger.info(
            f"Retrieved {len(messages)} chat messages for {video_id}, "
            f"next_token={'present' if next_page_token else 'None'}"
        )
        return chat_response.model_dump()

    except HttpError as e:
        logger.error(f"YouTube API error getting live chat messages: {e}")
        handle_youtube_api_error(e)
        # handle_youtube_api_error always raises, but type checker needs this
        raise


__all__ = [
    "get_live_chat_id",
    "get_live_chat_messages",
    "is_live",
]
