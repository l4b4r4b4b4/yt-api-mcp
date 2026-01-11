"""YouTube comment retrieval tools.

This module provides tools for fetching comments from YouTube videos with
intelligent caching and error handling.
"""

from __future__ import annotations

from typing import Any

from googleapiclient.errors import HttpError

from app.tools.youtube.client import get_youtube_service
from app.tools.youtube.models import CommentData


async def get_video_comments(video_id: str, max_results: int = 20) -> dict[str, Any]:
    """Get top comments for a YouTube video.

    Retrieves top-level comments (no replies) sorted by relevance.
    Returns empty list if comments are disabled for the video.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ").
        max_results: Maximum number of comments to return (1-100, default: 20).

    Returns:
        Dictionary with:
            - video_id: The video ID
            - comments: List of CommentData objects
            - total_returned: Number of comments returned

    Raises:
        ValueError: If video_id is invalid or other API errors occur.

    Example:
        >>> comments = await get_video_comments("nLwbNhSxLd4", max_results=10)
        >>> print(comments["comments"][0]["author"])
        >>> print(len(comments["comments"]))
    """
    # Clamp max_results to valid range (1-100)
    max_results = min(max(1, max_results), 100)

    try:
        youtube = get_youtube_service()

        # Fetch comments from YouTube API
        response = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                textFormat="plainText",
                order="relevance",  # Get most relevant comments first
            )
            .execute()
        )

        # Parse comments from response
        comments = []
        for item in response.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]["snippet"]

            comment_data = CommentData(
                author=top_comment["authorDisplayName"],
                text=top_comment["textDisplay"],
                like_count=int(top_comment.get("likeCount", 0)),
                published_at=top_comment["publishedAt"],
                reply_count=int(item["snippet"].get("totalReplyCount", 0)),
            )

            comments.append(comment_data.model_dump())

        return {
            "video_id": video_id,
            "comments": comments,
            "total_returned": len(comments),
        }

    except HttpError as e:
        # Handle comments disabled gracefully (very common on YouTube)
        error_content = (
            e.content.decode("utf-8")
            if isinstance(e.content, bytes)
            else str(e.content)
        )
        if "commentsDisabled" in error_content or "commentsDisabled" in str(e):
            return {
                "video_id": video_id,
                "comments": [],
                "total_returned": 0,
            }

        # Handle other HTTP errors
        error_msg = f"Failed to get video comments: {e!s}"
        raise ValueError(error_msg) from e


__all__ = ["get_video_comments"]
