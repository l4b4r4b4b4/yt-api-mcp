"""Comment extraction functions for YouTube MCP ELT pipeline.

This module provides raw data extraction from the YouTube API for comment-related
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

from app.tools.youtube.comments import get_video_comments
from app.tools.youtube.load.cache import comments_key, extract_or_cache
from app.tools.youtube.load.namespaces import RAW_COMMENTS

logger = logging.getLogger(__name__)


async def extract_comments_raw(
    video_id: str,
    max_results: int = 100,
    cache: Any | None = None,
) -> dict[str, Any]:
    """Extract raw comments from a YouTube video.

    Wraps the get_video_comments function for the ELT pipeline.
    Returns raw comment data without any transformation.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ").
        max_results: Maximum number of comments (1-100, default: 100).
        cache: Optional RefCache instance for caching results.

    Returns:
        Dictionary containing:
        - video_id: The video ID
        - comments: List of comment dictionaries with:
            - author: Comment author display name
            - text: Comment text content
            - like_count: Number of likes on the comment
            - published_at: ISO 8601 timestamp
            - reply_count: Number of replies
        - total_returned: Number of comments returned

    Raises:
        ValueError: If video_id is invalid or API error occurs.

    Example:
        >>> result = await extract_comments_raw("dQw4w9WgXcQ", max_results=50)
        >>> print(len(result["comments"]))
        50
        >>> print(result["comments"][0]["text"])
        "Great video!"

    Note:
        This function costs 1 quota unit per request.
        Returns empty list if comments are disabled on the video.
        Results are cached when cache parameter is provided.
    """
    logger.info(f"Extracting comments for video: {video_id}, max_results={max_results}")

    # If no cache, extract directly
    if cache is None:
        return await get_video_comments(video_id, max_results=max_results)

    # Build cache key
    key = comments_key(video_id, max_results)

    # Extract or use cached
    async def extractor() -> dict[str, Any]:
        return await get_video_comments(video_id, max_results=max_results)

    return await extract_or_cache(RAW_COMMENTS, key, extractor, cache)


async def extract_comments_batch(
    video_ids: list[str],
    max_results_per_video: int = 50,
    cache: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract comments from multiple videos.

    Fetches comments for multiple videos, returning a dictionary
    keyed by video_id. Useful for analyzing comment patterns
    across a set of videos.

    Args:
        video_ids: List of YouTube video IDs.
        max_results_per_video: Max comments per video (1-100, default: 50).
        cache: Optional RefCache instance for caching results.

    Returns:
        Dictionary mapping video_id to comment result dictionary.
        Videos with disabled comments will have empty comment lists.

    Example:
        >>> results = await extract_comments_batch(["id1", "id2", "id3"])
        >>> print(len(results["id1"]["comments"]))
        50
        >>> print(results["id2"]["total_returned"])
        0  # Comments disabled

    Note:
        This function costs 1 quota unit per video.
        Failed videos are included with empty results.
        Each video's comments are cached individually for reusability.
    """
    if not video_ids:
        return {}

    logger.info(
        f"Extracting comments for {len(video_ids)} videos, "
        f"max_results_per_video={max_results_per_video}"
    )

    results = {}
    success_count = 0

    for video_id in video_ids:
        try:
            result = await extract_comments_raw(
                video_id, max_results=max_results_per_video, cache=cache
            )
            results[video_id] = result
            if result.get("total_returned", 0) > 0:
                success_count += 1
        except Exception as e:
            logger.warning(f"Failed to get comments for video {video_id}: {e}")
            # Include empty result for failed videos
            results[video_id] = {
                "video_id": video_id,
                "comments": [],
                "total_returned": 0,
                "error": str(e),
            }

    logger.info(
        f"Successfully extracted comments from {success_count}/{len(video_ids)} videos"
    )
    return results


async def extract_top_comments(
    video_id: str,
    max_results: int = 20,
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract top comments from a video (convenience wrapper).

    Returns just the comment list (not the full result dict).
    Useful when you only need the comments themselves.

    Args:
        video_id: YouTube video ID.
        max_results: Maximum number of comments (1-100, default: 20).
        cache: Optional RefCache instance for caching results.

    Returns:
        List of comment dictionaries, sorted by relevance.
        Empty list if comments are disabled.

    Example:
        >>> comments = await extract_top_comments("dQw4w9WgXcQ", max_results=10)
        >>> for comment in comments[:3]:
        ...     print(f"{comment['author']}: {comment['text'][:50]}...")
    """
    result = await extract_comments_raw(video_id, max_results=max_results, cache=cache)
    return result.get("comments", [])


__all__ = [
    "extract_comments_batch",
    "extract_comments_raw",
    "extract_top_comments",
]
