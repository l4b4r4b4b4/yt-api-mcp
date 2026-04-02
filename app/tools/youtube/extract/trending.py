"""Trending video extraction functions for YouTube MCP ELT pipeline.

This module provides raw data extraction from the YouTube API for trending
video data. These functions return unprocessed data suitable for caching
in the Load layer.

Extract layer principles:
1. Return raw data exactly as received from API
2. No transformation or analysis
3. Designed for caching - consistent output format
4. Quota-consuming operations that benefit from caching

Note:
    YouTube's trending API (videos.list with chart=mostPopular) requires
    specific parameters and returns region-specific results.
"""

from __future__ import annotations

import logging
from typing import Any

from googleapiclient.errors import HttpError

from app.tools.youtube.client import get_youtube_service, handle_youtube_api_error
from app.tools.youtube.load.cache import extract_or_cache, trending_key
from app.tools.youtube.load.namespaces import RAW_TRENDING
from app.tools.youtube.models import VideoSearchResult

logger = logging.getLogger(__name__)

# YouTube category IDs for common content types
# See: https://developers.google.com/youtube/v3/docs/videoCategories/list
CATEGORY_IDS = {
    "film_animation": "1",
    "autos_vehicles": "2",
    "music": "10",
    "pets_animals": "15",
    "sports": "17",
    "travel_events": "19",
    "gaming": "20",
    "people_blogs": "22",
    "comedy": "23",
    "entertainment": "24",
    "news_politics": "25",
    "howto_style": "26",
    "education": "27",
    "science_technology": "28",
    "nonprofits_activism": "29",
}


async def extract_trending_raw(
    region: str = "US",
    category_id: str | None = None,
    max_results: int = 50,
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract raw trending videos from YouTube API.

    Fetches the most popular (trending) videos for a specific region
    and optionally a specific category.

    Args:
        region: ISO 3166-1 alpha-2 country code (e.g., "US", "GB", "DE").
        category_id: Optional YouTube category ID (e.g., "28" for Science & Tech).
            Use CATEGORY_IDS dict for common categories.
        max_results: Maximum number of results (1-50, default: 50).
        cache: Optional RefCache instance for caching results.

    Returns:
        List of raw video dictionaries containing:
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
        >>> # Get trending videos in the US
        >>> videos = await extract_trending_raw("US", max_results=20)
        >>> print(len(videos))
        20

        >>> # Get trending tech videos in Germany
        >>> tech_videos = await extract_trending_raw(
        ...     region="DE",
        ...     category_id=CATEGORY_IDS["science_technology"],
        ...     max_results=10
        ... )
        >>> print(tech_videos[0]["title"])
        "Top Tech News..."

    Note:
        This function costs 1 quota unit per request (very cheap!).
        Unlike search (100 units), trending is cost-effective.
        Results are cached when cache parameter is provided.
    """
    # Clamp max_results to valid range
    max_results = max(1, min(50, max_results))

    logger.info(
        f"Extracting trending videos: region={region}, "
        f"category_id={category_id}, max_results={max_results}"
    )

    # If no cache, extract directly
    if cache is None:
        return await _fetch_trending_videos(region, category_id, max_results)

    # Build cache key
    key = trending_key(region, category_id)

    # Extract or use cached
    async def extractor() -> list[dict[str, Any]]:
        return await _fetch_trending_videos(region, category_id, max_results)

    return await extract_or_cache(RAW_TRENDING, key, extractor, cache)


async def _fetch_trending_videos(
    region: str,
    category_id: str | None,
    max_results: int,
) -> list[dict[str, Any]]:
    """Internal function to fetch trending videos from YouTube API.

    This is separated from extract_trending_raw to allow caching.
    """
    try:
        youtube = get_youtube_service()

        # Build request parameters
        request_params = {
            "part": "snippet",
            "chart": "mostPopular",
            "regionCode": region,
            "maxResults": max_results,
        }

        # Add category filter if specified
        if category_id:
            request_params["videoCategoryId"] = category_id

        # Execute request
        request = youtube.videos().list(**request_params)
        response = request.execute()

        # Parse results
        results = []
        for item in response.get("items", []):
            video_id = item["id"]
            snippet = item.get("snippet", {})

            # Extract thumbnail (prefer high quality)
            thumbnails = snippet.get("thumbnails", {})
            thumbnail_url = (
                thumbnails.get("high", {}).get("url")
                or thumbnails.get("medium", {}).get("url")
                or thumbnails.get("default", {}).get("url", "")
            )

            # Build video result using existing model
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

        logger.info(
            f"Found {len(results)} trending videos for region={region}, "
            f"category_id={category_id}"
        )
        return results

    except HttpError as e:
        logger.error(f"YouTube API error during trending fetch: {e}")
        handle_youtube_api_error(e)
        raise


async def extract_trending_by_category(
    region: str = "US",
    categories: list[str] | None = None,
    max_results_per_category: int = 10,
    cache: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Extract trending videos across multiple categories.

    Fetches trending videos for multiple content categories,
    returning results organized by category.

    Args:
        region: ISO 3166-1 alpha-2 country code (default: "US").
        categories: List of category keys from CATEGORY_IDS.
            If None, uses ["science_technology", "education", "howto_style"].
        max_results_per_category: Max videos per category (1-50, default: 10).
        cache: Optional RefCache instance for caching results.

    Returns:
        Dictionary mapping category name to list of video dictionaries.

    Example:
        >>> results = await extract_trending_by_category(
        ...     region="US",
        ...     categories=["science_technology", "gaming"],
        ...     max_results_per_category=5
        ... )
        >>> print(len(results["science_technology"]))
        5
        >>> print(len(results["gaming"]))
        5

    Note:
        This function costs 1 quota unit per category.
        Total cost = len(categories) quota units.
        Each category's results are cached individually for reusability.
    """
    if categories is None:
        categories = ["science_technology", "education", "howto_style"]

    logger.info(
        f"Extracting trending videos for {len(categories)} categories in {region}"
    )

    results = {}
    for category in categories:
        category_id = CATEGORY_IDS.get(category)
        if not category_id:
            logger.warning(f"Unknown category: {category}, skipping")
            continue

        try:
            videos = await extract_trending_raw(
                region=region,
                category_id=category_id,
                max_results=max_results_per_category,
                cache=cache,
            )
            results[category] = videos
        except Exception as e:
            logger.warning(f"Failed to get trending for category {category}: {e}")
            results[category] = []

    logger.info(f"Successfully extracted trending for {len(results)} categories")
    return results


__all__ = [
    "CATEGORY_IDS",
    "extract_trending_by_category",
    "extract_trending_raw",
]
