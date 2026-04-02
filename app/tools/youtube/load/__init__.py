"""Load layer for YouTube MCP ELT pipeline.

This module provides caching functionality for the ELT pipeline,
storing raw data from the Extract layer into RefCache for efficient
reuse across multiple Transform operations.

Key components:
- cache: Core caching functions (extract_or_cache, load_to_cache, get_cached)
- namespaces: Namespace constants for organizing cached data
- ttl: TTL configurations for different data types

The Load layer enables:
1. Quota efficiency - Extract once, transform many times
2. Fast subsequent requests - No API calls needed
3. Data reusability - Same raw data feeds multiple analyses
4. Clear organization - Hierarchical namespace structure

Example:
    >>> from app.tools.youtube.load import (
    ...     extract_or_cache,
    ...     build_cache_key,
    ...     RAW_SEARCH_VIDEOS,
    ... )
    >>>
    >>> key = build_cache_key(RAW_SEARCH_VIDEOS, "kubernetes")
    >>> videos = await extract_or_cache(RAW_SEARCH_VIDEOS, key, fetch_func)
"""

from __future__ import annotations

from app.tools.youtube.load.cache import (
    build_batch_key,
    build_cache_key,
    channel_info_batch_key,
    channel_info_key,
    channel_search_key,
    channel_videos_key,
    comments_key,
    extract_or_cache,
    get_cached,
    hash_key,
    invalidate_cache,
    invalidate_namespace,
    load_to_cache,
    transcript_key,
    trending_key,
    video_details_batch_key,
    video_details_key,
    video_search_key,
)
from app.tools.youtube.load.namespaces import (
    ALL_NAMESPACES,
    INTELLIGENCE_BENCHMARKS,
    INTELLIGENCE_COMPETITION,
    INTELLIGENCE_CONTENT_GAPS,
    INTELLIGENCE_NAMESPACES,
    INTELLIGENCE_NAMING,
    INTELLIGENCE_NICHE,
    INTELLIGENCE_TITLES,
    INTELLIGENCE_TRENDING,
    RAW_CHANNEL_INFO,
    RAW_CHANNEL_INFO_BATCH,
    RAW_CHANNEL_VIDEOS,
    RAW_COMMENTS,
    RAW_NAMESPACES,
    RAW_SEARCH_CHANNELS,
    RAW_SEARCH_VIDEOS,
    RAW_TRANSCRIPTS,
    RAW_TRENDING,
    RAW_VIDEO_DETAILS,
    RAW_VIDEO_DETAILS_BATCH,
    TRANSFORM_GAPS,
    TRANSFORM_NAMESPACES,
    TRANSFORM_PATTERNS,
    TRANSFORM_SCORES,
    TRANSFORM_STATISTICS,
)
from app.tools.youtube.load.ttl import (
    ALL_TTL,
    INTELLIGENCE_BENCHMARKS_TTL,
    INTELLIGENCE_COMPETITION_TTL,
    INTELLIGENCE_CONTENT_GAPS_TTL,
    INTELLIGENCE_NAME_CHECK_TTL,
    INTELLIGENCE_NAMING_TTL,
    INTELLIGENCE_NICHE_TTL,
    INTELLIGENCE_TITLES_TTL,
    INTELLIGENCE_TRENDING_TTL,
    INTELLIGENCE_TTL,
    RAW_CHANNEL_INFO_TTL,
    RAW_CHANNEL_VIDEOS_TTL,
    RAW_COMMENTS_TTL,
    RAW_SEARCH_CHANNELS_TTL,
    RAW_SEARCH_VIDEOS_TTL,
    RAW_TRANSCRIPTS_TTL,
    RAW_TRENDING_TTL,
    RAW_TTL,
    RAW_VIDEO_DETAILS_TTL,
    TRANSFORM_GAPS_TTL,
    TRANSFORM_PATTERNS_TTL,
    TRANSFORM_SCORES_TTL,
    TRANSFORM_STATISTICS_TTL,
    TRANSFORM_TTL,
    get_ttl,
    get_ttl_seconds,
)

__all__ = [
    # Namespace constants - Raw
    "ALL_NAMESPACES",
    # TTL constants - Raw
    "ALL_TTL",
    # Namespace constants - Intelligence
    "INTELLIGENCE_BENCHMARKS",
    # TTL constants - Intelligence
    "INTELLIGENCE_BENCHMARKS_TTL",
    "INTELLIGENCE_COMPETITION",
    "INTELLIGENCE_COMPETITION_TTL",
    "INTELLIGENCE_CONTENT_GAPS",
    "INTELLIGENCE_CONTENT_GAPS_TTL",
    "INTELLIGENCE_NAMESPACES",
    "INTELLIGENCE_NAME_CHECK_TTL",
    "INTELLIGENCE_NAMING",
    "INTELLIGENCE_NAMING_TTL",
    "INTELLIGENCE_NICHE",
    "INTELLIGENCE_NICHE_TTL",
    "INTELLIGENCE_TITLES",
    "INTELLIGENCE_TITLES_TTL",
    "INTELLIGENCE_TRENDING",
    "INTELLIGENCE_TRENDING_TTL",
    "INTELLIGENCE_TTL",
    "RAW_CHANNEL_INFO",
    "RAW_CHANNEL_INFO_BATCH",
    "RAW_CHANNEL_INFO_TTL",
    "RAW_CHANNEL_VIDEOS",
    "RAW_CHANNEL_VIDEOS_TTL",
    "RAW_COMMENTS",
    "RAW_COMMENTS_TTL",
    "RAW_NAMESPACES",
    "RAW_SEARCH_CHANNELS",
    "RAW_SEARCH_CHANNELS_TTL",
    "RAW_SEARCH_VIDEOS",
    "RAW_SEARCH_VIDEOS_TTL",
    "RAW_TRANSCRIPTS",
    "RAW_TRANSCRIPTS_TTL",
    "RAW_TRENDING",
    "RAW_TRENDING_TTL",
    "RAW_TTL",
    "RAW_VIDEO_DETAILS",
    "RAW_VIDEO_DETAILS_BATCH",
    "RAW_VIDEO_DETAILS_TTL",
    # Namespace constants - Transform
    "TRANSFORM_GAPS",
    # TTL constants - Transform
    "TRANSFORM_GAPS_TTL",
    "TRANSFORM_NAMESPACES",
    "TRANSFORM_PATTERNS",
    "TRANSFORM_PATTERNS_TTL",
    "TRANSFORM_SCORES",
    "TRANSFORM_SCORES_TTL",
    "TRANSFORM_STATISTICS",
    "TRANSFORM_STATISTICS_TTL",
    "TRANSFORM_TTL",
    # Cache functions
    "build_batch_key",
    "build_cache_key",
    "channel_info_batch_key",
    "channel_info_key",
    "channel_search_key",
    "channel_videos_key",
    "comments_key",
    "extract_or_cache",
    "get_cached",
    "get_ttl",
    "get_ttl_seconds",
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
