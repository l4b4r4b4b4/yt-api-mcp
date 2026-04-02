"""TTL (Time-To-Live) configurations for YouTube MCP ELT pipeline.

This module defines cache expiration times for different data types in the
ELT pipeline. TTLs are optimized based on:

1. Data volatility - How often the data changes
2. Quota cost - More expensive data is cached longer
3. Freshness requirements - How stale is acceptable

TTL Strategy:
- Raw data: Longer TTLs (expensive to re-fetch, costs quota)
- Transform data: Medium TTLs (cheap to recompute from cached raw)
- Intelligence: Varies by report type (some need freshness, some don't)
"""

from __future__ import annotations

from datetime import timedelta

# =============================================================================
# RAW DATA TTLs (Extract Layer)
# =============================================================================
# These determine how long raw API responses are cached.
# Longer TTLs = fewer API calls = better quota efficiency.

RAW_SEARCH_VIDEOS_TTL = timedelta(hours=6)
"""Video search results. Moderate TTL - new videos appear frequently."""

RAW_SEARCH_CHANNELS_TTL = timedelta(hours=6)
"""Channel search results. Same as video search."""

RAW_VIDEO_DETAILS_TTL = timedelta(hours=24)
"""Video metadata/statistics. Long TTL - stats change slowly."""

RAW_CHANNEL_INFO_TTL = timedelta(hours=24)
"""Channel metadata/statistics. Long TTL - stats change slowly."""

RAW_CHANNEL_VIDEOS_TTL = timedelta(hours=6)
"""Channel video list. Moderate - new uploads matter."""

RAW_COMMENTS_TTL = timedelta(minutes=30)
"""Video comments. Short TTL - comments are dynamic."""

RAW_TRENDING_TTL = timedelta(hours=6)
"""Trending videos. Moderate - trending changes throughout day."""

RAW_TRANSCRIPTS_TTL = timedelta(days=7)
"""Video transcripts. Very long TTL - transcripts rarely change."""

# Convenience dict for programmatic access
RAW_TTL = {
    "youtube.raw.search_videos": RAW_SEARCH_VIDEOS_TTL,
    "youtube.raw.search_channels": RAW_SEARCH_CHANNELS_TTL,
    "youtube.raw.video_details": RAW_VIDEO_DETAILS_TTL,
    "youtube.raw.video_details_batch": RAW_VIDEO_DETAILS_TTL,
    "youtube.raw.channel_info": RAW_CHANNEL_INFO_TTL,
    "youtube.raw.channel_info_batch": RAW_CHANNEL_INFO_TTL,
    "youtube.raw.channel_videos": RAW_CHANNEL_VIDEOS_TTL,
    "youtube.raw.comments": RAW_COMMENTS_TTL,
    "youtube.raw.trending": RAW_TRENDING_TTL,
    "youtube.raw.transcripts": RAW_TRANSCRIPTS_TTL,
}
"""TTL lookup by namespace string."""


# =============================================================================
# TRANSFORM DATA TTLs (Transform Layer)
# =============================================================================
# Transform results are derived from raw data.
# Shorter TTLs are OK because recomputing is cheap (no API calls).

TRANSFORM_STATISTICS_TTL = timedelta(hours=6)
"""Statistical aggregations. Recompute when underlying data refreshes."""

TRANSFORM_PATTERNS_TTL = timedelta(hours=12)
"""Pattern analysis. Slightly longer - patterns are more stable."""

TRANSFORM_SCORES_TTL = timedelta(hours=6)
"""Scoring results. Same as statistics."""

TRANSFORM_GAPS_TTL = timedelta(hours=12)
"""Gap analysis. More stable, longer TTL."""

TRANSFORM_TTL = {
    "youtube.transform.statistics": TRANSFORM_STATISTICS_TTL,
    "youtube.transform.patterns": TRANSFORM_PATTERNS_TTL,
    "youtube.transform.scores": TRANSFORM_SCORES_TTL,
    "youtube.transform.gaps": TRANSFORM_GAPS_TTL,
}
"""TTL lookup by transform namespace string."""


# =============================================================================
# INTELLIGENCE DATA TTLs (Intelligence Layer)
# =============================================================================
# Final analysis reports. TTLs vary by use case.

INTELLIGENCE_NICHE_TTL = timedelta(hours=12)
"""Niche analysis reports. Medium TTL - comprehensive reports."""

INTELLIGENCE_COMPETITION_TTL = timedelta(hours=24)
"""Competition analysis. Long TTL - competitive landscape is stable."""

INTELLIGENCE_CONTENT_GAPS_TTL = timedelta(hours=12)
"""Content gap reports. Medium TTL - gaps evolve moderately."""

INTELLIGENCE_BENCHMARKS_TTL = timedelta(hours=24)
"""Benchmark reports. Long TTL - benchmarks are statistical averages."""

INTELLIGENCE_TITLES_TTL = timedelta(hours=24)
"""Title analysis. Long TTL - title patterns are stable."""

INTELLIGENCE_NAMING_TTL = timedelta(hours=24)
"""Naming analysis. Long TTL - naming patterns are stable."""

INTELLIGENCE_TRENDING_TTL = timedelta(hours=6)
"""Trending analysis. Shorter TTL - trending changes frequently."""

INTELLIGENCE_NAME_CHECK_TTL = timedelta(hours=1)
"""Channel name availability check. Short TTL - names can be taken."""

INTELLIGENCE_TTL = {
    "youtube.intelligence.niche": INTELLIGENCE_NICHE_TTL,
    "youtube.intelligence.competition": INTELLIGENCE_COMPETITION_TTL,
    "youtube.intelligence.content_gaps": INTELLIGENCE_CONTENT_GAPS_TTL,
    "youtube.intelligence.benchmarks": INTELLIGENCE_BENCHMARKS_TTL,
    "youtube.intelligence.titles": INTELLIGENCE_TITLES_TTL,
    "youtube.intelligence.naming": INTELLIGENCE_NAMING_TTL,
    "youtube.intelligence.trending": INTELLIGENCE_TRENDING_TTL,
}
"""TTL lookup by intelligence namespace string."""


# =============================================================================
# COMBINED TTL LOOKUP
# =============================================================================

ALL_TTL = {**RAW_TTL, **TRANSFORM_TTL, **INTELLIGENCE_TTL}
"""Combined TTL lookup for all namespaces."""


def get_ttl(namespace: str) -> timedelta:
    """Get TTL for a given namespace.

    Args:
        namespace: The cache namespace string.

    Returns:
        TTL timedelta for the namespace.

    Raises:
        KeyError: If namespace is not recognized.

    Example:
        >>> ttl = get_ttl("youtube.raw.video_details")
        >>> print(ttl)
        1 day, 0:00:00
    """
    if namespace not in ALL_TTL:
        raise KeyError(
            f"Unknown namespace: {namespace}. "
            f"Valid namespaces: {sorted(ALL_TTL.keys())}"
        )
    return ALL_TTL[namespace]


def get_ttl_seconds(namespace: str) -> int:
    """Get TTL in seconds for a given namespace.

    Args:
        namespace: The cache namespace string.

    Returns:
        TTL in seconds for the namespace.

    Raises:
        KeyError: If namespace is not recognized.

    Example:
        >>> seconds = get_ttl_seconds("youtube.raw.video_details")
        >>> print(seconds)
        86400
    """
    return int(get_ttl(namespace).total_seconds())


__all__ = [
    "ALL_TTL",
    "INTELLIGENCE_BENCHMARKS_TTL",
    "INTELLIGENCE_COMPETITION_TTL",
    "INTELLIGENCE_CONTENT_GAPS_TTL",
    "INTELLIGENCE_NAME_CHECK_TTL",
    "INTELLIGENCE_NAMING_TTL",
    "INTELLIGENCE_NICHE_TTL",
    "INTELLIGENCE_TITLES_TTL",
    "INTELLIGENCE_TRENDING_TTL",
    "INTELLIGENCE_TTL",
    "RAW_CHANNEL_INFO_TTL",
    "RAW_CHANNEL_VIDEOS_TTL",
    "RAW_COMMENTS_TTL",
    "RAW_SEARCH_CHANNELS_TTL",
    "RAW_SEARCH_VIDEOS_TTL",
    "RAW_TRANSCRIPTS_TTL",
    "RAW_TRENDING_TTL",
    "RAW_TTL",
    "RAW_VIDEO_DETAILS_TTL",
    "TRANSFORM_GAPS_TTL",
    "TRANSFORM_PATTERNS_TTL",
    "TRANSFORM_SCORES_TTL",
    "TRANSFORM_STATISTICS_TTL",
    "TRANSFORM_TTL",
    "get_ttl",
    "get_ttl_seconds",
]
