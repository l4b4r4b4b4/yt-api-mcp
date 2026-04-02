"""RefCache namespace constants for YouTube MCP ELT pipeline.

This module defines standardized namespace strings for organizing cached data
in the RefCache system. Namespaces follow a hierarchical structure:

- youtube.raw.*    : Raw data from API (quota-consuming extractions)
- youtube.transform.*  : Transformed/aggregated data (computed from raw)
- youtube.intelligence.*  : High-level analysis results

Using consistent namespaces enables:
- Clear data organization
- Easy cache invalidation by category
- Quota tracking and optimization
- Debugging and monitoring
"""

from __future__ import annotations

# =============================================================================
# RAW DATA NAMESPACES (Extract Layer)
# =============================================================================
# These store unprocessed data directly from YouTube API calls.
# Caching these saves API quota on repeated requests.

RAW_SEARCH_VIDEOS = "youtube.raw.search_videos"
"""Search results for video queries. Key: query string."""

RAW_SEARCH_CHANNELS = "youtube.raw.search_channels"
"""Search results for channel queries. Key: query string."""

RAW_VIDEO_DETAILS = "youtube.raw.video_details"
"""Detailed video metadata (statistics, duration, tags). Key: video_id."""

RAW_VIDEO_DETAILS_BATCH = "youtube.raw.video_details_batch"
"""Batch video details. Key: sorted comma-separated video_ids."""

RAW_CHANNEL_INFO = "youtube.raw.channel_info"
"""Detailed channel metadata (statistics, branding). Key: channel_id."""

RAW_CHANNEL_INFO_BATCH = "youtube.raw.channel_info_batch"
"""Batch channel info. Key: sorted comma-separated channel_ids."""

RAW_CHANNEL_VIDEOS = "youtube.raw.channel_videos"
"""Videos from a specific channel. Key: channel_id."""

RAW_COMMENTS = "youtube.raw.comments"
"""Video comments. Key: video_id."""

RAW_TRENDING = "youtube.raw.trending"
"""Trending videos. Key: region_category (e.g., 'US_28')."""

RAW_TRANSCRIPTS = "youtube.raw.transcripts"
"""Video transcripts. Key: video_id_language (e.g., 'abc123_en')."""

# =============================================================================
# TRANSFORM NAMESPACES (Transform Layer)
# =============================================================================
# These store computed/aggregated data derived from raw data.
# Cheaper to recompute than raw data, so shorter TTLs are acceptable.

TRANSFORM_STATISTICS = "youtube.transform.statistics"
"""Statistical aggregations (percentiles, averages). Key: varies by analysis."""

TRANSFORM_PATTERNS = "youtube.transform.patterns"
"""Pattern analysis results (title patterns, naming). Key: varies by analysis."""

TRANSFORM_SCORES = "youtube.transform.scores"
"""Scoring results (competition, opportunity). Key: varies by analysis."""

TRANSFORM_GAPS = "youtube.transform.gaps"
"""Content gap analysis results. Key: topic or channel_id."""

# =============================================================================
# INTELLIGENCE NAMESPACES (Intelligence Layer)
# =============================================================================
# These store high-level analysis reports that combine multiple transforms.
# Final output for MCP tools.

INTELLIGENCE_NICHE = "youtube.intelligence.niche"
"""Full niche analysis reports. Key: topic_region."""

INTELLIGENCE_COMPETITION = "youtube.intelligence.competition"
"""Channel competition analysis. Key: topic."""

INTELLIGENCE_CONTENT_GAPS = "youtube.intelligence.content_gaps"
"""Content gap reports. Key: topic."""

INTELLIGENCE_BENCHMARKS = "youtube.intelligence.benchmarks"
"""Niche benchmark reports. Key: topic_segment."""

INTELLIGENCE_TITLES = "youtube.intelligence.titles"
"""Title analysis reports. Key: topic."""

INTELLIGENCE_NAMING = "youtube.intelligence.naming"
"""Channel naming analysis. Key: topic."""

INTELLIGENCE_TRENDING = "youtube.intelligence.trending"
"""Trending analysis with insights. Key: region_category."""

# =============================================================================
# NAMESPACE GROUPS (for batch operations)
# =============================================================================

RAW_NAMESPACES = [
    RAW_SEARCH_VIDEOS,
    RAW_SEARCH_CHANNELS,
    RAW_VIDEO_DETAILS,
    RAW_VIDEO_DETAILS_BATCH,
    RAW_CHANNEL_INFO,
    RAW_CHANNEL_INFO_BATCH,
    RAW_CHANNEL_VIDEOS,
    RAW_COMMENTS,
    RAW_TRENDING,
    RAW_TRANSCRIPTS,
]
"""All raw data namespaces."""

TRANSFORM_NAMESPACES = [
    TRANSFORM_STATISTICS,
    TRANSFORM_PATTERNS,
    TRANSFORM_SCORES,
    TRANSFORM_GAPS,
]
"""All transform namespaces."""

INTELLIGENCE_NAMESPACES = [
    INTELLIGENCE_NICHE,
    INTELLIGENCE_COMPETITION,
    INTELLIGENCE_CONTENT_GAPS,
    INTELLIGENCE_BENCHMARKS,
    INTELLIGENCE_TITLES,
    INTELLIGENCE_NAMING,
    INTELLIGENCE_TRENDING,
]
"""All intelligence namespaces."""

ALL_NAMESPACES = RAW_NAMESPACES + TRANSFORM_NAMESPACES + INTELLIGENCE_NAMESPACES
"""All YouTube MCP namespaces."""


__all__ = [
    "ALL_NAMESPACES",
    "INTELLIGENCE_BENCHMARKS",
    "INTELLIGENCE_COMPETITION",
    "INTELLIGENCE_CONTENT_GAPS",
    "INTELLIGENCE_NAMESPACES",
    "INTELLIGENCE_NAMING",
    "INTELLIGENCE_NICHE",
    "INTELLIGENCE_TITLES",
    "INTELLIGENCE_TRENDING",
    "RAW_CHANNEL_INFO",
    "RAW_CHANNEL_INFO_BATCH",
    "RAW_CHANNEL_VIDEOS",
    "RAW_COMMENTS",
    "RAW_NAMESPACES",
    "RAW_SEARCH_CHANNELS",
    "RAW_SEARCH_VIDEOS",
    "RAW_TRANSCRIPTS",
    "RAW_TRENDING",
    "RAW_VIDEO_DETAILS",
    "RAW_VIDEO_DETAILS_BATCH",
    "TRANSFORM_GAPS",
    "TRANSFORM_NAMESPACES",
    "TRANSFORM_PATTERNS",
    "TRANSFORM_SCORES",
    "TRANSFORM_STATISTICS",
]
