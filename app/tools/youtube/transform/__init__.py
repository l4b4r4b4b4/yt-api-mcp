"""Transform layer for YouTube MCP ELT pipeline.

This module provides transformation and analysis functions for processing
raw YouTube data into actionable insights. The Transform layer operates
entirely on cached data from the Load layer - no API calls.

Key principles:
1. No API calls - only process cached data
2. Pure functions - no side effects
3. Return structured results for Intelligence layer
4. Computationally cheap (can be recomputed from cached raw data)

Transform modules:
- statistics: Statistical aggregations and calculations
- patterns: Title, naming, and content pattern extraction
- scoring: Competition, opportunity, and performance scoring
- gaps: Content gap identification algorithms
- utils: Common text analysis utilities

Example:
    >>> from app.tools.youtube.transform import (
    ...     aggregate_video_statistics,
    ...     analyze_title_patterns,
    ...     score_competition,
    ...     identify_content_gaps,
    ... )
    >>>
    >>> # Transform raw video data into statistics
    >>> stats = aggregate_video_statistics(raw_videos)
    >>>
    >>> # Analyze title patterns
    >>> patterns = analyze_title_patterns([v["title"] for v in raw_videos])
    >>>
    >>> # Score competition
    >>> competition = score_competition(raw_channels, raw_videos)
"""

from __future__ import annotations

from app.tools.youtube.transform.gaps import (
    analyze_comment_questions,
    analyze_demand_supply_mismatch,
    check_related_topic_coverage,
    deduplicate_gaps,
    estimate_gap_effort,
    find_uncovered_topics,
    generate_gap_report,
    identify_content_gaps,
    prioritize_gaps,
)
from app.tools.youtube.transform.patterns import (
    analyze_channel_naming_patterns,
    analyze_content_duration_patterns,
    analyze_title_patterns,
    categorize_channel_name,
    categorize_tags,
    extract_tag_patterns,
    find_title_keywords_by_performance,
    parse_iso8601_duration,
)
from app.tools.youtube.transform.scoring import (
    calculate_saturation_score,
    generate_competition_recommendations,
    generate_opportunity_recommendations,
    rank_videos_by_performance,
    score_channel_strength,
    score_competition,
    score_opportunity,
)
from app.tools.youtube.transform.statistics import (
    aggregate_channel_statistics,
    aggregate_video_statistics,
    calculate_basic_stats,
    calculate_correlation,
    calculate_distribution,
    calculate_engagement_rate,
    calculate_growth_metrics,
    calculate_percentiles,
    identify_outliers,
)
from app.tools.youtube.transform.utils import (
    calculate_text_similarity,
    contains_emoji,
    contains_number,
    contains_question,
    extract_title_structure,
    extract_words,
    normalize_text,
    safe_float,
    safe_int,
    truncate_text,
    word_frequency,
)

__all__ = [
    # Statistics
    "aggregate_channel_statistics",
    "aggregate_video_statistics",
    # Patterns
    "analyze_channel_naming_patterns",
    # Gaps
    "analyze_comment_questions",
    "analyze_content_duration_patterns",
    "analyze_demand_supply_mismatch",
    "analyze_title_patterns",
    "calculate_basic_stats",
    "calculate_correlation",
    "calculate_distribution",
    "calculate_engagement_rate",
    "calculate_growth_metrics",
    "calculate_percentiles",
    # Scoring
    "calculate_saturation_score",
    # Utils
    "calculate_text_similarity",
    "categorize_channel_name",
    "categorize_tags",
    "check_related_topic_coverage",
    "contains_emoji",
    "contains_number",
    "contains_question",
    "deduplicate_gaps",
    "estimate_gap_effort",
    "extract_tag_patterns",
    "extract_title_structure",
    "extract_words",
    "find_title_keywords_by_performance",
    "find_uncovered_topics",
    "generate_competition_recommendations",
    "generate_gap_report",
    "generate_opportunity_recommendations",
    "identify_content_gaps",
    "identify_outliers",
    "normalize_text",
    "parse_iso8601_duration",
    "prioritize_gaps",
    "rank_videos_by_performance",
    "safe_float",
    "safe_int",
    "score_channel_strength",
    "score_competition",
    "score_opportunity",
    "truncate_text",
    "word_frequency",
]
