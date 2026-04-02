"""Intelligence layer for YouTube MCP ELT pipeline.

This module provides high-level intelligence tools that orchestrate the
full ELT pipeline to produce actionable market intelligence reports.

The Intelligence layer:
1. Orchestrates Extract → Load → Transform flow
2. Combines multiple transform results into comprehensive reports
3. Provides strategic recommendations and insights
4. Caches final reports for efficient reuse

Intelligence modules:
- niche: Comprehensive niche analysis (analyze_niche)
- competition: Channel competition analysis
- content: Content gaps and title analysis
- trending: Trending video analysis
- naming: Channel naming analysis and availability
- benchmarks: Niche performance benchmarks

Example:
    >>> from app.tools.youtube.intelligence import (
    ...     analyze_niche,
    ...     analyze_channel_competition,
    ...     find_content_gaps,
    ...     get_niche_benchmarks,
    ... )
    >>>
    >>> # Full niche analysis
    >>> result = await analyze_niche("kubernetes", region="US")
    >>>
    >>> # Competition analysis
    >>> competition = await analyze_channel_competition("kubernetes")
"""

from __future__ import annotations

from app.tools.youtube.intelligence.benchmarks import (
    calculate_benchmark_percentile,
    compare_to_benchmarks,
    get_growth_benchmarks,
    get_niche_benchmarks,
)
from app.tools.youtube.intelligence.competition import (
    analyze_channel_competition,
    analyze_competitor_content,
    find_market_leaders,
    rank_competitors,
)
from app.tools.youtube.intelligence.content import (
    analyze_content_format_performance,
    analyze_successful_titles,
    find_content_gaps,
    generate_title_suggestions,
)
from app.tools.youtube.intelligence.naming import (
    analyze_channel_names,
    analyze_name_brandability,
    check_channel_name_availability,
    suggest_channel_names,
)
from app.tools.youtube.intelligence.niche import (
    analyze_niche,
    compare_niches,
    get_niche_summary,
)
from app.tools.youtube.intelligence.trending import (
    analyze_trending_patterns,
    compare_trending_regions,
    get_trending_videos,
    identify_trending_opportunities,
)

__all__ = [
    # Competition analysis
    "analyze_channel_competition",
    # Naming analysis
    "analyze_channel_names",
    "analyze_competitor_content",
    # Content analysis
    "analyze_content_format_performance",
    "analyze_name_brandability",
    # Niche analysis
    "analyze_niche",
    "analyze_successful_titles",
    # Trending analysis
    "analyze_trending_patterns",
    # Benchmarks
    "calculate_benchmark_percentile",
    "check_channel_name_availability",
    "compare_niches",
    "compare_to_benchmarks",
    "compare_trending_regions",
    "find_content_gaps",
    "find_market_leaders",
    "generate_title_suggestions",
    "get_growth_benchmarks",
    "get_niche_benchmarks",
    "get_niche_summary",
    "get_trending_videos",
    "identify_trending_opportunities",
    "rank_competitors",
    "suggest_channel_names",
]
