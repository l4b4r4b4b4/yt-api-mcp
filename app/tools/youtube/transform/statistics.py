"""Statistical analysis functions for YouTube MCP Transform layer.

This module provides statistical analysis functions for computing aggregations,
percentiles, trends, and other statistical measures from raw YouTube data.

Transform layer principles:
1. No API calls - only process cached data
2. Pure functions - no side effects
3. Return structured results for Intelligence layer
4. Computationally cheap (can be recomputed)
"""

from __future__ import annotations

import statistics
from typing import Any

from app.tools.youtube.transform.utils import safe_int


def calculate_percentiles(
    values: list[int | float],
    percentiles: list[int] | None = None,
) -> dict[str, float]:
    """Calculate percentile values for a list of numbers.

    Args:
        values: List of numeric values.
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 90]).

    Returns:
        Dictionary mapping percentile names to values (e.g., {"p25": 100, "p50": 200}).

    Example:
        >>> views = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        >>> calculate_percentiles(views)
        {"p25": 275.0, "p50": 550.0, "p75": 825.0, "p90": 910.0}
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 90]

    if not values:
        return {f"p{p}": 0.0 for p in percentiles}

    sorted_values = sorted(values)
    result = {}

    for p in percentiles:
        # Calculate percentile index
        index = (p / 100) * (len(sorted_values) - 1)
        lower_idx = int(index)
        upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
        fraction = index - lower_idx

        # Linear interpolation
        value = sorted_values[lower_idx] + fraction * (
            sorted_values[upper_idx] - sorted_values[lower_idx]
        )
        result[f"p{p}"] = round(value, 2)

    return result


def calculate_basic_stats(values: list[int | float]) -> dict[str, float]:
    """Calculate basic statistical measures.

    Args:
        values: List of numeric values.

    Returns:
        Dictionary with count, sum, min, max, mean, median, and std_dev.

    Example:
        >>> calculate_basic_stats([10, 20, 30, 40, 50])
        {
            "count": 5,
            "sum": 150,
            "min": 10,
            "max": 50,
            "mean": 30.0,
            "median": 30.0,
            "std_dev": 15.81
        }
    """
    if not values:
        return {
            "count": 0,
            "sum": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
        }

    return {
        "count": len(values),
        "sum": sum(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "std_dev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
    }


def calculate_engagement_rate(
    views: int,
    likes: int,
    comments: int,
) -> float:
    """Calculate engagement rate for a video.

    Standard engagement rate formula:
    engagement_rate = ((likes + comments) / views) * 100

    Args:
        views: Number of views.
        likes: Number of likes.
        comments: Number of comments.

    Returns:
        Engagement rate as a percentage (0-100+).

    Example:
        >>> calculate_engagement_rate(views=10000, likes=500, comments=50)
        5.5
    """
    if views <= 0:
        return 0.0

    engagement = ((likes + comments) / views) * 100
    return round(engagement, 2)


def aggregate_video_statistics(videos: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate statistics across multiple videos.

    Computes summary statistics for a collection of video data,
    including view counts, likes, comments, and engagement.

    Args:
        videos: List of video dictionaries with statistics fields.
            Expected fields: view_count, like_count, comment_count.

    Returns:
        Dictionary with aggregated statistics including:
        - sample_size: Number of videos analyzed
        - views: Basic stats and percentiles for view counts
        - likes: Basic stats and percentiles for like counts
        - comments: Basic stats and percentiles for comment counts
        - engagement: Average engagement rate and distribution

    Example:
        >>> videos = [
        ...     {"view_count": "1000", "like_count": "50", "comment_count": "10"},
        ...     {"view_count": "5000", "like_count": "250", "comment_count": "25"},
        ... ]
        >>> stats = aggregate_video_statistics(videos)
        >>> print(stats["views"]["mean"])
        3000.0
    """
    if not videos:
        return {
            "sample_size": 0,
            "views": calculate_basic_stats([]),
            "likes": calculate_basic_stats([]),
            "comments": calculate_basic_stats([]),
            "engagement": {"mean": 0.0, "percentiles": {}},
        }

    # Extract numeric values
    view_counts = [safe_int(v.get("view_count", 0)) for v in videos]
    like_counts = [safe_int(v.get("like_count", 0)) for v in videos]
    comment_counts = [safe_int(v.get("comment_count", 0)) for v in videos]

    # Calculate engagement rates
    engagement_rates = []
    for video in videos:
        views = safe_int(video.get("view_count", 0))
        likes = safe_int(video.get("like_count", 0))
        comments = safe_int(video.get("comment_count", 0))
        if views > 0:
            engagement_rates.append(calculate_engagement_rate(views, likes, comments))

    # Build result
    return {
        "sample_size": len(videos),
        "views": {
            **calculate_basic_stats(view_counts),
            "percentiles": calculate_percentiles(view_counts),
        },
        "likes": {
            **calculate_basic_stats(like_counts),
            "percentiles": calculate_percentiles(like_counts),
        },
        "comments": {
            **calculate_basic_stats(comment_counts),
            "percentiles": calculate_percentiles(comment_counts),
        },
        "engagement": {
            "mean": round(statistics.mean(engagement_rates), 2)
            if engagement_rates
            else 0.0,
            "percentiles": calculate_percentiles(engagement_rates),
        },
    }


def aggregate_channel_statistics(channels: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate statistics across multiple channels.

    Computes summary statistics for a collection of channel data,
    including subscriber counts, video counts, and total views.

    Args:
        channels: List of channel dictionaries with statistics fields.
            Expected fields: subscriber_count, video_count, view_count.

    Returns:
        Dictionary with aggregated statistics including:
        - sample_size: Number of channels analyzed
        - subscribers: Basic stats and percentiles
        - video_counts: Basic stats for videos per channel
        - total_views: Basic stats for total channel views

    Example:
        >>> channels = [
        ...     {"subscriber_count": "1000", "video_count": "50", "view_count": "100000"},
        ...     {"subscriber_count": "5000", "video_count": "100", "view_count": "500000"},
        ... ]
        >>> stats = aggregate_channel_statistics(channels)
        >>> print(stats["subscribers"]["mean"])
        3000.0
    """
    if not channels:
        return {
            "sample_size": 0,
            "subscribers": calculate_basic_stats([]),
            "video_counts": calculate_basic_stats([]),
            "total_views": calculate_basic_stats([]),
        }

    # Extract numeric values
    subscriber_counts = [safe_int(c.get("subscriber_count", 0)) for c in channels]
    video_counts = [safe_int(c.get("video_count", 0)) for c in channels]
    view_counts = [safe_int(c.get("view_count", 0)) for c in channels]

    return {
        "sample_size": len(channels),
        "subscribers": {
            **calculate_basic_stats(subscriber_counts),
            "percentiles": calculate_percentiles(subscriber_counts),
        },
        "video_counts": calculate_basic_stats(video_counts),
        "total_views": {
            **calculate_basic_stats(view_counts),
            "percentiles": calculate_percentiles(view_counts),
        },
    }


def calculate_growth_metrics(
    recent_values: list[int | float],
    older_values: list[int | float],
) -> dict[str, float]:
    """Calculate growth metrics between two time periods.

    Args:
        recent_values: Values from recent time period.
        older_values: Values from older time period.

    Returns:
        Dictionary with growth metrics:
        - absolute_change: Difference between means
        - percent_change: Percentage change
        - trend: "increasing", "decreasing", or "stable"

    Example:
        >>> recent = [500, 600, 700, 800]
        >>> older = [300, 400, 500, 600]
        >>> calculate_growth_metrics(recent, older)
        {"absolute_change": 200.0, "percent_change": 44.44, "trend": "increasing"}
    """
    if not recent_values or not older_values:
        return {
            "absolute_change": 0.0,
            "percent_change": 0.0,
            "trend": "unknown",
        }

    recent_mean = statistics.mean(recent_values)
    older_mean = statistics.mean(older_values)

    absolute_change = recent_mean - older_mean

    if older_mean > 0:
        percent_change = (absolute_change / older_mean) * 100
    else:
        percent_change = 100.0 if recent_mean > 0 else 0.0

    # Determine trend
    threshold = 5.0  # 5% change threshold for "stable"
    if percent_change > threshold:
        trend = "increasing"
    elif percent_change < -threshold:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "absolute_change": round(absolute_change, 2),
        "percent_change": round(percent_change, 2),
        "trend": trend,
    }


def calculate_distribution(
    values: list[int | float],
    bucket_count: int = 5,
) -> list[dict[str, Any]]:
    """Calculate value distribution across buckets.

    Divides the value range into equal buckets and counts values in each.

    Args:
        values: List of numeric values.
        bucket_count: Number of buckets (default: 5).

    Returns:
        List of bucket dictionaries with min, max, count, and percentage.

    Example:
        >>> values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        >>> calculate_distribution(values, bucket_count=5)
        [
            {"min": 10, "max": 28, "count": 2, "percentage": 20.0},
            {"min": 28, "max": 46, "count": 2, "percentage": 20.0},
            ...
        ]
    """
    if not values or bucket_count < 1:
        return []

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [
            {
                "min": min_val,
                "max": max_val,
                "count": len(values),
                "percentage": 100.0,
            }
        ]

    bucket_size = (max_val - min_val) / bucket_count
    buckets = []

    for i in range(bucket_count):
        bucket_min = min_val + (i * bucket_size)
        bucket_max = min_val + ((i + 1) * bucket_size)

        # Count values in this bucket
        if i == bucket_count - 1:
            # Last bucket includes max value
            count = sum(1 for v in values if bucket_min <= v <= bucket_max)
        else:
            count = sum(1 for v in values if bucket_min <= v < bucket_max)

        buckets.append(
            {
                "min": round(bucket_min, 2),
                "max": round(bucket_max, 2),
                "count": count,
                "percentage": round((count / len(values)) * 100, 2),
            }
        )

    return buckets


def identify_outliers(
    values: list[int | float],
    method: str = "iqr",
) -> dict[str, Any]:
    """Identify outliers in a dataset.

    Args:
        values: List of numeric values.
        method: Outlier detection method ("iqr" or "zscore").

    Returns:
        Dictionary with:
        - lower_bound: Values below this are outliers
        - upper_bound: Values above this are outliers
        - outliers: List of outlier values
        - outlier_indices: Indices of outliers in original list
        - count: Number of outliers

    Example:
        >>> values = [10, 20, 30, 40, 50, 1000]  # 1000 is outlier
        >>> identify_outliers(values)
        {"outliers": [1000], "count": 1, ...}
    """
    if not values or len(values) < 4:
        return {
            "lower_bound": 0,
            "upper_bound": 0,
            "outliers": [],
            "outlier_indices": [],
            "count": 0,
        }

    sorted_vals = sorted(values)

    if method == "iqr":
        # IQR method
        q1_idx = len(sorted_vals) // 4
        q3_idx = 3 * len(sorted_vals) // 4
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    else:  # zscore
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        threshold = 3  # 3 standard deviations

        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

    # Find outliers
    outliers = []
    outlier_indices = []
    for i, v in enumerate(values):
        if v < lower_bound or v > upper_bound:
            outliers.append(v)
            outlier_indices.append(i)

    return {
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outliers": outliers,
        "outlier_indices": outlier_indices,
        "count": len(outliers),
    }


def calculate_correlation(
    values_x: list[int | float],
    values_y: list[int | float],
) -> float:
    """Calculate Pearson correlation coefficient.

    Args:
        values_x: First list of values.
        values_y: Second list of values (same length as values_x).

    Returns:
        Correlation coefficient between -1 and 1.

    Example:
        >>> views = [100, 200, 300, 400, 500]
        >>> likes = [10, 20, 30, 40, 50]
        >>> calculate_correlation(views, likes)
        1.0  # Perfect positive correlation
    """
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return 0.0

    len(values_x)
    mean_x = statistics.mean(values_x)
    mean_y = statistics.mean(values_y)

    # Calculate covariance and standard deviations
    covariance = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y, strict=False)
    )
    std_x = (sum((x - mean_x) ** 2 for x in values_x)) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in values_y)) ** 0.5

    if std_x == 0 or std_y == 0:
        return 0.0

    correlation = covariance / (std_x * std_y)
    return round(correlation, 4)


__all__ = [
    "aggregate_channel_statistics",
    "aggregate_video_statistics",
    "calculate_basic_stats",
    "calculate_correlation",
    "calculate_distribution",
    "calculate_engagement_rate",
    "calculate_growth_metrics",
    "calculate_percentiles",
    "identify_outliers",
]
