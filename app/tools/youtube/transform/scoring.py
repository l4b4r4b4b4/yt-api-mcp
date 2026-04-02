"""Scoring functions for YouTube MCP Transform layer.

This module provides scoring functions for calculating competition levels,
opportunity scores, engagement ratings, and other comparative metrics
from YouTube data.

Transform layer principles:
1. No API calls - only process cached data
2. Pure functions - no side effects
3. Return structured results for Intelligence layer
4. Computationally cheap (can be recomputed)
"""

from __future__ import annotations

from typing import Any

from app.tools.youtube.transform.statistics import (
    calculate_engagement_rate,
)
from app.tools.youtube.transform.utils import safe_int


def score_competition(
    channels: list[dict[str, Any]],
    videos: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate competition score for a niche.

    Evaluates how competitive a niche is based on channel and video metrics.
    Higher scores indicate more competition (harder to enter).

    Scoring factors:
    1. Number of established channels (high subscriber counts)
    2. Average channel size
    3. Upload frequency
    4. Video view averages
    5. Engagement rates

    Args:
        channels: List of channel dictionaries with statistics.
        videos: List of video dictionaries with statistics.

    Returns:
        Dictionary with:
        - score: Competition score (1-10, higher = more competitive)
        - level: Competition level ("low", "medium", "high", "very_high")
        - factors: Breakdown of scoring factors
        - barrier_to_entry: Assessment of difficulty to enter
        - recommendations: List of strategic recommendations

    Example:
        >>> channels = [{"subscriber_count": "1000000", ...}, ...]
        >>> videos = [{"view_count": "50000", ...}, ...]
        >>> result = score_competition(channels, videos)
        >>> print(result["score"])
        7.5
        >>> print(result["level"])
        "high"
    """
    if not channels and not videos:
        return {
            "score": 1.0,
            "level": "low",
            "factors": {},
            "barrier_to_entry": "low",
            "recommendations": ["Niche appears underserved - good opportunity"],
        }

    factors = {}
    scores = []

    # Factor 1: Channel count and sizes
    if channels:
        subscriber_counts = [safe_int(c.get("subscriber_count", 0)) for c in channels]
        large_channels = sum(1 for s in subscriber_counts if s > 100000)
        medium_channels = sum(1 for s in subscriber_counts if 10000 <= s <= 100000)

        # Score based on established channels
        channel_score = min(10, (large_channels * 2) + (medium_channels * 0.5))
        factors["established_channels"] = {
            "large_channels": large_channels,
            "medium_channels": medium_channels,
            "score": round(channel_score, 2),
        }
        scores.append(channel_score)

        # Factor 2: Average channel size
        avg_subscribers = sum(subscriber_counts) / len(subscriber_counts)
        size_score = min(10, (avg_subscribers / 100000) * 5)
        factors["average_channel_size"] = {
            "avg_subscribers": round(avg_subscribers, 0),
            "score": round(size_score, 2),
        }
        scores.append(size_score)

    # Factor 3: Video performance
    if videos:
        view_counts = [safe_int(v.get("view_count", 0)) for v in videos]
        avg_views = sum(view_counts) / len(view_counts) if view_counts else 0

        # Higher average views = more competition
        view_score = min(10, (avg_views / 50000) * 5)
        factors["video_performance"] = {
            "avg_views": round(avg_views, 0),
            "score": round(view_score, 2),
        }
        scores.append(view_score)

        # Factor 4: Engagement levels
        engagement_rates = []
        for video in videos:
            views = safe_int(video.get("view_count", 0))
            likes = safe_int(video.get("like_count", 0))
            comments = safe_int(video.get("comment_count", 0))
            if views > 0:
                engagement_rates.append(
                    calculate_engagement_rate(views, likes, comments)
                )

        if engagement_rates:
            avg_engagement = sum(engagement_rates) / len(engagement_rates)
            # Higher engagement = more invested audience = harder to compete
            engagement_score = min(10, avg_engagement * 2)
            factors["audience_engagement"] = {
                "avg_engagement_rate": round(avg_engagement, 2),
                "score": round(engagement_score, 2),
            }
            scores.append(engagement_score)

    # Calculate final score
    final_score = sum(scores) / len(scores) if scores else 1.0
    final_score = round(max(1.0, min(10.0, final_score)), 2)

    # Determine level
    if final_score <= 3:
        level = "low"
        barrier = "low"
    elif final_score <= 5:
        level = "medium"
        barrier = "moderate"
    elif final_score <= 7:
        level = "high"
        barrier = "significant"
    else:
        level = "very_high"
        barrier = "very_high"

    # Generate recommendations
    recommendations = generate_competition_recommendations(final_score, level, factors)

    return {
        "score": final_score,
        "level": level,
        "factors": factors,
        "barrier_to_entry": barrier,
        "recommendations": recommendations,
    }


def generate_competition_recommendations(
    score: float,
    level: str,
    factors: dict[str, Any],
) -> list[str]:
    """Generate strategic recommendations based on competition analysis.

    Args:
        score: Competition score (1-10).
        level: Competition level string.
        factors: Dictionary of scoring factors.

    Returns:
        List of recommendation strings.
    """
    recommendations = []

    if level == "low":
        recommendations.append(
            "Low competition - excellent opportunity for early entry"
        )
        recommendations.append(
            "Focus on establishing authority before competitors arrive"
        )
    elif level == "medium":
        recommendations.append("Moderate competition - differentiation is key")
        recommendations.append("Find a unique angle or underserved sub-niche")
    elif level == "high":
        recommendations.append(
            "High competition - requires strong differentiation strategy"
        )
        recommendations.append("Consider targeting a more specific sub-niche")
        recommendations.append("Focus on quality and consistency to stand out")
    else:  # very_high
        recommendations.append(
            "Very high competition - challenging market for new entrants"
        )
        recommendations.append("Consider a unique format or perspective")
        recommendations.append("Collaboration with established creators may help")

    # Factor-specific recommendations
    if (
        "established_channels" in factors
        and factors["established_channels"]["large_channels"] > 5
    ):
        recommendations.append("Many large channels exist - find gaps in their content")

    if "audience_engagement" in factors:
        eng_rate = factors["audience_engagement"].get("avg_engagement_rate", 0)
        if eng_rate > 5:
            recommendations.append(
                "High engagement audience - focus on community building"
            )

    return recommendations


def score_opportunity(
    videos: list[dict[str, Any]],
    channels: list[dict[str, Any]],
    content_gaps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate opportunity score for a niche.

    Evaluates how much opportunity exists in a niche based on
    demand signals, competition gaps, and growth potential.

    Args:
        videos: List of video dictionaries with statistics.
        channels: List of channel dictionaries with statistics.
        content_gaps: Optional list of identified content gaps.

    Returns:
        Dictionary with:
        - score: Opportunity score (1-10, higher = more opportunity)
        - level: Opportunity level ("low", "medium", "high", "excellent")
        - factors: Breakdown of opportunity factors
        - growth_potential: Assessment of growth potential
        - recommendations: List of opportunity-specific recommendations

    Example:
        >>> result = score_opportunity(videos, channels)
        >>> print(result["score"])
        6.5
        >>> print(result["level"])
        "high"
    """
    if not videos and not channels:
        return {
            "score": 5.0,
            "level": "medium",
            "factors": {},
            "growth_potential": "unknown",
            "recommendations": ["Insufficient data to assess opportunity"],
        }

    factors = {}
    scores = []

    # Factor 1: Demand indicators (high views but room for more content)
    if videos:
        view_counts = [safe_int(v.get("view_count", 0)) for v in videos]
        avg_views = sum(view_counts) / len(view_counts) if view_counts else 0

        # Good views = demand exists
        demand_score = min(10, (avg_views / 10000) * 3)
        factors["demand_level"] = {
            "avg_views": round(avg_views, 0),
            "score": round(demand_score, 2),
        }
        scores.append(demand_score)

    # Factor 2: Competition inverse (less competition = more opportunity)
    if channels:
        subscriber_counts = [safe_int(c.get("subscriber_count", 0)) for c in channels]
        large_channels = sum(1 for s in subscriber_counts if s > 100000)

        # Fewer large channels = more opportunity
        competition_inverse = max(0, 10 - (large_channels * 1.5))
        factors["competition_gap"] = {
            "large_channels": large_channels,
            "score": round(competition_inverse, 2),
        }
        scores.append(competition_inverse)

    # Factor 3: Engagement potential
    if videos:
        engagement_rates = []
        for video in videos:
            views = safe_int(video.get("view_count", 0))
            likes = safe_int(video.get("like_count", 0))
            comments = safe_int(video.get("comment_count", 0))
            if views > 0:
                engagement_rates.append(
                    calculate_engagement_rate(views, likes, comments)
                )

        if engagement_rates:
            avg_engagement = sum(engagement_rates) / len(engagement_rates)
            # Moderate engagement is ideal - audience engages but not saturated
            if avg_engagement < 2:
                engagement_opportunity = 4  # Low engagement = less interested audience
            elif avg_engagement < 4:
                engagement_opportunity = 8  # Sweet spot
            elif avg_engagement < 6:
                engagement_opportunity = 6  # Good but competitive
            else:
                engagement_opportunity = 4  # Very high = saturated

            factors["engagement_opportunity"] = {
                "avg_engagement_rate": round(avg_engagement, 2),
                "score": round(engagement_opportunity, 2),
            }
            scores.append(engagement_opportunity)

    # Factor 4: Content gaps (if provided)
    if content_gaps:
        gap_count = len(content_gaps)
        high_opportunity_gaps = sum(
            1 for g in content_gaps if g.get("opportunity_score", 0) > 7
        )

        gap_score = min(10, gap_count * 1.5 + high_opportunity_gaps * 2)
        factors["content_gaps"] = {
            "total_gaps": gap_count,
            "high_opportunity_gaps": high_opportunity_gaps,
            "score": round(gap_score, 2),
        }
        scores.append(gap_score)

    # Calculate final score
    final_score = sum(scores) / len(scores) if scores else 5.0
    final_score = round(max(1.0, min(10.0, final_score)), 2)

    # Determine level
    if final_score <= 3:
        level = "low"
        growth_potential = "limited"
    elif final_score <= 5:
        level = "medium"
        growth_potential = "moderate"
    elif final_score <= 7:
        level = "high"
        growth_potential = "strong"
    else:
        level = "excellent"
        growth_potential = "exceptional"

    # Generate recommendations
    recommendations = generate_opportunity_recommendations(final_score, level, factors)

    return {
        "score": final_score,
        "level": level,
        "factors": factors,
        "growth_potential": growth_potential,
        "recommendations": recommendations,
    }


def generate_opportunity_recommendations(
    score: float,
    level: str,
    factors: dict[str, Any],
) -> list[str]:
    """Generate recommendations based on opportunity analysis.

    Args:
        score: Opportunity score (1-10).
        level: Opportunity level string.
        factors: Dictionary of opportunity factors.

    Returns:
        List of recommendation strings.
    """
    recommendations = []

    if level == "excellent":
        recommendations.append("Excellent opportunity - act quickly")
        recommendations.append("Strong demand with manageable competition")
    elif level == "high":
        recommendations.append("High opportunity - good market entry point")
        recommendations.append("Focus on consistent content to capture audience")
    elif level == "medium":
        recommendations.append("Moderate opportunity - requires strategic approach")
        recommendations.append("Identify unique value proposition")
    else:  # low
        recommendations.append("Limited opportunity in current state")
        recommendations.append("Consider adjacent niches or new angles")

    # Factor-specific recommendations
    if "content_gaps" in factors:
        gap_count = factors["content_gaps"].get("total_gaps", 0)
        if gap_count > 5:
            recommendations.append(
                f"{gap_count} content gaps identified - prioritize high-demand topics"
            )

    if "demand_level" in factors:
        avg_views = factors["demand_level"].get("avg_views", 0)
        if avg_views > 50000:
            recommendations.append(
                "Strong audience demand - focus on SEO and discovery"
            )

    return recommendations


def score_channel_strength(channel: dict[str, Any]) -> dict[str, Any]:
    """Score a channel's overall strength.

    Evaluates a single channel's strength based on multiple metrics.

    Args:
        channel: Channel dictionary with statistics.

    Returns:
        Dictionary with:
        - score: Channel strength score (1-10)
        - level: Strength level
        - metrics: Individual metric scores
    """
    metrics = {}
    scores = []

    # Subscriber score
    subscribers = safe_int(channel.get("subscriber_count", 0))
    if subscribers < 1000:
        sub_score = 1
    elif subscribers < 10000:
        sub_score = 3
    elif subscribers < 100000:
        sub_score = 5
    elif subscribers < 1000000:
        sub_score = 7
    else:
        sub_score = 10

    metrics["subscribers"] = {
        "count": subscribers,
        "score": sub_score,
    }
    scores.append(sub_score)

    # Video count score
    video_count = safe_int(channel.get("video_count", 0))
    video_score = min(10, video_count / 10)  # 100+ videos = max score
    metrics["video_count"] = {
        "count": video_count,
        "score": round(video_score, 2),
    }
    scores.append(video_score)

    # Views per subscriber (efficiency)
    total_views = safe_int(channel.get("view_count", 0))
    if subscribers > 0:
        views_per_sub = total_views / subscribers
        efficiency_score = min(10, views_per_sub / 100)
    else:
        efficiency_score = 0

    metrics["efficiency"] = {
        "views_per_subscriber": round(views_per_sub if subscribers > 0 else 0, 2),
        "score": round(efficiency_score, 2),
    }
    scores.append(efficiency_score)

    # Calculate final score
    final_score = sum(scores) / len(scores) if scores else 1.0
    final_score = round(max(1.0, min(10.0, final_score)), 2)

    # Determine level
    if final_score <= 3:
        level = "emerging"
    elif final_score <= 5:
        level = "growing"
    elif final_score <= 7:
        level = "established"
    else:
        level = "dominant"

    return {
        "score": final_score,
        "level": level,
        "metrics": metrics,
    }


def rank_videos_by_performance(
    videos: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rank videos by composite performance score.

    Creates a ranking of videos based on weighted performance metrics.

    Args:
        videos: List of video dictionaries with statistics.
        weights: Optional weights for metrics (default: balanced).
            Keys: "views", "likes", "comments", "engagement"

    Returns:
        List of videos with added "performance_score" and "rank" fields,
        sorted by score descending.

    Example:
        >>> ranked = rank_videos_by_performance(videos)
        >>> print(ranked[0]["rank"])
        1
        >>> print(ranked[0]["performance_score"])
        8.5
    """
    if not videos:
        return []

    # Default weights
    if weights is None:
        weights = {
            "views": 0.4,
            "likes": 0.2,
            "comments": 0.2,
            "engagement": 0.2,
        }

    # Calculate max values for normalization
    view_counts = [safe_int(v.get("view_count", 0)) for v in videos]
    like_counts = [safe_int(v.get("like_count", 0)) for v in videos]
    comment_counts = [safe_int(v.get("comment_count", 0)) for v in videos]

    max_views = max(view_counts) if view_counts else 1
    max_likes = max(like_counts) if like_counts else 1
    max_comments = max(comment_counts) if comment_counts else 1

    # Score each video
    scored_videos = []
    for video in videos:
        views = safe_int(video.get("view_count", 0))
        likes = safe_int(video.get("like_count", 0))
        comments = safe_int(video.get("comment_count", 0))

        # Normalize to 0-10 scale
        view_score = (views / max_views) * 10 if max_views > 0 else 0
        like_score = (likes / max_likes) * 10 if max_likes > 0 else 0
        comment_score = (comments / max_comments) * 10 if max_comments > 0 else 0
        engagement = calculate_engagement_rate(views, likes, comments)
        engagement_score = min(10, engagement * 2)

        # Weighted composite score
        composite = (
            view_score * weights.get("views", 0.25)
            + like_score * weights.get("likes", 0.25)
            + comment_score * weights.get("comments", 0.25)
            + engagement_score * weights.get("engagement", 0.25)
        )

        video_with_score = {
            **video,
            "performance_score": round(composite, 2),
            "score_breakdown": {
                "views": round(view_score, 2),
                "likes": round(like_score, 2),
                "comments": round(comment_score, 2),
                "engagement": round(engagement_score, 2),
            },
        }
        scored_videos.append(video_with_score)

    # Sort by score and add rank
    scored_videos.sort(key=lambda x: x["performance_score"], reverse=True)
    for i, video in enumerate(scored_videos):
        video["rank"] = i + 1

    return scored_videos


def calculate_saturation_score(
    niche_videos: list[dict[str, Any]],
    niche_channels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate market saturation score for a niche.

    Estimates how saturated a niche is based on content volume
    and creator density.

    Args:
        niche_videos: Videos in the niche.
        niche_channels: Channels in the niche.

    Returns:
        Dictionary with saturation assessment.
    """
    if not niche_videos and not niche_channels:
        return {
            "score": 0.0,
            "level": "unsaturated",
            "indicators": {},
        }

    indicators = {}

    # Video volume indicator
    video_count = len(niche_videos)
    if video_count < 20:
        video_saturation = 2
    elif video_count < 50:
        video_saturation = 4
    elif video_count < 100:
        video_saturation = 6
    elif video_count < 200:
        video_saturation = 8
    else:
        video_saturation = 10

    indicators["video_volume"] = {
        "count": video_count,
        "score": video_saturation,
    }

    # Creator density
    channel_count = len(niche_channels)
    if channel_count < 5:
        creator_saturation = 2
    elif channel_count < 10:
        creator_saturation = 4
    elif channel_count < 20:
        creator_saturation = 6
    elif channel_count < 50:
        creator_saturation = 8
    else:
        creator_saturation = 10

    indicators["creator_density"] = {
        "count": channel_count,
        "score": creator_saturation,
    }

    # Calculate overall saturation
    saturation_score = (video_saturation + creator_saturation) / 2

    if saturation_score <= 3:
        level = "unsaturated"
    elif saturation_score <= 5:
        level = "emerging"
    elif saturation_score <= 7:
        level = "mature"
    else:
        level = "saturated"

    return {
        "score": round(saturation_score, 2),
        "level": level,
        "indicators": indicators,
    }


__all__ = [
    "calculate_saturation_score",
    "generate_competition_recommendations",
    "generate_opportunity_recommendations",
    "rank_videos_by_performance",
    "score_channel_strength",
    "score_competition",
    "score_opportunity",
]
