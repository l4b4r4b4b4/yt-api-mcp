"""Content gap identification functions for YouTube MCP Transform layer.

This module provides functions for identifying content gaps and underserved
topics by analyzing existing content, comments, and search patterns.

Transform layer principles:
1. No API calls - only process cached data
2. Pure functions - no side effects
3. Return structured results for Intelligence layer
4. Computationally cheap (can be recomputed)
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.tools.youtube.transform.statistics import calculate_basic_stats
from app.tools.youtube.transform.utils import (
    extract_words,
    normalize_text,
    safe_int,
)


def identify_content_gaps(
    videos: list[dict[str, Any]],
    comments: list[dict[str, Any]] | None = None,
    related_topics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Identify content gaps from existing videos and audience feedback.

    Analyzes video content and comments to find topics that are
    underserved or frequently requested by the audience.

    Gap identification strategies:
    1. Question analysis from comments
    2. Low-coverage topics mentioned in comments
    3. Related topics not covered in videos
    4. Topics with high engagement but low content volume

    Args:
        videos: List of video dictionaries with titles, descriptions, tags.
        comments: Optional list of comment dictionaries with text.
        related_topics: Optional list of related topic strings to check.

    Returns:
        List of content gap dictionaries, each containing:
        - topic: The gap topic/subtopic
        - evidence: Why this is identified as a gap
        - demand_signals: Evidence of audience demand
        - existing_coverage: Level of existing content
        - opportunity_score: Score from 1-10
        - suggested_content: Content format suggestions

    Example:
        >>> gaps = identify_content_gaps(videos, comments)
        >>> print(gaps[0])
        {
            "topic": "advanced kubernetes networking",
            "evidence": "Frequently asked in comments",
            "demand_signals": ["15 questions about networking"],
            "existing_coverage": "low",
            "opportunity_score": 8.5,
            "suggested_content": ["Deep dive tutorial", "Practical examples"]
        }
    """
    gaps = []

    # Strategy 1: Analyze questions from comments
    if comments:
        question_gaps = analyze_comment_questions(comments, videos)
        gaps.extend(question_gaps)

    # Strategy 2: Find topics mentioned in comments but not in videos
    if comments and videos:
        topic_gaps = find_uncovered_topics(videos, comments)
        gaps.extend(topic_gaps)

    # Strategy 3: Check related topics not covered
    if related_topics and videos:
        related_gaps = check_related_topic_coverage(videos, related_topics)
        gaps.extend(related_gaps)

    # Strategy 4: Find high-demand low-supply topics
    if videos:
        demand_gaps = analyze_demand_supply_mismatch(videos)
        gaps.extend(demand_gaps)

    # Deduplicate and sort by opportunity score
    unique_gaps = deduplicate_gaps(gaps)
    unique_gaps.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)

    return unique_gaps


def analyze_comment_questions(
    comments: list[dict[str, Any]],
    videos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract content gaps from questions in comments.

    Identifies questions that appear frequently in comments but
    may not be adequately addressed by existing content.

    Args:
        comments: List of comment dictionaries with text.
        videos: List of video dictionaries for context.

    Returns:
        List of gap dictionaries from question analysis.
    """
    if not comments:
        return []

    # Extract questions from comments
    question_texts = []
    question_keywords: list[str] = []

    for comment in comments:
        text = comment.get("text", "")
        if not text:
            continue

        # Check if comment contains a question
        if "?" in text or any(
            text.lower().startswith(w)
            for w in ["how", "what", "why", "when", "where", "which", "can", "is"]
        ):
            question_texts.append(text)
            # Extract key terms from question
            words = extract_words(text, min_length=4)
            question_keywords.extend(words)

    if not question_keywords:
        return []

    # Find frequently asked topics
    keyword_counts = Counter(question_keywords)
    frequent_topics = keyword_counts.most_common(10)

    # Check coverage in existing videos
    video_content = " ".join(
        v.get("title", "") + " " + v.get("description", "") for v in videos
    ).lower()

    gaps = []
    for topic, count in frequent_topics:
        # Skip common words
        if topic in {"this", "that", "what", "have", "does", "would", "about"}:
            continue

        # Check if topic is covered in videos
        coverage_count = video_content.count(topic)

        if coverage_count < 3:  # Low coverage threshold
            opportunity_score = min(10, count * 1.5 + (3 - coverage_count) * 2)

            gaps.append(
                {
                    "topic": topic,
                    "evidence": f"Asked {count} times in comments with low video coverage",
                    "demand_signals": [f"{count} questions mentioning '{topic}'"],
                    "existing_coverage": "low" if coverage_count < 2 else "moderate",
                    "opportunity_score": round(opportunity_score, 2),
                    "suggested_content": [
                        f"Tutorial: Understanding {topic}",
                        f"FAQ: Common {topic} questions answered",
                    ],
                }
            )

    return gaps


def find_uncovered_topics(
    videos: list[dict[str, Any]],
    comments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find topics mentioned in comments but not covered in videos.

    Identifies discrepancies between audience interests (comments)
    and creator content (videos).

    Args:
        videos: List of video dictionaries.
        comments: List of comment dictionaries.

    Returns:
        List of gap dictionaries for uncovered topics.
    """
    if not videos or not comments:
        return []

    # Build vocabulary from videos
    video_vocabulary = set()
    for video in videos:
        title_words = extract_words(video.get("title", ""), min_length=4)
        desc_words = extract_words(video.get("description", ""), min_length=4)
        video_vocabulary.update(title_words)
        video_vocabulary.update(desc_words)

    # Build vocabulary from comments
    comment_words: list[str] = []
    for comment in comments:
        text = comment.get("text", "")
        words = extract_words(text, min_length=4)
        comment_words.extend(words)

    # Find frequently mentioned comment topics not in video vocabulary
    comment_freq = Counter(comment_words)
    uncovered = []

    for word, count in comment_freq.most_common(30):
        if word not in video_vocabulary and count >= 3:
            # This topic is discussed by audience but not in videos
            opportunity_score = min(10, count * 0.8)
            uncovered.append(
                {
                    "topic": word,
                    "evidence": f"Mentioned {count} times by audience, not covered in videos",
                    "demand_signals": [f"{count} comment mentions"],
                    "existing_coverage": "none",
                    "opportunity_score": round(opportunity_score, 2),
                    "suggested_content": [
                        f"Introduction to {word}",
                        f"Audience request: {word} explained",
                    ],
                }
            )

    return uncovered[:5]  # Limit to top 5


def check_related_topic_coverage(
    videos: list[dict[str, Any]],
    related_topics: list[str],
) -> list[dict[str, Any]]:
    """Check if related topics are covered in existing videos.

    Args:
        videos: List of video dictionaries.
        related_topics: List of related topic strings to check.

    Returns:
        List of gap dictionaries for uncovered related topics.
    """
    if not videos or not related_topics:
        return []

    # Build searchable content from videos
    all_content = " ".join(
        v.get("title", "").lower()
        + " "
        + v.get("description", "").lower()
        + " "
        + " ".join(v.get("tags", []))
        for v in videos
    )

    gaps = []
    for topic in related_topics:
        topic_lower = topic.lower()
        topic_words = topic_lower.split()

        # Check coverage (all words must appear)
        covered = all(word in all_content for word in topic_words if len(word) > 3)

        if not covered:
            gaps.append(
                {
                    "topic": topic,
                    "evidence": "Related topic not covered in existing content",
                    "demand_signals": ["Related to main niche"],
                    "existing_coverage": "none",
                    "opportunity_score": 6.0,  # Moderate - we don't know demand
                    "suggested_content": [
                        f"Complete guide to {topic}",
                        f"{topic} for beginners",
                    ],
                }
            )

    return gaps


def analyze_demand_supply_mismatch(
    videos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find topics with high demand (views) but low supply (content).

    Analyzes view counts to identify topics where few videos
    get disproportionately high views, indicating unmet demand.

    Args:
        videos: List of video dictionaries with view_count and title.

    Returns:
        List of gap dictionaries for high-demand low-supply topics.
    """
    if not videos or len(videos) < 5:
        return []

    # Group videos by topic keywords
    topic_videos: dict[str, list[dict[str, Any]]] = {}

    for video in videos:
        title = video.get("title", "")
        words = extract_words(title, min_length=4)

        for word in words:
            if word not in topic_videos:
                topic_videos[word] = []
            topic_videos[word].append(video)

    # Find topics with few videos but high average views
    gaps = []
    overall_stats = calculate_basic_stats(
        [safe_int(v.get("view_count", 0)) for v in videos]
    )
    overall_avg = overall_stats["mean"]

    for topic, topic_vids in topic_videos.items():
        if len(topic_vids) < 2 or len(topic_vids) > 10:
            continue  # Skip very rare or very common topics

        topic_views = [safe_int(v.get("view_count", 0)) for v in topic_vids]
        topic_avg = sum(topic_views) / len(topic_views) if topic_views else 0

        # High demand indicator: topic avg views > 1.5x overall avg
        # Low supply indicator: few videos on this topic
        if topic_avg > overall_avg * 1.5 and len(topic_vids) <= 5:
            opportunity_score = min(
                10, (topic_avg / overall_avg) * 2 + (5 - len(topic_vids))
            )

            gaps.append(
                {
                    "topic": topic,
                    "evidence": f"High views ({int(topic_avg):,}) with only {len(topic_vids)} videos",
                    "demand_signals": [
                        f"Avg views {int(topic_avg):,} vs overall {int(overall_avg):,}",
                        f"Only {len(topic_vids)} videos on this topic",
                    ],
                    "existing_coverage": "low",
                    "opportunity_score": round(opportunity_score, 2),
                    "suggested_content": [
                        f"More content about {topic}",
                        f"{topic} deep dive series",
                    ],
                }
            )

    return sorted(gaps, key=lambda x: x["opportunity_score"], reverse=True)[:5]


def deduplicate_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate gaps and merge similar ones.

    Args:
        gaps: List of gap dictionaries.

    Returns:
        Deduplicated list of gaps.
    """
    if not gaps:
        return []

    seen_topics = set()
    unique_gaps = []

    for gap in gaps:
        topic = normalize_text(gap.get("topic", ""))
        if topic and topic not in seen_topics:
            seen_topics.add(topic)
            unique_gaps.append(gap)

    return unique_gaps


def prioritize_gaps(
    gaps: list[dict[str, Any]],
    criteria: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Prioritize content gaps based on weighted criteria.

    Args:
        gaps: List of gap dictionaries.
        criteria: Optional weights for prioritization factors.
            Keys: "opportunity", "demand", "difficulty"

    Returns:
        Prioritized list of gaps with priority scores.
    """
    if not gaps:
        return []

    if criteria is None:
        criteria = {
            "opportunity": 0.5,
            "demand": 0.3,
            "difficulty": 0.2,
        }

    prioritized = []
    for gap in gaps:
        # Calculate priority score
        opportunity = gap.get("opportunity_score", 5)

        # Estimate demand from signals
        demand_signals = gap.get("demand_signals", [])
        demand = min(10, len(demand_signals) * 3)

        # Estimate difficulty (inverse of coverage)
        coverage = gap.get("existing_coverage", "moderate")
        if coverage == "none":
            difficulty_score = 8  # Easier - no competition
        elif coverage == "low":
            difficulty_score = 6
        elif coverage == "moderate":
            difficulty_score = 4
        else:
            difficulty_score = 2  # Harder - lots of competition

        priority_score = (
            opportunity * criteria.get("opportunity", 0.5)
            + demand * criteria.get("demand", 0.3)
            + difficulty_score * criteria.get("difficulty", 0.2)
        )

        prioritized.append(
            {
                **gap,
                "priority_score": round(priority_score, 2),
                "priority_breakdown": {
                    "opportunity": opportunity,
                    "demand": demand,
                    "difficulty": difficulty_score,
                },
            }
        )

    prioritized.sort(key=lambda x: x["priority_score"], reverse=True)
    return prioritized


def estimate_gap_effort(gap: dict[str, Any]) -> dict[str, Any]:
    """Estimate effort required to fill a content gap.

    Args:
        gap: Gap dictionary with topic and evidence.

    Returns:
        Dictionary with effort estimates.
    """
    topic = gap.get("topic", "")
    coverage = gap.get("existing_coverage", "moderate")
    suggested = gap.get("suggested_content", [])

    # Base effort on coverage level
    if coverage == "none":
        research_effort = "low"  # Less to research
        production_effort = "medium"
    elif coverage == "low":
        research_effort = "medium"
        production_effort = "medium"
    else:
        research_effort = "high"  # Lots to differentiate from
        production_effort = "high"

    # Estimate time based on content suggestions
    estimated_hours = 4 + len(suggested) * 2  # Base + per suggested content

    return {
        "topic": topic,
        "research_effort": research_effort,
        "production_effort": production_effort,
        "estimated_hours": estimated_hours,
        "content_pieces": len(suggested),
        "recommendation": (
            "Start with a comprehensive overview"
            if coverage == "none"
            else "Focus on unique angle or deeper insights"
        ),
    }


def generate_gap_report(gaps: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a comprehensive content gap report.

    Args:
        gaps: List of identified content gaps.

    Returns:
        Report dictionary with summary and recommendations.
    """
    if not gaps:
        return {
            "summary": "No significant content gaps identified",
            "total_gaps": 0,
            "high_priority": [],
            "quick_wins": [],
            "long_term": [],
            "recommendations": ["Continue monitoring audience feedback"],
        }

    # Categorize gaps
    high_priority = [g for g in gaps if g.get("opportunity_score", 0) >= 7]
    quick_wins = [
        g
        for g in gaps
        if g.get("existing_coverage") == "none" and g.get("opportunity_score", 0) >= 5
    ]
    long_term = [
        g
        for g in gaps
        if g.get("existing_coverage") != "none" and g.get("opportunity_score", 0) >= 6
    ]

    # Generate recommendations
    recommendations = []
    if high_priority:
        recommendations.append(
            f"Prioritize {len(high_priority)} high-opportunity gaps immediately"
        )
    if quick_wins:
        recommendations.append(
            f"{len(quick_wins)} quick wins with no existing coverage"
        )
    if len(gaps) > 10:
        recommendations.append("Many gaps found - focus on top 5 first")

    return {
        "summary": f"Identified {len(gaps)} content gaps",
        "total_gaps": len(gaps),
        "high_priority": high_priority[:5],
        "quick_wins": quick_wins[:3],
        "long_term": long_term[:3],
        "average_opportunity_score": round(
            sum(g.get("opportunity_score", 0) for g in gaps) / len(gaps), 2
        ),
        "recommendations": recommendations,
    }


__all__ = [
    "analyze_comment_questions",
    "analyze_demand_supply_mismatch",
    "check_related_topic_coverage",
    "deduplicate_gaps",
    "estimate_gap_effort",
    "find_uncovered_topics",
    "generate_gap_report",
    "identify_content_gaps",
    "prioritize_gaps",
]
