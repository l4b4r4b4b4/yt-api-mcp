"""Pattern extraction functions for YouTube MCP Transform layer.

This module provides pattern analysis functions for extracting patterns
from titles, channel names, tags, and other text content from YouTube data.

Transform layer principles:
1. No API calls - only process cached data
2. Pure functions - no side effects
3. Return structured results for Intelligence layer
4. Computationally cheap (can be recomputed)
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.tools.youtube.transform.utils import (
    contains_emoji,
    contains_number,
    contains_question,
    extract_title_structure,
    extract_words,
    normalize_text,
    safe_int,
    word_frequency,
)


def analyze_title_patterns(titles: list[str]) -> dict[str, Any]:
    """Analyze patterns across a collection of video titles.

    Extracts common patterns, structures, word frequencies, and
    characteristics from a list of video titles.

    Args:
        titles: List of video title strings.

    Returns:
        Dictionary with pattern analysis including:
        - sample_size: Number of titles analyzed
        - avg_length: Average title length in characters
        - length_range: Min and max title lengths
        - structure_patterns: Distribution of title structures
        - common_words: Most frequent words
        - uses_numbers: Percentage using numbers
        - uses_questions: Percentage using questions
        - uses_emoji: Percentage using emojis

    Example:
        >>> titles = [
        ...     "How to Learn Python in 2024",
        ...     "Top 10 JavaScript Tips",
        ...     "React vs Vue: Which is Better?",
        ... ]
        >>> patterns = analyze_title_patterns(titles)
        >>> print(patterns["structure_patterns"])
        {"how_to": 33.33, "listicle": 33.33, "comparison": 33.33}
    """
    if not titles:
        return {
            "sample_size": 0,
            "avg_length": 0,
            "length_range": {"min": 0, "max": 0},
            "structure_patterns": {},
            "common_words": [],
            "uses_numbers": 0.0,
            "uses_questions": 0.0,
            "uses_emoji": 0.0,
        }

    # Calculate lengths
    lengths = [len(t) for t in titles]
    avg_length = sum(lengths) / len(lengths)

    # Extract structure patterns
    structure_counts: Counter[str] = Counter()
    for title in titles:
        structure = extract_title_structure(title)
        if structure:
            structure_counts[structure] += 1

    # Convert to percentages
    structure_patterns = {
        structure: round((count / len(titles)) * 100, 2)
        for structure, count in structure_counts.most_common()
    }

    # Count characteristics
    num_with_numbers = sum(1 for t in titles if contains_number(t))
    num_with_questions = sum(1 for t in titles if contains_question(t))
    num_with_emoji = sum(1 for t in titles if contains_emoji(t))

    # Get common words
    common = word_frequency(titles, top_n=20)

    return {
        "sample_size": len(titles),
        "avg_length": round(avg_length, 1),
        "length_range": {"min": min(lengths), "max": max(lengths)},
        "structure_patterns": structure_patterns,
        "common_words": [{"word": w, "count": c} for w, c in common],
        "uses_numbers": round((num_with_numbers / len(titles)) * 100, 2),
        "uses_questions": round((num_with_questions / len(titles)) * 100, 2),
        "uses_emoji": round((num_with_emoji / len(titles)) * 100, 2),
    }


def analyze_channel_naming_patterns(channel_names: list[str]) -> dict[str, Any]:
    """Analyze naming patterns across YouTube channels.

    Categorizes channel names and extracts patterns for naming strategy.

    Args:
        channel_names: List of channel name strings.

    Returns:
        Dictionary with naming pattern analysis including:
        - sample_size: Number of names analyzed
        - avg_length: Average name length
        - categories: Distribution of naming categories
        - common_words: Most frequent words in names
        - word_count_distribution: Distribution of word counts

    Example:
        >>> names = ["TechWorld with Nana", "Fireship", "Google Cloud Platform"]
        >>> patterns = analyze_channel_naming_patterns(names)
        >>> print(patterns["categories"])
        {"personal_brand": 33.33, "creative_abstract": 33.33, "company_brand": 33.33}
    """
    if not channel_names:
        return {
            "sample_size": 0,
            "avg_length": 0,
            "categories": {},
            "common_words": [],
            "word_count_distribution": {},
        }

    # Categorize names
    categories: Counter[str] = Counter()
    word_counts: list[int] = []

    for name in channel_names:
        category = categorize_channel_name(name)
        categories[category] += 1
        word_counts.append(len(name.split()))

    # Category percentages
    category_pcts = {
        cat: round((count / len(channel_names)) * 100, 2)
        for cat, count in categories.most_common()
    }

    # Word count distribution
    word_count_dist: Counter[int] = Counter(word_counts)
    word_count_distribution = {
        f"{wc}_words": round((count / len(channel_names)) * 100, 2)
        for wc, count in sorted(word_count_dist.items())
    }

    # Common words
    common = word_frequency(channel_names, top_n=15)

    return {
        "sample_size": len(channel_names),
        "avg_length": round(sum(len(n) for n in channel_names) / len(channel_names), 1),
        "categories": category_pcts,
        "common_words": [{"word": w, "count": c} for w, c in common],
        "word_count_distribution": word_count_distribution,
    }


def categorize_channel_name(name: str) -> str:
    """Categorize a channel name by type.

    Args:
        name: Channel name to categorize.

    Returns:
        Category string: "personal_brand", "topic_keyword",
        "creative_abstract", or "company_brand".

    Example:
        >>> categorize_channel_name("TechWorld with Nana")
        "personal_brand"
        >>> categorize_channel_name("Python Tutorials")
        "topic_keyword"
        >>> categorize_channel_name("Fireship")
        "creative_abstract"
    """
    name_lower = name.lower()
    words = name.split()

    # Personal brand indicators
    personal_indicators = [
        "with",
        "by",
        "'s",
        "official",
    ]
    # Check for "FirstName LastName" pattern (2 capitalized words)
    if (
        len(words) == 2
        and all(w[0].isupper() for w in words if w)
        and all(w.isalpha() for w in words)
    ):
        # Could be a personal name
        return "personal_brand"

    for indicator in personal_indicators:
        if indicator in name_lower:
            return "personal_brand"

    # Company brand indicators
    company_indicators = [
        "inc",
        "corp",
        "llc",
        "ltd",
        "official",
        "platform",
        "cloud",
        "microsoft",
        "google",
        "amazon",
        "meta",
        "apple",
    ]
    for indicator in company_indicators:
        if indicator in name_lower:
            return "company_brand"

    # Topic keyword indicators
    topic_indicators = [
        "tutorial",
        "learn",
        "academy",
        "course",
        "school",
        "programming",
        "coding",
        "dev",
        "tech",
        "science",
        "news",
        "tips",
        "guide",
        "how",
    ]
    for indicator in topic_indicators:
        if indicator in name_lower:
            return "topic_keyword"

    # Default to creative/abstract for short, unique names
    if len(words) <= 2 and len(name) <= 20:
        return "creative_abstract"

    return "topic_keyword"


def extract_tag_patterns(videos: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract patterns from video tags across multiple videos.

    Args:
        videos: List of video dictionaries with "tags" field.

    Returns:
        Dictionary with tag pattern analysis including:
        - sample_size: Number of videos with tags
        - avg_tags_per_video: Average number of tags
        - common_tags: Most frequently used tags
        - tag_categories: Tags grouped by apparent category

    Example:
        >>> videos = [
        ...     {"tags": ["python", "programming", "tutorial"]},
        ...     {"tags": ["python", "beginner", "coding"]},
        ... ]
        >>> patterns = extract_tag_patterns(videos)
        >>> print(patterns["common_tags"][0])
        {"tag": "python", "count": 2, "percentage": 100.0}
    """
    if not videos:
        return {
            "sample_size": 0,
            "avg_tags_per_video": 0,
            "common_tags": [],
            "tag_categories": {},
        }

    # Collect all tags
    all_tags: list[str] = []
    videos_with_tags = 0
    tag_counts_per_video: list[int] = []

    for video in videos:
        tags = video.get("tags", [])
        if tags:
            videos_with_tags += 1
            tag_counts_per_video.append(len(tags))
            all_tags.extend([normalize_text(t) for t in tags])

    if not all_tags:
        return {
            "sample_size": 0,
            "avg_tags_per_video": 0,
            "common_tags": [],
            "tag_categories": {},
        }

    # Count tag frequencies
    tag_counter = Counter(all_tags)
    common_tags = [
        {
            "tag": tag,
            "count": count,
            "percentage": round((count / videos_with_tags) * 100, 2),
        }
        for tag, count in tag_counter.most_common(30)
    ]

    # Categorize tags
    tag_categories = categorize_tags(list(tag_counter.keys()))

    return {
        "sample_size": videos_with_tags,
        "avg_tags_per_video": round(
            sum(tag_counts_per_video) / len(tag_counts_per_video), 1
        )
        if tag_counts_per_video
        else 0,
        "common_tags": common_tags,
        "tag_categories": tag_categories,
    }


def categorize_tags(tags: list[str]) -> dict[str, list[str]]:
    """Categorize tags into groups.

    Args:
        tags: List of tag strings.

    Returns:
        Dictionary mapping category names to lists of tags.

    Example:
        >>> tags = ["python", "tutorial", "beginner", "2024"]
        >>> categorize_tags(tags)
        {
            "technology": ["python"],
            "content_type": ["tutorial"],
            "audience": ["beginner"],
            "temporal": ["2024"]
        }
    """
    categories: dict[str, list[str]] = {
        "technology": [],
        "content_type": [],
        "audience": [],
        "temporal": [],
        "other": [],
    }

    technology_keywords = [
        "python",
        "javascript",
        "java",
        "react",
        "node",
        "docker",
        "kubernetes",
        "aws",
        "linux",
        "git",
        "sql",
        "api",
        "web",
        "mobile",
        "cloud",
        "ai",
        "ml",
        "machine learning",
        "devops",
        "backend",
        "frontend",
    ]

    content_type_keywords = [
        "tutorial",
        "guide",
        "course",
        "how to",
        "tips",
        "tricks",
        "review",
        "explained",
        "introduction",
        "basics",
        "advanced",
        "deep dive",
    ]

    audience_keywords = [
        "beginner",
        "intermediate",
        "advanced",
        "pro",
        "expert",
        "starter",
        "newbie",
    ]

    for tag in tags:
        tag_lower = tag.lower()
        categorized = False

        # Check for year/temporal
        if (tag.isdigit() and len(tag) == 4) or any(
            year in tag_lower for year in ["2023", "2024", "2025"]
        ):
            categories["temporal"].append(tag)
            categorized = True

        # Check technology
        if not categorized:
            for tech in technology_keywords:
                if tech in tag_lower:
                    categories["technology"].append(tag)
                    categorized = True
                    break

        # Check content type
        if not categorized:
            for content in content_type_keywords:
                if content in tag_lower:
                    categories["content_type"].append(tag)
                    categorized = True
                    break

        # Check audience
        if not categorized:
            for audience in audience_keywords:
                if audience in tag_lower:
                    categories["audience"].append(tag)
                    categorized = True
                    break

        # Default to other
        if not categorized:
            categories["other"].append(tag)

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def analyze_content_duration_patterns(
    videos: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze video duration patterns.

    Args:
        videos: List of video dictionaries with "duration" field (ISO 8601).

    Returns:
        Dictionary with duration pattern analysis including:
        - sample_size: Number of videos analyzed
        - avg_duration_minutes: Average duration in minutes
        - duration_distribution: Distribution across categories
        - short_form_ratio: Percentage of videos under 60 seconds

    Example:
        >>> videos = [
        ...     {"duration": "PT5M30S"},
        ...     {"duration": "PT10M"},
        ...     {"duration": "PT45S"},
        ... ]
        >>> patterns = analyze_content_duration_patterns(videos)
        >>> print(patterns["short_form_ratio"])
        33.33
    """
    if not videos:
        return {
            "sample_size": 0,
            "avg_duration_minutes": 0,
            "duration_distribution": {},
            "short_form_ratio": 0,
        }

    durations_seconds: list[int] = []

    for video in videos:
        duration_str = video.get("duration", "")
        if duration_str:
            seconds = parse_iso8601_duration(duration_str)
            if seconds > 0:
                durations_seconds.append(seconds)

    if not durations_seconds:
        return {
            "sample_size": 0,
            "avg_duration_minutes": 0,
            "duration_distribution": {},
            "short_form_ratio": 0,
        }

    # Categorize durations
    categories = {
        "shorts": 0,  # < 60 seconds
        "short": 0,  # 1-5 minutes
        "medium": 0,  # 5-15 minutes
        "long": 0,  # 15-30 minutes
        "very_long": 0,  # > 30 minutes
    }

    for seconds in durations_seconds:
        if seconds < 60:
            categories["shorts"] += 1
        elif seconds < 300:
            categories["short"] += 1
        elif seconds < 900:
            categories["medium"] += 1
        elif seconds < 1800:
            categories["long"] += 1
        else:
            categories["very_long"] += 1

    # Convert to percentages
    distribution = {
        cat: round((count / len(durations_seconds)) * 100, 2)
        for cat, count in categories.items()
        if count > 0
    }

    avg_minutes = (sum(durations_seconds) / len(durations_seconds)) / 60

    return {
        "sample_size": len(durations_seconds),
        "avg_duration_minutes": round(avg_minutes, 2),
        "duration_distribution": distribution,
        "short_form_ratio": distribution.get("shorts", 0),
    }


def parse_iso8601_duration(duration: str) -> int:
    """Parse ISO 8601 duration string to seconds.

    Args:
        duration: ISO 8601 duration string (e.g., "PT1H30M45S").

    Returns:
        Duration in seconds.

    Example:
        >>> parse_iso8601_duration("PT1H30M45S")
        5445
        >>> parse_iso8601_duration("PT5M30S")
        330
        >>> parse_iso8601_duration("PT45S")
        45
    """
    if not duration or not duration.startswith("PT"):
        return 0

    duration = duration[2:]  # Remove "PT" prefix
    total_seconds = 0

    # Extract hours
    if "H" in duration:
        hours_part, duration = duration.split("H")
        total_seconds += safe_int(hours_part) * 3600

    # Extract minutes
    if "M" in duration:
        minutes_part, duration = duration.split("M")
        total_seconds += safe_int(minutes_part) * 60

    # Extract seconds
    if "S" in duration:
        seconds_part = duration.replace("S", "")
        total_seconds += safe_int(seconds_part)

    return total_seconds


def find_title_keywords_by_performance(
    videos: list[dict[str, Any]],
    metric: str = "view_count",
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Find keywords correlated with high performance.

    Analyzes which title words appear in high-performing videos.

    Args:
        videos: List of video dictionaries with title and metric fields.
        metric: Performance metric to use ("view_count", "like_count").
        top_n: Number of keywords to return.

    Returns:
        List of keyword dictionaries with word, avg_metric, and count.

    Example:
        >>> videos = [
        ...     {"title": "Python Tutorial", "view_count": "10000"},
        ...     {"title": "Python Basics", "view_count": "5000"},
        ...     {"title": "Java Tutorial", "view_count": "3000"},
        ... ]
        >>> keywords = find_title_keywords_by_performance(videos)
        >>> print(keywords[0])
        {"word": "python", "avg_views": 7500.0, "count": 2}
    """
    if not videos:
        return []

    # Collect word -> metric values
    word_metrics: dict[str, list[int]] = {}

    for video in videos:
        title = video.get("title", "")
        metric_value = safe_int(video.get(metric, 0))

        if not title or metric_value <= 0:
            continue

        words = extract_words(title, min_length=3)
        for word in set(words):  # Unique words per title
            if word not in word_metrics:
                word_metrics[word] = []
            word_metrics[word].append(metric_value)

    # Calculate averages
    word_stats = []
    for word, values in word_metrics.items():
        if len(values) >= 2:  # Require at least 2 occurrences
            avg_value = sum(values) / len(values)
            word_stats.append(
                {
                    "word": word,
                    f"avg_{metric}": round(avg_value, 2),
                    "count": len(values),
                }
            )

    # Sort by average metric value
    word_stats.sort(key=lambda x: x[f"avg_{metric}"], reverse=True)

    return word_stats[:top_n]


__all__ = [
    "analyze_channel_naming_patterns",
    "analyze_content_duration_patterns",
    "analyze_title_patterns",
    "categorize_channel_name",
    "categorize_tags",
    "extract_tag_patterns",
    "find_title_keywords_by_performance",
    "parse_iso8601_duration",
]
