"""Common utilities for YouTube MCP Transform layer.

This module provides shared utility functions used across transform modules
for text analysis, data processing, and common operations.

Key utilities:
- Text normalization and cleaning
- Word frequency analysis
- Common pattern detection
- Statistical helpers
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize text for analysis.

    Performs basic text normalization:
    - Lowercase
    - Remove extra whitespace
    - Strip leading/trailing whitespace

    Args:
        text: Input text to normalize.

    Returns:
        Normalized text string.

    Example:
        >>> normalize_text("  Hello   World  ")
        "hello world"
    """
    if not text:
        return ""
    # Lowercase and normalize whitespace
    return " ".join(text.lower().split())


def extract_words(text: str, min_length: int = 3) -> list[str]:
    """Extract words from text.

    Extracts alphabetic words from text, optionally filtering by length.

    Args:
        text: Input text.
        min_length: Minimum word length to include (default: 3).

    Returns:
        List of extracted words (lowercase).

    Example:
        >>> extract_words("Hello World! This is a test.")
        ["hello", "world", "this", "test"]
    """
    if not text:
        return []

    # Extract words (alphabetic only)
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    # Filter by length
    return [w for w in words if len(w) >= min_length]


def word_frequency(texts: list[str], top_n: int = 20) -> list[tuple[str, int]]:
    """Calculate word frequency across multiple texts.

    Args:
        texts: List of text strings to analyze.
        top_n: Number of top words to return (default: 20).

    Returns:
        List of (word, count) tuples, sorted by frequency.

    Example:
        >>> titles = ["Python Tutorial", "Python Basics", "Java Tutorial"]
        >>> word_frequency(titles, top_n=3)
        [("python", 2), ("tutorial", 2), ("java", 1)]
    """
    all_words: list[str] = []

    for text in texts:
        all_words.extend(extract_words(text))

    counter = Counter(all_words)
    return counter.most_common(top_n)


def contains_number(text: str) -> bool:
    """Check if text contains any numbers.

    Args:
        text: Input text to check.

    Returns:
        True if text contains digits, False otherwise.

    Example:
        >>> contains_number("Top 10 Tips")
        True
        >>> contains_number("Best Practices")
        False
    """
    return bool(re.search(r"\d", text))


def contains_question(text: str) -> bool:
    """Check if text contains a question.

    Detects questions by looking for question marks or question words.

    Args:
        text: Input text to check.

    Returns:
        True if text appears to be a question, False otherwise.

    Example:
        >>> contains_question("How to learn Python?")
        True
        >>> contains_question("What is Docker")
        True
        >>> contains_question("Python Tutorial")
        False
    """
    if "?" in text:
        return True

    # Check for question words at start
    question_words = [
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "can",
        "could",
        "would",
        "should",
        "is",
        "are",
        "do",
        "does",
        "did",
        "will",
    ]

    text_lower = text.lower().strip()
    return any(text_lower.startswith(word + " ") for word in question_words)


def contains_emoji(text: str) -> bool:
    """Check if text contains emoji characters.

    Args:
        text: Input text to check.

    Returns:
        True if text contains emojis, False otherwise.

    Example:
        >>> contains_emoji("Great video! 🎉")
        True
        >>> contains_emoji("Great video!")
        False
    """
    # Emoji regex pattern (simplified, covers most common emojis)
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # Emoticons
        "\U0001f300-\U0001f5ff"  # Misc Symbols and Pictographs
        "\U0001f680-\U0001f6ff"  # Transport and Map
        "\U0001f700-\U0001f77f"  # Alchemical Symbols
        "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
        "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
        "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        "\U0001fa00-\U0001fa6f"  # Chess Symbols
        "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027b0"  # Dingbats
        "\U000024c2-\U0001f251"  # Enclosed characters
        "]+",
        flags=re.UNICODE,
    )
    return bool(emoji_pattern.search(text))


def extract_title_structure(title: str) -> str | None:
    """Identify the structure pattern of a title.

    Detects common title structures like "How to X", "X vs Y", "Top N X".

    Args:
        title: Video title to analyze.

    Returns:
        Structure pattern name, or None if no known pattern.

    Example:
        >>> extract_title_structure("How to Learn Python in 2024")
        "how_to"
        >>> extract_title_structure("React vs Vue: Which is Better?")
        "comparison"
        >>> extract_title_structure("Top 10 JavaScript Tips")
        "listicle"
    """
    title_lower = title.lower().strip()

    # How to / Tutorial patterns
    if title_lower.startswith(("how to ", "how i ", "learn to ")):
        return "how_to"

    # Comparison patterns
    if " vs " in title_lower or " vs. " in title_lower:
        return "comparison"

    # Listicle patterns (Top N, N things, N ways, etc.)
    if re.match(r"^(top\s+)?\d+\s+", title_lower):
        return "listicle"

    # Question patterns
    if contains_question(title):
        return "question"

    # Tutorial/Guide patterns
    if any(
        word in title_lower
        for word in ["tutorial", "guide", "course", "complete", "full"]
    ):
        return "tutorial"

    # Review patterns
    if any(word in title_lower for word in ["review", "honest", "opinion"]):
        return "review"

    # News/Update patterns
    if any(
        word in title_lower
        for word in ["update", "news", "announcement", "breaking", "new in"]
    ):
        return "news"

    return None


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int.

    Args:
        value: Value to convert.
        default: Default value if conversion fails (default: 0).

    Returns:
        Integer value or default.

    Example:
        >>> safe_int("123")
        123
        >>> safe_int("abc")
        0
        >>> safe_int(None, -1)
        -1
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float.

    Args:
        value: Value to convert.
        default: Default value if conversion fails (default: 0.0).

    Returns:
        Float value or default.

    Example:
        >>> safe_float("3.14")
        3.14
        >>> safe_float("abc")
        0.0
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length.

    Args:
        text: Text to truncate.
        max_length: Maximum length (default: 100).
        suffix: Suffix to add when truncated (default: "...").

    Returns:
        Truncated text with suffix if needed.

    Example:
        >>> truncate_text("This is a long text", max_length=10)
        "This is..."
    """
    if not text or len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple word-based similarity between texts.

    Uses Jaccard similarity on word sets.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Similarity score between 0.0 and 1.0.

    Example:
        >>> calculate_text_similarity("python tutorial", "python course")
        0.333...  # 1 shared word / 3 total unique words
    """
    words1 = set(extract_words(text1))
    words2 = set(extract_words(text2))

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


__all__ = [
    "calculate_text_similarity",
    "contains_emoji",
    "contains_number",
    "contains_question",
    "extract_title_structure",
    "extract_words",
    "normalize_text",
    "safe_float",
    "safe_int",
    "truncate_text",
    "word_frequency",
]
