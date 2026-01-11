"""YouTube transcript tools - retrieve and manage video transcripts.

This module provides tools for retrieving YouTube video transcripts with
intelligent caching and multiple access patterns (list, preview, full, chunked).
Uses the youtube-transcript-api library for transcript retrieval.
"""

from __future__ import annotations

from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from app.tools.youtube.models import (
    AvailableTranscripts,
    FullTranscript,
    TranscriptChunk,
    TranscriptEntry,
    TranscriptInfo,
    TranscriptPreview,
)


class TranscriptError(Exception):
    """Base exception for transcript-related errors."""

    pass


async def list_available_transcripts(video_id: str) -> dict[str, Any]:
    """List all available transcript languages for a YouTube video.

    Discovers which transcript languages are available for a video,
    including both manual and auto-generated transcripts.
    Results should be cached permanently via the youtube.content namespace.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")

    Returns:
        AvailableTranscripts dictionary with:
        - video_id: The video ID
        - available_languages: List of language codes
        - transcript_info: Detailed info for each transcript (language,
          language_code, is_generated, is_translatable)

    Raises:
        ValueError: If video_id is empty or invalid format
        TranscriptError: If transcripts are disabled or video not found

    Example:
        >>> transcripts = await list_available_transcripts("dQw4w9WgXcQ")
        >>> print(transcripts["available_languages"])
        ["en", "es", "fr"]
        >>> print(transcripts["transcript_info"][0]["language"])
        "English"
    """
    if not video_id or not isinstance(video_id, str):
        raise ValueError("video_id must be a non-empty string")

    # Basic validation of video ID format (11 characters)
    if len(video_id) != 11:
        raise ValueError(
            f"Invalid video ID format: {video_id}. "
            "YouTube video IDs are 11 characters long."
        )

    try:
        # Get the transcript list for this video
        transcript_list = YouTubeTranscriptApi().list(video_id)

        available_languages: list[str] = []
        transcript_info: list[dict[str, Any]] = []

        # Iterate through all available transcripts
        for transcript in transcript_list:
            language_code = transcript.language_code
            available_languages.append(language_code)

            info = TranscriptInfo(
                language=transcript.language,
                language_code=language_code,
                is_generated=transcript.is_generated,
                is_translatable=transcript.is_translatable,
            )
            transcript_info.append(info.model_dump())

        # Build the result
        result = AvailableTranscripts(
            video_id=video_id,
            available_languages=available_languages,
            transcript_info=transcript_info,
        )

        return result.model_dump()

    except TranscriptsDisabled as e:
        raise TranscriptError(
            f"Transcripts are disabled for video {video_id}. "
            "The video owner has disabled captions/transcripts."
        ) from e
    except NoTranscriptFound as e:
        raise TranscriptError(
            f"No transcripts found for video {video_id}. "
            "This video may not have any captions available."
        ) from e
    except Exception as e:
        raise TranscriptError(
            f"Failed to list transcripts for video {video_id}: {e!s}"
        ) from e


async def get_video_transcript_preview(
    video_id: str,
    language: str = "",
    max_chars: int = 2000,
) -> dict[str, Any]:
    """Get a preview of a YouTube video transcript.

    Retrieves the first N characters of a video transcript for quick preview.
    Useful for LLMs to decide if they need the full transcript.
    Results should be cached permanently via the youtube.content namespace.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")
        language: ISO 639-1 language code (e.g., "en", "es"). If empty,
            uses first available transcript (prefers manual over auto-generated)
        max_chars: Maximum characters to return in preview (default: 2000)

    Returns:
        TranscriptPreview dictionary with:
        - video_id: The video ID
        - language: Language code used
        - preview: First N characters of transcript
        - total_length: Total character count of full transcript
        - is_truncated: Whether preview is truncated

    Raises:
        ValueError: If video_id is invalid or max_chars <= 0
        TranscriptError: If transcripts unavailable or language not found

    Example:
        >>> preview = await get_video_transcript_preview("dQw4w9WgXcQ", max_chars=500)
        >>> print(preview["preview"][:50])
        "We're no strangers to love..."
        >>> print(preview["is_truncated"])
        True
    """
    if not video_id or not isinstance(video_id, str):
        raise ValueError("video_id must be a non-empty string")

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    # Fetch the full transcript data
    transcript_data = await _fetch_transcript_data(video_id, language)

    # Extract full text
    full_text = transcript_data["full_text"]
    language_code = transcript_data["language_code"]
    total_length = len(full_text)

    # Create preview (truncate if needed)
    is_truncated = total_length > max_chars
    preview_text = full_text[:max_chars] if is_truncated else full_text

    # Build the result
    result = TranscriptPreview(
        video_id=video_id,
        language=language_code,
        preview=preview_text,
        total_length=total_length,
        is_truncated=is_truncated,
    )

    return result.model_dump()


async def get_full_transcript(
    video_id: str,
    language: str = "",
) -> dict[str, Any]:
    """Get the complete transcript for a YouTube video.

    Retrieves the full transcript with all entries and timestamps.
    For large transcripts, RefCache will automatically return a reference
    with preview that can be paginated using get_cached_result.
    Results should be cached permanently via the youtube.content namespace.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")
        language: ISO 639-1 language code (e.g., "en", "es"). If empty,
            uses first available transcript (prefers manual over auto-generated)

    Returns:
        FullTranscript dictionary with:
        - video_id: The video ID
        - language: Language code used
        - transcript: List of TranscriptEntry dicts with text, start, duration
        - full_text: Complete transcript as plain text (concatenated)

    Raises:
        ValueError: If video_id is invalid
        TranscriptError: If transcripts unavailable or language not found

    Example:
        >>> full = await get_full_transcript("dQw4w9WgXcQ", language="en")
        >>> print(len(full["transcript"]))
        150
        >>> print(full["transcript"][0])
        {"text": "We're no strangers to love", "start": 0.0, "duration": 2.5}
        >>> print(full["full_text"][:50])
        "We're no strangers to love You know the rules..."

    Note:
        - RefCache may return a preview + reference for large transcripts
        - Use get_cached_result() to retrieve full data if needed
        - Use get_transcript_chunk() for entry-by-entry pagination
    """
    if not video_id or not isinstance(video_id, str):
        raise ValueError("video_id must be a non-empty string")

    # Fetch the full transcript data
    transcript_data = await _fetch_transcript_data(video_id, language)

    # Build TranscriptEntry models for each entry
    entries: list[dict[str, Any]] = []
    for entry_data in transcript_data["entries"]:
        entry = TranscriptEntry(
            text=entry_data["text"],
            start=entry_data["start"],
            duration=entry_data["duration"],
        )
        entries.append(entry.model_dump())

    # Build the result
    result = FullTranscript(
        video_id=video_id,
        language=transcript_data["language_code"],
        transcript=entries,
        full_text=transcript_data["full_text"],
    )

    return result.model_dump()


async def get_transcript_chunk(
    video_id: str,
    start_index: int = 0,
    chunk_size: int = 50,
    language: str = "",
) -> dict[str, Any]:
    """Get a chunk of transcript entries for pagination.

    Retrieves a subset of transcript entries for large transcripts.
    Useful for iterating through transcripts entry-by-entry without
    loading the entire transcript into memory.
    Results should be cached permanently via the youtube.content namespace.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")
        start_index: Starting index of entries to return (0-based, default: 0)
        chunk_size: Number of entries to return (default: 50)
        language: ISO 639-1 language code (e.g., "en", "es"). If empty,
            uses first available transcript

    Returns:
        TranscriptChunk dictionary with:
        - video_id: The video ID
        - language: Language code used
        - start_index: Starting index of this chunk
        - chunk_size: Number of entries in this chunk
        - entries: List of TranscriptEntry dicts in this chunk
        - total_entries: Total number of entries in full transcript
        - has_more: Whether more entries are available after this chunk

    Raises:
        ValueError: If video_id invalid, start_index < 0, or chunk_size <= 0
        TranscriptError: If transcripts unavailable or index out of bounds

    Example:
        >>> chunk = await get_transcript_chunk("dQw4w9WgXcQ", start_index=0, chunk_size=10)
        >>> print(len(chunk["entries"]))
        10
        >>> print(chunk["total_entries"])
        150
        >>> print(chunk["has_more"])
        True
        >>> # Get next chunk
        >>> chunk2 = await get_transcript_chunk("dQw4w9WgXcQ", start_index=10, chunk_size=10)
    """
    if not video_id or not isinstance(video_id, str):
        raise ValueError("video_id must be a non-empty string")

    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    # Fetch the full transcript data
    transcript_data = await _fetch_transcript_data(video_id, language)

    # Get all entries
    all_entries = transcript_data["entries"]
    total_entries = len(all_entries)

    # Validate start_index
    if start_index >= total_entries:
        raise ValueError(
            f"start_index {start_index} out of bounds. "
            f"Total entries: {total_entries} (valid range: 0-{total_entries - 1})"
        )

    # Calculate end index
    end_index = min(start_index + chunk_size, total_entries)

    # Extract the chunk
    chunk_entries_data = all_entries[start_index:end_index]

    # Build TranscriptEntry models for the chunk
    chunk_entries: list[dict[str, Any]] = []
    for entry_data in chunk_entries_data:
        entry = TranscriptEntry(
            text=entry_data["text"],
            start=entry_data["start"],
            duration=entry_data["duration"],
        )
        chunk_entries.append(entry.model_dump())

    # Determine if there are more entries after this chunk
    has_more = end_index < total_entries

    # Build the result
    result = TranscriptChunk(
        video_id=video_id,
        language=transcript_data["language_code"],
        start_index=start_index,
        chunk_size=len(chunk_entries),
        entries=chunk_entries,
        total_entries=total_entries,
        has_more=has_more,
    )

    return result.model_dump()


# =============================================================================
# Internal Helper Functions
# =============================================================================


async def _fetch_transcript_data(video_id: str, language: str = "") -> dict[str, Any]:
    """Internal helper to fetch and parse transcript data.

    This function is shared by all transcript tools to avoid duplication.
    It handles language selection, transcript fetching, and entry parsing.

    Args:
        video_id: YouTube video ID
        language: Optional language code. If empty, uses first available.

    Returns:
        Dictionary with:
        - language_code: The language code used
        - is_generated: Whether this is auto-generated
        - entries: List of dicts with text, start, duration
        - full_text: Concatenated text of all entries

    Raises:
        TranscriptError: If transcripts unavailable or language not found
    """
    try:
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi().list(video_id)

        # Select the appropriate transcript
        if language:
            # Try to get the specific language requested
            try:
                transcript = transcript_list.find_transcript([language])
            except NoTranscriptFound as e:
                # Get available languages for error message
                available = [t.language_code for t in transcript_list]
                raise TranscriptError(
                    f"No transcript found for language '{language}'. "
                    f"Available languages: {', '.join(available)}. "
                    "Use list_available_transcripts() to see all options."
                ) from e
        else:
            # Get first available transcript (YouTube API returns manual first)
            # We need to iterate to get the first one
            first_transcript = None
            for t in transcript_list:
                if first_transcript is None:
                    first_transcript = t
                    break

            if first_transcript is None:
                raise TranscriptError(f"No transcripts available for video {video_id}")

            transcript = first_transcript

        # Fetch the transcript data
        transcript_entries = transcript.fetch()

        # Parse entries and build full text
        entries: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        for item in transcript_entries:
            # Handle both dict-like and object-like items
            if isinstance(item, dict):
                text = item.get("text", "")
                start = item.get("start", 0.0)
                duration = item.get("duration", 0.0)
            elif hasattr(item, "text"):
                text = getattr(item, "text", "")
                start = getattr(item, "start", 0.0)
                duration = getattr(item, "duration", 0.0)
            else:
                # Skip items we can't parse
                continue

            # Add to entries list
            entries.append(
                {
                    "text": text,
                    "start": float(start),
                    "duration": float(duration),
                }
            )

            # Add to full text
            full_text_parts.append(text)

        # Join all text with spaces
        full_text = " ".join(full_text_parts)

        return {
            "language_code": transcript.language_code,
            "is_generated": transcript.is_generated,
            "entries": entries,
            "full_text": full_text,
        }

    except TranscriptsDisabled as e:
        raise TranscriptError(
            f"Transcripts are disabled for video {video_id}. "
            "The video owner has disabled captions/transcripts."
        ) from e
    except NoTranscriptFound as e:
        # This should only be raised if no transcripts exist at all
        raise TranscriptError(
            f"No transcripts found for video {video_id}. "
            "This video may not have any captions available."
        ) from e
    except TranscriptError:
        # Re-raise our own errors
        raise
    except Exception as e:
        raise TranscriptError(
            f"Failed to fetch transcript for video {video_id}: {e!s}"
        ) from e


__all__ = [
    "TranscriptError",
    "get_full_transcript",
    "get_transcript_chunk",
    "get_video_transcript_preview",
    "list_available_transcripts",
]
