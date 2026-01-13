"""Transcript-aware text chunker for semantic search.

Provides a transcript-specific chunking strategy that:
- Groups transcript entries by target character count
- Preserves timestamp boundaries (never splits mid-entry)
- Calculates start_time and end_time for each chunk
- Stores rich metadata for filtering and timestamped playback URLs

Unlike generic text splitters, this chunker understands transcript structure
and maintains the temporal relationship between text and video position.

Example:
    >>> from app.tools.youtube.semantic.chunker import TranscriptChunker
    >>> from app.tools.youtube.semantic.config import get_semantic_config
    >>> chunker = TranscriptChunker(get_semantic_config())
    >>> chunks = chunker.chunk_transcript(
    ...     transcript_entries=[
    ...         {"text": "Hello world", "start": 0.0, "duration": 2.5},
    ...         {"text": "This is a test", "start": 2.5, "duration": 3.0},
    ...     ],
    ...     video_metadata={
    ...         "video_id": "abc123",
    ...         "video_title": "Test Video",
    ...         "channel_id": "UC123",
    ...         "channel_title": "Test Channel",
    ...     },
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

if TYPE_CHECKING:
    from app.tools.youtube.semantic.config import SemanticSearchConfig


class TranscriptChunker:
    """Transcript-aware text chunker that preserves timestamps.

    Groups transcript entries into chunks of approximately chunk_size characters,
    preserving entry boundaries and calculating time ranges for each chunk.

    Attributes:
        config: Semantic search configuration with chunk_size and chunk_overlap.
    """

    def __init__(self, config: SemanticSearchConfig) -> None:
        """Initialize the transcript chunker.

        Args:
            config: Semantic search configuration with chunking parameters.
        """
        self.config = config

    def chunk_transcript(
        self,
        transcript_entries: list[dict[str, Any]],
        video_metadata: dict[str, str],
    ) -> list[Document]:
        """Chunk a transcript into Documents with timestamp metadata.

        Groups transcript entries into chunks of approximately chunk_size characters,
        preserving entry boundaries. Each chunk includes metadata for filtering
        and generating timestamped video URLs.

        Args:
            transcript_entries: List of transcript entries, each with:
                - text: The transcript text
                - start: Start time in seconds
                - duration: Duration in seconds
            video_metadata: Video information to include in chunk metadata:
                - video_id: YouTube video ID
                - video_title: Video title
                - channel_id: YouTube channel ID
                - channel_title: Channel name
                - video_url: Direct link to video (optional)
                - published_at: ISO 8601 timestamp (optional)
                - language: Transcript language code (optional)

        Returns:
            List of LangChain Document objects, each with:
                - page_content: Concatenated transcript text
                - metadata: Rich metadata including timestamps and video info
        """
        if not transcript_entries:
            return []

        documents: list[Document] = []
        chunk_index = 0

        # Current chunk state
        current_entries: list[dict[str, Any]] = []
        current_text_length = 0

        for entry in transcript_entries:
            entry_text = str(entry.get("text", "")).strip()
            entry_length = len(entry_text)

            # Skip empty entries
            if not entry_text:
                continue

            # Check if adding this entry would exceed chunk_size
            # Account for space separator between entries
            separator_length = 1 if current_entries else 0
            would_exceed = (
                current_text_length + separator_length + entry_length
                > self.config.chunk_size
            )

            # If current chunk is not empty and would exceed, finalize it
            if current_entries and would_exceed:
                doc = self._finalize_chunk(
                    entries=current_entries,
                    video_metadata=video_metadata,
                    chunk_index=chunk_index,
                )
                documents.append(doc)
                chunk_index += 1

                # Handle overlap: keep trailing entries that fit in overlap
                current_entries, current_text_length = self._apply_overlap(
                    current_entries
                )

            # Add entry to current chunk
            current_entries.append(entry)
            separator_length = 1 if len(current_entries) > 1 else 0
            current_text_length += separator_length + entry_length

        # Finalize the last chunk if there are remaining entries
        if current_entries:
            doc = self._finalize_chunk(
                entries=current_entries,
                video_metadata=video_metadata,
                chunk_index=chunk_index,
            )
            documents.append(doc)

        return documents

    def _finalize_chunk(
        self,
        entries: list[dict[str, Any]],
        video_metadata: dict[str, str],
        chunk_index: int,
    ) -> Document:
        """Create a Document from a list of transcript entries.

        Args:
            entries: List of transcript entries in this chunk.
            video_metadata: Video metadata to include.
            chunk_index: Index of this chunk in the transcript.

        Returns:
            LangChain Document with concatenated text and rich metadata.
        """
        # Concatenate entry texts with spaces
        texts = [str(e.get("text", "")).strip() for e in entries]
        page_content = " ".join(texts)

        # Calculate timestamps
        first_entry = entries[0]
        last_entry = entries[-1]

        start_time = float(first_entry.get("start", 0.0))
        last_start = float(last_entry.get("start", 0.0))
        last_duration = float(last_entry.get("duration", 0.0))
        end_time = last_start + last_duration

        # Get video_id for timestamp URL
        video_id = video_metadata.get("video_id", "")

        # Build metadata
        metadata: dict[str, Any] = {
            # Video identifiers
            "video_id": video_id,
            "channel_id": video_metadata.get("channel_id", ""),
            # Display info
            "video_title": video_metadata.get("video_title", ""),
            "channel_title": video_metadata.get("channel_title", ""),
            "video_url": video_metadata.get(
                "video_url", f"https://www.youtube.com/watch?v={video_id}"
            ),
            # Timestamps
            "start_time": start_time,
            "end_time": end_time,
            "timestamp_url": self._create_timestamp_url(video_id, start_time),
            # Chunk info
            "chunk_index": chunk_index,
        }

        # Add optional fields if present
        if "published_at" in video_metadata:
            metadata["published_at"] = video_metadata["published_at"]
        if "language" in video_metadata:
            metadata["language"] = video_metadata["language"]

        return Document(page_content=page_content, metadata=metadata)

    def _apply_overlap(
        self,
        entries: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Get entries to carry over for overlap.

        Keeps trailing entries from the previous chunk that fit within
        the configured chunk_overlap character count.

        Args:
            entries: Entries from the previous chunk.

        Returns:
            Tuple of (entries_to_keep, total_text_length).
        """
        if self.config.chunk_overlap <= 0:
            return [], 0

        # Work backwards from the end to find entries that fit in overlap
        overlap_entries: list[dict[str, Any]] = []
        overlap_length = 0

        for entry in reversed(entries):
            entry_text = str(entry.get("text", "")).strip()
            entry_length = len(entry_text)

            # Account for separator
            separator_length = 1 if overlap_entries else 0
            new_length = overlap_length + separator_length + entry_length

            if new_length <= self.config.chunk_overlap:
                overlap_entries.insert(0, entry)
                overlap_length = new_length
            else:
                # Stop if we can't fit more
                break

        return overlap_entries, overlap_length

    def _create_timestamp_url(self, video_id: str, start_time: float) -> str:
        """Create a YouTube URL with timestamp.

        Args:
            video_id: YouTube video ID.
            start_time: Start time in seconds.

        Returns:
            YouTube URL with timestamp parameter (e.g., https://youtube.com/watch?v=abc&t=123).
        """
        seconds = int(start_time)
        return f"https://www.youtube.com/watch?v={video_id}&t={seconds}"
