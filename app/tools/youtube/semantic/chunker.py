"""Transcript-aware text chunker for semantic search.

Provides a transcript-specific chunking strategy that:
- Uses **token-based** sizing for accurate embedding model limits
- Supports **chapter-aware** chunking when YouTube chapters are available
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

Example with chapters:
    >>> chunks = chunker.chunk_transcript(
    ...     transcript_entries=[...],
    ...     video_metadata={...},
    ...     chapters=[
    ...         {"start_time": 0.0, "title": "Introduction"},
    ...         {"start_time": 120.0, "title": "Getting Started"},
    ...     ],
    ... )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from app.tools.youtube.semantic.tokenizers import TokenizerProtocol, create_tokenizer

if TYPE_CHECKING:
    from app.tools.youtube.semantic.config import SemanticSearchConfig

logger = logging.getLogger(__name__)


class TranscriptChunker:
    """Transcript-aware text chunker that uses token-based sizing.

    Groups transcript entries into chunks of approximately chunk_size tokens,
    preserving entry boundaries and calculating time ranges for each chunk.
    Optionally respects YouTube chapter boundaries for better semantic grouping.

    Attributes:
        config: Semantic search configuration with chunk_size and chunk_overlap.
        tokenizer: Tokenizer instance for counting tokens.
    """

    def __init__(
        self,
        config: SemanticSearchConfig,
        tokenizer: TokenizerProtocol | None = None,
    ) -> None:
        """Initialize the transcript chunker.

        Args:
            config: Semantic search configuration with chunking parameters.
            tokenizer: Optional tokenizer instance. If not provided, creates one
                based on config.tokenizer_model.
        """
        self.config = config
        self.tokenizer = tokenizer or create_tokenizer(config.tokenizer_model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: The text to tokenize.

        Returns:
            Number of tokens.
        """
        return self.tokenizer.count_tokens(text)

    def chunk_transcript(
        self,
        transcript_entries: list[dict[str, Any]],
        video_metadata: dict[str, str],
        chapters: list[dict[str, Any]] | None = None,
    ) -> list[Document]:
        """Chunk a transcript into Documents with timestamp metadata.

        Groups transcript entries into chunks of approximately chunk_size tokens,
        preserving entry boundaries. When chapters are provided, respects chapter
        boundaries as natural semantic divisions.

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
            chapters: Optional list of chapter markers, each with:
                - start_time: Start time in seconds (float)
                - title: Chapter title (str)
                When provided, chunks respect chapter boundaries.

        Returns:
            List of LangChain Document objects, each with:
                - page_content: Concatenated transcript text
                - metadata: Rich metadata including timestamps and video info
        """
        if not transcript_entries:
            return []

        # Filter out empty entries
        entries = [
            entry for entry in transcript_entries if str(entry.get("text", "")).strip()
        ]

        if not entries:
            return []

        # Use chapter-aware chunking if chapters provided
        if chapters and len(chapters) > 0:
            return self._chunk_with_chapters(entries, video_metadata, chapters)

        # Otherwise, use standard token-based chunking
        return self._chunk_without_chapters(entries, video_metadata)

    def _chunk_without_chapters(
        self,
        entries: list[dict[str, Any]],
        video_metadata: dict[str, str],
    ) -> list[Document]:
        """Chunk entries without chapter awareness.

        Args:
            entries: Filtered transcript entries.
            video_metadata: Video metadata for chunk metadata.

        Returns:
            List of Document chunks.
        """
        documents: list[Document] = []
        chunk_index = 0

        # Current chunk state
        current_entries: list[dict[str, Any]] = []
        current_token_count = 0

        for entry in entries:
            entry_text = str(entry.get("text", "")).strip()
            entry_tokens = self.count_tokens(entry_text)

            # Calculate tokens with separator
            separator_tokens = self.count_tokens(" ") if current_entries else 0
            total_would_be = current_token_count + separator_tokens + entry_tokens

            # Check if adding this entry would exceed chunk_size
            would_exceed = total_would_be > self.config.chunk_size

            # If current chunk is not empty and would exceed, finalize it
            if current_entries and would_exceed:
                doc = self._finalize_chunk(
                    entries=current_entries,
                    video_metadata=video_metadata,
                    chunk_index=chunk_index,
                    chapter_info=None,
                )
                documents.append(doc)
                chunk_index += 1

                # Handle overlap: keep trailing entries that fit in overlap
                current_entries, current_token_count = self._apply_overlap(
                    current_entries
                )

            # Add entry to current chunk
            current_entries.append(entry)
            separator_tokens = self.count_tokens(" ") if len(current_entries) > 1 else 0
            current_token_count += separator_tokens + entry_tokens

        # Finalize the last chunk if there are remaining entries
        if current_entries:
            doc = self._finalize_chunk(
                entries=current_entries,
                video_metadata=video_metadata,
                chunk_index=chunk_index,
                chapter_info=None,
            )
            documents.append(doc)

        return documents

    def _chunk_with_chapters(
        self,
        entries: list[dict[str, Any]],
        video_metadata: dict[str, str],
        chapters: list[dict[str, Any]],
    ) -> list[Document]:
        """Chunk entries respecting chapter boundaries.

        Args:
            entries: Filtered transcript entries.
            video_metadata: Video metadata for chunk metadata.
            chapters: Chapter markers with start_time and title.

        Returns:
            List of Document chunks.
        """
        documents: list[Document] = []
        chunk_index = 0

        # Sort chapters by start time
        sorted_chapters = sorted(chapters, key=lambda c: float(c.get("start_time", 0)))

        # Group entries by chapter
        chapter_entries: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

        for chapter_idx, chapter in enumerate(sorted_chapters):
            chapter_start = float(chapter.get("start_time", 0))

            # Find next chapter start (or infinity if last chapter)
            if chapter_idx + 1 < len(sorted_chapters):
                next_chapter_start = float(
                    sorted_chapters[chapter_idx + 1].get("start_time", 0)
                )
            else:
                next_chapter_start = float("inf")

            # Collect entries in this chapter's time range
            chapter_entry_list = [
                entry
                for entry in entries
                if chapter_start <= float(entry.get("start", 0)) < next_chapter_start
            ]

            if chapter_entry_list:
                chapter_entries.append((chapter, chapter_entry_list))

        # Handle entries before first chapter (if any)
        if sorted_chapters:
            first_chapter_start = float(sorted_chapters[0].get("start_time", 0))
            pre_chapter_entries = [
                entry
                for entry in entries
                if float(entry.get("start", 0)) < first_chapter_start
            ]
            if pre_chapter_entries:
                # Insert at beginning with a "Pre-chapter" marker
                pre_chapter = {"start_time": 0.0, "title": ""}
                chapter_entries.insert(0, (pre_chapter, pre_chapter_entries))

        # Process each chapter
        for chapter_idx, (chapter, chapter_entry_list) in enumerate(chapter_entries):
            chapter_info = {
                "chapter_title": str(chapter.get("title", "")),
                "chapter_index": chapter_idx,
            }

            # Chunk within this chapter (with overlap within chapter, but not across)
            current_entries: list[dict[str, Any]] = []
            current_token_count = 0

            for entry in chapter_entry_list:
                entry_text = str(entry.get("text", "")).strip()
                entry_tokens = self.count_tokens(entry_text)

                separator_tokens = self.count_tokens(" ") if current_entries else 0
                total_would_be = current_token_count + separator_tokens + entry_tokens

                would_exceed = total_would_be > self.config.chunk_size

                if current_entries and would_exceed:
                    doc = self._finalize_chunk(
                        entries=current_entries,
                        video_metadata=video_metadata,
                        chunk_index=chunk_index,
                        chapter_info=chapter_info,
                    )
                    documents.append(doc)
                    chunk_index += 1

                    # Apply overlap within chapter
                    current_entries, current_token_count = self._apply_overlap(
                        current_entries
                    )

                current_entries.append(entry)
                separator_tokens = (
                    self.count_tokens(" ") if len(current_entries) > 1 else 0
                )
                current_token_count += separator_tokens + entry_tokens

            # Finalize remaining entries in this chapter
            if current_entries:
                doc = self._finalize_chunk(
                    entries=current_entries,
                    video_metadata=video_metadata,
                    chunk_index=chunk_index,
                    chapter_info=chapter_info,
                )
                documents.append(doc)
                chunk_index += 1

        return documents

    def _finalize_chunk(
        self,
        entries: list[dict[str, Any]],
        video_metadata: dict[str, str],
        chunk_index: int,
        chapter_info: dict[str, Any] | None,
    ) -> Document:
        """Create a Document from a list of transcript entries.

        Args:
            entries: List of transcript entries in this chunk.
            video_metadata: Video metadata to include.
            chunk_index: Index of this chunk in the transcript.
            chapter_info: Optional chapter metadata (chapter_title, chapter_index).

        Returns:
            LangChain Document with concatenated text and rich metadata.
        """
        # Concatenate entry texts with spaces
        texts = [str(e.get("text", "")).strip() for e in entries]
        page_content = " ".join(texts)

        # Calculate token count for metadata
        token_count = self.count_tokens(page_content)

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
            # Content type discriminator
            "content_type": "transcript",
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
            "token_count": token_count,
        }

        # Add chapter info if present
        if chapter_info:
            if chapter_info.get("chapter_title"):
                metadata["chapter_title"] = chapter_info["chapter_title"]
            if "chapter_index" in chapter_info:
                metadata["chapter_index"] = chapter_info["chapter_index"]

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
        the configured chunk_overlap token count.

        Args:
            entries: Entries from the previous chunk.

        Returns:
            Tuple of (entries_to_keep, total_token_count).
        """
        if self.config.chunk_overlap <= 0:
            return [], 0

        # Work backwards from the end to find entries that fit in overlap
        overlap_entries: list[dict[str, Any]] = []
        overlap_tokens = 0

        for entry in reversed(entries):
            entry_text = str(entry.get("text", "")).strip()
            entry_tokens = self.count_tokens(entry_text)

            # Account for separator
            separator_tokens = self.count_tokens(" ") if overlap_entries else 0
            new_total = overlap_tokens + separator_tokens + entry_tokens

            if new_total <= self.config.chunk_overlap:
                overlap_entries.insert(0, entry)
                overlap_tokens = new_total
            else:
                # Stop if we can't fit more
                break

        return overlap_entries, overlap_tokens

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
