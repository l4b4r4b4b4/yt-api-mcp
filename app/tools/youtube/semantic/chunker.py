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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

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
        transcript_entries: list[dict[str, float | str]],
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
                - published_at: ISO 8601 timestamp (optional)
                - language: Transcript language code (optional)

        Returns:
            List of LangChain Document objects, each with:
                - page_content: Concatenated transcript text
                - metadata: Rich metadata including timestamps and video info

        Raises:
            NotImplementedError: This is a placeholder for Task-04.
        """
        # TODO: Implement in Task-04
        # 1. Iterate through transcript entries
        # 2. Group entries until chunk_size is reached
        # 3. Calculate start_time (first entry's start) and end_time (last entry's start + duration)
        # 4. Create Document with concatenated text and metadata
        # 5. Handle chunk_overlap by including entries from end of previous chunk
        raise NotImplementedError("TranscriptChunker will be implemented in Task-04")

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
