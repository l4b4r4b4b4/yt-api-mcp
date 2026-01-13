"""Unit tests for semantic transcript chunker.

Tests the TranscriptChunker with token-based chunking and chapter awareness.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tools.youtube.semantic.chunker import TranscriptChunker
from app.tools.youtube.semantic.config import SemanticSearchConfig


class TestTranscriptChunkerInit:
    """Tests for TranscriptChunker initialization."""

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        config = SemanticSearchConfig(chunk_size=256, chunk_overlap=50)
        chunker = TranscriptChunker(config)

        assert chunker.config == config
        assert chunker.tokenizer is not None

    def test_init_with_custom_tokenizer(self) -> None:
        """Test initialization with custom tokenizer."""
        config = SemanticSearchConfig()
        mock_tokenizer = MagicMock()
        mock_tokenizer.count_tokens.return_value = 10

        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        assert chunker.tokenizer == mock_tokenizer

    def test_init_creates_tokenizer_from_config(self) -> None:
        """Test that tokenizer is created based on config.tokenizer_model."""
        config = SemanticSearchConfig(tokenizer_model="cl100k_base")

        with patch(
            "app.tools.youtube.semantic.chunker.create_tokenizer"
        ) as mock_create:
            mock_tokenizer = MagicMock()
            mock_create.return_value = mock_tokenizer

            chunker = TranscriptChunker(config)

            mock_create.assert_called_once_with("cl100k_base")
            assert chunker.tokenizer == mock_tokenizer


class TestCountTokens:
    """Tests for token counting."""

    def test_count_tokens_delegates_to_tokenizer(self) -> None:
        """Test that count_tokens delegates to tokenizer."""
        config = SemanticSearchConfig()
        mock_tokenizer = MagicMock()
        mock_tokenizer.count_tokens.return_value = 42

        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)
        result = chunker.count_tokens("test text")

        assert result == 42
        mock_tokenizer.count_tokens.assert_called_once_with("test text")

    def test_count_tokens_empty_string(self) -> None:
        """Test counting tokens of empty string."""
        config = SemanticSearchConfig()
        mock_tokenizer = MagicMock()
        mock_tokenizer.count_tokens.return_value = 0

        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)
        result = chunker.count_tokens("")

        assert result == 0


class TestBasicChunking:
    """Tests for basic token-based chunking without chapters."""

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer that counts words as tokens."""
        mock = MagicMock()

        def count_tokens(text: str) -> int:
            if not text:
                return 0
            return len(text.split())

        mock.count_tokens.side_effect = count_tokens
        return mock

    @pytest.fixture
    def video_metadata(self) -> dict[str, str]:
        """Sample video metadata."""
        return {
            "video_id": "abc123",
            "video_title": "Test Video",
            "channel_id": "UC123",
            "channel_title": "Test Channel",
        }

    def test_empty_transcript_returns_empty_list(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that empty transcript returns empty list."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        result = chunker.chunk_transcript([], video_metadata)

        assert result == []

    def test_transcript_with_only_empty_entries(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test transcript with only empty text entries."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [
            {"text": "", "start": 0.0, "duration": 1.0},
            {"text": "   ", "start": 1.0, "duration": 1.0},
        ]
        result = chunker.chunk_transcript(entries, video_metadata)

        assert result == []

    def test_single_entry_becomes_single_chunk(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that a single entry becomes a single chunk."""
        config = SemanticSearchConfig(chunk_size=100, chunk_overlap=0)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [{"text": "Hello world", "start": 0.0, "duration": 2.5}]
        result = chunker.chunk_transcript(entries, video_metadata)

        assert len(result) == 1
        assert result[0].page_content == "Hello world"
        assert result[0].metadata["chunk_index"] == 0

    def test_multiple_entries_within_limit_single_chunk(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that multiple entries within limit form single chunk."""
        # Each entry is ~2-3 tokens, chunk_size=50 should fit all
        config = SemanticSearchConfig(chunk_size=50, chunk_overlap=0)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [
            {"text": "Hello world", "start": 0.0, "duration": 1.0},
            {"text": "This is", "start": 1.0, "duration": 1.0},
            {"text": "a test", "start": 2.0, "duration": 1.0},
        ]
        result = chunker.chunk_transcript(entries, video_metadata)

        assert len(result) == 1
        assert result[0].page_content == "Hello world This is a test"

    def test_entries_exceed_limit_creates_multiple_chunks(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that entries exceeding limit create multiple chunks."""
        # chunk_size=50 tokens, but we'll create entries that exceed it
        config = SemanticSearchConfig(chunk_size=50, chunk_overlap=0)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        # Create entries that will exceed 50 tokens total
        entries = [
            {
                "text": " ".join(["word"] * 30),
                "start": 0.0,
                "duration": 1.0,
            },  # 30 tokens
            {
                "text": " ".join(["more"] * 30),
                "start": 1.0,
                "duration": 1.0,
            },  # 30 tokens
            {
                "text": " ".join(["text"] * 30),
                "start": 2.0,
                "duration": 1.0,
            },  # 30 tokens
        ]
        result = chunker.chunk_transcript(entries, video_metadata)

        # Should create multiple chunks since entries exceed 50 tokens
        assert len(result) >= 2

    def test_chunk_overlap_includes_trailing_entries(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that chunk overlap includes entries from previous chunk."""
        config = SemanticSearchConfig(chunk_size=50, chunk_overlap=20)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        # Create entries that will trigger chunking and overlap
        entries = [
            {
                "text": " ".join(["word"] * 30),
                "start": 0.0,
                "duration": 1.0,
            },  # 30 tokens
            {
                "text": " ".join(["more"] * 15),
                "start": 1.0,
                "duration": 1.0,
            },  # 15 tokens
            {
                "text": " ".join(["text"] * 30),
                "start": 2.0,
                "duration": 1.0,
            },  # 30 tokens
        ]
        result = chunker.chunk_transcript(entries, video_metadata)

        # With overlap, some content should appear in multiple chunks
        assert len(result) >= 2

    def test_metadata_includes_content_type(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that metadata includes content_type discriminator."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [{"text": "Test content", "start": 0.0, "duration": 1.0}]
        result = chunker.chunk_transcript(entries, video_metadata)

        metadata = result[0].metadata
        assert metadata["content_type"] == "transcript"

    def test_metadata_includes_video_info(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that metadata includes all video information."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [{"text": "Test content", "start": 0.0, "duration": 1.0}]
        result = chunker.chunk_transcript(entries, video_metadata)

        metadata = result[0].metadata
        assert metadata["video_id"] == "abc123"
        assert metadata["video_title"] == "Test Video"
        assert metadata["channel_id"] == "UC123"
        assert metadata["channel_title"] == "Test Channel"

    def test_metadata_includes_timestamps(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that metadata includes correct timestamps."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [
            {"text": "First", "start": 10.5, "duration": 2.0},
            {"text": "Second", "start": 12.5, "duration": 3.0},
        ]
        result = chunker.chunk_transcript(entries, video_metadata)

        metadata = result[0].metadata
        assert metadata["start_time"] == 10.5
        assert metadata["end_time"] == 15.5  # 12.5 + 3.0

    def test_metadata_includes_timestamp_url(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that timestamp_url is correctly generated."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [{"text": "Test", "start": 65.7, "duration": 1.0}]
        result = chunker.chunk_transcript(entries, video_metadata)

        metadata = result[0].metadata
        assert (
            metadata["timestamp_url"] == "https://www.youtube.com/watch?v=abc123&t=65"
        )

    def test_metadata_includes_token_count(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that token_count is included in metadata."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [{"text": "Hello world test", "start": 0.0, "duration": 1.0}]
        result = chunker.chunk_transcript(entries, video_metadata)

        metadata = result[0].metadata
        assert "token_count" in metadata
        assert metadata["token_count"] == 3  # 3 words = 3 tokens in mock

    def test_chunk_index_increments(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that chunk_index increments correctly."""
        config = SemanticSearchConfig(chunk_size=50, chunk_overlap=0)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        # Create entries that will each become their own chunk
        entries = [
            {
                "text": " ".join(["one"] * 40),
                "start": 0.0,
                "duration": 1.0,
            },  # 40 tokens
            {
                "text": " ".join(["two"] * 40),
                "start": 1.0,
                "duration": 1.0,
            },  # 40 tokens
            {
                "text": " ".join(["three"] * 40),
                "start": 2.0,
                "duration": 1.0,
            },  # 40 tokens
        ]
        result = chunker.chunk_transcript(entries, video_metadata)

        for i, doc in enumerate(result):
            assert doc.metadata["chunk_index"] == i

    def test_optional_metadata_propagated(self, mock_tokenizer: MagicMock) -> None:
        """Test that optional metadata fields are propagated."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        video_metadata = {
            "video_id": "abc123",
            "video_title": "Test",
            "channel_id": "UC123",
            "channel_title": "Channel",
            "published_at": "2024-01-01T00:00:00Z",
            "language": "en",
        }
        entries = [{"text": "Test", "start": 0.0, "duration": 1.0}]
        result = chunker.chunk_transcript(entries, video_metadata)

        metadata = result[0].metadata
        assert metadata["published_at"] == "2024-01-01T00:00:00Z"
        assert metadata["language"] == "en"


class TestChapterAwareChunking:
    """Tests for chapter-aware chunking."""

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer that counts words as tokens."""
        mock = MagicMock()

        def count_tokens(text: str) -> int:
            if not text:
                return 0
            return len(text.split())

        mock.count_tokens.side_effect = count_tokens
        return mock

    @pytest.fixture
    def video_metadata(self) -> dict[str, str]:
        """Sample video metadata."""
        return {
            "video_id": "abc123",
            "video_title": "Test Video",
            "channel_id": "UC123",
            "channel_title": "Test Channel",
        }

    def test_chapters_respected_as_boundaries(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that chapter boundaries are respected."""
        config = SemanticSearchConfig(chunk_size=100, chunk_overlap=0)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [
            {"text": "Intro content", "start": 0.0, "duration": 5.0},
            {"text": "More intro", "start": 5.0, "duration": 5.0},
            {"text": "Chapter two content", "start": 10.0, "duration": 5.0},
            {"text": "More chapter two", "start": 15.0, "duration": 5.0},
        ]
        chapters = [
            {"start_time": 0.0, "title": "Introduction"},
            {"start_time": 10.0, "title": "Chapter Two"},
        ]

        result = chunker.chunk_transcript(entries, video_metadata, chapters=chapters)

        # Should have at least 2 chunks (one per chapter)
        assert len(result) >= 2

        # First chunk should be intro content
        assert "Intro" in result[0].page_content

        # Check chapter metadata
        assert result[0].metadata["chapter_title"] == "Introduction"
        assert result[0].metadata["chapter_index"] == 0

    def test_chapter_title_in_metadata(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that chapter_title is included in metadata."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [{"text": "Content", "start": 0.0, "duration": 1.0}]
        chapters = [{"start_time": 0.0, "title": "My Chapter"}]

        result = chunker.chunk_transcript(entries, video_metadata, chapters=chapters)

        assert result[0].metadata["chapter_title"] == "My Chapter"

    def test_chapter_index_in_metadata(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that chapter_index is included in metadata."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [
            {"text": "First chapter", "start": 0.0, "duration": 5.0},
            {"text": "Second chapter", "start": 10.0, "duration": 5.0},
        ]
        chapters = [
            {"start_time": 0.0, "title": "One"},
            {"start_time": 10.0, "title": "Two"},
        ]

        result = chunker.chunk_transcript(entries, video_metadata, chapters=chapters)

        assert result[0].metadata["chapter_index"] == 0
        assert result[1].metadata["chapter_index"] == 1

    def test_large_chapter_splits_into_multiple_chunks(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that large chapters are split into multiple chunks."""
        config = SemanticSearchConfig(chunk_size=50, chunk_overlap=0)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        # Create entries that will exceed chunk_size when combined
        entries = [
            {
                "text": " ".join(["one"] * 30),
                "start": 0.0,
                "duration": 1.0,
            },  # 30 tokens
            {
                "text": " ".join(["two"] * 30),
                "start": 1.0,
                "duration": 1.0,
            },  # 30 tokens
            {
                "text": " ".join(["three"] * 30),
                "start": 2.0,
                "duration": 1.0,
            },  # 30 tokens
        ]
        chapters = [{"start_time": 0.0, "title": "Big Chapter"}]

        result = chunker.chunk_transcript(entries, video_metadata, chapters=chapters)

        # Should have multiple chunks from same chapter
        assert len(result) >= 2
        # All chunks should have same chapter title
        for doc in result:
            assert doc.metadata["chapter_title"] == "Big Chapter"
            assert doc.metadata["chapter_index"] == 0

    def test_small_chapter_preserved(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that small chapters are preserved as single chunks."""
        config = SemanticSearchConfig(chunk_size=100, chunk_overlap=0)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [
            {"text": "tiny", "start": 0.0, "duration": 1.0},
            {"text": "also small", "start": 10.0, "duration": 1.0},
        ]
        chapters = [
            {"start_time": 0.0, "title": "Tiny Chapter"},
            {"start_time": 10.0, "title": "Also Small"},
        ]

        result = chunker.chunk_transcript(entries, video_metadata, chapters=chapters)

        assert len(result) == 2
        assert result[0].page_content == "tiny"
        assert result[1].page_content == "also small"

    def test_no_overlap_across_chapters(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that overlap does not cross chapter boundaries."""
        config = SemanticSearchConfig(chunk_size=50, chunk_overlap=20)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [
            {"text": "chapter one end content here", "start": 0.0, "duration": 5.0},
            {"text": "chapter two start content here", "start": 10.0, "duration": 5.0},
        ]
        chapters = [
            {"start_time": 0.0, "title": "One"},
            {"start_time": 10.0, "title": "Two"},
        ]

        result = chunker.chunk_transcript(entries, video_metadata, chapters=chapters)

        # Chapter two should not contain chapter one content
        for doc in result:
            if doc.metadata.get("chapter_title") == "Two":
                assert "chapter one" not in doc.page_content.lower()

    def test_empty_chapters_list_treated_as_no_chapters(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that empty chapters list behaves like no chapters."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [{"text": "Test content", "start": 0.0, "duration": 1.0}]

        result_no_chapters = chunker.chunk_transcript(entries, video_metadata)
        result_empty_chapters = chunker.chunk_transcript(
            entries, video_metadata, chapters=[]
        )

        # Both should produce same result (no chapter metadata)
        assert len(result_no_chapters) == len(result_empty_chapters)
        assert "chapter_title" not in result_no_chapters[0].metadata
        assert "chapter_title" not in result_empty_chapters[0].metadata

    def test_entries_before_first_chapter(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that entries before first chapter are handled."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [
            {"text": "Pre-chapter content", "start": 0.0, "duration": 5.0},
            {"text": "Chapter content", "start": 10.0, "duration": 5.0},
        ]
        chapters = [{"start_time": 10.0, "title": "First Chapter"}]

        result = chunker.chunk_transcript(entries, video_metadata, chapters=chapters)

        # Should have 2 chunks
        assert len(result) == 2
        # First chunk (pre-chapter) should have empty or no chapter title
        # Second chunk should have the chapter title
        assert result[1].metadata.get("chapter_title") == "First Chapter"

    def test_chapters_sorted_by_start_time(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that chapters are sorted by start time internally."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [
            {"text": "First", "start": 0.0, "duration": 5.0},
            {"text": "Second", "start": 10.0, "duration": 5.0},
        ]
        # Chapters provided out of order
        chapters = [
            {"start_time": 10.0, "title": "Chapter B"},
            {"start_time": 0.0, "title": "Chapter A"},
        ]

        result = chunker.chunk_transcript(entries, video_metadata, chapters=chapters)

        # Should be sorted correctly
        assert result[0].metadata["chapter_title"] == "Chapter A"
        assert result[1].metadata["chapter_title"] == "Chapter B"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer that counts words as tokens."""
        mock = MagicMock()

        def count_tokens(text: str) -> int:
            if not text:
                return 0
            return len(text.split())

        mock.count_tokens.side_effect = count_tokens
        return mock

    @pytest.fixture
    def video_metadata(self) -> dict[str, str]:
        """Sample video metadata."""
        return {
            "video_id": "abc123",
            "video_title": "Test Video",
            "channel_id": "UC123",
            "channel_title": "Test Channel",
        }

    def test_single_large_entry_kept_whole(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that single entry larger than chunk_size is kept whole."""
        config = SemanticSearchConfig(chunk_size=50, chunk_overlap=0)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        # Entry has 100 tokens, chunk_size is 50
        large_text = " ".join(["word"] * 100)
        entries = [
            {
                "text": large_text,
                "start": 0.0,
                "duration": 1.0,
            }
        ]
        result = chunker.chunk_transcript(entries, video_metadata)

        # Should still be kept as single chunk (never split mid-entry)
        assert len(result) == 1
        assert result[0].metadata["token_count"] == 100

    def test_entry_with_missing_duration(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test handling of entry with missing duration."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [{"text": "Test", "start": 5.0}]  # No duration
        result = chunker.chunk_transcript(entries, video_metadata)

        assert len(result) == 1
        assert result[0].metadata["start_time"] == 5.0
        assert result[0].metadata["end_time"] == 5.0  # start + 0 duration

    def test_entry_with_missing_start(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test handling of entry with missing start time."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [{"text": "Test", "duration": 1.0}]  # No start
        result = chunker.chunk_transcript(entries, video_metadata)

        assert len(result) == 1
        assert result[0].metadata["start_time"] == 0.0
        assert result[0].metadata["end_time"] == 1.0

    def test_whitespace_only_entries_filtered(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that whitespace-only entries are filtered out."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [
            {"text": "Valid", "start": 0.0, "duration": 1.0},
            {"text": "   ", "start": 1.0, "duration": 1.0},
            {"text": "\t\n", "start": 2.0, "duration": 1.0},
            {"text": "Also valid", "start": 3.0, "duration": 1.0},
        ]
        result = chunker.chunk_transcript(entries, video_metadata)

        assert len(result) == 1
        assert result[0].page_content == "Valid Also valid"

    def test_chapter_with_no_entries_skipped(
        self, mock_tokenizer: MagicMock, video_metadata: dict[str, str]
    ) -> None:
        """Test that chapters with no entries are skipped."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        entries = [
            {"text": "In chapter one", "start": 0.0, "duration": 5.0},
            # No entries in 10-20 range (chapter two)
            {"text": "In chapter three", "start": 20.0, "duration": 5.0},
        ]
        chapters = [
            {"start_time": 0.0, "title": "One"},
            {"start_time": 10.0, "title": "Two (empty)"},
            {"start_time": 20.0, "title": "Three"},
        ]

        result = chunker.chunk_transcript(entries, video_metadata, chapters=chapters)

        # Should only have chunks for chapters with content
        chapter_titles = [doc.metadata.get("chapter_title") for doc in result]
        assert "One" in chapter_titles
        assert "Three" in chapter_titles
        assert "Two (empty)" not in chapter_titles

    def test_video_url_generated_when_not_provided(
        self, mock_tokenizer: MagicMock
    ) -> None:
        """Test that video_url is generated if not in metadata."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        video_metadata = {
            "video_id": "xyz789",
            "video_title": "Test",
            "channel_id": "UC123",
            "channel_title": "Channel",
        }
        entries = [{"text": "Test", "start": 0.0, "duration": 1.0}]
        result = chunker.chunk_transcript(entries, video_metadata)

        assert (
            result[0].metadata["video_url"] == "https://www.youtube.com/watch?v=xyz789"
        )

    def test_video_url_preserved_when_provided(self, mock_tokenizer: MagicMock) -> None:
        """Test that provided video_url is preserved."""
        config = SemanticSearchConfig(chunk_size=100)
        chunker = TranscriptChunker(config, tokenizer=mock_tokenizer)

        video_metadata = {
            "video_id": "xyz789",
            "video_title": "Test",
            "channel_id": "UC123",
            "channel_title": "Channel",
            "video_url": "https://youtu.be/xyz789",
        }
        entries = [{"text": "Test", "start": 0.0, "duration": 1.0}]
        result = chunker.chunk_transcript(entries, video_metadata)

        assert result[0].metadata["video_url"] == "https://youtu.be/xyz789"


class TestIntegration:
    """Integration tests with real tiktoken tokenizer."""

    @pytest.fixture
    def video_metadata(self) -> dict[str, str]:
        """Sample video metadata."""
        return {
            "video_id": "abc123",
            "video_title": "Test Video",
            "channel_id": "UC123",
            "channel_title": "Test Channel",
        }

    def test_with_real_tokenizer(self, video_metadata: dict[str, str]) -> None:
        """Test chunking with real tiktoken tokenizer."""
        config = SemanticSearchConfig(
            chunk_size=50,
            chunk_overlap=10,
            tokenizer_model="cl100k_base",
        )
        chunker = TranscriptChunker(config)

        entries = [
            {
                "text": "Welcome to this video about NixOS.",
                "start": 0.0,
                "duration": 3.0,
            },
            {
                "text": "Today we will learn about garbage collection.",
                "start": 3.0,
                "duration": 4.0,
            },
            {
                "text": "Nix stores all packages in the Nix store.",
                "start": 7.0,
                "duration": 3.0,
            },
            {
                "text": "Over time, old generations accumulate.",
                "start": 10.0,
                "duration": 3.0,
            },
        ]
        result = chunker.chunk_transcript(entries, video_metadata)

        assert len(result) >= 1
        assert all(doc.metadata["token_count"] > 0 for doc in result)
        assert all("video_id" in doc.metadata for doc in result)

    def test_transcript_like_content(self, video_metadata: dict[str, str]) -> None:
        """Test with realistic transcript content."""
        config = SemanticSearchConfig(
            chunk_size=100,
            chunk_overlap=20,
            tokenizer_model="cl100k_base",
        )
        chunker = TranscriptChunker(config)

        # Simulate a typical YouTube transcript
        entries = [
            {
                "text": "Hey everyone, welcome back to the channel.",
                "start": 0.0,
                "duration": 2.5,
            },
            {
                "text": "In today's video, we're going to be talking about",
                "start": 2.5,
                "duration": 2.0,
            },
            {
                "text": "how to set up your development environment.",
                "start": 4.5,
                "duration": 2.0,
            },
            {
                "text": "First, let's start with installing the prerequisites.",
                "start": 6.5,
                "duration": 2.5,
            },
            {
                "text": "You'll need Python 3.12 or later.",
                "start": 9.0,
                "duration": 2.0,
            },
            {
                "text": "Make sure you have pip installed as well.",
                "start": 11.0,
                "duration": 2.0,
            },
        ]
        result = chunker.chunk_transcript(entries, video_metadata)

        # Should produce reasonable chunks
        assert len(result) >= 1
        # Content should be preserved
        all_content = " ".join(doc.page_content for doc in result)
        assert "Python" in all_content
        assert "development environment" in all_content
