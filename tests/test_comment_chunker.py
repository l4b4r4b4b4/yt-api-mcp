"""Unit tests for semantic comment chunker.

Tests the CommentChunker which converts YouTube comments into LangChain
Documents for semantic search.
"""

from __future__ import annotations

import pytest

from app.tools.youtube.semantic.comment_chunker import CommentChunker


class TestCommentChunkerBasics:
    """Tests for basic CommentChunker functionality."""

    @pytest.fixture
    def chunker(self) -> CommentChunker:
        """Create a CommentChunker instance."""
        return CommentChunker()

    @pytest.fixture
    def video_metadata(self) -> dict[str, str]:
        """Sample video metadata."""
        return {
            "video_id": "abc123",
            "video_title": "Test Video",
            "channel_id": "UC123",
            "channel_title": "Test Channel",
            "video_url": "https://www.youtube.com/watch?v=abc123",
        }

    def test_empty_comments_returns_empty_list(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that empty comments list returns empty Documents list."""
        result = chunker.chunk_comments([], video_metadata)

        assert result == []

    def test_single_comment_becomes_single_document(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that a single comment becomes a single Document."""
        comments = [
            {
                "text": "Great video!",
                "author": "User1",
                "like_count": 10,
                "reply_count": 2,
                "published_at": "2024-01-15T10:30:00Z",
            }
        ]

        result = chunker.chunk_comments(comments, video_metadata)

        assert len(result) == 1
        assert result[0].page_content == "Great video!"

    def test_multiple_comments_become_multiple_documents(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that multiple comments become multiple Documents."""
        comments = [
            {"text": "First comment", "author": "User1", "like_count": 5},
            {"text": "Second comment", "author": "User2", "like_count": 10},
            {"text": "Third comment", "author": "User3", "like_count": 15},
        ]

        result = chunker.chunk_comments(comments, video_metadata)

        assert len(result) == 3
        assert result[0].page_content == "First comment"
        assert result[1].page_content == "Second comment"
        assert result[2].page_content == "Third comment"


class TestCommentChunkerMetadata:
    """Tests for metadata in chunked comments."""

    @pytest.fixture
    def chunker(self) -> CommentChunker:
        """Create a CommentChunker instance."""
        return CommentChunker()

    @pytest.fixture
    def video_metadata(self) -> dict[str, str]:
        """Sample video metadata."""
        return {
            "video_id": "xyz789",
            "video_title": "Nix Tutorial",
            "channel_id": "UCnix",
            "channel_title": "Vimjoyer",
            "video_url": "https://www.youtube.com/watch?v=xyz789",
        }

    def test_metadata_includes_content_type(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that metadata includes content_type: 'comment'."""
        comments = [{"text": "Test", "author": "User", "like_count": 0}]

        result = chunker.chunk_comments(comments, video_metadata)

        assert result[0].metadata["content_type"] == "comment"

    def test_metadata_includes_video_info(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that metadata includes video information."""
        comments = [{"text": "Test", "author": "User", "like_count": 0}]

        result = chunker.chunk_comments(comments, video_metadata)

        metadata = result[0].metadata
        assert metadata["video_id"] == "xyz789"
        assert metadata["video_title"] == "Nix Tutorial"
        assert metadata["channel_id"] == "UCnix"
        assert metadata["channel_title"] == "Vimjoyer"
        assert metadata["video_url"] == "https://www.youtube.com/watch?v=xyz789"

    def test_metadata_includes_author(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that metadata includes comment author."""
        comments = [{"text": "Helpful!", "author": "NixLearner42", "like_count": 5}]

        result = chunker.chunk_comments(comments, video_metadata)

        assert result[0].metadata["author"] == "NixLearner42"

    def test_metadata_includes_like_count(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that metadata includes like count as integer."""
        comments = [{"text": "Amazing", "author": "User", "like_count": 42}]

        result = chunker.chunk_comments(comments, video_metadata)

        assert result[0].metadata["like_count"] == 42
        assert isinstance(result[0].metadata["like_count"], int)

    def test_metadata_includes_reply_count(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that metadata includes reply count."""
        comments = [
            {
                "text": "Discussion starter",
                "author": "User",
                "like_count": 10,
                "reply_count": 7,
            }
        ]

        result = chunker.chunk_comments(comments, video_metadata)

        assert result[0].metadata["reply_count"] == 7

    def test_metadata_includes_published_at(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that metadata includes comment publish timestamp."""
        comments = [
            {
                "text": "Posted today",
                "author": "User",
                "like_count": 1,
                "published_at": "2025-01-14T12:00:00Z",
            }
        ]

        result = chunker.chunk_comments(comments, video_metadata)

        assert result[0].metadata["published_at"] == "2025-01-14T12:00:00Z"

    def test_metadata_includes_chunk_index(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that metadata includes chunk_index (position in list)."""
        comments = [
            {"text": "First", "author": "A", "like_count": 0},
            {"text": "Second", "author": "B", "like_count": 0},
            {"text": "Third", "author": "C", "like_count": 0},
        ]

        result = chunker.chunk_comments(comments, video_metadata)

        assert result[0].metadata["chunk_index"] == 0
        assert result[1].metadata["chunk_index"] == 1
        assert result[2].metadata["chunk_index"] == 2


class TestCommentChunkerEdgeCases:
    """Tests for edge cases in comment chunking."""

    @pytest.fixture
    def chunker(self) -> CommentChunker:
        """Create a CommentChunker instance."""
        return CommentChunker()

    @pytest.fixture
    def video_metadata(self) -> dict[str, str]:
        """Sample video metadata."""
        return {
            "video_id": "test123",
            "video_title": "Test",
            "channel_id": "UCtest",
            "channel_title": "Test Channel",
        }

    def test_empty_text_comments_filtered(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that comments with empty text are filtered out."""
        comments = [
            {"text": "", "author": "User1", "like_count": 0},
            {"text": "Valid comment", "author": "User2", "like_count": 5},
            {"text": "   ", "author": "User3", "like_count": 0},  # Whitespace only
        ]

        result = chunker.chunk_comments(comments, video_metadata)

        assert len(result) == 1
        assert result[0].page_content == "Valid comment"

    def test_missing_optional_fields_default_gracefully(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that missing optional fields use sensible defaults."""
        comments = [{"text": "Minimal comment"}]  # Only text, no other fields

        result = chunker.chunk_comments(comments, video_metadata)

        assert len(result) == 1
        metadata = result[0].metadata
        assert metadata["author"] == ""
        assert metadata["like_count"] == 0
        assert metadata["reply_count"] == 0
        assert metadata["published_at"] == ""

    def test_video_url_generated_when_not_provided(
        self, chunker: CommentChunker
    ) -> None:
        """Test that video URL is generated from video_id if not provided."""
        video_metadata = {
            "video_id": "genurl123",
            "video_title": "Test",
        }
        comments = [{"text": "Test comment", "author": "User", "like_count": 0}]

        result = chunker.chunk_comments(comments, video_metadata)

        assert (
            result[0].metadata["video_url"]
            == "https://www.youtube.com/watch?v=genurl123"
        )

    def test_whitespace_stripped_from_comment_text(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that leading/trailing whitespace is stripped from comment text."""
        comments = [{"text": "  Padded comment  \n", "author": "User", "like_count": 0}]

        result = chunker.chunk_comments(comments, video_metadata)

        assert result[0].page_content == "Padded comment"

    def test_like_count_converted_to_int(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that like_count string is converted to int."""
        comments = [{"text": "Test", "author": "User", "like_count": "25"}]

        result = chunker.chunk_comments(comments, video_metadata)

        assert result[0].metadata["like_count"] == 25
        assert isinstance(result[0].metadata["like_count"], int)

    def test_chunk_index_skips_filtered_comments(
        self, chunker: CommentChunker, video_metadata: dict[str, str]
    ) -> None:
        """Test that chunk_index reflects original position, not filtered position."""
        comments = [
            {"text": "", "author": "Filtered1", "like_count": 0},  # idx 0, filtered
            {"text": "Valid 1", "author": "User1", "like_count": 5},  # idx 1
            {"text": "", "author": "Filtered2", "like_count": 0},  # idx 2, filtered
            {"text": "Valid 2", "author": "User2", "like_count": 10},  # idx 3
        ]

        result = chunker.chunk_comments(comments, video_metadata)

        # chunk_index should preserve original positions
        assert len(result) == 2
        assert result[0].metadata["chunk_index"] == 1
        assert result[1].metadata["chunk_index"] == 3

    def test_missing_video_id_uses_empty_string(self, chunker: CommentChunker) -> None:
        """Test handling of missing video_id in metadata."""
        video_metadata = {"video_title": "No ID Video"}
        comments = [{"text": "Test", "author": "User", "like_count": 0}]

        result = chunker.chunk_comments(comments, video_metadata)

        assert result[0].metadata["video_id"] == ""
        # URL should be generated with empty ID
        assert "watch?v=" in result[0].metadata["video_url"]


class TestCommentChunkerIntegration:
    """Integration tests with realistic comment data."""

    def test_realistic_youtube_comments(self) -> None:
        """Test with realistic YouTube comment data structure."""
        chunker = CommentChunker()

        video_metadata = {
            "video_id": "nLwbNhSxLd4",
            "video_title": "NixOS Full Guide",
            "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw",
            "channel_title": "Vimjoyer",
            "video_url": "https://www.youtube.com/watch?v=nLwbNhSxLd4",
        }

        comments = [
            {
                "text": "This is the best NixOS tutorial I've ever seen!",
                "author": "LinuxEnthusiast",
                "like_count": 127,
                "reply_count": 8,
                "published_at": "2024-06-15T09:30:00Z",
            },
            {
                "text": "Can you do a video about Nix flakes?",
                "author": "FlakeLearner",
                "like_count": 45,
                "reply_count": 3,
                "published_at": "2024-06-16T14:22:00Z",
            },
            {
                "text": "Finally understand home-manager, thanks!",
                "author": "HomeManagerFan",
                "like_count": 89,
                "reply_count": 0,
                "published_at": "2024-06-17T20:15:00Z",
            },
        ]

        result = chunker.chunk_comments(comments, video_metadata)

        assert len(result) == 3

        # Check first comment
        doc0 = result[0]
        assert doc0.page_content == "This is the best NixOS tutorial I've ever seen!"
        assert doc0.metadata["content_type"] == "comment"
        assert doc0.metadata["author"] == "LinuxEnthusiast"
        assert doc0.metadata["like_count"] == 127
        assert doc0.metadata["video_title"] == "NixOS Full Guide"

        # Check that all have correct content_type
        for doc in result:
            assert doc.metadata["content_type"] == "comment"
            assert doc.metadata["channel_title"] == "Vimjoyer"
