"""Tests for semantic search MCP tools.

Tests the semantic_search_transcripts tool with auto-indexing behavior,
scope determination (channels, videos, both, neither), and search functionality.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.youtube.semantic.indexer import IndexingResult
from app.tools.youtube.semantic.tools import (
    _describe_scope,
    delete_indexed_video,
    get_indexed_videos,
    get_indexer,
    index_channel_transcripts,
    index_video_transcript,
    semantic_search_all,
    semantic_search_comments,
    semantic_search_transcripts,
)


class TestGetIndexer:
    """Tests for the get_indexer factory function."""

    def test_returns_transcript_indexer(self) -> None:
        """get_indexer returns a TranscriptIndexer instance."""
        with (
            patch("app.tools.youtube.semantic.store.get_vector_store") as mock_store,
            patch(
                "app.tools.youtube.semantic.chunker.TranscriptChunker"
            ) as mock_chunker,
            patch(
                "app.tools.youtube.semantic.config.get_semantic_config"
            ) as mock_config,
        ):
            mock_store.return_value = MagicMock()
            mock_chunker.return_value = MagicMock()
            mock_config.return_value = MagicMock()

            indexer = get_indexer()

            assert indexer is not None
            mock_store.assert_called_once()
            mock_chunker.assert_called_once()

    def test_creates_new_instance_each_call(self) -> None:
        """get_indexer creates a new instance each call (no caching)."""
        with (
            patch("app.tools.youtube.semantic.store.get_vector_store") as mock_store,
            patch(
                "app.tools.youtube.semantic.chunker.TranscriptChunker"
            ) as mock_chunker,
            patch(
                "app.tools.youtube.semantic.config.get_semantic_config"
            ) as mock_config,
        ):
            mock_store.return_value = MagicMock()
            mock_chunker.return_value = MagicMock()
            mock_config.return_value = MagicMock()

            indexer1 = get_indexer()
            indexer2 = get_indexer()

            # Each call creates a new instance (no caching)
            assert indexer1 is not indexer2
            # Should be called twice
            assert mock_store.call_count == 2


class TestDescribeScope:
    """Tests for the _describe_scope helper function."""

    def test_no_scope(self) -> None:
        """Returns 'all indexed content' when no scope provided."""
        result = _describe_scope(None, None)
        assert result == "all indexed content"

    def test_channels_only(self) -> None:
        """Describes channel scope correctly."""
        result = _describe_scope(["ch1", "ch2"], None)
        assert result == "2 channel(s)"

    def test_videos_only(self) -> None:
        """Describes video scope correctly."""
        result = _describe_scope(None, ["v1", "v2", "v3"])
        assert result == "3 specific video(s)"

    def test_both_channels_and_videos(self) -> None:
        """Describes combined scope correctly."""
        result = _describe_scope(["ch1"], ["v1", "v2"])
        assert result == "1 channel(s) + 2 specific video(s)"

    def test_empty_lists_treated_as_no_scope(self) -> None:
        """Empty lists are falsy and treated as no scope."""
        result = _describe_scope([], [])
        assert result == "all indexed content"


class TestIndexChannelTranscripts:
    """Tests for the index_channel_transcripts tool."""

    @pytest.mark.asyncio
    async def test_calls_indexer_with_correct_params(self) -> None:
        """Passes parameters correctly to indexer.index_channel."""
        mock_indexer = MagicMock()
        mock_result = IndexingResult(
            indexed_count=5,
            chunk_count=50,
            skipped_count=2,
            error_count=1,
            errors=["error1"],
            video_ids=["v1", "v2", "v3", "v4", "v5"],
        )
        mock_indexer.index_channel = AsyncMock(return_value=mock_result)

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await index_channel_transcripts(
                channel_id="UCtest123",
                max_videos=25,
                language="de",
                force_reindex=True,
            )

        mock_indexer.index_channel.assert_called_once_with(
            channel_id="UCtest123",
            max_videos=25,
            language="de",
            force_reindex=True,
        )
        assert result["indexed_count"] == 5
        assert result["chunk_count"] == 50
        assert result["skipped_count"] == 2
        assert result["error_count"] == 1

    @pytest.mark.asyncio
    async def test_returns_dict_format(self) -> None:
        """Returns IndexingResult.to_dict() format."""
        mock_indexer = MagicMock()
        mock_result = IndexingResult(indexed_count=1, chunk_count=10)
        mock_indexer.index_channel = AsyncMock(return_value=mock_result)

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await index_channel_transcripts(channel_id="UCtest")

        assert isinstance(result, dict)
        assert "indexed_count" in result
        assert "chunk_count" in result
        assert "skipped_count" in result
        assert "error_count" in result
        assert "errors" in result
        assert "video_ids" in result


class TestIndexVideoTranscript:
    """Tests for the index_video_transcript tool."""

    @pytest.mark.asyncio
    async def test_calls_indexer_with_correct_params(self) -> None:
        """Passes parameters correctly to indexer.index_video."""
        mock_indexer = MagicMock()
        mock_result = IndexingResult(indexed_count=1, chunk_count=15)
        mock_indexer.index_video = AsyncMock(return_value=mock_result)

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await index_video_transcript(
                video_id="dQw4w9WgXcQ",
                language="fr",
                force_reindex=True,
            )

        mock_indexer.index_video.assert_called_once_with(
            video_id="dQw4w9WgXcQ",
            language="fr",
            force_reindex=True,
        )
        assert result["indexed_count"] == 1
        assert result["chunk_count"] == 15


class TestSemanticSearchTranscripts:
    """Tests for the semantic_search_transcripts tool with auto-indexing."""

    @pytest.fixture
    def mock_indexer(self) -> MagicMock:
        """Create a mock indexer with common setup."""
        indexer = MagicMock()
        indexer.is_video_indexed = MagicMock(return_value=False)
        indexer.index_video = AsyncMock(
            return_value=IndexingResult(indexed_count=1, chunk_count=10)
        )
        indexer.vector_store = MagicMock()
        return indexer

    @pytest.fixture
    def mock_search_results(self) -> list[tuple[MagicMock, float]]:
        """Create mock search results."""
        doc = MagicMock()
        doc.page_content = "This is the matching transcript text"
        doc.metadata = {
            "video_id": "v1",
            "video_title": "Test Video",
            "video_url": "https://youtube.com/watch?v=v1",
            "start_time": 120.5,
            "end_time": 135.0,
            "timestamp_url": "https://youtube.com/watch?v=v1&t=120",
            "channel_id": "ch1",
            "channel_title": "Test Channel",
            "language": "en",
            "chunk_index": 5,
        }
        return [(doc, 0.15)]

    @pytest.mark.asyncio
    async def test_search_with_no_scope_searches_all(
        self, mock_indexer: MagicMock, mock_search_results: list
    ) -> None:
        """When no scope provided, searches all indexed content without auto-indexing."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_transcripts(query="test query")

        # Should not attempt to index anything
        mock_indexer.index_video.assert_not_called()

        # Should search with no filter
        mock_indexer.vector_store.similarity_search_with_score.assert_called_once()
        call_kwargs = mock_indexer.vector_store.similarity_search_with_score.call_args[
            1
        ]
        assert call_kwargs["filter"] is None

        assert result["scope"] == "all indexed content"
        assert result["indexing_stats"]["videos_checked"] == 0

    @pytest.mark.asyncio
    async def test_search_with_video_ids_auto_indexes(
        self, mock_indexer: MagicMock, mock_search_results: list
    ) -> None:
        """Auto-indexes missing videos when video_ids provided."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_transcripts(
                query="test query",
                video_ids=["v1", "v2"],
            )

        # Should check and index both videos
        assert mock_indexer.is_video_indexed.call_count == 2
        assert mock_indexer.index_video.call_count == 2

        # Should filter by video IDs (compare as sets since order is non-deterministic)
        call_kwargs = mock_indexer.vector_store.similarity_search_with_score.call_args[
            1
        ]
        assert "filter" in call_kwargs
        assert "video_id" in call_kwargs["filter"]
        assert "$in" in call_kwargs["filter"]["video_id"]
        assert set(call_kwargs["filter"]["video_id"]["$in"]) == {"v1", "v2"}

        assert result["indexing_stats"]["videos_checked"] == 2
        assert result["indexing_stats"]["videos_indexed"] == 2

    @pytest.mark.asyncio
    async def test_search_with_channel_ids_fetches_and_indexes(
        self, mock_indexer: MagicMock, mock_search_results: list
    ) -> None:
        """Fetches channel videos and auto-indexes when channel_ids provided."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_search_results
        )

        mock_channel_videos = [
            {"video_id": "cv1", "title": "Video 1"},
            {"video_id": "cv2", "title": "Video 2"},
        ]

        with (
            patch(
                "app.tools.youtube.semantic.tools.get_indexer",
                return_value=mock_indexer,
            ),
            patch(
                "app.tools.youtube.search.get_channel_videos",
                new_callable=AsyncMock,
                return_value=mock_channel_videos,
            ) as mock_get_videos,
        ):
            result = await semantic_search_transcripts(
                query="test query",
                channel_ids=["ch1"],
                max_videos_per_channel=25,
            )

        # Should fetch videos from channel
        mock_get_videos.assert_called_once_with("ch1", max_results=25)

        # Should index the channel's videos
        assert mock_indexer.index_video.call_count == 2

        assert result["scope"] == "1 channel(s)"

    @pytest.mark.asyncio
    async def test_search_skips_already_indexed_videos(
        self, mock_indexer: MagicMock, mock_search_results: list
    ) -> None:
        """Skips indexing for videos already in the index."""
        # One video is already indexed, one is not
        # Use a dict to make behavior deterministic regardless of iteration order
        indexed_videos = {"v1"}

        def is_indexed(video_id: str) -> bool:
            return video_id in indexed_videos

        mock_indexer.is_video_indexed = MagicMock(side_effect=is_indexed)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_transcripts(
                query="test",
                video_ids=["v1", "v2"],
            )

        # Should only index the non-indexed video (v2)
        assert mock_indexer.index_video.call_count == 1
        # Check that the one indexed video was v2 (the not-already-indexed one)
        indexed_call_args = [
            call.args[0] for call in mock_indexer.index_video.call_args_list
        ]
        assert indexed_call_args == ["v2"]

        assert result["indexing_stats"]["videos_already_indexed"] == 1
        assert result["indexing_stats"]["videos_indexed"] == 1

    @pytest.mark.asyncio
    async def test_search_handles_indexing_failures(
        self, mock_indexer: MagicMock, mock_search_results: list
    ) -> None:
        """Counts failed indexing attempts correctly."""
        # index_video returns error result
        mock_indexer.index_video = AsyncMock(
            return_value=IndexingResult(error_count=1, errors=["No transcript"])
        )
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_transcripts(
                query="test",
                video_ids=["v1"],
            )

        assert result["indexing_stats"]["videos_failed"] == 1
        assert result["indexing_stats"]["videos_indexed"] == 0

    @pytest.mark.asyncio
    async def test_search_formats_results_correctly(
        self, mock_indexer: MagicMock, mock_search_results: list
    ) -> None:
        """Formats search results with all expected fields."""
        mock_indexer.is_video_indexed = MagicMock(return_value=True)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_transcripts(
                query="test query",
                video_ids=["v1"],
            )

        assert result["query"] == "test query"
        assert result["total_results"] == 1
        assert len(result["results"]) == 1

        hit = result["results"][0]
        assert hit["video_id"] == "v1"
        assert hit["video_title"] == "Test Video"
        assert hit["text"] == "This is the matching transcript text"
        assert hit["start_time"] == 120.5
        assert hit["end_time"] == 135.0
        assert hit["timestamp_url"] == "https://youtube.com/watch?v=v1&t=120"
        assert hit["score"] == 0.15
        assert hit["channel_id"] == "ch1"
        assert hit["channel_title"] == "Test Channel"

    @pytest.mark.asyncio
    async def test_search_respects_min_score_filter(
        self, mock_indexer: MagicMock
    ) -> None:
        """Filters results by min_score threshold."""
        # Create results with different scores
        doc1 = MagicMock()
        doc1.page_content = "Good match"
        doc1.metadata = {"video_id": "v1"}

        doc2 = MagicMock()
        doc2.page_content = "Poor match"
        doc2.metadata = {"video_id": "v2"}

        mock_results = [(doc1, 0.1), (doc2, 0.8)]  # Lower is better for cosine

        mock_indexer.is_video_indexed = MagicMock(return_value=True)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_transcripts(
                query="test",
                video_ids=["v1", "v2"],
                min_score=0.5,  # Only keep scores <= 0.5
            )

        # Should filter out doc2 (score 0.8 > 0.5)
        assert result["total_results"] == 1
        assert result["results"][0]["text"] == "Good match"

    @pytest.mark.asyncio
    async def test_search_handles_vector_store_error(
        self, mock_indexer: MagicMock
    ) -> None:
        """Returns error information when vector search fails."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            side_effect=Exception("Vector store connection failed")
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_transcripts(query="test")

        assert result["total_results"] == 0
        assert result["results"] == []
        assert "error" in result
        assert "Vector store connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_search_combines_channel_and_video_scope(
        self, mock_indexer: MagicMock, mock_search_results: list
    ) -> None:
        """Combines videos from channels and explicit video_ids."""
        mock_indexer.is_video_indexed = MagicMock(return_value=True)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_search_results
        )

        mock_channel_videos = [{"video_id": "cv1"}, {"video_id": "cv2"}]

        with (
            patch(
                "app.tools.youtube.semantic.tools.get_indexer",
                return_value=mock_indexer,
            ),
            patch(
                "app.tools.youtube.search.get_channel_videos",
                new_callable=AsyncMock,
                return_value=mock_channel_videos,
            ),
        ):
            result = await semantic_search_transcripts(
                query="test",
                channel_ids=["ch1"],
                video_ids=["v1", "v2"],
            )

        # Should check 4 videos total (2 from channel + 2 explicit)
        assert result["indexing_stats"]["videos_checked"] == 4

        # Filter should include all 4 video IDs
        call_kwargs = mock_indexer.vector_store.similarity_search_with_score.call_args[
            1
        ]
        filter_video_ids = set(call_kwargs["filter"]["video_id"]["$in"])
        assert filter_video_ids == {"cv1", "cv2", "v1", "v2"}

        assert result["scope"] == "1 channel(s) + 2 specific video(s)"

    @pytest.mark.asyncio
    async def test_search_uses_custom_k_value(
        self, mock_indexer: MagicMock, mock_search_results: list
    ) -> None:
        """Passes k parameter to similarity search."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            await semantic_search_transcripts(query="test", k=25)

        call_kwargs = mock_indexer.vector_store.similarity_search_with_score.call_args[
            1
        ]
        assert call_kwargs["k"] == 25

    @pytest.mark.asyncio
    async def test_search_uses_custom_language(
        self, mock_indexer: MagicMock, mock_search_results: list
    ) -> None:
        """Uses specified language for auto-indexing."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            await semantic_search_transcripts(
                query="test",
                video_ids=["v1"],
                language="de",
            )

        mock_indexer.index_video.assert_called_with("v1", language="de")

    @pytest.mark.asyncio
    async def test_search_handles_channel_fetch_error(
        self, mock_indexer: MagicMock, mock_search_results: list
    ) -> None:
        """Continues with other channels if one fails to fetch."""
        mock_indexer.is_video_indexed = MagicMock(return_value=True)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_search_results
        )

        async def mock_get_videos(channel_id: str, max_results: int) -> list[dict]:
            if channel_id == "bad_channel":
                raise Exception("Channel not found")
            return [{"video_id": f"{channel_id}_v1"}]

        with (
            patch(
                "app.tools.youtube.semantic.tools.get_indexer",
                return_value=mock_indexer,
            ),
            patch(
                "app.tools.youtube.search.get_channel_videos",
                side_effect=mock_get_videos,
            ),
        ):
            result = await semantic_search_transcripts(
                query="test",
                channel_ids=["good_channel", "bad_channel"],
            )

        # Should still have results from good_channel
        assert result["indexing_stats"]["videos_checked"] == 1


class TestSemanticSearchComments:
    """Tests for the semantic_search_comments tool with auto-indexing."""

    @pytest.fixture
    def mock_indexer(self) -> MagicMock:
        """Create a mock indexer with common setup for comment search."""
        indexer = MagicMock()
        indexer.is_video_comments_indexed = MagicMock(return_value=False)
        indexer.index_video_comments = AsyncMock(
            return_value=IndexingResult(indexed_count=1, chunk_count=15)
        )
        indexer.vector_store = MagicMock()
        return indexer

    @pytest.fixture
    def mock_comment_search_results(self) -> list[tuple[MagicMock, float]]:
        """Create mock comment search results."""
        doc = MagicMock()
        doc.page_content = "This is a great tutorial, thank you!"
        doc.metadata = {
            "content_type": "comment",
            "video_id": "v1",
            "video_title": "Test Video",
            "video_url": "https://youtube.com/watch?v=v1",
            "channel_id": "ch1",
            "channel_title": "Test Channel",
            "author": "TestUser123",
            "like_count": 42,
            "reply_count": 3,
            "published_at": "2024-01-15T10:30:00Z",
            "chunk_index": 5,
        }
        return [(doc, 0.12)]

    @pytest.mark.asyncio
    async def test_search_with_no_scope_searches_all_comments(
        self, mock_indexer: MagicMock, mock_comment_search_results: list
    ) -> None:
        """When no scope provided, searches all indexed comments without auto-indexing."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_comment_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_comments(query="great tutorial")

        # Should not attempt to index anything
        mock_indexer.index_video_comments.assert_not_called()

        # Should search with content_type filter only
        mock_indexer.vector_store.similarity_search_with_score.assert_called_once()
        call_kwargs = mock_indexer.vector_store.similarity_search_with_score.call_args[
            1
        ]
        assert call_kwargs["filter"] == {"content_type": "comment"}

        assert result["scope"] == "all indexed content"
        assert result["indexing_stats"]["videos_checked"] == 0

    @pytest.mark.asyncio
    async def test_search_with_video_ids_auto_indexes(
        self, mock_indexer: MagicMock, mock_comment_search_results: list
    ) -> None:
        """Auto-indexes missing video comments when video_ids provided."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_comment_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_comments(
                query="test query",
                video_ids=["v1", "v2"],
            )

        # Should check and index both videos
        assert mock_indexer.is_video_comments_indexed.call_count == 2
        assert mock_indexer.index_video_comments.call_count == 2

        # Should filter by content_type AND video IDs using $and
        call_kwargs = mock_indexer.vector_store.similarity_search_with_score.call_args[
            1
        ]
        assert "filter" in call_kwargs
        filter_dict = call_kwargs["filter"]
        assert "$and" in filter_dict
        # Extract the conditions from $and
        conditions = filter_dict["$and"]
        assert {"content_type": "comment"} in conditions
        # Find the video_id condition
        video_condition = next(c for c in conditions if "video_id" in c)
        assert set(video_condition["video_id"]["$in"]) == {"v1", "v2"}

        assert result["indexing_stats"]["videos_checked"] == 2
        assert result["indexing_stats"]["videos_indexed"] == 2

    @pytest.mark.asyncio
    async def test_search_with_channel_ids_fetches_and_indexes(
        self, mock_indexer: MagicMock, mock_comment_search_results: list
    ) -> None:
        """Fetches channel videos and auto-indexes comments when channel_ids provided."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_comment_search_results
        )

        mock_channel_videos = [
            {"video_id": "cv1", "title": "Video 1"},
            {"video_id": "cv2", "title": "Video 2"},
        ]

        with (
            patch(
                "app.tools.youtube.semantic.tools.get_indexer",
                return_value=mock_indexer,
            ),
            patch(
                "app.tools.youtube.search.get_channel_videos",
                new_callable=AsyncMock,
                return_value=mock_channel_videos,
            ) as mock_get_videos,
        ):
            result = await semantic_search_comments(
                query="test query",
                channel_ids=["ch1"],
                max_videos_per_channel=25,
            )

        # Should fetch videos from channel
        mock_get_videos.assert_called_once_with("ch1", max_results=25)

        # Should index the channel's videos' comments
        assert mock_indexer.index_video_comments.call_count == 2

        assert result["scope"] == "1 channel(s)"

    @pytest.mark.asyncio
    async def test_search_skips_already_indexed_videos(
        self, mock_indexer: MagicMock, mock_comment_search_results: list
    ) -> None:
        """Skips indexing for videos whose comments are already indexed."""
        # One video is already indexed, one is not
        indexed_videos = {"v1"}

        def is_indexed(video_id: str) -> bool:
            return video_id in indexed_videos

        mock_indexer.is_video_comments_indexed = MagicMock(side_effect=is_indexed)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_comment_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_comments(
                query="test",
                video_ids=["v1", "v2"],
            )

        # Should only index the non-indexed video (v2)
        assert mock_indexer.index_video_comments.call_count == 1
        indexed_call_args = [
            call.args[0] for call in mock_indexer.index_video_comments.call_args_list
        ]
        assert indexed_call_args == ["v2"]

        assert result["indexing_stats"]["videos_already_indexed"] == 1
        assert result["indexing_stats"]["videos_indexed"] == 1

    @pytest.mark.asyncio
    async def test_search_handles_indexing_failures(
        self, mock_indexer: MagicMock, mock_comment_search_results: list
    ) -> None:
        """Counts failed comment indexing attempts correctly."""
        # index_video_comments returns error result
        mock_indexer.index_video_comments = AsyncMock(
            return_value=IndexingResult(error_count=1, errors=["Comments disabled"])
        )
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_comment_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_comments(
                query="test",
                video_ids=["v1"],
            )

        assert result["indexing_stats"]["videos_failed"] == 1
        assert result["indexing_stats"]["videos_indexed"] == 0

    @pytest.mark.asyncio
    async def test_search_formats_results_correctly(
        self, mock_indexer: MagicMock, mock_comment_search_results: list
    ) -> None:
        """Formats comment search results with all expected fields."""
        mock_indexer.is_video_comments_indexed = MagicMock(return_value=True)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_comment_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_comments(
                query="great tutorial",
                video_ids=["v1"],
            )

        assert result["query"] == "great tutorial"
        assert result["total_results"] == 1
        assert len(result["results"]) == 1

        hit = result["results"][0]
        assert hit["video_id"] == "v1"
        assert hit["video_title"] == "Test Video"
        assert hit["text"] == "This is a great tutorial, thank you!"
        assert hit["author"] == "TestUser123"
        assert hit["like_count"] == 42
        assert hit["reply_count"] == 3
        assert hit["published_at"] == "2024-01-15T10:30:00Z"
        assert hit["score"] == 0.12
        assert hit["channel_id"] == "ch1"
        assert hit["channel_title"] == "Test Channel"
        assert hit["chunk_index"] == 5

    @pytest.mark.asyncio
    async def test_search_respects_min_score_filter(
        self, mock_indexer: MagicMock
    ) -> None:
        """Filters comment results by min_score threshold."""
        # Create results with different scores
        doc1 = MagicMock()
        doc1.page_content = "Great video!"
        doc1.metadata = {"video_id": "v1", "author": "User1"}

        doc2 = MagicMock()
        doc2.page_content = "Random comment"
        doc2.metadata = {"video_id": "v2", "author": "User2"}

        mock_results = [(doc1, 0.1), (doc2, 0.8)]  # Lower is better for cosine

        mock_indexer.is_video_comments_indexed = MagicMock(return_value=True)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_comments(
                query="test",
                video_ids=["v1", "v2"],
                min_score=0.5,  # Only keep scores <= 0.5
            )

        # Should filter out doc2 (score 0.8 > 0.5)
        assert result["total_results"] == 1
        assert result["results"][0]["text"] == "Great video!"

    @pytest.mark.asyncio
    async def test_search_handles_vector_store_error(
        self, mock_indexer: MagicMock
    ) -> None:
        """Returns error information when vector search fails."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            side_effect=Exception("Vector store connection failed")
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_comments(query="test")

        assert result["total_results"] == 0
        assert result["results"] == []
        assert "error" in result
        assert "Vector store connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_search_combines_channel_and_video_scope(
        self, mock_indexer: MagicMock, mock_comment_search_results: list
    ) -> None:
        """Combines videos from channels and explicit video_ids."""
        mock_indexer.is_video_comments_indexed = MagicMock(return_value=True)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_comment_search_results
        )

        mock_channel_videos = [{"video_id": "cv1"}, {"video_id": "cv2"}]

        with (
            patch(
                "app.tools.youtube.semantic.tools.get_indexer",
                return_value=mock_indexer,
            ),
            patch(
                "app.tools.youtube.search.get_channel_videos",
                new_callable=AsyncMock,
                return_value=mock_channel_videos,
            ),
        ):
            result = await semantic_search_comments(
                query="test",
                channel_ids=["ch1"],
                video_ids=["v1", "v2"],
            )

        # Should check 4 videos total (2 from channel + 2 explicit)
        assert result["indexing_stats"]["videos_checked"] == 4

        # Filter should include $and with content_type and all 4 video IDs
        call_kwargs = mock_indexer.vector_store.similarity_search_with_score.call_args[
            1
        ]
        filter_dict = call_kwargs["filter"]
        assert "$and" in filter_dict
        conditions = filter_dict["$and"]
        video_condition = next(c for c in conditions if "video_id" in c)
        filter_video_ids = set(video_condition["video_id"]["$in"])
        assert filter_video_ids == {"cv1", "cv2", "v1", "v2"}

        assert result["scope"] == "1 channel(s) + 2 specific video(s)"

    @pytest.mark.asyncio
    async def test_search_uses_custom_k_value(
        self, mock_indexer: MagicMock, mock_comment_search_results: list
    ) -> None:
        """Passes k parameter to similarity search."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_comment_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            await semantic_search_comments(query="test", k=25)

        call_kwargs = mock_indexer.vector_store.similarity_search_with_score.call_args[
            1
        ]
        assert call_kwargs["k"] == 25

    @pytest.mark.asyncio
    async def test_search_uses_custom_max_comments(
        self, mock_indexer: MagicMock, mock_comment_search_results: list
    ) -> None:
        """Uses specified max_comments_per_video for auto-indexing."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_comment_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            await semantic_search_comments(
                query="test",
                video_ids=["v1"],
                max_comments_per_video=50,
            )

        mock_indexer.index_video_comments.assert_called_with("v1", max_comments=50)

    @pytest.mark.asyncio
    async def test_search_handles_channel_fetch_error(
        self, mock_indexer: MagicMock, mock_comment_search_results: list
    ) -> None:
        """Continues with other channels if one fails to fetch."""
        mock_indexer.is_video_comments_indexed = MagicMock(return_value=True)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_comment_search_results
        )

        async def mock_get_videos(channel_id: str, max_results: int) -> list[dict]:
            if channel_id == "bad_channel":
                raise Exception("Channel not found")
            return [{"video_id": f"{channel_id}_v1"}]

        with (
            patch(
                "app.tools.youtube.semantic.tools.get_indexer",
                return_value=mock_indexer,
            ),
            patch(
                "app.tools.youtube.search.get_channel_videos",
                side_effect=mock_get_videos,
            ),
        ):
            result = await semantic_search_comments(
                query="test",
                channel_ids=["good_channel", "bad_channel"],
            )

        # Should still have results from good_channel
        assert result["indexing_stats"]["videos_checked"] == 1


class TestSemanticSearchAll:
    """Tests for the semantic_search_all tool with unified search."""

    @pytest.fixture
    def mock_indexer(self) -> MagicMock:
        """Create a mock indexer for unified search."""
        indexer = MagicMock()
        indexer.is_video_indexed = MagicMock(return_value=False)
        indexer.is_video_comments_indexed = MagicMock(return_value=False)
        indexer.index_video = AsyncMock(
            return_value=IndexingResult(indexed_count=1, chunk_count=10)
        )
        indexer.index_video_comments = AsyncMock(
            return_value=IndexingResult(indexed_count=1, chunk_count=15)
        )
        indexer.vector_store = MagicMock()
        return indexer

    @pytest.fixture
    def mock_mixed_search_results(self) -> list[tuple[MagicMock, float]]:
        """Create mock results with both transcripts and comments."""
        transcript_doc = MagicMock()
        transcript_doc.page_content = "This is transcript text about garbage collection"
        transcript_doc.metadata = {
            "content_type": "transcript",
            "video_id": "v1",
            "video_title": "Test Video",
            "video_url": "https://youtube.com/watch?v=v1",
            "channel_id": "ch1",
            "channel_title": "Test Channel",
            "start_time": 120.5,
            "end_time": 135.0,
            "timestamp_url": "https://youtube.com/watch?v=v1&t=120",
            "language": "en",
            "chunk_index": 5,
        }

        comment_doc = MagicMock()
        comment_doc.page_content = "Great explanation of garbage collection!"
        comment_doc.metadata = {
            "content_type": "comment",
            "video_id": "v1",
            "video_title": "Test Video",
            "video_url": "https://youtube.com/watch?v=v1",
            "channel_id": "ch1",
            "channel_title": "Test Channel",
            "author": "NixFan42",
            "like_count": 15,
            "reply_count": 2,
            "published_at": "2024-01-15T10:30:00Z",
            "chunk_index": 3,
        }

        return [(transcript_doc, 0.10), (comment_doc, 0.18)]

    @pytest.mark.asyncio
    async def test_search_all_content_types_by_default(
        self, mock_indexer: MagicMock, mock_mixed_search_results: list
    ) -> None:
        """Searches both transcripts and comments when no content_types specified."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_mixed_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_all(query="garbage collection")

        # Should search without content_type filter (all types)
        mock_indexer.vector_store.similarity_search_with_score.assert_called_once()
        call_kwargs = mock_indexer.vector_store.similarity_search_with_score.call_args[
            1
        ]
        assert call_kwargs["filter"] is None

        assert result["total_results"] == 2
        assert set(result["content_types_searched"]) == {"transcript", "comment"}

    @pytest.mark.asyncio
    async def test_search_filters_by_content_type(
        self, mock_indexer: MagicMock, mock_mixed_search_results: list
    ) -> None:
        """Filters to specified content types."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=[mock_mixed_search_results[0]]  # Only transcript
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_all(
                query="test",
                content_types=["transcript"],
            )

        call_kwargs = mock_indexer.vector_store.similarity_search_with_score.call_args[
            1
        ]
        assert call_kwargs["filter"] == {"content_type": "transcript"}
        assert result["content_types_searched"] == ["transcript"]

    @pytest.mark.asyncio
    async def test_search_indexes_both_types_when_scoped(
        self, mock_indexer: MagicMock, mock_mixed_search_results: list
    ) -> None:
        """Indexes both transcripts and comments when video scope provided."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_mixed_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_all(
                query="test",
                video_ids=["v1"],
            )

        # Should index both transcripts and comments
        mock_indexer.index_video.assert_called_once_with("v1", language="en")
        mock_indexer.index_video_comments.assert_called_once_with(
            "v1", max_comments=100
        )

        assert result["indexing_stats"]["transcripts"]["videos_indexed"] == 1
        assert result["indexing_stats"]["comments"]["videos_indexed"] == 1

    @pytest.mark.asyncio
    async def test_search_only_indexes_requested_types(
        self, mock_indexer: MagicMock, mock_mixed_search_results: list
    ) -> None:
        """Only indexes content types that are being searched."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=[mock_mixed_search_results[0]]
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_all(
                query="test",
                content_types=["transcript"],
                video_ids=["v1"],
            )

        # Should only index transcripts, not comments
        mock_indexer.index_video.assert_called_once()
        mock_indexer.index_video_comments.assert_not_called()

        assert result["indexing_stats"]["transcripts"]["videos_checked"] == 1
        assert result["indexing_stats"]["comments"]["videos_checked"] == 0

    @pytest.mark.asyncio
    async def test_search_formats_transcript_results(
        self, mock_indexer: MagicMock, mock_mixed_search_results: list
    ) -> None:
        """Transcript results include timestamp fields."""
        mock_indexer.is_video_indexed = MagicMock(return_value=True)
        mock_indexer.is_video_comments_indexed = MagicMock(return_value=True)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=[mock_mixed_search_results[0]]  # Only transcript
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_all(
                query="test",
                video_ids=["v1"],
            )

        hit = result["results"][0]
        assert hit["content_type"] == "transcript"
        assert hit["start_time"] == 120.5
        assert hit["end_time"] == 135.0
        assert hit["timestamp_url"] == "https://youtube.com/watch?v=v1&t=120"
        assert hit["language"] == "en"
        # Should not have comment-specific fields
        assert "author" not in hit
        assert "like_count" not in hit

    @pytest.mark.asyncio
    async def test_search_formats_comment_results(
        self, mock_indexer: MagicMock, mock_mixed_search_results: list
    ) -> None:
        """Comment results include author and engagement fields."""
        mock_indexer.is_video_indexed = MagicMock(return_value=True)
        mock_indexer.is_video_comments_indexed = MagicMock(return_value=True)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=[mock_mixed_search_results[1]]  # Only comment
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_all(
                query="test",
                video_ids=["v1"],
            )

        hit = result["results"][0]
        assert hit["content_type"] == "comment"
        assert hit["author"] == "NixFan42"
        assert hit["like_count"] == 15
        assert hit["reply_count"] == 2
        assert hit["published_at"] == "2024-01-15T10:30:00Z"
        # Should not have transcript-specific fields
        assert "start_time" not in hit
        assert "timestamp_url" not in hit

    @pytest.mark.asyncio
    async def test_search_skips_already_indexed(
        self, mock_indexer: MagicMock, mock_mixed_search_results: list
    ) -> None:
        """Skips indexing for content already indexed."""
        mock_indexer.is_video_indexed = MagicMock(return_value=True)
        mock_indexer.is_video_comments_indexed = MagicMock(return_value=True)
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_mixed_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_all(
                query="test",
                video_ids=["v1"],
            )

        mock_indexer.index_video.assert_not_called()
        mock_indexer.index_video_comments.assert_not_called()

        assert result["indexing_stats"]["transcripts"]["videos_already_indexed"] == 1
        assert result["indexing_stats"]["comments"]["videos_already_indexed"] == 1

    @pytest.mark.asyncio
    async def test_search_handles_vector_store_error(
        self, mock_indexer: MagicMock
    ) -> None:
        """Returns error info when search fails."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            side_effect=Exception("Connection failed")
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_all(query="test")

        assert result["total_results"] == 0
        assert "error" in result
        assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_search_respects_min_score(
        self, mock_indexer: MagicMock, mock_mixed_search_results: list
    ) -> None:
        """Filters results by min_score threshold."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_mixed_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_all(
                query="test",
                min_score=0.15,  # Should filter out comment (0.18)
            )

        assert result["total_results"] == 1
        assert result["results"][0]["content_type"] == "transcript"

    @pytest.mark.asyncio
    async def test_search_with_channel_scope(
        self, mock_indexer: MagicMock, mock_mixed_search_results: list
    ) -> None:
        """Fetches videos from channels and indexes both types."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_mixed_search_results
        )

        mock_channel_videos = [{"video_id": "cv1"}, {"video_id": "cv2"}]

        with (
            patch(
                "app.tools.youtube.semantic.tools.get_indexer",
                return_value=mock_indexer,
            ),
            patch(
                "app.tools.youtube.search.get_channel_videos",
                new_callable=AsyncMock,
                return_value=mock_channel_videos,
            ) as mock_get_videos,
        ):
            result = await semantic_search_all(
                query="test",
                channel_ids=["ch1"],
                max_videos_per_channel=25,
            )

        mock_get_videos.assert_called_once_with("ch1", max_results=25)

        # Should index both types for both videos
        assert mock_indexer.index_video.call_count == 2
        assert mock_indexer.index_video_comments.call_count == 2

        assert result["scope"] == "1 channel(s)"

    @pytest.mark.asyncio
    async def test_search_ignores_invalid_content_types(
        self, mock_indexer: MagicMock, mock_mixed_search_results: list
    ) -> None:
        """Falls back to all types when invalid types provided."""
        mock_indexer.vector_store.similarity_search_with_score = MagicMock(
            return_value=mock_mixed_search_results
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await semantic_search_all(
                query="test",
                content_types=["invalid", "nonsense"],
            )

        # Should fall back to searching all valid types
        assert set(result["content_types_searched"]) == {"transcript", "comment"}


class TestGetIndexedVideos:
    """Tests for the get_indexed_videos utility tool."""

    @pytest.fixture
    def mock_indexer(self) -> MagicMock:
        """Create a mock indexer for utility tests."""
        indexer = MagicMock()
        indexer.get_indexed_video_ids_by_content_type = AsyncMock(
            return_value=["v1", "v2", "v3"]
        )
        return indexer

    @pytest.mark.asyncio
    async def test_returns_all_indexed_videos(self, mock_indexer: MagicMock) -> None:
        """Returns list of all indexed video IDs."""
        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await get_indexed_videos()

        mock_indexer.get_indexed_video_ids_by_content_type.assert_called_once_with(
            content_type=None,
            channel_id=None,
        )

        assert result["video_ids"] == ["v1", "v2", "v3"]
        assert result["total_count"] == 3
        assert result["channel_filter"] is None
        assert result["content_type_filter"] is None

    @pytest.mark.asyncio
    async def test_filters_by_channel_id(self, mock_indexer: MagicMock) -> None:
        """Passes channel_id filter to indexer."""
        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await get_indexed_videos(channel_id="ch1")

        mock_indexer.get_indexed_video_ids_by_content_type.assert_called_once_with(
            content_type=None,
            channel_id="ch1",
        )

        assert result["channel_filter"] == "ch1"

    @pytest.mark.asyncio
    async def test_filters_by_content_type(self, mock_indexer: MagicMock) -> None:
        """Passes content_type filter to indexer."""
        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await get_indexed_videos(content_type="comment")

        mock_indexer.get_indexed_video_ids_by_content_type.assert_called_once_with(
            content_type="comment",
            channel_id=None,
        )

        assert result["content_type_filter"] == "comment"

    @pytest.mark.asyncio
    async def test_handles_error(self, mock_indexer: MagicMock) -> None:
        """Returns error info when indexer fails."""
        mock_indexer.get_indexed_video_ids_by_content_type = AsyncMock(
            side_effect=Exception("Database error")
        )

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await get_indexed_videos()

        assert result["video_ids"] == []
        assert result["total_count"] == 0
        assert "error" in result
        assert "Database error" in result["error"]


class TestDeleteIndexedVideo:
    """Tests for the delete_indexed_video utility tool."""

    @pytest.fixture
    def mock_indexer(self) -> MagicMock:
        """Create a mock indexer for deletion tests."""
        indexer = MagicMock()
        indexer.delete_video = AsyncMock(return_value=10)
        indexer.delete_video_comments = AsyncMock(return_value=25)
        return indexer

    @pytest.mark.asyncio
    async def test_deletes_all_content_by_default(
        self, mock_indexer: MagicMock
    ) -> None:
        """Deletes both transcripts and comments when no type specified."""
        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await delete_indexed_video("v1")

        mock_indexer.delete_video.assert_called_once_with("v1")
        mock_indexer.delete_video_comments.assert_called_once_with("v1")

        assert result["video_id"] == "v1"
        assert result["transcripts_deleted"] == 10
        assert result["comments_deleted"] == 25
        assert result["total_deleted"] == 35
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_deletes_only_transcripts(self, mock_indexer: MagicMock) -> None:
        """Deletes only transcripts when content_type is 'transcript'."""
        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await delete_indexed_video("v1", content_type="transcript")

        mock_indexer.delete_video.assert_called_once_with("v1")
        mock_indexer.delete_video_comments.assert_not_called()

        assert result["transcripts_deleted"] == 10
        assert result["comments_deleted"] == 0
        assert result["total_deleted"] == 10
        assert result["content_type_filter"] == "transcript"

    @pytest.mark.asyncio
    async def test_deletes_only_comments(self, mock_indexer: MagicMock) -> None:
        """Deletes only comments when content_type is 'comment'."""
        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await delete_indexed_video("v1", content_type="comment")

        mock_indexer.delete_video.assert_not_called()
        mock_indexer.delete_video_comments.assert_called_once_with("v1")

        assert result["transcripts_deleted"] == 0
        assert result["comments_deleted"] == 25
        assert result["total_deleted"] == 25
        assert result["content_type_filter"] == "comment"

    @pytest.mark.asyncio
    async def test_handles_error(self, mock_indexer: MagicMock) -> None:
        """Returns error info when deletion fails."""
        mock_indexer.delete_video = AsyncMock(side_effect=Exception("Delete failed"))

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await delete_indexed_video("v1")

        assert result["success"] is False
        assert "error" in result
        assert "Delete failed" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_zero_when_nothing_to_delete(
        self, mock_indexer: MagicMock
    ) -> None:
        """Returns zero counts when video not indexed."""
        mock_indexer.delete_video = AsyncMock(return_value=0)
        mock_indexer.delete_video_comments = AsyncMock(return_value=0)

        with patch(
            "app.tools.youtube.semantic.tools.get_indexer",
            return_value=mock_indexer,
        ):
            result = await delete_indexed_video("not_indexed")

        assert result["total_deleted"] == 0
        assert result["success"] is True
