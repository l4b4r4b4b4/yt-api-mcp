"""Tests for the TranscriptIndexer batch indexing functionality.

This module tests the TranscriptIndexer class methods including:
- is_video_indexed(): Check if a video has chunks in the store
- delete_video(): Remove all chunks for a video
- index_video(): Index a single video's transcript
- index_channel(): Batch index all videos from a channel
- get_indexed_video_ids(): List indexed video IDs
- get_chunk_count(): Count chunks in the store

Tests use mocks for YouTube API calls and real ChromaDB with ephemeral storage.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.tools.youtube.semantic.chunker import TranscriptChunker
from app.tools.youtube.semantic.config import SemanticSearchConfig
from app.tools.youtube.semantic.indexer import (
    IndexingResult,
    TranscriptIndexer,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def semantic_config() -> SemanticSearchConfig:
    """Create a test semantic search configuration."""
    return SemanticSearchConfig(
        embedding_model="nomic-embed-text-v1.5",
        embedding_dimensionality=64,  # Small for fast tests
        chunk_size=100,
        chunk_overlap=20,
    )


@pytest.fixture
def mock_chunker(semantic_config: SemanticSearchConfig) -> MagicMock:
    """Create a mock TranscriptChunker."""
    chunker = MagicMock(spec=TranscriptChunker)

    # Default behavior: return 3 documents per video
    def chunk_transcript(
        transcript_entries: list[dict[str, Any]],
        video_metadata: dict[str, str],
        chapters: list[dict[str, Any]] | None = None,
    ) -> list[Document]:
        video_id = video_metadata.get("video_id", "unknown")
        return [
            Document(
                page_content=f"Chunk {i} for {video_id}",
                metadata={
                    "video_id": video_id,
                    "channel_id": video_metadata.get("channel_id", ""),
                    "chunk_index": i,
                    "start_time": i * 10.0,
                    "end_time": (i + 1) * 10.0,
                    **video_metadata,
                },
            )
            for i in range(3)
        ]

    chunker.chunk_transcript.side_effect = chunk_transcript
    return chunker


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock vector store with in-memory storage."""
    store = MagicMock()

    # Internal storage for testing
    storage: dict[str, dict[str, Any]] = {}

    # Mock the underlying collection
    collection = MagicMock()

    def mock_get(
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Simulate ChromaDB get() method."""
        results: dict[str, Any] = {"ids": [], "metadatas": []}

        for doc_id, doc_data in storage.items():
            # Apply where filter
            if where:
                match = all(
                    doc_data.get("metadata", {}).get(key) == value
                    for key, value in where.items()
                )
                if not match:
                    continue

            results["ids"].append(doc_id)
            if include is None or "metadatas" in include:
                results["metadatas"].append(doc_data.get("metadata", {}))

            # Apply limit
            if limit and len(results["ids"]) >= limit:
                break

        return results

    def mock_delete(where: dict[str, Any] | None = None) -> None:
        """Simulate ChromaDB delete() method."""
        if not where:
            return

        to_delete = []
        for doc_id, doc_data in storage.items():
            match = all(
                doc_data.get("metadata", {}).get(key) == value
                for key, value in where.items()
            )
            if match:
                to_delete.append(doc_id)

        for doc_id in to_delete:
            del storage[doc_id]

    def mock_count() -> int:
        """Simulate ChromaDB count() method."""
        return len(storage)

    collection.get.side_effect = mock_get
    collection.delete.side_effect = mock_delete
    collection.count.side_effect = mock_count

    store._collection = collection
    store._storage = storage  # Expose for testing

    def mock_add_documents(documents: list[Document]) -> list[str]:
        """Simulate adding documents to the store."""
        ids = []
        for i, doc in enumerate(documents):
            doc_id = f"doc_{len(storage)}_{i}"
            storage[doc_id] = {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            ids.append(doc_id)
        return ids

    store.add_documents.side_effect = mock_add_documents

    return store


@pytest.fixture
def indexer(
    mock_vector_store: MagicMock,
    mock_chunker: MagicMock,
) -> TranscriptIndexer:
    """Create a TranscriptIndexer with mocked dependencies."""
    return TranscriptIndexer(
        vector_store=mock_vector_store,
        chunker=mock_chunker,
    )


@pytest.fixture
def sample_transcript() -> dict[str, Any]:
    """Sample transcript data."""
    return {
        "video_id": "test_video_1",
        "language": "en",
        "transcript": [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "This is a test", "start": 2.0, "duration": 3.0},
            {"text": "Transcript content", "start": 5.0, "duration": 2.5},
        ],
        "full_text": "Hello world This is a test Transcript content",
    }


@pytest.fixture
def sample_video_details() -> dict[str, Any]:
    """Sample video details."""
    return {
        "video_id": "test_video_1",
        "title": "Test Video Title",
        "description": "A test video",
        "channel_id": "UCtest123456789012345",
        "channel_title": "Test Channel",
        "url": "https://www.youtube.com/watch?v=test_video_1",
        "published_at": "2024-01-15T10:00:00Z",
    }


@pytest.fixture
def sample_channel_videos() -> list[dict[str, Any]]:
    """Sample list of videos from a channel."""
    return [
        {
            "video_id": f"video_{i}",
            "title": f"Video {i} Title",
            "description": f"Description for video {i}",
            "channel_title": "Test Channel",
            "url": f"https://www.youtube.com/watch?v=video_{i}",
            "published_at": f"2024-01-{15 - i:02d}T10:00:00Z",
            "thumbnail": "https://example.com/thumb.jpg",
        }
        for i in range(5)
    ]


# =============================================================================
# IndexingResult Tests
# =============================================================================


class TestIndexingResult:
    """Tests for the IndexingResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values are initialized correctly."""
        result = IndexingResult()

        assert result.indexed_count == 0
        assert result.chunk_count == 0
        assert result.skipped_count == 0
        assert result.error_count == 0
        assert result.errors == []
        assert result.video_ids == []

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = IndexingResult(
            indexed_count=5,
            chunk_count=25,
            skipped_count=2,
            error_count=1,
            errors=["Video xyz failed"],
            video_ids=["a", "b", "c", "d", "e"],
        )

        as_dict = result.to_dict()

        assert as_dict["indexed_count"] == 5
        assert as_dict["chunk_count"] == 25
        assert as_dict["skipped_count"] == 2
        assert as_dict["error_count"] == 1
        assert as_dict["errors"] == ["Video xyz failed"]
        assert as_dict["video_ids"] == ["a", "b", "c", "d", "e"]

    def test_merge(self) -> None:
        """Test merging two IndexingResults."""
        result1 = IndexingResult(
            indexed_count=3,
            chunk_count=15,
            skipped_count=1,
            error_count=0,
            errors=[],
            video_ids=["a", "b", "c"],
        )

        result2 = IndexingResult(
            indexed_count=2,
            chunk_count=10,
            skipped_count=0,
            error_count=1,
            errors=["Error in video d"],
            video_ids=["d", "e"],
        )

        result1.merge(result2)

        assert result1.indexed_count == 5
        assert result1.chunk_count == 25
        assert result1.skipped_count == 1
        assert result1.error_count == 1
        assert result1.errors == ["Error in video d"]
        assert result1.video_ids == ["a", "b", "c", "d", "e"]


# =============================================================================
# is_video_indexed Tests
# =============================================================================


class TestIsVideoIndexed:
    """Tests for the is_video_indexed method."""

    def test_video_indexed_returns_true(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test returns True when video has chunks in store."""
        # Add some chunks for a video
        mock_vector_store._storage["doc1"] = {
            "content": "Chunk 1",
            "metadata": {"video_id": "existing_video"},
        }

        assert indexer.is_video_indexed("existing_video") is True

    def test_video_not_indexed_returns_false(
        self,
        indexer: TranscriptIndexer,
    ) -> None:
        """Test returns False when video has no chunks."""
        assert indexer.is_video_indexed("nonexistent_video") is False

    def test_empty_store_returns_false(
        self,
        indexer: TranscriptIndexer,
    ) -> None:
        """Test returns False when store is empty."""
        assert indexer.is_video_indexed("any_video") is False

    def test_different_video_returns_false(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test returns False when only other videos are indexed."""
        mock_vector_store._storage["doc1"] = {
            "content": "Chunk 1",
            "metadata": {"video_id": "other_video"},
        }

        assert indexer.is_video_indexed("target_video") is False


# =============================================================================
# delete_video Tests
# =============================================================================


class TestDeleteVideo:
    """Tests for the delete_video method."""

    @pytest.mark.asyncio
    async def test_delete_video_removes_chunks(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test deleting a video removes all its chunks."""
        # Add chunks for the video
        mock_vector_store._storage["doc1"] = {
            "content": "Chunk 1",
            "metadata": {"video_id": "video_to_delete"},
        }
        mock_vector_store._storage["doc2"] = {
            "content": "Chunk 2",
            "metadata": {"video_id": "video_to_delete"},
        }
        mock_vector_store._storage["doc3"] = {
            "content": "Chunk 3",
            "metadata": {"video_id": "other_video"},
        }

        deleted = await indexer.delete_video("video_to_delete")

        assert deleted == 2
        assert len(mock_vector_store._storage) == 1
        assert "doc3" in mock_vector_store._storage

    @pytest.mark.asyncio
    async def test_delete_video_returns_count(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test returns correct count of deleted chunks."""
        # Add 5 chunks for the video
        for i in range(5):
            mock_vector_store._storage[f"doc{i}"] = {
                "content": f"Chunk {i}",
                "metadata": {"video_id": "video_with_5_chunks"},
            }

        deleted = await indexer.delete_video("video_with_5_chunks")

        assert deleted == 5
        assert len(mock_vector_store._storage) == 0

    @pytest.mark.asyncio
    async def test_delete_video_not_found_returns_zero(
        self,
        indexer: TranscriptIndexer,
    ) -> None:
        """Test returns 0 when video not in store."""
        deleted = await indexer.delete_video("nonexistent_video")

        assert deleted == 0


# =============================================================================
# index_video Tests
# =============================================================================


class TestIndexVideo:
    """Tests for the index_video method."""

    @pytest.mark.asyncio
    async def test_index_video_creates_chunks(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
        sample_transcript: dict[str, Any],
        sample_video_details: dict[str, Any],
    ) -> None:
        """Test indexing a video creates chunks in the store."""
        with (
            patch(
                "app.tools.youtube.transcripts.get_full_transcript",
                new_callable=AsyncMock,
            ) as mock_transcript,
            patch(
                "app.tools.youtube.metadata.get_video_details",
                new_callable=AsyncMock,
            ) as mock_details,
        ):
            mock_transcript.return_value = sample_transcript
            mock_details.return_value = sample_video_details

            result = await indexer.index_video("test_video_1")

            assert result.indexed_count == 1
            assert result.chunk_count == 3  # Mock chunker creates 3 chunks
            assert result.error_count == 0
            assert "test_video_1" in result.video_ids
            assert len(mock_vector_store._storage) == 3

    @pytest.mark.asyncio
    async def test_index_video_no_transcript(
        self,
        indexer: TranscriptIndexer,
        sample_video_details: dict[str, Any],
    ) -> None:
        """Test handles missing transcript gracefully."""
        with (
            patch(
                "app.tools.youtube.transcripts.get_full_transcript",
                new_callable=AsyncMock,
            ) as mock_transcript,
            patch(
                "app.tools.youtube.metadata.get_video_details",
                new_callable=AsyncMock,
            ) as mock_details,
        ):
            mock_transcript.side_effect = Exception("No transcript available")
            mock_details.return_value = sample_video_details

            result = await indexer.index_video("video_without_transcript")

            assert result.indexed_count == 0
            assert result.error_count == 1
            assert len(result.errors) == 1
            assert "No transcript available" in result.errors[0]

    @pytest.mark.asyncio
    async def test_index_video_skips_if_indexed(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test skips indexing if video already indexed (force=False)."""
        # Pre-populate with existing chunks
        mock_vector_store._storage["existing"] = {
            "content": "Existing chunk",
            "metadata": {"video_id": "already_indexed"},
        }

        result = await indexer.index_video("already_indexed", force_reindex=False)

        assert result.indexed_count == 0
        assert result.skipped_count == 1
        assert result.error_count == 0

    @pytest.mark.asyncio
    async def test_index_video_force_reindex(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
        sample_transcript: dict[str, Any],
        sample_video_details: dict[str, Any],
    ) -> None:
        """Test force_reindex deletes existing and re-indexes."""
        # Pre-populate with existing chunks
        mock_vector_store._storage["old1"] = {
            "content": "Old chunk 1",
            "metadata": {"video_id": "video_to_reindex"},
        }
        mock_vector_store._storage["old2"] = {
            "content": "Old chunk 2",
            "metadata": {"video_id": "video_to_reindex"},
        }

        with (
            patch(
                "app.tools.youtube.transcripts.get_full_transcript",
                new_callable=AsyncMock,
            ) as mock_transcript,
            patch(
                "app.tools.youtube.metadata.get_video_details",
                new_callable=AsyncMock,
            ) as mock_details,
        ):
            sample_transcript["video_id"] = "video_to_reindex"
            sample_video_details["video_id"] = "video_to_reindex"
            mock_transcript.return_value = sample_transcript
            mock_details.return_value = sample_video_details

            result = await indexer.index_video("video_to_reindex", force_reindex=True)

            assert result.indexed_count == 1
            assert result.chunk_count == 3
            # Old chunks should be deleted, new ones added
            assert len(mock_vector_store._storage) == 3

    @pytest.mark.asyncio
    async def test_index_video_uses_provided_metadata(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
        mock_chunker: MagicMock,
        sample_transcript: dict[str, Any],
    ) -> None:
        """Test uses provided video_info instead of fetching."""
        with patch(
            "app.tools.youtube.transcripts.get_full_transcript",
            new_callable=AsyncMock,
        ) as mock_transcript:
            mock_transcript.return_value = sample_transcript

            video_info = {
                "video_id": "test_video_1",
                "title": "Provided Title",
                "channel_title": "Provided Channel",
            }

            result = await indexer.index_video(
                "test_video_1",
                video_info=video_info,
                channel_id="UCprovided",
            )

            assert result.indexed_count == 1

            # Check chunker was called with provided metadata
            call_args = mock_chunker.chunk_transcript.call_args
            video_metadata = call_args.kwargs.get("video_metadata") or call_args.args[1]
            assert video_metadata["video_title"] == "Provided Title"
            assert video_metadata["channel_id"] == "UCprovided"

    @pytest.mark.asyncio
    async def test_index_video_api_error(
        self,
        indexer: TranscriptIndexer,
    ) -> None:
        """Test handles API errors gracefully."""
        with patch(
            "app.tools.youtube.metadata.get_video_details",
            new_callable=AsyncMock,
        ) as mock_details:
            mock_details.side_effect = Exception("API quota exceeded")

            result = await indexer.index_video("video_causing_error")

            assert result.indexed_count == 0
            assert result.error_count == 1
            assert "API quota exceeded" in result.errors[0]


# =============================================================================
# index_channel Tests
# =============================================================================


class TestIndexChannel:
    """Tests for the index_channel method."""

    @pytest.mark.asyncio
    async def test_index_channel_indexes_videos(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
        sample_channel_videos: list[dict[str, Any]],
        sample_transcript: dict[str, Any],
    ) -> None:
        """Test indexing a channel indexes all its videos."""
        with (
            patch(
                "app.tools.youtube.search.get_channel_videos",
                new_callable=AsyncMock,
            ) as mock_channel,
            patch(
                "app.tools.youtube.transcripts.get_full_transcript",
                new_callable=AsyncMock,
            ) as mock_transcript,
            patch(
                "app.tools.youtube.metadata.get_video_details",
                new_callable=AsyncMock,
            ) as mock_details,
        ):
            mock_channel.return_value = sample_channel_videos
            mock_transcript.return_value = sample_transcript
            mock_details.return_value = {"video_id": "test", "title": "Test"}

            result = await indexer.index_channel("UCtest123456789012345")

            assert result.indexed_count == 5
            assert result.chunk_count == 15  # 5 videos * 3 chunks each
            assert result.error_count == 0
            assert len(result.video_ids) == 5

    @pytest.mark.asyncio
    async def test_index_channel_skips_indexed(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
        sample_channel_videos: list[dict[str, Any]],
        sample_transcript: dict[str, Any],
    ) -> None:
        """Test skips already indexed videos."""
        # Pre-index 2 videos
        mock_vector_store._storage["existing1"] = {
            "content": "Chunk",
            "metadata": {"video_id": "video_0"},
        }
        mock_vector_store._storage["existing2"] = {
            "content": "Chunk",
            "metadata": {"video_id": "video_1"},
        }

        with (
            patch(
                "app.tools.youtube.search.get_channel_videos",
                new_callable=AsyncMock,
            ) as mock_channel,
            patch(
                "app.tools.youtube.transcripts.get_full_transcript",
                new_callable=AsyncMock,
            ) as mock_transcript,
            patch(
                "app.tools.youtube.metadata.get_video_details",
                new_callable=AsyncMock,
            ) as mock_details,
        ):
            mock_channel.return_value = sample_channel_videos
            mock_transcript.return_value = sample_transcript
            mock_details.return_value = {"video_id": "test", "title": "Test"}

            result = await indexer.index_channel(
                "UCtest123456789012345", force_reindex=False
            )

            assert result.indexed_count == 3  # Only 3 new videos
            assert result.skipped_count == 2  # 2 already indexed
            assert result.error_count == 0

    @pytest.mark.asyncio
    async def test_index_channel_force_reindex(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
        sample_channel_videos: list[dict[str, Any]],
        sample_transcript: dict[str, Any],
    ) -> None:
        """Test force_reindex re-indexes all videos."""
        # Pre-index 2 videos
        mock_vector_store._storage["existing1"] = {
            "content": "Chunk",
            "metadata": {"video_id": "video_0"},
        }
        mock_vector_store._storage["existing2"] = {
            "content": "Chunk",
            "metadata": {"video_id": "video_1"},
        }

        with (
            patch(
                "app.tools.youtube.search.get_channel_videos",
                new_callable=AsyncMock,
            ) as mock_channel,
            patch(
                "app.tools.youtube.transcripts.get_full_transcript",
                new_callable=AsyncMock,
            ) as mock_transcript,
            patch(
                "app.tools.youtube.metadata.get_video_details",
                new_callable=AsyncMock,
            ) as mock_details,
        ):
            mock_channel.return_value = sample_channel_videos
            mock_transcript.return_value = sample_transcript
            mock_details.return_value = {"video_id": "test", "title": "Test"}

            result = await indexer.index_channel(
                "UCtest123456789012345", force_reindex=True
            )

            assert result.indexed_count == 5  # All 5 re-indexed
            assert result.skipped_count == 0  # None skipped

    @pytest.mark.asyncio
    async def test_index_channel_handles_errors(
        self,
        indexer: TranscriptIndexer,
        sample_channel_videos: list[dict[str, Any]],
        sample_transcript: dict[str, Any],
    ) -> None:
        """Test continues on individual video failures."""
        with (
            patch(
                "app.tools.youtube.search.get_channel_videos",
                new_callable=AsyncMock,
            ) as mock_channel,
            patch(
                "app.tools.youtube.transcripts.get_full_transcript",
                new_callable=AsyncMock,
            ) as mock_transcript,
            patch(
                "app.tools.youtube.metadata.get_video_details",
                new_callable=AsyncMock,
            ) as mock_details,
        ):
            mock_channel.return_value = sample_channel_videos

            # Make transcript fail for 2 videos
            def transcript_side_effect(video_id: str, language: str = "en"):
                if video_id in ["video_1", "video_3"]:
                    raise Exception("No transcript")
                return sample_transcript

            mock_transcript.side_effect = transcript_side_effect
            mock_details.return_value = {"video_id": "test", "title": "Test"}

            result = await indexer.index_channel("UCtest123456789012345")

            assert result.indexed_count == 3  # 3 succeeded
            assert result.error_count == 2  # 2 failed
            assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_index_channel_respects_max_videos(
        self,
        indexer: TranscriptIndexer,
        sample_channel_videos: list[dict[str, Any]],
        sample_transcript: dict[str, Any],
    ) -> None:
        """Test respects max_videos parameter."""
        with (
            patch(
                "app.tools.youtube.search.get_channel_videos",
                new_callable=AsyncMock,
            ) as mock_channel,
            patch(
                "app.tools.youtube.transcripts.get_full_transcript",
                new_callable=AsyncMock,
            ) as mock_transcript,
            patch(
                "app.tools.youtube.metadata.get_video_details",
                new_callable=AsyncMock,
            ) as mock_details,
        ):
            mock_channel.return_value = sample_channel_videos[:2]  # Return only 2
            mock_transcript.return_value = sample_transcript
            mock_details.return_value = {"video_id": "test", "title": "Test"}

            result = await indexer.index_channel("UCtest123456789012345", max_videos=2)

            # Verify get_channel_videos was called with max_results=2
            mock_channel.assert_called_once_with("UCtest123456789012345", max_results=2)
            assert result.indexed_count == 2

    @pytest.mark.asyncio
    async def test_index_channel_progress_callback(
        self,
        indexer: TranscriptIndexer,
        sample_channel_videos: list[dict[str, Any]],
        sample_transcript: dict[str, Any],
    ) -> None:
        """Test progress callback is called correctly."""
        progress_calls: list[dict[str, Any]] = []

        def on_progress(
            video_id: str,
            index: int,
            total: int,
            status: str,
            chunks: int = 0,
            error: str | None = None,
        ) -> None:
            progress_calls.append(
                {
                    "video_id": video_id,
                    "index": index,
                    "total": total,
                    "status": status,
                    "chunks": chunks,
                    "error": error,
                }
            )

        with (
            patch(
                "app.tools.youtube.search.get_channel_videos",
                new_callable=AsyncMock,
            ) as mock_channel,
            patch(
                "app.tools.youtube.transcripts.get_full_transcript",
                new_callable=AsyncMock,
            ) as mock_transcript,
            patch(
                "app.tools.youtube.metadata.get_video_details",
                new_callable=AsyncMock,
            ) as mock_details,
        ):
            # Use only 2 videos for simplicity
            mock_channel.return_value = sample_channel_videos[:2]
            mock_transcript.return_value = sample_transcript
            mock_details.return_value = {"video_id": "test", "title": "Test"}

            await indexer.index_channel(
                "UCtest123456789012345",
                max_videos=2,
                on_progress=on_progress,
            )

        # Should have 4 calls: started+completed for each of 2 videos
        assert len(progress_calls) == 4

        # Check first video calls
        assert progress_calls[0]["status"] == "started"
        assert progress_calls[0]["index"] == 0
        assert progress_calls[0]["total"] == 2

        assert progress_calls[1]["status"] == "completed"
        assert progress_calls[1]["chunks"] == 3

    @pytest.mark.asyncio
    async def test_index_channel_fetch_error(
        self,
        indexer: TranscriptIndexer,
    ) -> None:
        """Test handles channel fetch error gracefully."""
        with patch(
            "app.tools.youtube.search.get_channel_videos",
            new_callable=AsyncMock,
        ) as mock_channel:
            mock_channel.side_effect = Exception("Channel not found")

            result = await indexer.index_channel("UCinvalid")

            assert result.indexed_count == 0
            assert result.error_count == 1
            assert "Channel not found" in result.errors[0]


# =============================================================================
# get_indexed_video_ids Tests
# =============================================================================


class TestGetIndexedVideoIds:
    """Tests for the get_indexed_video_ids method."""

    @pytest.mark.asyncio
    async def test_returns_unique_video_ids(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test returns unique video IDs from the store."""
        # Add chunks for multiple videos (some with multiple chunks)
        mock_vector_store._storage["doc1"] = {
            "content": "Chunk 1",
            "metadata": {"video_id": "video_a", "channel_id": "UCtest"},
        }
        mock_vector_store._storage["doc2"] = {
            "content": "Chunk 2",
            "metadata": {"video_id": "video_a", "channel_id": "UCtest"},
        }
        mock_vector_store._storage["doc3"] = {
            "content": "Chunk 1",
            "metadata": {"video_id": "video_b", "channel_id": "UCtest"},
        }
        mock_vector_store._storage["doc4"] = {
            "content": "Chunk 1",
            "metadata": {"video_id": "video_c", "channel_id": "UCother"},
        }

        video_ids = await indexer.get_indexed_video_ids()

        assert len(video_ids) == 3
        assert set(video_ids) == {"video_a", "video_b", "video_c"}

    @pytest.mark.asyncio
    async def test_filters_by_channel(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test filters by channel_id when provided."""
        mock_vector_store._storage["doc1"] = {
            "content": "Chunk",
            "metadata": {"video_id": "video_a", "channel_id": "UCtest"},
        }
        mock_vector_store._storage["doc2"] = {
            "content": "Chunk",
            "metadata": {"video_id": "video_b", "channel_id": "UCother"},
        }

        video_ids = await indexer.get_indexed_video_ids(channel_id="UCtest")

        assert video_ids == ["video_a"]

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_store(
        self,
        indexer: TranscriptIndexer,
    ) -> None:
        """Test returns empty list when store is empty."""
        video_ids = await indexer.get_indexed_video_ids()

        assert video_ids == []


# =============================================================================
# get_chunk_count Tests
# =============================================================================


class TestGetChunkCount:
    """Tests for the get_chunk_count method."""

    def test_returns_total_count(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test returns total chunk count."""
        mock_vector_store._storage["doc1"] = {"content": "Chunk 1", "metadata": {}}
        mock_vector_store._storage["doc2"] = {"content": "Chunk 2", "metadata": {}}
        mock_vector_store._storage["doc3"] = {"content": "Chunk 3", "metadata": {}}

        count = indexer.get_chunk_count()

        assert count == 3

    def test_returns_count_for_video(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test returns chunk count for specific video."""
        mock_vector_store._storage["doc1"] = {
            "content": "Chunk",
            "metadata": {"video_id": "video_a"},
        }
        mock_vector_store._storage["doc2"] = {
            "content": "Chunk",
            "metadata": {"video_id": "video_a"},
        }
        mock_vector_store._storage["doc3"] = {
            "content": "Chunk",
            "metadata": {"video_id": "video_b"},
        }

        count_a = indexer.get_chunk_count(video_id="video_a")
        count_b = indexer.get_chunk_count(video_id="video_b")

        assert count_a == 2
        assert count_b == 1

    def test_returns_zero_for_empty_store(
        self,
        indexer: TranscriptIndexer,
    ) -> None:
        """Test returns 0 for empty store."""
        count = indexer.get_chunk_count()

        assert count == 0


# =============================================================================
# Integration-Style Tests (Still Mocked but Full Flow)
# =============================================================================


class TestIndexerIntegration:
    """Integration-style tests for the complete indexing workflow."""

    @pytest.mark.asyncio
    async def test_full_indexing_workflow(
        self,
        indexer: TranscriptIndexer,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test complete workflow: index, check, delete."""
        video_id = "integration_test_video"
        transcript = {
            "video_id": video_id,
            "language": "en",
            "transcript": [
                {"text": "Test content", "start": 0.0, "duration": 5.0},
            ],
            "full_text": "Test content",
        }
        video_details = {
            "video_id": video_id,
            "title": "Integration Test",
            "channel_title": "Test Channel",
        }

        with (
            patch(
                "app.tools.youtube.transcripts.get_full_transcript",
                new_callable=AsyncMock,
            ) as mock_transcript,
            patch(
                "app.tools.youtube.metadata.get_video_details",
                new_callable=AsyncMock,
            ) as mock_details,
        ):
            mock_transcript.return_value = transcript
            mock_details.return_value = video_details

            # 1. Initially not indexed
            assert indexer.is_video_indexed(video_id) is False
            assert indexer.get_chunk_count() == 0

            # 2. Index the video
            result = await indexer.index_video(video_id)
            assert result.indexed_count == 1
            assert result.chunk_count == 3

            # 3. Now indexed
            assert indexer.is_video_indexed(video_id) is True
            assert indexer.get_chunk_count() == 3
            assert indexer.get_chunk_count(video_id) == 3

            # 4. Get indexed video IDs
            video_ids = await indexer.get_indexed_video_ids()
            assert video_id in video_ids

            # 5. Skip if try to re-index
            result2 = await indexer.index_video(video_id, force_reindex=False)
            assert result2.skipped_count == 1
            assert result2.indexed_count == 0

            # 6. Delete the video
            deleted = await indexer.delete_video(video_id)
            assert deleted == 3

            # 7. No longer indexed
            assert indexer.is_video_indexed(video_id) is False
            assert indexer.get_chunk_count() == 0
