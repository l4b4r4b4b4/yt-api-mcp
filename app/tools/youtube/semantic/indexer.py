"""Batch indexing logic for semantic transcript search.

Provides functionality to index YouTube video transcripts into the vector store,
with support for:
- Batch processing of multiple videos
- Progress tracking and error handling
- Incremental indexing (skip already indexed videos)
- Force re-indexing when needed

The indexer coordinates between transcript fetching, chunking, embedding,
and vector store insertion.

Example:
    >>> from app.tools.youtube.semantic.indexer import TranscriptIndexer
    >>> from app.tools.youtube.semantic.config import get_semantic_config
    >>> from app.tools.youtube.semantic.store import get_vector_store
    >>> from app.tools.youtube.semantic.chunker import TranscriptChunker
    >>>
    >>> config = get_semantic_config()
    >>> indexer = TranscriptIndexer(
    ...     vector_store=get_vector_store(),
    ...     chunker=TranscriptChunker(config),
    ... )
    >>> result = await indexer.index_channel("UCuAXFkgsw1L7xaCfnd5JJOw")
    >>> print(f"Indexed {result.indexed_count} videos")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_chroma import Chroma

    from app.tools.youtube.semantic.chunker import TranscriptChunker
    from app.tools.youtube.semantic.comment_chunker import CommentChunker

logger = logging.getLogger(__name__)


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""

    def __call__(
        self,
        video_id: str,
        index: int,
        total: int,
        status: str,
        chunks: int = 0,
        error: str | None = None,
    ) -> None:
        """Called when progress is made on indexing.

        Args:
            video_id: The video being processed.
            index: Current video index (0-based).
            total: Total number of videos to process.
            status: Status string - "started", "completed", "skipped", "error".
            chunks: Number of chunks created (for "completed" status).
            error: Error message (for "error" status).
        """
        ...


@dataclass
class IndexingResult:
    """Result of a batch indexing operation.

    Attributes:
        indexed_count: Number of videos successfully indexed.
        chunk_count: Total number of chunks created across all videos.
        skipped_count: Number of videos skipped (already indexed or no transcript).
        error_count: Number of videos that failed to index.
        errors: List of error messages with video IDs.
        video_ids: List of successfully indexed video IDs.
    """

    indexed_count: int = 0
    chunk_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    errors: list[str] = field(default_factory=list)
    video_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, int | list[str]]:
        """Convert to dictionary for API responses.

        Returns:
            Dictionary representation of indexing results.
        """
        return {
            "indexed_count": self.indexed_count,
            "chunk_count": self.chunk_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "errors": self.errors,
            "video_ids": self.video_ids,
        }

    def merge(self, other: IndexingResult) -> None:
        """Merge another IndexingResult into this one.

        Args:
            other: Another IndexingResult to merge in.
        """
        self.indexed_count += other.indexed_count
        self.chunk_count += other.chunk_count
        self.skipped_count += other.skipped_count
        self.error_count += other.error_count
        self.errors.extend(other.errors)
        self.video_ids.extend(other.video_ids)


class TranscriptIndexer:
    """Indexes YouTube video transcripts and comments into a vector store.

    Coordinates content fetching, chunking, and vector store insertion
    for batch indexing operations. Supports both transcripts and comments
    with distinct content_type metadata for filtering.

    Attributes:
        vector_store: ChromaDB vector store for storing embeddings.
        chunker: Transcript chunker for creating chunks with metadata.
        comment_chunker: Comment chunker for creating comment documents.
    """

    def __init__(
        self,
        vector_store: Chroma,
        chunker: TranscriptChunker,
        comment_chunker: CommentChunker | None = None,
    ) -> None:
        """Initialize the transcript indexer.

        Args:
            vector_store: ChromaDB vector store instance.
            chunker: TranscriptChunker instance for chunking transcripts.
            comment_chunker: Optional CommentChunker instance. If None, creates one.
        """
        self.vector_store = vector_store
        self.chunker = chunker

        # Lazy import to avoid circular imports
        if comment_chunker is None:
            from app.tools.youtube.semantic.comment_chunker import CommentChunker

            comment_chunker = CommentChunker()
        self.comment_chunker = comment_chunker

    async def index_channel(
        self,
        channel_id: str,
        max_videos: int = 50,
        language: str = "en",
        force_reindex: bool = False,
        on_progress: ProgressCallback | Callable[..., None] | None = None,
    ) -> IndexingResult:
        """Index all video transcripts from a YouTube channel.

        Fetches videos from the channel, retrieves transcripts, chunks them,
        and adds them to the vector store with rich metadata.

        Args:
            channel_id: YouTube channel ID to index.
            max_videos: Maximum number of videos to index (default: 50).
            language: Preferred transcript language code (default: "en").
            force_reindex: If True, re-index videos even if already indexed.
            on_progress: Optional callback for progress updates.

        Returns:
            IndexingResult with counts and any errors encountered.

        Example:
            >>> result = await indexer.index_channel("UCuAXFkgsw1L7xaCfnd5JJOw")
            >>> print(f"Indexed {result.indexed_count} videos, {result.chunk_count} chunks")
        """
        # Import here to avoid circular imports
        from app.tools.youtube.search import get_channel_videos

        logger.info(
            f"Starting channel indexing: channel_id={channel_id}, "
            f"max_videos={max_videos}, language={language}, force_reindex={force_reindex}"
        )

        result = IndexingResult()

        try:
            # Fetch video list from channel
            videos = await get_channel_videos(channel_id, max_results=max_videos)
            logger.info(f"Found {len(videos)} videos in channel {channel_id}")
        except Exception as e:
            error_msg = f"Failed to fetch videos from channel {channel_id}: {e!s}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.error_count = 1
            return result

        total_videos = len(videos)

        # Process each video
        for idx, video_info in enumerate(videos):
            video_id = video_info["video_id"]

            # Notify progress callback
            if on_progress:
                on_progress(
                    video_id=video_id,
                    index=idx,
                    total=total_videos,
                    status="started",
                )

            # Index the video
            video_result = await self.index_video(
                video_id=video_id,
                language=language,
                force_reindex=force_reindex,
                channel_id=channel_id,
                video_info=video_info,
            )

            # Merge results
            result.merge(video_result)

            # Notify progress callback
            if on_progress:
                if video_result.error_count > 0:
                    on_progress(
                        video_id=video_id,
                        index=idx,
                        total=total_videos,
                        status="error",
                        error=video_result.errors[0] if video_result.errors else None,
                    )
                elif video_result.skipped_count > 0:
                    on_progress(
                        video_id=video_id,
                        index=idx,
                        total=total_videos,
                        status="skipped",
                    )
                else:
                    on_progress(
                        video_id=video_id,
                        index=idx,
                        total=total_videos,
                        status="completed",
                        chunks=video_result.chunk_count,
                    )

        logger.info(
            f"Channel indexing complete: indexed={result.indexed_count}, "
            f"chunks={result.chunk_count}, skipped={result.skipped_count}, "
            f"errors={result.error_count}"
        )

        return result

    async def index_video(
        self,
        video_id: str,
        language: str = "en",
        force_reindex: bool = False,
        channel_id: str | None = None,
        video_info: dict[str, Any] | None = None,
    ) -> IndexingResult:
        """Index a single video's transcript.

        Args:
            video_id: YouTube video ID to index.
            language: Preferred transcript language code (default: "en").
            force_reindex: If True, re-index even if already indexed.
            channel_id: Optional channel ID (avoids extra API call if known).
            video_info: Optional video metadata (avoids extra API call if known).

        Returns:
            IndexingResult for the single video.

        Example:
            >>> result = await indexer.index_video("dQw4w9WgXcQ")
            >>> print(f"Created {result.chunk_count} chunks")
        """
        # Import here to avoid circular imports
        from app.tools.youtube.metadata import get_video_details
        from app.tools.youtube.transcripts import get_full_transcript

        result = IndexingResult()

        logger.debug(f"Indexing video: video_id={video_id}, language={language}")

        # Check if already indexed (unless force_reindex)
        if not force_reindex and self.is_video_indexed(video_id):
            logger.debug(f"Video {video_id} already indexed, skipping")
            result.skipped_count = 1
            return result

        # If force_reindex, delete existing chunks first
        if force_reindex:
            deleted = await self.delete_video(video_id)
            if deleted > 0:
                logger.debug(f"Deleted {deleted} existing chunks for video {video_id}")

        # Get video metadata if not provided
        if video_info is None:
            try:
                video_info = await get_video_details(video_id)
            except Exception as e:
                error_msg = f"[{video_id}] Failed to get video details: {e!s}"
                logger.warning(error_msg)
                result.errors.append(error_msg)
                result.error_count = 1
                return result

        # Get transcript
        try:
            transcript_data = await get_full_transcript(video_id, language=language)
        except Exception as e:
            error_msg = f"[{video_id}] No transcript available: {e!s}"
            logger.warning(error_msg)
            result.errors.append(error_msg)
            result.error_count = 1
            return result

        # Build video metadata for chunks
        video_metadata = {
            "video_id": video_id,
            "video_title": video_info.get("title", ""),
            "channel_id": channel_id or video_info.get("channel_id", ""),
            "channel_title": video_info.get("channel_title", ""),
            "video_url": video_info.get(
                "url", f"https://www.youtube.com/watch?v={video_id}"
            ),
            "published_at": video_info.get("published_at", ""),
            "language": transcript_data.get("language", language),
        }

        # Chunk the transcript
        transcript_entries = transcript_data.get("transcript", [])
        if not transcript_entries:
            error_msg = f"[{video_id}] Transcript is empty"
            logger.warning(error_msg)
            result.errors.append(error_msg)
            result.error_count = 1
            return result

        try:
            documents = self.chunker.chunk_transcript(
                transcript_entries=transcript_entries,
                video_metadata=video_metadata,
            )
        except Exception as e:
            error_msg = f"[{video_id}] Failed to chunk transcript: {e!s}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.error_count = 1
            return result

        if not documents:
            error_msg = f"[{video_id}] No chunks created from transcript"
            logger.warning(error_msg)
            result.errors.append(error_msg)
            result.error_count = 1
            return result

        # Add to vector store
        try:
            self.vector_store.add_documents(documents)
            logger.debug(f"Added {len(documents)} chunks for video {video_id}")
        except Exception as e:
            error_msg = f"[{video_id}] Failed to add to vector store: {e!s}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.error_count = 1
            return result

        # Success
        result.indexed_count = 1
        result.chunk_count = len(documents)
        result.video_ids.append(video_id)

        logger.info(f"Indexed video {video_id}: {len(documents)} chunks")

        return result

    def is_video_indexed(self, video_id: str) -> bool:
        """Check if a video is already indexed in the vector store.

        Args:
            video_id: YouTube video ID to check.

        Returns:
            True if the video has chunks in the vector store.

        Example:
            >>> if not indexer.is_video_indexed("dQw4w9WgXcQ"):
            ...     await indexer.index_video("dQw4w9WgXcQ")
        """
        try:
            # Access the underlying ChromaDB collection
            collection = self.vector_store._collection

            # Query for any documents with this video_id
            # Using get() with where filter and limit for efficiency
            results = collection.get(
                where={"video_id": video_id},
                limit=1,
                include=[],  # Don't need embeddings or documents, just IDs
            )

            # If any IDs returned, video is indexed
            has_chunks = bool(results.get("ids"))
            logger.debug(f"Video {video_id} indexed: {has_chunks}")
            return has_chunks

        except Exception as e:
            logger.warning(f"Error checking if video {video_id} is indexed: {e!s}")
            # On error, assume not indexed to allow indexing attempt
            return False

    async def delete_video(self, video_id: str) -> int:
        """Delete all chunks for a video from the vector store.

        Args:
            video_id: YouTube video ID to delete.

        Returns:
            Number of chunks deleted.

        Example:
            >>> deleted = await indexer.delete_video("dQw4w9WgXcQ")
            >>> print(f"Deleted {deleted} chunks")
        """
        try:
            # Access the underlying ChromaDB collection
            collection = self.vector_store._collection

            # First, count how many chunks exist
            existing = collection.get(
                where={"video_id": video_id},
                include=[],  # Only need IDs
            )
            chunk_count = len(existing.get("ids", []))

            if chunk_count == 0:
                logger.debug(f"No chunks to delete for video {video_id}")
                return 0

            # Delete all chunks with this video_id
            collection.delete(where={"video_id": video_id})

            logger.info(f"Deleted {chunk_count} chunks for video {video_id}")
            return chunk_count

        except Exception as e:
            logger.error(f"Error deleting video {video_id} chunks: {e!s}")
            return 0

    async def get_indexed_video_ids(self, channel_id: str | None = None) -> list[str]:
        """Get list of video IDs that have been indexed.

        Args:
            channel_id: Optional filter by channel ID.

        Returns:
            List of unique video IDs in the index.

        Example:
            >>> video_ids = await indexer.get_indexed_video_ids("UCxyz...")
            >>> print(f"Found {len(video_ids)} indexed videos")
        """
        try:
            collection = self.vector_store._collection

            # Build where filter
            where_filter = {"channel_id": channel_id} if channel_id else None

            # Get all documents (just metadata)
            results = collection.get(
                where=where_filter,
                include=["metadatas"],
            )

            # Extract unique video IDs
            video_ids = set()
            for metadata in results.get("metadatas", []):
                if metadata and "video_id" in metadata:
                    video_ids.add(metadata["video_id"])

            return sorted(video_ids)

        except Exception as e:
            logger.error(f"Error getting indexed video IDs: {e!s}")
            return []

    def get_chunk_count(self, video_id: str | None = None) -> int:
        """Get count of chunks in the vector store.

        Args:
            video_id: Optional filter by video ID.

        Returns:
            Number of chunks (all or for specific video).

        Example:
            >>> total = indexer.get_chunk_count()
            >>> per_video = indexer.get_chunk_count("dQw4w9WgXcQ")
        """
        try:
            collection = self.vector_store._collection

            if video_id:
                results = collection.get(
                    where={"video_id": video_id},
                    include=[],
                )
                return len(results.get("ids", []))
            else:
                return collection.count()

        except Exception as e:
            logger.error(f"Error getting chunk count: {e!s}")
            return 0

    def is_video_comments_indexed(self, video_id: str) -> bool:
        """Check if a video's comments are already indexed in the vector store.

        Args:
            video_id: YouTube video ID to check.

        Returns:
            True if the video has comment chunks in the vector store.

        Example:
            >>> if not indexer.is_video_comments_indexed("dQw4w9WgXcQ"):
            ...     await indexer.index_video_comments("dQw4w9WgXcQ")
        """
        try:
            collection = self.vector_store._collection

            # Query for documents with this video_id AND content_type="comment"
            results = collection.get(
                where={
                    "$and": [
                        {"video_id": video_id},
                        {"content_type": "comment"},
                    ]
                },
                limit=1,
                include=[],
            )

            has_comments = bool(results.get("ids"))
            logger.debug(f"Video {video_id} comments indexed: {has_comments}")
            return has_comments

        except Exception as e:
            logger.warning(
                f"Error checking if video {video_id} comments are indexed: {e!s}"
            )
            return False

    async def index_video_comments(
        self,
        video_id: str,
        max_comments: int = 100,
        force_reindex: bool = False,
        channel_id: str | None = None,
        video_info: dict[str, Any] | None = None,
    ) -> IndexingResult:
        """Index comments from a single video.

        Args:
            video_id: YouTube video ID to index comments from.
            max_comments: Maximum number of comments to fetch (default: 100).
            force_reindex: If True, re-index even if already indexed.
            channel_id: Optional channel ID (avoids extra API call if known).
            video_info: Optional video metadata (avoids extra API call if known).

        Returns:
            IndexingResult for the comment indexing operation.

        Example:
            >>> result = await indexer.index_video_comments("dQw4w9WgXcQ")
            >>> print(f"Indexed {result.chunk_count} comments")
        """
        from app.tools.youtube.comments import get_video_comments
        from app.tools.youtube.metadata import get_video_details

        result = IndexingResult()

        logger.debug(
            f"Indexing comments: video_id={video_id}, max_comments={max_comments}"
        )

        # Check if already indexed (unless force_reindex)
        if not force_reindex and self.is_video_comments_indexed(video_id):
            logger.debug(f"Video {video_id} comments already indexed, skipping")
            result.skipped_count = 1
            return result

        # If force_reindex, delete existing comment chunks first
        if force_reindex:
            deleted = await self.delete_video_comments(video_id)
            if deleted > 0:
                logger.debug(
                    f"Deleted {deleted} existing comment chunks for video {video_id}"
                )

        # Get video metadata if not provided
        if video_info is None:
            try:
                video_info = await get_video_details(video_id)
            except Exception as e:
                error_msg = f"[{video_id}] Failed to get video details: {e!s}"
                logger.warning(error_msg)
                result.errors.append(error_msg)
                result.error_count = 1
                return result

        # Get comments
        try:
            comments_data = await get_video_comments(video_id, max_results=max_comments)
            comments = comments_data.get("comments", [])
        except Exception as e:
            error_msg = f"[{video_id}] Failed to get comments: {e!s}"
            logger.warning(error_msg)
            result.errors.append(error_msg)
            result.error_count = 1
            return result

        if not comments:
            logger.debug(f"[{video_id}] No comments available")
            result.skipped_count = 1
            return result

        # Build video metadata for chunks
        video_metadata = {
            "video_id": video_id,
            "video_title": video_info.get("title", ""),
            "channel_id": channel_id or video_info.get("channel_id", ""),
            "channel_title": video_info.get("channel_title", ""),
            "video_url": video_info.get(
                "url", f"https://www.youtube.com/watch?v={video_id}"
            ),
        }

        # Create Documents from comments
        try:
            documents = self.comment_chunker.chunk_comments(
                comments=comments,
                video_metadata=video_metadata,
            )
        except Exception as e:
            error_msg = f"[{video_id}] Failed to chunk comments: {e!s}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.error_count = 1
            return result

        if not documents:
            logger.debug(f"[{video_id}] No documents created from comments")
            result.skipped_count = 1
            return result

        # Add to vector store
        try:
            self.vector_store.add_documents(documents)
            logger.debug(f"Added {len(documents)} comment chunks for video {video_id}")
        except Exception as e:
            error_msg = f"[{video_id}] Failed to add comments to vector store: {e!s}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.error_count = 1
            return result

        # Success
        result.indexed_count = 1
        result.chunk_count = len(documents)
        result.video_ids.append(video_id)

        logger.info(f"Indexed video {video_id} comments: {len(documents)} chunks")

        return result

    async def delete_video_comments(self, video_id: str) -> int:
        """Delete all comment chunks for a video from the vector store.

        Args:
            video_id: YouTube video ID whose comments to delete.

        Returns:
            Number of comment chunks deleted.

        Example:
            >>> deleted = await indexer.delete_video_comments("dQw4w9WgXcQ")
            >>> print(f"Deleted {deleted} comment chunks")
        """
        try:
            collection = self.vector_store._collection

            # Count existing comment chunks
            existing = collection.get(
                where={
                    "$and": [
                        {"video_id": video_id},
                        {"content_type": "comment"},
                    ]
                },
                include=[],
            )
            chunk_count = len(existing.get("ids", []))

            if chunk_count == 0:
                logger.debug(f"No comment chunks to delete for video {video_id}")
                return 0

            # Delete all comment chunks with this video_id
            collection.delete(
                where={
                    "$and": [
                        {"video_id": video_id},
                        {"content_type": "comment"},
                    ]
                }
            )

            logger.info(f"Deleted {chunk_count} comment chunks for video {video_id}")
            return chunk_count

        except Exception as e:
            logger.error(f"Error deleting video {video_id} comment chunks: {e!s}")
            return 0

    async def get_indexed_video_ids_by_content_type(
        self,
        content_type: str | None = None,
        channel_id: str | None = None,
    ) -> list[str]:
        """Get list of video IDs indexed for a specific content type.

        Args:
            content_type: Filter by content type ("transcript" or "comment").
                If None, returns all indexed videos regardless of type.
            channel_id: Optional filter by channel ID.

        Returns:
            List of unique video IDs in the index.

        Example:
            >>> comment_videos = await indexer.get_indexed_video_ids_by_content_type(
            ...     content_type="comment"
            ... )
        """
        try:
            collection = self.vector_store._collection

            # Build where filter
            filters = []
            if content_type:
                filters.append({"content_type": content_type})
            if channel_id:
                filters.append({"channel_id": channel_id})

            if len(filters) > 1:
                where_filter = {"$and": filters}
            elif len(filters) == 1:
                where_filter = filters[0]
            else:
                where_filter = None

            # Get all documents (just metadata)
            results = collection.get(
                where=where_filter,
                include=["metadatas"],
            )

            # Extract unique video IDs
            video_ids = set()
            for metadata in results.get("metadatas", []):
                if metadata and "video_id" in metadata:
                    video_ids.add(metadata["video_id"])

            return sorted(video_ids)

        except Exception as e:
            logger.error(f"Error getting indexed video IDs: {e!s}")
            return []


__all__ = [
    "IndexingResult",
    "ProgressCallback",
    "TranscriptIndexer",
]
