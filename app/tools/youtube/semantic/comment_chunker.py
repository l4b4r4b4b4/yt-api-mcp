"""Comment chunker for semantic search over YouTube comments.

Provides a simple chunker that converts YouTube comments into LangChain
Documents with appropriate metadata for semantic search. Unlike transcripts,
comments are typically short and self-contained, so each comment becomes
its own Document without further splitting.

Example:
    >>> from app.tools.youtube.semantic.comment_chunker import CommentChunker
    >>> chunker = CommentChunker()
    >>> documents = chunker.chunk_comments(
    ...     comments=[
    ...         {"text": "Great video!", "author": "User1", "like_count": 10},
    ...         {"text": "Very helpful", "author": "User2", "like_count": 5},
    ...     ],
    ...     video_metadata={"video_id": "abc123", "video_title": "Tutorial"},
    ... )
    >>> len(documents)
    2
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


class CommentChunker:
    """Converts YouTube comments into Documents for semantic search.

    Unlike TranscriptChunker which groups transcript entries into token-based
    chunks, CommentChunker creates one Document per comment since comments
    are typically short and self-contained.

    Each Document includes:
    - page_content: The comment text
    - metadata: Video info, author, engagement metrics, content_type

    Attributes:
        None (stateless chunker)
    """

    def chunk_comments(
        self,
        comments: list[dict[str, Any]],
        video_metadata: dict[str, str],
    ) -> list[Document]:
        """Convert a list of comments into Documents with metadata.

        Args:
            comments: List of comment dictionaries from get_video_comments,
                each containing:
                - text: Comment text content
                - author: Comment author display name
                - like_count: Number of likes (int)
                - reply_count: Number of replies (int)
                - published_at: ISO 8601 timestamp when posted
            video_metadata: Video information to include in each Document:
                - video_id: YouTube video ID (required)
                - video_title: Video title
                - channel_id: YouTube channel ID
                - channel_title: Channel name
                - video_url: Direct link to video
                - published_at: Video publish date (optional, distinct from comment date)

        Returns:
            List of LangChain Document objects, one per comment.
            Each Document has:
            - page_content: The comment text
            - metadata: Combined video info and comment-specific data

        Example:
            >>> chunker = CommentChunker()
            >>> comments = [
            ...     {
            ...         "text": "This helped me understand Nix flakes!",
            ...         "author": "NixUser42",
            ...         "like_count": 15,
            ...         "reply_count": 2,
            ...         "published_at": "2024-01-15T10:30:00Z",
            ...     }
            ... ]
            >>> video_metadata = {
            ...     "video_id": "dQw4w9WgXcQ",
            ...     "video_title": "Nix Flakes Tutorial",
            ...     "channel_id": "UC123",
            ...     "channel_title": "Vimjoyer",
            ... }
            >>> docs = chunker.chunk_comments(comments, video_metadata)
            >>> docs[0].metadata["content_type"]
            'comment'
            >>> docs[0].metadata["author"]
            'NixUser42'
        """
        if not comments:
            return []

        documents: list[Document] = []
        video_id = video_metadata.get("video_id", "")

        for idx, comment in enumerate(comments):
            # Skip comments with empty text
            text = str(comment.get("text", "")).strip()
            if not text:
                continue

            # Build metadata with content_type discriminator
            metadata: dict[str, Any] = {
                # Content type discriminator (for filtering in search)
                "content_type": "comment",
                # Video identifiers
                "video_id": video_id,
                "channel_id": video_metadata.get("channel_id", ""),
                # Display info
                "video_title": video_metadata.get("video_title", ""),
                "channel_title": video_metadata.get("channel_title", ""),
                "video_url": video_metadata.get(
                    "video_url", f"https://www.youtube.com/watch?v={video_id}"
                ),
                # Comment-specific metadata
                "author": comment.get("author", ""),
                "like_count": int(comment.get("like_count", 0)),
                "reply_count": int(comment.get("reply_count", 0)),
                "published_at": comment.get("published_at", ""),
                # Position in comment list
                "chunk_index": idx,
            }

            document = Document(page_content=text, metadata=metadata)
            documents.append(document)

        return documents


__all__ = ["CommentChunker"]
