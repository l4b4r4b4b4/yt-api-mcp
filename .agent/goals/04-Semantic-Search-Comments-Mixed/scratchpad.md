# Goal 04: Semantic Search for Comments & Mixed Content

**Status:** 🟢 Complete
**Priority:** High
**Created:** 2025-01-14
**Updated:** 2025-01-15
**Parent:** [Goals Index](../scratchpad.md)
**Depends On:** Goal 02 (Semantic Transcript Search) - 🟢 Complete

---

## Objective

Extend the semantic search system (v0.0.2) to support:
1. **Comment Search** - Semantic search over YouTube video comments
2. **Mixed Search** - Unified search across transcripts AND comments
3. **Utility Tools** - Index management (list, delete indexed content)

## Use Cases

1. "Find comments discussing Nix flakes issues on Vimjoyer videos"
2. "Search all content (transcripts + comments) about garbage collection"
3. "What videos have I indexed?" / "Remove this video from the index"

---

## Architecture

### Content Type Discrimination

Add `content_type` metadata field to all indexed documents:
- `"transcript"` - Transcript chunks (existing)
- `"comment"` - Comment chunks (new)

This enables filtering in ChromaDB:
```python
# Search only transcripts
filter={"content_type": "transcript"}

# Search only comments
filter={"content_type": "comment"}

# Search all (no filter or explicit $in)
filter={"content_type": {"$in": ["transcript", "comment"]}}
```

### Comment Chunking Strategy

Unlike transcripts, comments are typically short and self-contained.

**Approach:** Each comment becomes its own Document (no grouping)
- Comments are usually < 500 characters
- Preserves author attribution per document
- Engagement metrics (like_count) as metadata
- Simple and efficient

**Alternative considered (rejected):** Group 5-10 comments per chunk
- Loses author attribution clarity
- Aggregating like_counts is meaningless
- Comments lack the temporal continuity of transcripts

### Metadata Schema

**Transcript Metadata (existing + content_type):**
```python
{
    "content_type": "transcript",  # NEW
    "video_id": str,
    "channel_id": str,
    "channel_title": str,
    "video_title": str,
    "video_url": str,
    "published_at": str,
    "start_time": float,
    "end_time": float,
    "timestamp_url": str,
    "language": str,
    "chunk_index": int,
}
```

**Comment Metadata (new):**
```python
{
    "content_type": "comment",     # Discriminator
    "video_id": str,
    "channel_id": str,
    "channel_title": str,
    "video_title": str,
    "video_url": str,
    "published_at": str,           # When comment was posted
    "author": str,                 # Comment author display name
    "like_count": int,             # Engagement metric
    "reply_count": int,            # Number of replies
    "chunk_index": int,            # Position in comment list
}
```

### New/Modified Files

```
app/tools/youtube/semantic/
├── chunker.py              # MODIFY: Add content_type to TranscriptChunker
├── comment_chunker.py      # NEW: CommentChunker class
├── indexer.py              # MODIFY: Add comment indexing methods
└── tools.py                # MODIFY: Add new search/utility tools
```

---

## MCP Tools

### New Tools

#### 1. `semantic_search_comments`

Search indexed comments using natural language with auto-indexing.

**Parameters:**
- `query` (required): Natural language search query
- `channel_ids` (optional): Filter by channels
- `video_ids` (optional): Filter by specific videos
- `k` (optional): Number of results (default: 10)
- `max_comments_per_video` (optional): Max comments to index per video (default: 100)
- `min_score` (optional): Minimum similarity score

**Returns:**
- `results`: List with video_id, video_title, text, author, like_count, score
- `indexing_stats`: Auto-indexing statistics
- `scope`: Search scope description

#### 2. `semantic_search_all`

Search across all content types (transcripts + comments).

**Parameters:**
- `query` (required): Natural language search query
- `content_types` (optional): List of types to search (default: all)
- `channel_ids` (optional): Filter by channels
- `video_ids` (optional): Filter by specific videos
- `k` (optional): Number of results (default: 10)
- `language` (optional): Transcript language (for transcript results)
- `min_score` (optional): Minimum similarity score

**Returns:**
- `results`: Unified list with content_type in each result
- `indexing_stats`: What was indexed during search
- `scope`: Search scope description

### Utility Tools (from Task-07)

#### 3. `get_indexed_videos`

List videos that have been indexed for semantic search.

**Parameters:**
- `channel_id` (optional): Filter by channel
- `content_type` (optional): Filter by content type

**Returns:**
- `videos`: List of indexed video IDs with metadata
- `total_count`: Total indexed videos
- `by_content_type`: Breakdown by content type

#### 4. `delete_indexed_video`

Remove a video from the semantic search index.

**Parameters:**
- `video_id` (required): Video to remove
- `content_type` (optional): Remove only specific type (default: all)

**Returns:**
- `video_id`: Deleted video ID
- `chunks_deleted`: Number of chunks removed
- `by_content_type`: Breakdown of deleted chunks

---

## Task Breakdown

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| Task-01 | Add content_type to transcript metadata | 🟢 | Added to TranscriptChunker |
| Task-02 | Create CommentChunker class | 🟢 | 18 tests passing |
| Task-03 | Extend indexer for comment indexing | 🟢 | 12 new tests, 42 total |
| Task-04 | Implement semantic_search_comments tool | 🟢 | 12 tests, 386 total passing |
| Task-05 | Implement semantic_search_all tool | 🟢 | 11 tests, 397 total passing |
| Task-06 | Implement utility tools | 🟢 | 9 tests, 406 total passing |
| Task-07 | Register tools in server.py | 🟢 | 5 new tools registered |

---

## Success Criteria

- [x] `content_type` field added to all indexed documents
- [x] Comments can be indexed and searched semantically
- [x] Mixed search returns results from both content types
- [x] Results clearly indicate content_type for each match
- [x] Utility tools work for listing and deleting indexed content
- [x] Auto-indexing works for comments (like transcripts)
- [x] Tests cover new functionality (≥73% coverage maintained)
- [x] Tools registered in MCP server
- [x] **User validated in real environment**


---

## API Quota Considerations

**Comment fetching:**
- `get_video_comments`: 1 quota unit per call
- Max 100 comments per call
- For 50 videos × 100 comments = 50 quota units

**Total for full channel index (transcripts + comments):**
- Video list: ~1-2 calls = ~100 units
- Video details: 50 × 1 = 50 units
- Transcripts: 0 (third-party API)
- Comments: 50 × 1 = 50 units
- **Total: ~200 units** (well within 10,000 daily limit)

---

## Implementation Notes

### CommentChunker Design

```python
class CommentChunker:
    """Creates Documents from YouTube comments.

    Unlike TranscriptChunker, each comment becomes its own Document
    since comments are short and self-contained.
    """

    def chunk_comments(
        self,
        comments: list[dict[str, Any]],
        video_metadata: dict[str, str],
    ) -> list[Document]:
        """Convert comments to Documents with metadata.

        Args:
            comments: List from get_video_comments
            video_metadata: Video info to include

        Returns:
            List of Documents, one per comment
        """
        documents = []
        for idx, comment in enumerate(comments):
            doc = Document(
                page_content=comment["text"],
                metadata={
                    "content_type": "comment",
                    "video_id": video_metadata["video_id"],
                    "channel_id": video_metadata.get("channel_id", ""),
                    "channel_title": video_metadata.get("channel_title", ""),
                    "video_title": video_metadata.get("video_title", ""),
                    "video_url": video_metadata.get("video_url", ""),
                    "published_at": comment.get("published_at", ""),
                    "author": comment.get("author", ""),
                    "like_count": comment.get("like_count", 0),
                    "reply_count": comment.get("reply_count", 0),
                    "chunk_index": idx,
                },
            )
            documents.append(doc)
        return documents
```

### Indexer Extension

```python
class TranscriptIndexer:  # Rename to ContentIndexer?

    async def index_video_comments(
        self,
        video_id: str,
        max_comments: int = 100,
        force_reindex: bool = False,
    ) -> IndexingResult:
        """Index comments from a video."""
        ...

    def is_video_comments_indexed(self, video_id: str) -> bool:
        """Check if video comments are indexed."""
        ...

    async def get_indexed_video_ids(
        self,
        channel_id: str | None = None,
        content_type: str | None = None,
    ) -> list[str]:
        """Get list of indexed video IDs with optional filters."""
        ...
```

---

## Version Target

**Next Release:** v0.0.3
- Semantic comment search
- Mixed content search
- Index management utilities

---

## Session Log

### 2025-01-15: Goal Validated - Complete 🎉
- **User Validation:** All 5 new MCP tools tested and working
  - `get_indexed_videos`: ✅ Returns 16 indexed videos, filtering by content_type works
  - `semantic_search_comments`: ✅ Auto-indexed comments, returns relevant results with author/like_count
  - `semantic_search_all`: ✅ Unified search works, returns results with content_type field
  - `delete_indexed_video`: ✅ Successfully deleted 20 transcript chunks, 77 chunks from another video
  - Filtering by `content_type="comment"` correctly shows only newly indexed content
- **Note:** YouTube transcript API rate-limited during testing (expected behavior)
- **Goal Status:** 🟢 Complete - ready for v0.0.3 release

### 2025-01-14: Task 7 Complete - Goal Implemented
- **Task-07:** Registered all new tools in `app/server.py`
  - `semantic_search_comments` - comment search with auto-indexing
  - `semantic_search_all` - unified transcript + comment search
  - `get_indexed_videos` - list indexed video IDs
  - `delete_indexed_video` - remove videos from index
  - Updated server instructions with new tool documentation
  - All 406 tests pass, linting clean
- **Goal Status:** 🟠 Implemented - awaiting user validation

### 2025-01-14: Task 6 Complete
- **Task-06:** Implemented utility tools `get_indexed_videos` and `delete_indexed_video`
  - `get_indexed_videos`: Lists indexed video IDs with channel/content_type filters
  - `delete_indexed_video`: Deletes transcripts and/or comments for a video
  - Both support `content_type` parameter for type-specific operations
  - 9 new tests (4 for get, 5 for delete)
  - All 406 tests pass, linting clean

### 2025-01-14: Task 5 Complete
- **Task-05:** Implemented `semantic_search_all` unified search tool
  - Searches across both transcripts and comments in single query
  - Supports `content_types` parameter to filter by type
  - Auto-indexes both types when scope provided
  - Results include `content_type` field with type-specific metadata
  - Transcript results: start_time, end_time, timestamp_url, language
  - Comment results: author, like_count, reply_count, published_at
  - 11 new tests in `TestSemanticSearchAll` class
  - All 397 tests pass, linting clean

### 2025-01-14: Task 4 Complete
- **Task-04:** Implemented `semantic_search_comments` tool with tests
  - Tool supports auto-indexing of video comments
  - Filters by `content_type: "comment"` using `$and` with video scope
  - Returns author, like_count, reply_count in results
  - 12 new tests in `TestSemanticSearchComments` class
  - Tests cover: no scope, video/channel scope, already indexed, failures,
    result formatting, min_score, vector store errors, combined scope, k value,
    max_comments_per_video, channel fetch errors
  - All 386 tests pass, linting clean

### 2025-01-14: Tasks 1-3 Complete
- **Task-01:** Added `content_type: "transcript"` to TranscriptChunker metadata
  - 1 new test added to test_semantic_chunker.py
  - All 36 chunker tests pass
- **Task-02:** Created CommentChunker class
  - New file: `app/tools/youtube/semantic/comment_chunker.py`
  - Creates one Document per comment (no complex chunking needed)
  - Includes author, like_count, reply_count in metadata
  - 18 new tests in test_comment_chunker.py, all passing
- **Task-03:** Extended TranscriptIndexer for comments
  - Added `is_video_comments_indexed()` method
  - Added `index_video_comments()` method
  - Added `delete_video_comments()` method
  - Added `get_indexed_video_ids_by_content_type()` method
  - Updated mock_vector_store to handle $and filters
  - 12 new tests, 42 total indexer tests passing
- Linting clean, all tests pass

### 2025-01-14: Goal Created
- Defined architecture for comment semantic search
- Designed metadata schema with content_type discriminator
- Planned 7 tasks for implementation
- Removed backward compatibility requirement (single user)
- Ready for implementation pending approval
