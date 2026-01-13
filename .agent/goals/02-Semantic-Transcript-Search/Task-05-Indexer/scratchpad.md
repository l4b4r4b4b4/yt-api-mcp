# Task-05: Batch Indexer

**Status:** 🟠 Implemented
**Priority:** High
**Created:** 2025-01-13
**Updated:** 2025-01-13
**Parent:** [Goal 02: Semantic Transcript Search](../scratchpad.md)

---

## Objective

Implement the `TranscriptIndexer` class that orchestrates:
1. Fetching videos from a YouTube channel
2. Retrieving transcripts for each video
3. Chunking transcripts with the `TranscriptChunker`
4. Embedding and storing chunks in the vector store

---

## Implementation Plan

### Phase 1: Channel Video Fetching

1. [x] Create `get_channel_videos()` function in `search.py`
   - Use YouTube API search with `channelId` filter
   - Return list of video IDs with basic metadata
   - Support pagination for channels with many videos

### Phase 2: Indexer Core Methods

2. [x] Implement `is_video_indexed()` method
   - Query vector store for any documents with matching `video_id`
   - Return True/False efficiently (limit=1 query)

3. [x] Implement `delete_video()` method
   - Remove all chunks for a video from vector store
   - Use ChromaDB's delete with `where={"video_id": video_id}`
   - Return count of deleted chunks

4. [x] Implement `index_video()` method
   - Fetch transcript using `get_full_transcript()`
   - Get video metadata using `get_video_details()`
   - Chunk transcript with `TranscriptChunker`
   - Add chunks to vector store
   - Handle errors gracefully (no transcript, API errors)

5. [x] Implement `index_channel()` method
   - Fetch video list using `get_channel_videos()`
   - Check which videos are already indexed (unless `force_reindex`)
   - Call `index_video()` for each unindexed video
   - Track progress and accumulate results
   - Return `IndexingResult`

### Phase 3: Progress Tracking

6. [x] Add optional progress callback support
   - `on_video_start(video_id, index, total)`
   - `on_video_complete(video_id, chunks_created, error)`
   - Useful for UI progress bars

### Phase 4: Testing

7. [x] Unit tests for `is_video_indexed()`
8. [x] Unit tests for `delete_video()`
9. [x] Unit tests for `index_video()` with mocked dependencies
10. [x] Unit tests for `index_channel()` with mocked dependencies
11. [x] Error handling tests (no transcript, API errors)
12. [ ] Integration test with real vector store (marked `@pytest.mark.slow`)

---

## Design Decisions

### Channel Video Fetching Strategy

**Decision:** Use YouTube API search with `channelId` filter, not uploads playlist.

**Rationale:**
- Search API is simpler (single request with `channelId` param)
- Uploads playlist requires two API calls (get playlist ID, then list items)
- Search returns all publicly visible videos
- Can sort by date (`order="date"`) to get most recent first
- Quota cost is 100 units per search (same as playlist approach)

### Vector Store Queries for Existing Videos

**Decision:** Query with `where={"video_id": video_id}` and `limit=1`.

**Rationale:**
- We only need to know if ANY chunks exist for a video
- `limit=1` makes the query efficient
- ChromaDB's `get()` with `where` filter is the right API

### Error Handling Strategy

**Decision:** Continue on individual video failures, accumulate errors.

**Rationale:**
- One video without a transcript shouldn't fail the entire channel
- Collect all errors in `IndexingResult.errors`
- User can see which videos failed and why
- Can retry specific videos later

### Delete Before Re-index

**Decision:** When `force_reindex=True`, delete existing chunks before re-indexing.

**Rationale:**
- Prevents duplicate chunks in the vector store
- Clean slate for the video's content
- Chunk IDs might change (different tokenizer, config changes)

---

## API Design

### get_channel_videos() (New in search.py)

```python
async def get_channel_videos(
    channel_id: str,
    max_results: int = 50,
    order: str = "date",
) -> list[dict[str, Any]]:
    """Fetch videos from a YouTube channel.

    Args:
        channel_id: YouTube channel ID.
        max_results: Maximum videos to return (1-50).
        order: Sort order - "date", "rating", "viewCount", "title".

    Returns:
        List of video info dicts with video_id, title, published_at, etc.
    """
```

### TranscriptIndexer Methods

```python
class TranscriptIndexer:
    def __init__(
        self,
        vector_store: Chroma,
        chunker: TranscriptChunker,
    ) -> None: ...

    async def index_channel(
        self,
        channel_id: str,
        max_videos: int = 50,
        language: str = "en",
        force_reindex: bool = False,
        on_progress: Callable | None = None,
    ) -> IndexingResult: ...

    async def index_video(
        self,
        video_id: str,
        language: str = "en",
        force_reindex: bool = False,
    ) -> IndexingResult: ...

    def is_video_indexed(self, video_id: str) -> bool: ...

    async def delete_video(self, video_id: str) -> int: ...
```

---

## Metadata for Indexed Chunks

Each chunk stored in the vector store includes:

```python
metadata = {
    # Video identifiers
    "video_id": str,
    "channel_id": str,

    # Display info
    "video_title": str,
    "channel_title": str,
    "video_url": str,
    "published_at": str,

    # Timestamps
    "start_time": float,
    "end_time": float,
    "timestamp_url": str,

    # Chunk info
    "chunk_index": int,
    "token_count": int,

    # Optional chapter info
    "chapter_title": str | None,
    "chapter_index": int | None,

    # Transcript info
    "language": str,
}
```

---

## Files to Modify/Create

1. **app/tools/youtube/search.py**
   - Add `get_channel_videos()` function
   - Update `__all__`

2. **app/tools/youtube/semantic/indexer.py**
   - Implement all `TranscriptIndexer` methods
   - Replace `NotImplementedError` placeholders

3. **tests/test_semantic_indexer.py** (NEW)
   - Unit tests for all indexer methods
   - Mock YouTube API calls
   - Mock vector store operations

4. **app/tools/youtube/semantic/__init__.py**
   - May need to export `get_channel_videos` if used elsewhere

---

## Test Cases

### is_video_indexed()
- `test_video_indexed_returns_true` - Video with chunks returns True
- `test_video_not_indexed_returns_false` - Unknown video returns False
- `test_video_indexed_empty_store` - Empty store returns False

### delete_video()
- `test_delete_video_removes_chunks` - Removes all chunks for video
- `test_delete_video_returns_count` - Returns correct count
- `test_delete_video_not_found` - Returns 0 for unknown video

### index_video()
- `test_index_video_creates_chunks` - Happy path
- `test_index_video_no_transcript` - Handles missing transcript
- `test_index_video_skip_indexed` - Skips already indexed (force=False)
- `test_index_video_force_reindex` - Re-indexes when force=True
- `test_index_video_api_error` - Handles API errors gracefully

### index_channel()
- `test_index_channel_indexes_videos` - Happy path
- `test_index_channel_skips_indexed` - Skips already indexed videos
- `test_index_channel_force_reindex` - Re-indexes all when force=True
- `test_index_channel_handles_errors` - Continues on individual failures
- `test_index_channel_returns_result` - Returns correct IndexingResult
- `test_index_channel_respects_max_videos` - Limits video count

### get_channel_videos()
- `test_get_channel_videos_returns_list` - Returns video list
- `test_get_channel_videos_max_results` - Respects max_results
- `test_get_channel_videos_invalid_channel` - Handles invalid channel

---

## Quota Considerations

**Per Channel Index (50 videos):**
- 1 search request (100 units) to get video list
- 50 video details requests (50 units)
- 0 units for transcripts (uses youtube-transcript-api)
- **Total: ~150 quota units**

**Daily Quota:** 10,000 units
**Channels per day:** ~66 channels (at 50 videos each)

---

## Acceptance Criteria

- [x] `get_channel_videos()` fetches videos from a channel
- [x] `is_video_indexed()` correctly detects indexed videos
- [x] `delete_video()` removes all chunks for a video
- [x] `index_video()` handles transcripts and errors
- [x] `index_channel()` orchestrates batch indexing
- [x] `IndexingResult` contains accurate counts
- [x] Error messages include video IDs for debugging
- [x] All new tests pass (44 new tests)
- [x] All 312 tests pass (was 268)
- [x] Linting passes (ruff check/format)

## Completion Summary

**What was done:**
- Created `get_channel_videos()` in `search.py` with channelId filter, order param
- Implemented full `TranscriptIndexer` class with all methods
- Added `ProgressCallback` protocol for progress tracking
- Added `get_indexed_video_ids()` and `get_chunk_count()` utility methods
- Created 44 new tests in `test_semantic_indexer.py` and `test_youtube_search.py`
- All 312 tests passing, linting clean

**Awaiting:** User validation in real usage

---

## References

- [YouTube API Search](https://developers.google.com/youtube/v3/docs/search/list)
- [ChromaDB Collection API](https://docs.trychroma.com/reference/py-collection)
- [LangChain Chroma VectorStore](https://python.langchain.com/docs/integrations/vectorstores/chroma)
- [Task-04 Chunker Scratchpad](../Task-04-Chunker/scratchpad.md)
