# Task-06: Implement `semantic_search_transcripts` MCP Tool (Revised)

**Status:** 🟢 Complete
**Priority:** High
**Created:** 2025-01-13
**Updated:** 2025-01-14
**Parent:** [Goal 02 Scratchpad](../scratchpad.md)

---

## Objective

Implement `semantic_search_transcripts` as the primary MCP tool for semantic transcript search. The tool should **automatically index missing transcripts** before searching, providing a seamless agent experience.

---

## Revised Design: Auto-Index on Search

### Key Insight

The agent shouldn't need to explicitly call an "index" tool. Indexing is an implementation detail. The agent just wants to search - the tool handles everything else transparently.

### Tool Signature

```python
async def semantic_search_transcripts(
    query: str,
    channel_ids: list[str] | None = None,  # Scope to these channels
    video_ids: list[str] | None = None,     # And/or specific videos
    k: int = 10,
    language: str = "en",
    max_videos_per_channel: int = 50,  # Limit when fetching channel videos
) -> dict[str, Any]:
    """Search transcripts using natural language.

    Automatically indexes any missing transcripts before searching.
    Can search across multiple channels and/or specific videos.
    """
```

### Flow

1. **Determine Scope:** Collect all video IDs to search over
   - If `video_ids` provided → use those directly
   - If `channel_ids` provided → fetch video lists from each channel
   - If both → union of both sets
   - If neither → search all indexed content (no auto-indexing)

2. **Auto-Index Missing:** For each video in scope
   - Check if already indexed via `is_video_indexed()`
   - Index missing ones transparently via `index_video()`
   - Track indexing stats for response

3. **Search with Filter:** Query vector store
   - Apply metadata filter on scoped video IDs
   - Use `similarity_search_with_score()` for ranking

4. **Return Results:** Ranked transcript segments
   - Include video metadata, timestamps, scores
   - Include indexing stats (how many were newly indexed)

---

## Implementation Plan

### Files to Modify

1. **`app/tools/youtube/semantic/tools.py`**
   - Rewrite `semantic_search_transcripts()` with auto-index logic
   - Keep `index_channel_transcripts()` for explicit bulk pre-warming
   - Keep `index_video_transcript()` for single video indexing
   - Add `get_indexer()` factory (already done)

2. **`app/server.py`**
   - Register `semantic_search_transcripts` as primary MCP tool
   - Optionally register `index_channel_transcripts` for pre-warming
   - Update server instructions

3. **`app/tools/youtube/semantic/__init__.py`**
   - Export new functions

4. **`tests/unit/test_semantic_tools.py`**
   - Test auto-index logic
   - Test scope determination (channels, videos, both, neither)
   - Test search with filters

### Implementation Details

```python
async def semantic_search_transcripts(
    query: str,
    channel_ids: list[str] | None = None,
    video_ids: list[str] | None = None,
    k: int = 10,
    language: str = "en",
    max_videos_per_channel: int = 50,
) -> dict[str, Any]:
    """Search transcripts with auto-indexing."""

    indexer = get_indexer()
    scoped_video_ids: set[str] = set()
    indexing_stats = {"checked": 0, "indexed": 0, "skipped": 0, "errors": 0}

    # Step 1: Determine scope
    if video_ids:
        scoped_video_ids.update(video_ids)

    if channel_ids:
        for channel_id in channel_ids:
            videos = await get_channel_videos(channel_id, max_results=max_videos_per_channel)
            scoped_video_ids.update(v["video_id"] for v in videos)

    # Step 2: Auto-index missing (only if scope is defined)
    if scoped_video_ids:
        for video_id in scoped_video_ids:
            indexing_stats["checked"] += 1
            if not indexer.is_video_indexed(video_id):
                result = await indexer.index_video(video_id, language=language)
                if result.indexed_count > 0:
                    indexing_stats["indexed"] += 1
                elif result.skipped_count > 0:
                    indexing_stats["skipped"] += 1
                else:
                    indexing_stats["errors"] += 1

    # Step 3: Build filter and search
    filter_dict = None
    if scoped_video_ids:
        filter_dict = {"video_id": {"$in": list(scoped_video_ids)}}

    results = indexer.vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter=filter_dict,
    )

    # Step 4: Format results
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "video_id": doc.metadata.get("video_id"),
            "video_title": doc.metadata.get("video_title"),
            "video_url": doc.metadata.get("video_url"),
            "text": doc.page_content,
            "start_time": doc.metadata.get("start_time"),
            "end_time": doc.metadata.get("end_time"),
            "timestamp_url": doc.metadata.get("timestamp_url"),
            "score": float(score),
            "channel_id": doc.metadata.get("channel_id"),
            "channel_title": doc.metadata.get("channel_title"),
        })

    return {
        "query": query,
        "results": formatted_results,
        "total_results": len(formatted_results),
        "indexing_stats": indexing_stats,
        "filters_applied": {
            "channel_ids": channel_ids,
            "video_ids": video_ids,
            "language": language,
        },
    }
```

---

## Success Criteria

- [x] `semantic_search_transcripts` auto-indexes missing videos
- [x] Supports `channel_ids`, `video_ids`, or both
- [x] Returns search results with timestamps and scores
- [x] Returns indexing stats (how many newly indexed)
- [x] Tool registered in `server.py` with `@mcp.tool`
- [x] Unit tests for scope determination and auto-index logic
- [x] Linting passes
- [x] All existing tests still pass
- [x] Test isolation fixed (no flaky tests)

---

## Progress Log

### 2025-01-14: Test Isolation Fixed (🟢)
- Added `clear_semantic_caches` autouse fixture to `tests/conftest.py`
  - Clears `lru_cache` on `get_vector_store`, `get_embeddings`, `get_semantic_config` before/after each test
  - Prevents cache pollution between tests
- Fixed test assertion in `test_search_with_video_ids_auto_indexes`
  - Set comparison instead of list comparison (order is non-deterministic)
- All 334 tests pass, linting clean
- Task complete and validated

### 2025-01-13: Implementation Complete (🟠)
- Full `semantic_search_transcripts()` with auto-indexing implemented
- Registered in `server.py` as MCP tool
- 22 new tests in `test_semantic_tools.py`
- Removed `@lru_cache` from `get_indexer()` (unnecessary, caused test issues)
- **Issue:** 1 flaky test due to test isolation with `get_vector_store()` lru_cache
  - Passes alone, fails in full suite sometimes
  - Next session: fix test isolation or add conftest fixture
- Linting passes

### 2025-01-13: Revised Approach
- Original plan exposed indexing as separate agent-facing tool
- User feedback: agent shouldn't need to manage indexing explicitly
- Revised to auto-index-on-search approach
- Supports multiple channels and/or specific videos
- Updated scratchpad with new implementation plan

### 2025-01-13: Initial Implementation
- Created `get_indexer()` factory function
- Implemented basic `index_channel_transcripts()` (kept for pre-warming)
- Implemented `index_video_transcript()` (kept for single video)
- Now revising `semantic_search_transcripts()` with auto-index

---

## Notes

- **First search is slow:** If indexing 50 videos, expect 1-2 minutes. Subsequent searches are fast.
- **No scope = search all:** If neither `channel_ids` nor `video_ids` provided, searches all indexed content without auto-indexing.
- **Pre-warming still available:** `index_channel_transcripts` exists for explicit bulk indexing if agent/user wants to pre-warm the index.
- **Quota usage:** ~1 unit per video for metadata, 0 for transcripts. 50 videos = ~50 units.

## Completed

Task-06 is now fully complete. The semantic search tool is implemented, tested, and validated.

### Future Considerations (for next tasks)
- Consider adding `content_type` to metadata for future comments search
- Explore cacheTag/cacheLife for RefCache integration with semantic results
