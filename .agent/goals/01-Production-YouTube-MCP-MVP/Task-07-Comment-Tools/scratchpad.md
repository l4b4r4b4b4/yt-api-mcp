# Task 07: Comment Tools

**Status:** 🟢 Complete
**Created:** 2025-01-08
**Updated:** 2025-01-08
**Validated:** 2025-01-08
**Dependencies:** Task 06 (Transcript Tools) ✅

---

## Objective

Implement YouTube comment retrieval tool with intelligent caching. This is simpler than transcripts - just one tool to fetch top-level comments with engagement metrics.

---

## Scope

### Single Comment Tool:
1. **`get_video_comments(video_id, max_results)`** - Get top comments with engagement data
   - Returns: List of comments with author, text, like_count, published_at
   - Use: `youtube.comments` namespace (12h TTL)
   - Default: max_results=20, max=100
   - Handle: Comments disabled gracefully (return empty list)

### Key Design Decisions:

**Caching Strategy:**
- Namespace: `youtube.comments` (12h TTL)
- Rationale: Comments are more volatile than video metadata but less than search results
- TTL: 43200 seconds (12 hours)

**Comment Ordering:**
- Fetch by relevance (YouTube's default)
- Only top-level comments (no replies in MVP)
- Limit to 100 max_results (API constraint)

**Error Handling:**
- `commentsDisabled` → Return empty list with no error
- Invalid video ID → Clear error message
- Network errors → Wrap with context
- Quota exceeded → Standard API error handling

**Data Structure:**
- Author display name
- Comment text (plainText format)
- Like count
- Published timestamp

---

## Implementation Plan

### Step 1: Create `app/tools/youtube/comments.py`

**Structure:**
```python
# Imports
from googleapiclient.errors import HttpError
from app.tools.youtube.client import get_youtube_service
from app.tools.youtube.models import VideoComment

# Single async function (decorator applied in server.py)
async def get_video_comments(video_id: str, max_results: int = 20) -> dict[str, Any]
```

**Implementation Details:**

**`get_video_comments(video_id, max_results)`:**
- Validate max_results (1-100 range)
- Call YouTube API `commentThreads().list()`
- Parameters:
  - part="snippet"
  - videoId=video_id
  - maxResults=max_results
  - textFormat="plainText"
- Parse response items
- Build `VideoComment` models for each
- Return dict with: video_id, comments (list), total_returned
- Handle commentsDisabled → return empty comments list
- Handle other errors → raise with context

### Step 2: Add Model to `app/tools/youtube/models.py`

```python
class VideoComment(BaseModel):
    """Single YouTube comment."""
    author: str
    text: str
    like_count: int
    published_at: str
```

### Step 3: Register Tool in `app/server.py`

Add after transcript tools:

```python
# YouTube comment tools
@mcp.tool
@cache.cached(namespace="youtube.comments", ttl=43200)  # 12h cache
async def get_video_comments(video_id: str, max_results: int = 20) -> dict[str, Any]:
    """Get top comments for a YouTube video with engagement metrics.

    Retrieves top-level comments (no replies) sorted by relevance.
    Comments are cached for 12 hours. Returns empty list if comments
    are disabled for the video.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")
        max_results: Maximum comments to return (1-100, default: 20)

    Returns:
        Dictionary with video_id, comments list, and total_returned.
        Each comment includes author, text, like_count, published_at.

    Example:
        >>> comments = get_video_comments("nLwbNhSxLd4", max_results=10)
        >>> print(comments["comments"][0]["author"])

    Note:
        - Costs 1 quota unit per request
        - Cached for 12h in youtube.comments namespace
        - Returns empty list if comments disabled (not an error)
    """
    result = await get_video_comments_impl(video_id, max_results)
    return result
```

### Step 4: Write Tests in `tests/test_youtube_comments.py`

**Test Structure (~8-10 tests):**

```python
# Fixtures
@pytest.fixture
def mock_youtube_service()  # Mock YouTube API client
@pytest.fixture
def sample_comments_response()  # Mock API response
@pytest.fixture
def comments_disabled_error()  # Mock commentsDisabled HttpError

# TestGetVideoComments (8-10 tests)
- test_get_comments_success
- test_get_comments_with_max_results
- test_get_comments_clamps_min_max (test 1-100 range)
- test_get_comments_disabled_returns_empty_list
- test_get_comments_invalid_video_id
- test_get_comments_no_comments_returns_empty
- test_get_comments_network_error
- test_get_comments_parses_response_correctly
```

### Step 5: Update Server Instructions

Add comment tool to server instructions in `app/server.py`:

```
Available YouTube Tools:
...
- get_full_transcript: ... (existing)
- get_transcript_chunk: ... (existing)
- get_video_comments: Get top comments with engagement metrics (cached 12h)

Comment Notes:
- Returns empty list if comments disabled (not an error)
- Only top-level comments (no replies)
- Sorted by relevance
- Costs 1 quota unit per request
```

---

## Tasks

- [x] Gather context - Read reference implementation
- [x] Document plan in this scratchpad
- [x] Get approval before implementation
- [x] Add `VideoComment` model to `app/tools/youtube/models.py` (already existed as CommentData)
- [x] Create `app/tools/youtube/comments.py` (99 lines)
- [x] Implement `get_video_comments()`
- [x] Update `app/tools/youtube/__init__.py` exports
- [x] Register tool in `app/server.py` with @cache.cached (5 min TTL)
- [x] Update server instructions with comment docs
- [x] Write test file `tests/test_youtube_comments.py` (9 tests, 284 lines)
- [x] Run tests: `pytest tests/test_youtube_comments.py -v` ✅ 9/9 passing
- [x] Run linting: `ruff check . --fix && ruff format .` ✅ Clean
- [x] All tests passing: 154/154 tests ✅
- [x] User validation in Zed ✅ (2025-01-08)
- [x] Document completion

---

## Success Criteria

- [x] Comment function implemented and tested
- [x] Uses Pydantic `CommentData` model
- [x] Handles errors: commentsDisabled (returns empty list), invalid IDs
- [x] All 9 unit tests pass
- [x] Linting passes (100% clean)
- [x] **User validation: Works in Zed with real API calls** ✅ Validated 2025-01-08
- [x] Caching configured (5 min TTL, youtube.comments namespace)

---

## Reference Implementation Notes

From `.agent/youtube_toolset.py` lines 290-337:

**Key Patterns:**
```python
@comment_cache.cached
def get_video_comments(video_id: str, max_results: int = 20) -> List[Dict[str, str]]:
    max_results = min(max(1, max_results), 100)  # Clamp 1-100

    comments_response = (
        youtube.commentThreads()
        .list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
        )
        .execute()
    )

    results = []
    for item in comments_response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]
        result = {
            "author": comment["authorDisplayName"],
            "text": comment["textDisplay"],
            "like_count": comment["likeCount"],
            "published_at": comment["publishedAt"],
        }
        results.append(result)
```

**Error Handling:**
```python
except HttpError as e:
    if "commentsDisabled" in str(e):
        logger.warning(f"Comments are disabled for video ID: {video_id}")
        return []  # Empty list, not an error
    raise ValueError(f"Failed to get video comments: {str(e)}")
```

---

## Testing Strategy

1. **Unit tests with mocks** - Mock YouTube API to avoid quota usage
2. **Edge cases** - Empty, disabled, invalid IDs
3. **Range validation** - Test max_results clamping (1-100)
4. **Error handling** - commentsDisabled, network errors
5. **Integration test in Zed** - Real API call with popular video

---

## Notes

- **Much simpler than transcripts** - Only one function, no chunking needed ✅
- **Comments disabled is common** - Must handle gracefully (empty list, not error) ✅
- **No pagination in MVP** - Just fetch max_results and return ✅
- **Only top-level comments** - No reply threads in MVP (can add later if needed) ✅
- **Good test candidate** - Popular videos like NixOS tutorials have lots of comments
- **Cache TTL changed to 5 minutes** - Better for trending videos with active engagement (user feedback)

After this task completes, we're at 80% done (8/10 tasks)! Only final polish and docs remain.

---

## Implementation Summary

**Files Created:**
- `app/tools/youtube/comments.py` (99 lines) - Single comment function
- `tests/test_youtube_comments.py` (284 lines) - 9 comprehensive tests

**Files Modified:**
- `app/tools/youtube/__init__.py` - Added comment exports
- `app/server.py` - Registered comment tool with 5 min cache
- Model already existed as `CommentData` in `models.py`

**Test Results:**
- ✅ 9/9 comment tests passing
- ✅ 154/154 total tests passing (+9 from Task 06)
- ✅ Linting clean (ruff)

**Key Implementation Details:**
1. **5-minute cache TTL** - Perfect for trending videos with active engagement
2. **Graceful commentsDisabled handling** - Returns empty list, not error
3. **Range clamping** - max_results 1-100 validated
4. **Robust error detection** - Checks both error content and string representation
5. **Complete metadata** - author, text, like_count, published_at, reply_count

**API Design:**
- `get_video_comments(video_id, max_results=20)` - Simple, focused function
- Returns dict with video_id, comments list, total_returned
- Handles all edge cases gracefully

## Validation Status

**Implementation:** ✅ Complete
- Code written: 99 lines (comments.py)
- Tests written: 9 tests (all passing)
- Linting: Clean
- Cache TTL: 5 minutes (user-requested change)

**User Validation:** ✅ Complete (2025-01-08)
- Zed restarted successfully
- Tested on video `nLwbNhSxLd4` (NixOS Ultimate Guide)
- Test: `get_video_comments("nLwbNhSxLd4", max_results=10)`

### Validation Results:

**1. get_video_comments(nLwbNhSxLd4, max_results=10)** ✅
- Returned 10 top comments sorted by relevance
- Top comment: @oglothenerd with 353 likes
- All metadata present: author, text, like_count, published_at, reply_count
- RefCache: 644 bytes, 3 items

**2. Caching Verification** ✅
- Second call returned same ref_id (instant cache hit)
- 5-minute cache working perfectly
- Great for trending videos with active engagement

**3. Real-world Usage** ✅
- Works on popular videos with active comments
- Comments include engagement metrics
- Handles various comment types (recent, high-engagement, etc.)

**Critical Success:** 5-minute cache TTL perfect for trending videos (user-requested change from 12h)!

## Next Steps

1. ✅ Task 07 Complete and validated
2. **Next: Task 08 - Live Streaming Features** (NEW - user requested)
   - search_live_videos() - Search for live streams
   - is_live() - Check live status
   - get_live_chat_id() - Get chat ID
   - get_live_chat_messages() - Get live chat
3. Then Task 09: Final Server Polish (renamed from Task 08)
4. Then Task 10: Documentation (renamed from Task 09)
5. Finally Task 11: Testing & Validation (renamed from Task 10)

---
