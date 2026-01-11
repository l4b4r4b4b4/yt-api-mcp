# Task 08: Live Streaming Features

**Status:** 🟢 Complete
**Created:** 2025-01-08
**Updated:** 2025-01-09
**Dependencies:** Task 07 (Comment Tools) ✅

---

## Objective

Implement YouTube live streaming tools to search for live streams, check live status, and retrieve live chat messages. Extends the MCP server with real-time content capabilities.

---

## Scope

### Four Live Streaming Tools:

1. **`search_live_videos(query, max_results)`** - Search for currently live streams
   - Extends existing search_videos with event_type filter
   - Returns: Live video results with viewer count
   - Use: `youtube.search` namespace (6h TTL - same as regular search)

2. **`is_live(video_id)`** - Check if a video is currently live
   - Fast check for live broadcast status
   - Returns: Boolean + live details (viewer count, scheduled time, etc.)
   - Use: `youtube.api` namespace (very short TTL: 30 seconds)

3. **`get_live_chat_id(video_id)`** - Get live chat ID for streaming video
   - Required before fetching chat messages
   - Returns: activeLiveChatId from liveStreamingDetails
   - Use: `youtube.api` namespace (5 min TTL)

4. **`get_live_chat_messages(video_id, max_results, page_token)`** - Get recent live chat messages
   - Fetches latest chat messages from live stream
   - **Uses pagination token** for efficient polling (only new messages)
   - Returns: List of chat messages + next_page_token for continuous polling
   - Use: Very short cache (30 seconds)
   - Note: MCP request/response requires manual polling (not true streaming)

### Key Design Decisions:

**Caching Strategy:**
- `search_live_videos`: 6h TTL (same as regular search) - search results change slowly
- `is_live`: 30 seconds TTL - live status changes quickly
- `get_live_chat_id`: 5 min TTL - chat ID doesn't change during stream
- `get_live_chat_messages`: 30 seconds TTL - chat is real-time but MCP is request/response

**Live Status Detection:**
- Use `videos.list` with `liveStreamingDetails` part
- Check `liveStreamingDetails.activeLiveChatId` for live status
- Return viewer count, scheduled time, actual start time

**Chat Message Handling:**
- Use `liveChatMessages.list` endpoint with pagination token support
- **First call:** No token → Get latest messages + return next_page_token
- **Subsequent calls:** Pass token → Get only NEW messages since last call
- Return top-level messages only (no threading in MVP)
- Include author display name, text, timestamp, next_page_token, polling_interval
- Limit to max_results (1-2000, default 200)
- Token-based approach prevents duplicates and enables efficient continuous polling

**Error Handling:**
- Video not live → Clear message (not an error)
- Chat disabled → Return empty list gracefully
- Invalid chat ID → Clear error message
- Quota exceeded → Standard API error handling

---

## Implementation Plan

### Step 1: Extend `app/tools/youtube/search.py`

Add live streaming search capability:

```python
async def search_live_videos(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search for currently live YouTube videos.

    Same as search_videos but filtered to only live streams.
    """
    # Call search API with eventType="live"
    # Return VideoSearchResult models
```

### Step 2: Create `app/tools/youtube/live.py`

**Structure:**
```python
# Imports
from googleapiclient.errors import HttpError
from app.tools.youtube.client import get_youtube_service
from app.tools.youtube.models import LiveStatus, LiveChatMessage

# Three async functions
async def is_live(video_id: str) -> dict[str, Any]
async def get_live_chat_id(video_id: str) -> dict[str, Any]
async def get_live_chat_messages(video_id: str, max_results: int = 200) -> dict[str, Any]
```

**Implementation Details:**

**`is_live(video_id)`:**
- Call `videos().list(part="liveStreamingDetails", id=video_id)`
- Check if `liveStreamingDetails` exists
- Check if `activeLiveChatId` is present (indicates currently live)
- Return LiveStatus with: is_live, viewer_count, scheduled_start_time, actual_start_time
- Handle video not found, not a live broadcast

**`get_live_chat_id(video_id)`:**
- Call `videos().list(part="liveStreamingDetails", id=video_id)`
- Extract `liveStreamingDetails.activeLiveChatId`
- Return dict with: video_id, live_chat_id, is_live
- Raise error if video is not live or chat not available

**`get_live_chat_messages(video_id, max_results, page_token)`:**
- First call `get_live_chat_id(video_id)` to get chat ID
- Call `liveChatMessages().list(liveChatId=chat_id, part="snippet,authorDetails", pageToken=page_token)`
- Parse messages into LiveChatMessage models
- Return dict with: video_id, messages, total_returned, next_page_token, polling_interval_millis
- **page_token handling:** None on first call → latest messages; token from previous call → only new messages
- Handle chat disabled, video not live

### Step 3: Add Models to `app/tools/youtube/models.py`

```python
class LiveStatus(BaseModel):
    """Live broadcast status for a video."""
    video_id: str
    is_live: bool
    viewer_count: int | None
    scheduled_start_time: str | None
    actual_start_time: str | None
    active_live_chat_id: str | None

class LiveChatMessage(BaseModel):
    """Single live chat message."""
    author: str
    text: str
    published_at: str
    author_channel_id: str

class LiveChatResponse(BaseModel):
    """Response from get_live_chat_messages with pagination."""
    video_id: str
    messages: list[LiveChatMessage]
    total_returned: int
    next_page_token: str | None
    polling_interval_millis: int
```

### Step 4: Register Tools in `app/server.py`

Add after comment tools:

```python
# YouTube live streaming tools
@mcp.tool
@cache.cached(namespace="youtube.search", ttl=21600)  # 6h cache
async def search_live_videos(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search for currently live YouTube videos. [docstring]"""
    result = await search_live_videos_impl(query, max_results)
    return result

@mcp.tool
@cache.cached(namespace="youtube.api", ttl=30)  # 30 second cache
async def is_live(video_id: str) -> dict[str, Any]:
    """Check if a video is currently live. [docstring]"""
    result = await is_live_impl(video_id)
    return result

@mcp.tool
@cache.cached(namespace="youtube.api", ttl=300)  # 5 min cache
async def get_live_chat_id(video_id: str) -> dict[str, Any]:
    """Get live chat ID for a streaming video. [docstring]"""
    result = await get_live_chat_id_impl(video_id)
    return result

@mcp.tool
@cache.cached(namespace="youtube.comments", ttl=30)  # 30 second cache
async def get_live_chat_messages(
    video_id: str,
    max_results: int = 200,
    page_token: str | None = None
) -> dict[str, Any]:
    """Get recent live chat messages with pagination support. [docstring]"""
    result = await get_live_chat_messages_impl(video_id, max_results, page_token)
    return result
```

### Step 5: Write Tests

**Test files:**
- Add to `tests/test_youtube_search.py` - live search tests (5 tests)
- Create `tests/test_youtube_live.py` - live status and chat tests (15-20 tests)

**Test Structure (~20-25 tests total):**

```python
# test_youtube_search.py additions (5 tests)
class TestSearchLiveVideos:
    - test_search_live_videos_success
    - test_search_live_videos_empty_results
    - test_search_live_videos_clamps_max_results
    - test_search_live_videos_handles_quota_error
    - test_search_live_videos_uses_event_type_filter

# test_youtube_live.py (15-20 tests)
class TestIsLive:
    - test_is_live_returns_true_for_live_video
    - test_is_live_returns_false_for_not_live
    - test_is_live_includes_viewer_count
    - test_is_live_includes_scheduled_time
    - test_is_live_video_not_found
    - test_is_live_not_a_broadcast

class TestGetLiveChatId:
    - test_get_chat_id_success
    - test_get_chat_id_video_not_live
    - test_get_chat_id_video_not_found
    - test_get_chat_id_no_chat_available

class TestGetLiveChatMessages:
    - test_get_messages_success
    - test_get_messages_with_page_token
    - test_get_messages_returns_next_page_token
    - test_get_messages_clamps_max_results
    - test_get_messages_video_not_live
    - test_get_messages_chat_disabled
    - test_get_messages_empty_chat
    - test_get_messages_parses_author_details
    - test_get_messages_includes_polling_interval
```

### Step 6: Update Server Instructions

Add live streaming tools to server instructions in `app/server.py`:

```
Available YouTube Tools:
...
- get_video_comments: ... (existing)

Live Streaming Tools:
- search_live_videos: Search for currently live streams (cached 6h)
- is_live: Check if video is currently live (cached 30s)
- get_live_chat_id: Get chat ID for live video (cached 5m)
- get_live_chat_messages: Get recent live chat (cached 30s)

Live Streaming Notes:
- Very short cache times for real-time data (30s for status/chat, 5min for chat ID)
- MCP is request/response (not true streaming) - agent must poll for new messages
- Use page_token from previous call to get only new messages (efficient polling)
- For continuous monitoring, poll every 30-60 seconds using returned next_page_token
- 30 second cache prevents excessive API usage during polling
- Same API quota as other tools (1 unit per request)
- For true real-time experience with instant updates, use YouTube web interface
```

---

## Tasks

- [x] Gather context - Read YouTube Live API docs
- [x] Document plan in this scratchpad
- [x] Get approval before implementation (approved with pagination)
- [x] Add `LiveStatus`, `LiveChatMessage`, and `LiveChatResponse` models to `app/tools/youtube/models.py`
- [x] Add `search_live_videos()` to `app/tools/youtube/search.py`
- [x] Create `app/tools/youtube/live.py` (371 lines - 3 functions with full docs)
- [x] Implement `is_live()`
- [x] Implement `get_live_chat_id()`
- [x] Implement `get_live_chat_messages()` with page_token support
- [x] Update `app/tools/youtube/__init__.py` exports
- [x] Register 4 tools in `app/server.py` with @cache.cached
- [x] Update server instructions with live streaming docs
- [x] Add 5 tests to `tests/test_youtube_search.py`
- [x] Create `tests/test_youtube_live.py` (19 tests including pagination)
- [x] Run tests: `pytest tests/test_youtube_live.py -v` (19/19 passing)
- [x] Run all tests: `pytest -v` (178/178 passing - was 154, now +24)
- [x] Run linting: `ruff check . --fix && ruff format .` (all checks passed)
- [x] All tests passing ✅
- [x] User validation in Zed (tested with LiveNOW from FOX live stream)
- [x] Document completion in this scratchpad

---

## Success Criteria

- [x] All 4 live streaming functions implemented and tested
- [x] Uses Pydantic models (LiveStatus, LiveChatMessage, LiveChatResponse)
- [x] Pagination with page_token working correctly (no duplicates, only new messages)
- [x] Handles errors: video not live, chat disabled, not found
- [x] All 24 unit tests pass (19 live tests + 5 search tests)
- [x] Linting passes (100% clean)
- [x] User validation: Works in Zed with real live stream (search → status → chat with polling)
- [x] Caching configured (30s for status/chat, 5m for chat ID, 6h for search)
- [x] Clear documentation about MCP polling pattern and limitations

---

## Reference Implementation Notes

**YouTube Live API Endpoints:**

```python
# Check live status
videos().list(
    part="liveStreamingDetails",
    id=video_id
)
# Returns: liveStreamingDetails with activeLiveChatId, concurrentViewers, etc.

# Search for live videos
search().list(
    q=query,
    part="snippet",
    type="video",
    eventType="live",  # Only currently live
    maxResults=max_results
)

# Get chat messages (with pagination)
liveChatMessages().list(
    liveChatId=chat_id,
    part="snippet,authorDetails",
    maxResults=max_results,
    pageToken=page_token  # None for first call, token from previous response for subsequent calls
)
# Returns: items (messages), nextPageToken, pollingIntervalMillis
```

**Key Patterns:**
- Check `liveStreamingDetails.activeLiveChatId` to confirm live status
- Chat ID remains same throughout stream
- Messages have `snippet` (text, timestamp) and `authorDetails` (name, channel)
- **Pagination workflow:**
  1. First call: No pageToken → Get latest messages + nextPageToken
  2. Store nextPageToken
  3. Second call: Pass nextPageToken → Get only NEW messages since first call + new nextPageToken
  4. Repeat step 3 for continuous polling
- YouTube recommends polling interval (pollingIntervalMillis in response)

**Error Handling:**
```python
# Video not live
if "liveStreamingDetails" not in video:
    return {"is_live": False, "video_id": video_id}

# Chat disabled
if not video["liveStreamingDetails"].get("activeLiveChatId"):
    return {"video_id": video_id, "messages": [], "total_returned": 0}
```

---

## Testing Strategy

1. **Unit tests with mocks** - Mock YouTube API to avoid quota usage
2. **Edge cases** - Not live, chat disabled, video not found
3. **Live search** - Empty results, quota errors
4. **Cache verification** - Very short TTLs working correctly
5. **Integration test in Zed** - Find actual live stream and test all tools

---

## Notes

- **MCP Limitation:** Request/response model means no true streaming - user must manually refresh to get new messages
- **Polling Pattern:** Could call `get_live_chat_messages()` repeatedly, but each call costs quota
- **Same API Service:** Uses YouTube Data API v3 - no new Google Console setup needed! ✅
- **API Key Works:** Read-only access to public live chats works with API key (no OAuth2 needed)
- **Quota Cost:** Each call costs 1 unit (same as other tools)
- **Cache Strategy:** Very short TTLs (30s) for live data to balance freshness vs quota

**Use Cases:**
- Monitor live streams for specific topics
- Check if creator is currently streaming
- Poll live chat activity with page_token for continuous feed
- Track viewer engagement during streams
- Efficient polling pattern: 30s cache + page_token = minimal quota usage + near real-time

**Pagination Benefits:**
- No duplicate messages between polls
- Only fetches new messages (efficient)
- Continuous feed experience (like following a conversation)
- Token expires ~30min if not used (graceful degradation to latest messages)

## Implementation Summary

**Code Added:**
- `app/tools/youtube/live.py` - 371 lines (3 async functions)
- `app/tools/youtube/models.py` - +83 lines (3 new models)
- `app/tools/youtube/search.py` - +98 lines (search_live_videos function)
- `app/server.py` - +163 lines (4 tool registrations + docs)
- `app/tools/youtube/__init__.py` - +10 lines (exports)
- `tests/test_youtube_live.py` - 549 lines (19 comprehensive tests)
- `tests/test_youtube_search.py` - +117 lines (5 live search tests)

**Total Lines:** ~1,391 lines of production code + tests

**Test Results:**
- 178 tests passing (was 154, added 24)
- 19 live streaming tests (is_live, get_live_chat_id, get_live_chat_messages)
- 5 search_live_videos tests
- All error paths tested (not live, chat disabled, not found, quota)
- Pagination fully tested (page_token, no duplicates, only new messages)
- 100% linting clean

**Features Delivered:**
1. ✅ Live stream search with eventType="live" filter
2. ✅ Live status detection with viewer count and timing
3. ✅ Live chat ID retrieval (cached 5min)
4. ✅ Live chat message polling with pagination (cached 30s)
5. ✅ Proper error handling (graceful not-live, clear errors)
6. ✅ MCP polling documentation (limitations clearly stated)

**✅ VALIDATED IN ZED - 2025-01-09**

Tested all 4 tools with real live stream (LiveNOW from FOX, video_id: e7AqeMm52LI):

1. ✅ `search_live_videos("news live", 5)` - Found 5 currently broadcasting streams
2. ✅ `is_live("e7AqeMm52LI")` - Returned is_live=true, viewer_count=2160, active_live_chat_id
3. ✅ `get_live_chat_messages("e7AqeMm52LI", 20)` - Retrieved 20 live chat messages with authors, timestamps
4. ✅ Pagination with page_token - Second call returned only NEW messages (no duplicates)
5. ✅ Cache hit verification - Repeat call returned same ref_id instantly (cache working)
6. ✅ Error handling - `is_live("dQw4w9WgXcQ")` gracefully returned is_live=false for regular video

All features working as designed in real-world conditions!

Task 08 complete! We're at 73% done (8/11 tasks). Next: Task 09 (Final Server Polish), then documentation and final validation.

---

## Next Steps After Completion

1. ✅ Task 08 marked as 🟢 Complete after successful validation
2. Update main scratchpad with progress (8/11 tasks = 73%)
3. Proceed to Task 09: Final Server Polish
4. Then Task 10: Documentation
5. Finally Task 11: Testing & Validation

---
