# Task 05: Metadata Tools - Video & Channel Details

**Status:** ⚪ Not Started
**Created:** 2025-01-08
**Dependencies:** Task 02 (Core YouTube Client), Task 04 (Local Dev Testing)

---

## Objective

Implement tools to retrieve detailed metadata about YouTube videos and channels, including statistics, descriptions, and other information not available in search results.

---

## Scope

### In Scope
- `get_video_details(video_id)` - Full video metadata
- `get_channel_info(channel_id)` - Channel statistics and details
- Pydantic models for structured responses
- 24-hour caching via `youtube.api` namespace
- Unit tests with mocked API responses
- Integration with server.py

### Out of Scope
- Playlist metadata (future enhancement)
- Video analytics (requires OAuth)
- Channel upload history (use search instead)
- Trending videos (separate tool)

---

## Implementation Plan

### Step 1: Review Existing Models (5 min)

Check what's already in `app/tools/youtube/models.py`:
- `VideoDetails` - Should have all video fields
- `ChannelInfo` - Should have all channel fields

**Fields needed:**

**VideoDetails:**
- Basic: video_id, title, description, url, thumbnail
- Statistics: view_count, like_count, comment_count
- Metadata: published_at, duration, tags, category_id
- Channel: channel_id, channel_title

**ChannelInfo:**
- Basic: channel_id, title, description, url, thumbnail
- Statistics: subscriber_count, video_count, view_count
- Metadata: published_at, country, custom_url

---

### Step 2: Create metadata.py Module (30 min)

**Location:** `app/tools/youtube/metadata.py`

**Structure:**
```python
"""YouTube metadata tools - video and channel details."""

from typing import Any
from app.tools.youtube.client import get_youtube_service
from app.tools.youtube.models import VideoDetails, ChannelInfo

async def get_video_details(video_id: str) -> dict[str, Any]:
    """Get detailed information about a YouTube video.

    Retrieves comprehensive video metadata including title, description,
    statistics (views, likes, comments), duration, tags, and channel info.
    Results are cached for 24 hours in the youtube.api namespace.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")

    Returns:
        VideoDetails dictionary with all available metadata

    Raises:
        ValueError: If video_id is invalid
        YouTubeAPIError: If video not found or API error

    Example:
        >>> details = await get_video_details("dQw4w9WgXcQ")
        >>> print(details["title"])
        "Never Gonna Give You Up"
    """
    # Implementation here
    pass


async def get_channel_info(channel_id: str) -> dict[str, Any]:
    """Get detailed information about a YouTube channel.

    Retrieves channel metadata including title, description, statistics
    (subscribers, videos, total views), and branding information.
    Results are cached for 24 hours in the youtube.api namespace.

    Args:
        channel_id: YouTube channel ID (e.g., "UCuAXFkgsw1L7xaCfnd5JJOw")

    Returns:
        ChannelInfo dictionary with all available metadata

    Raises:
        ValueError: If channel_id is invalid
        YouTubeAPIError: If channel not found or API error

    Example:
        >>> info = await get_channel_info("UCuAXFkgsw1L7xaCfnd5JJOw")
        >>> print(info["subscriber_count"])
        "1.5M"
    """
    # Implementation here
    pass
```

**Implementation Details:**

1. **get_video_details():**
   - Use `youtube.videos().list(part="snippet,statistics,contentDetails")`
   - Extract video_id, handle URL input (extract ID if needed)
   - Parse ISO 8601 duration to readable format
   - Handle missing fields (some videos disable likes/comments)
   - Return VideoDetails model as dict

2. **get_channel_info():**
   - Use `youtube.channels().list(part="snippet,statistics,brandingSettings")`
   - Extract channel_id, handle URL input
   - Format large numbers (1.5M subscribers)
   - Handle missing custom_url
   - Return ChannelInfo model as dict

**Error Handling:**
- Invalid ID format: `ValueError("Invalid video/channel ID")`
- Not found: `YouTubeAPIError("Video/Channel not found")`
- Quota exceeded: `YouTubeAPIError("API quota exceeded, resets at...")`
- Network errors: Retry once, then fail with clear message

---

### Step 3: Register Tools in server.py (10 min)

Add to `app/server.py`:

```python
from app.tools.youtube import get_video_details, get_channel_info

@mcp.tool
@cache.cached(namespace="youtube.api", ttl=86400)  # 24 hours
async def _get_video_details(video_id: str) -> dict[str, Any]:
    """Get detailed information about a YouTube video.

    Retrieves comprehensive video metadata including title, description,
    statistics (views, likes, comments), duration, tags, and channel info.
    Cached for 24 hours to minimize API quota usage.

    Args:
        video_id: YouTube video ID (from URL or search results)

    Returns:
        Video details with statistics and metadata

    Note:
        - Costs 1 quota unit per request (very cheap)
        - Cached for 24h in youtube.api namespace
    """
    result = await get_video_details(video_id)
    return result


@mcp.tool
@cache.cached(namespace="youtube.api", ttl=86400)  # 24 hours
async def _get_channel_info(channel_id: str) -> dict[str, Any]:
    """Get detailed information about a YouTube channel.

    Retrieves channel metadata including title, description, statistics
    (subscribers, videos, total views), and branding information.
    Cached for 24 hours to minimize API quota usage.

    Args:
        channel_id: YouTube channel ID (from search results or URL)

    Returns:
        Channel info with statistics and metadata

    Note:
        - Costs 1 quota unit per request (very cheap)
        - Cached for 24h in youtube.api namespace
    """
    result = await get_channel_info(channel_id)
    return result
```

**Update server instructions:**
Add to the tool list:
- `get_video_details`: Get full video metadata (views, likes, duration, etc.)
- `get_channel_info`: Get channel statistics (subscribers, video count, etc.)

---

### Step 4: Write Unit Tests (30 min)

**Location:** `tests/test_youtube_metadata.py`

**Test Coverage:**

1. **test_get_video_details_success**
   - Mock API response with full video data
   - Verify all fields populated correctly
   - Check duration parsing (PT15M30S → "15:30")

2. **test_get_video_details_missing_stats**
   - Mock video with disabled likes/comments
   - Verify graceful handling (None or 0)

3. **test_get_video_details_not_found**
   - Mock 404 response
   - Verify raises YouTubeAPIError with clear message

4. **test_get_video_details_invalid_id**
   - Pass invalid video ID format
   - Verify raises ValueError

5. **test_get_channel_info_success**
   - Mock API response with full channel data
   - Verify subscriber count formatting (1500000 → "1.5M")

6. **test_get_channel_info_no_custom_url**
   - Mock channel without custom URL
   - Verify graceful handling

7. **test_get_channel_info_not_found**
   - Mock 404 response
   - Verify raises YouTubeAPIError

8. **test_caching_video_details**
   - Call get_video_details twice with same ID
   - Verify API called once, second returns cached

9. **test_caching_channel_info**
   - Call get_channel_info twice with same ID
   - Verify API called once, second returns cached

**Fixtures:**
```python
@pytest.fixture
def mock_video_response():
    return {
        "items": [{
            "id": "test123",
            "snippet": {
                "title": "Test Video",
                "description": "Test description",
                "publishedAt": "2024-01-01T00:00:00Z",
                "channelId": "UCtest",
                "channelTitle": "Test Channel",
                "thumbnails": {...},
                "tags": ["test", "video"]
            },
            "statistics": {
                "viewCount": "1000",
                "likeCount": "50",
                "commentCount": "10"
            },
            "contentDetails": {
                "duration": "PT15M30S"
            }
        }]
    }
```

---

### Step 5: Integration Testing in Zed (10 min)

**Test 5.1: Get Video Details**
```
Get details for video ID "nLwbNhSxLd4" (NixOS guide from earlier)
```

**Expected:**
- Returns full video metadata
- Shows view count, like count, duration
- Includes channel info
- Response time: ~1 second (fresh API call)

**Test 5.2: Get Channel Info**
```
Get information about the Vimjoyer channel
```

**Expected:**
- Returns channel statistics
- Shows subscriber count, video count, total views
- Includes description and thumbnail
- Response time: ~1 second (fresh API call)

**Test 5.3: Cache Verification**
Repeat Test 5.1:
```
Get details for video ID "nLwbNhSxLd4" again
```

**Expected:**
- Instant response (<100ms)
- Same data as first call
- No new API call

**Test 5.4: Workflow Test**
```
1. Search for "Vimjoyer NixOS"
2. Get details for the first video result
3. Get info about Vimjoyer's channel
```

**Expected:**
- Smooth workflow without errors
- All data consistent (video's channel_id matches channel info)

---

## API Quota Cost Analysis

### Quota Usage:
- **get_video_details**: 1 unit per request
- **get_channel_info**: 1 unit per request
- **With 24h caching**: Effectively 0.04 units/hour per unique video/channel

### Comparison to Search:
- Search: 100 units per request
- Metadata: 1 unit per request
- **Metadata is 100x cheaper!**

### Expected Usage Pattern:
1. User searches for videos (100 units)
2. User gets details on 5 results (5 units)
3. User gets channel info for 2 channels (2 units)
4. **Total: 107 units** (vs 700 units without caching)

With caching, subsequent requests cost 0 units within 24h window.

---

## File Changes Summary

### New Files:
- `app/tools/youtube/metadata.py` (~150 lines)
- `tests/test_youtube_metadata.py` (~300 lines)

### Modified Files:
- `app/server.py` - Add 2 new tools with caching decorators
- `app/tools/youtube/__init__.py` - Export new functions

### Models (should already exist in models.py):
- `VideoDetails` - May need to add duration, tags fields
- `ChannelInfo` - Should be complete from Task 02

---

## Success Criteria

### Critical (Must Pass):
- [ ] `get_video_details` returns complete video metadata
- [ ] `get_channel_info` returns complete channel metadata
- [ ] Both tools work in Zed integration test
- [ ] Caching works (24h TTL, instant on repeat)
- [ ] All unit tests pass (9+ tests)
- [ ] Linting passes (ruff check/format)
- [ ] Type hints correct (no mypy errors)

### Important (Should Pass):
- [ ] Error handling graceful (clear messages)
- [ ] Duration parsing works (PT15M30S → "15:30")
- [ ] Large number formatting (1500000 → "1.5M")
- [ ] Works with video IDs from search results
- [ ] API quota usage is minimal (1 unit per call)

### Nice to Have:
- [ ] Handles edge cases (private videos, deleted channels)
- [ ] URL extraction (accept YouTube URLs, not just IDs)
- [ ] Rich error context (includes video/channel ID in errors)

---

## Known Challenges & Solutions

### Challenge 1: Duration Parsing
**Issue:** YouTube returns ISO 8601 format (PT15M30S)
**Solution:** Use `isodate` library or custom parser
```python
def parse_duration(iso_duration: str) -> str:
    """Convert PT15M30S to 15:30"""
    # Implementation
```

### Challenge 2: Missing Statistics
**Issue:** Some videos disable likes/comments
**Solution:** Return None or 0, document in model
```python
like_count: int | None  # None if disabled
```

### Challenge 3: Subscriber Count Hidden
**Issue:** Some channels hide subscriber counts
**Solution:** Return "Hidden" or None
```python
subscriber_count: str | None  # "1.5M" or "Hidden"
```

---

## Testing Strategy

### Unit Tests (Fast, No API):
- Mock all YouTube API calls
- Test data transformation logic
- Test error handling
- Test edge cases

### Integration Tests (Real API):
- Use known stable videos (official YouTube channels)
- Test with real API responses
- Verify caching behavior
- Run manually before committing

### Manual Validation (Zed):
- Test complete workflow (search → details → channel)
- Verify all fields displayed correctly
- Check performance (response times)
- Validate error messages are helpful

---

## Dependencies

### Python Packages (Already Installed):
- `google-api-python-client` - YouTube API client
- `pydantic` - Data validation
- `mcp-refcache` - Caching

### New Dependencies (if needed):
- `isodate` - ISO 8601 duration parsing (optional, can DIY)

---

## Next Steps After Completion

1. Update Task 05 scratchpad with completion status
2. Update goal scratchpad (5/11 tasks complete)
3. Move to Task 06: Transcript Tools
4. Consider: Should we add playlist metadata? (Out of scope for MVP)

---

## Notes

- Metadata tools are 100x cheaper than search (1 unit vs 100 units)
- 24h caching is aggressive but reasonable for metadata
- Consider shorter TTL (6h) if channels update stats frequently
- These tools enable rich video/channel analysis workflows
- Natural follow-up after search: search → details → transcripts

---

## Estimated Time

- **Step 1 (Review Models):** 5 minutes
- **Step 2 (Implement metadata.py):** 30 minutes
- **Step 3 (Register in server):** 10 minutes
- **Step 4 (Write tests):** 30 minutes
- **Step 5 (Integration test):** 10 minutes
- **Total:** ~1.5 hours

---

**Ready to implement! Let's build the metadata tools! 🚀**
