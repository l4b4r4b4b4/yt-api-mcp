# Task 06: Transcript Tools

**Status:** 🟢 Complete
**Created:** 2025-01-08
**Updated:** 2025-01-08
**Validated:** 2025-01-08
**Dependencies:** Task 05 (Metadata Tools) ✅

---

## Objective

Implement YouTube transcript retrieval tools with intelligent caching and chunking strategies. This is the most complex task, handling large transcript data with RefCache preview/pagination patterns.

---

## Scope

### Four Transcript Tools:
1. **`list_available_transcripts(video_id)`** - List all available transcript languages
   - Returns: `AvailableTranscripts` with language codes, names, auto-generated flags
   - Use: `youtube.content` namespace (permanent cache)

2. **`get_video_transcript_preview(video_id, language, max_chars)`** - Get preview
   - Returns: `TranscriptPreview` with first N characters + metadata
   - Use: `youtube.content` namespace (permanent cache)
   - Default: max_chars=2000

3. **`get_full_transcript(video_id, language)`** - Get complete transcript
   - Returns: `FullTranscript` with all entries + full_text
   - Use: `youtube.content` namespace (permanent cache)
   - RefCache handles large results automatically

4. **`get_transcript_chunk(video_id, start_index, chunk_size, language)`** - Paginate entries
   - Returns: `TranscriptChunk` with subset of entries
   - Use: `youtube.content` namespace (permanent cache)
   - For iterating through large transcripts entry-by-entry

### Key Design Decisions:

**Caching Strategy:**
- Namespace: `youtube.content` (permanent, deterministic=True)
- Rationale: Transcripts never change once published
- TTL: None (permanent cache)

**Language Handling:**
- If `language` not specified, use first available (prefer manual > auto-generated)
- Fall back to auto-generated if no manual transcript exists
- Raise clear errors if video has no transcripts

**Error Handling:**
- `TranscriptsDisabled` → Clear error message
- `NoTranscriptFound` → Suggest listing available languages
- Invalid video ID → Validate before API call
- Network errors → Wrap with context

**Data Flow:**
1. `list_available_transcripts()` → Discover languages
2. `get_video_transcript_preview()` → Quick peek (RefCache preview)
3. `get_full_transcript()` → Full data (RefCache handles large results)
4. `get_transcript_chunk()` → Paginate if needed

---

## Implementation Plan

### Step 1: Create `app/tools/youtube/transcripts.py`

**Structure:**
```python
# Imports
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# Models from app.tools.youtube.models
from app.tools.youtube.models import (
    AvailableTranscripts, TranscriptInfo, TranscriptEntry,
    TranscriptPreview, FullTranscript, TranscriptChunk
)

# Four async functions (no decorators here, applied in server.py)
async def list_available_transcripts(video_id: str) -> dict[str, Any]
async def get_video_transcript_preview(video_id: str, language: str = "", max_chars: int = 2000) -> dict[str, Any]
async def get_full_transcript(video_id: str, language: str = "") -> dict[str, Any]
async def get_transcript_chunk(video_id: str, start_index: int = 0, chunk_size: int = 50, language: str = "") -> dict[str, Any]
```

**Implementation Details:**

**`list_available_transcripts(video_id)`:**
- Call `YouTubeTranscriptApi.list_transcripts(video_id)`
- Iterate through transcript_list
- Build `TranscriptInfo` for each (language, language_code, is_generated, is_translatable)
- Return `AvailableTranscripts` model with video_id, available_languages list, transcript_info list
- Handle: TranscriptsDisabled, NoTranscriptFound, invalid video_id

**`get_video_transcript_preview(video_id, language, max_chars)`:**
- Call `_fetch_transcript_data(video_id, language)` (internal helper)
- Extract full_text from result
- Truncate to max_chars if needed
- Return `TranscriptPreview` with: video_id, language, preview, total_length, is_truncated
- Include metadata about full transcript (total entries, duration estimate)

**`get_full_transcript(video_id, language)`:**
- Call `_fetch_transcript_data(video_id, language)` (internal helper)
- Returns complete `FullTranscript` with all entries
- RefCache will automatically handle large results (>2KB) with preview/reference pattern
- Include: video_id, language, transcript (list[TranscriptEntry]), full_text (concatenated)

**`get_transcript_chunk(video_id, start_index, chunk_size, language)`:**
- Call `_fetch_transcript_data(video_id, language)` (internal helper)
- Extract transcript entries
- Validate start_index bounds
- Slice entries: entries[start_index:start_index+chunk_size]
- Return `TranscriptChunk` with: video_id, language, start_index, chunk_size, entries, total_entries, has_more

**`_fetch_transcript_data(video_id, language)` (internal helper):**
- If language specified: Try to get that specific language
- If not specified: Get first available (prefer manual over auto-generated)
- Fetch transcript using `YouTubeTranscriptApi.get_transcript()` or `.fetch()`
- Parse entries into `TranscriptEntry` models (text, start, duration)
- Return raw data dict for other functions to transform

### Step 2: Register Tools in `app/server.py`

Add after metadata tools:

```python
# YouTube transcript tools
@mcp.tool
@cache.cached(namespace="youtube.content")  # Permanent cache
async def _list_available_transcripts(video_id: str) -> dict[str, Any]:
    """List all available transcript languages for a video. [docstring]"""
    result = await list_available_transcripts(video_id)
    return result

@mcp.tool
@cache.cached(namespace="youtube.content")
async def _get_video_transcript_preview(
    video_id: str,
    language: str = "",
    max_chars: int = 2000,
) -> dict[str, Any]:
    """Get preview of video transcript. [docstring]"""
    result = await get_video_transcript_preview(video_id, language, max_chars)
    return result

@mcp.tool
@cache.cached(namespace="youtube.content")
async def _get_full_transcript(
    video_id: str,
    language: str = "",
) -> dict[str, Any]:
    """Get complete video transcript. [docstring]"""
    result = await get_full_transcript(video_id, language)
    return result

@mcp.tool
@cache.cached(namespace="youtube.content")
async def _get_transcript_chunk(
    video_id: str,
    start_index: int = 0,
    chunk_size: int = 50,
    language: str = "",
) -> dict[str, Any]:
    """Get chunk of transcript entries. [docstring]"""
    result = await get_transcript_chunk(video_id, start_index, chunk_size, language)
    return result
```

### Step 3: Write Tests in `tests/test_youtube_transcripts.py`

**Test Structure (~15-20 tests):**

```python
# Fixtures
@pytest.fixture
def mock_transcript_list()  # Mock YouTubeTranscriptApi.list_transcripts
@pytest.fixture
def mock_transcript_data()  # Mock transcript entries
@pytest.fixture
def sample_transcript_entries()  # Sample TranscriptEntry data

# TestListAvailableTranscripts (5-6 tests)
- test_list_available_transcripts_success
- test_list_available_transcripts_empty
- test_list_available_transcripts_disabled
- test_list_available_transcripts_invalid_id
- test_list_available_transcripts_multiple_languages

# TestGetVideoTranscriptPreview (4-5 tests)
- test_get_preview_success
- test_get_preview_truncates_long_text
- test_get_preview_short_text_not_truncated
- test_get_preview_with_language
- test_get_preview_transcripts_disabled

# TestGetFullTranscript (4-5 tests)
- test_get_full_transcript_success
- test_get_full_transcript_with_language
- test_get_full_transcript_no_language_uses_first
- test_get_full_transcript_disabled
- test_get_full_transcript_not_found

# TestGetTranscriptChunk (5-6 tests)
- test_get_chunk_success
- test_get_chunk_first_chunk
- test_get_chunk_middle_chunk
- test_get_chunk_last_chunk
- test_get_chunk_invalid_index_low
- test_get_chunk_invalid_index_high
```

### Step 4: Update Server Instructions

Add transcript tools to server instructions in `app/server.py`:

```
Available YouTube Tools:
...
- get_video_details: ... (existing)
- get_channel_info: ... (existing)
- list_available_transcripts: List available transcript languages (cached permanently)
- get_video_transcript_preview: Preview first N chars of transcript (cached permanently)
- get_full_transcript: Get complete transcript with timestamps (cached permanently)
- get_transcript_chunk: Paginate through transcript entries (cached permanently)

Transcript Notes:
- Transcripts cached permanently (content never changes)
- Use list_available_transcripts first to discover languages
- get_full_transcript may return RefCache preview for large transcripts
- Use get_transcript_chunk for pagination if needed
```

---

## Tasks

- [x] Gather context - Read reference implementation
- [x] Document plan in this scratchpad
- [x] Get approval before implementation
- [x] Create `app/tools/youtube/transcripts.py` (474 lines)
- [x] Implement `list_available_transcripts()`
- [x] Implement internal `_fetch_transcript_data()` helper
- [x] Implement `get_video_transcript_preview()`
- [x] Implement `get_full_transcript()`
- [x] Implement `get_transcript_chunk()`
- [x] Update `app/tools/youtube/__init__.py` exports
- [x] Register 4 tools in `app/server.py` with @cache.cached
- [x] Update server instructions with transcript docs
- [x] Write test file `tests/test_youtube_transcripts.py` (26 tests, 568 lines)
- [x] Run tests: `pytest tests/test_youtube_transcripts.py -v` ✅ 26/26 passing
- [x] Run linting: `ruff check . --fix && ruff format .` ✅ Clean
- [x] All tests passing: 145/145 tests ✅
- [x] User validation in Zed ✅ (2025-01-08)
- [x] Document implementation status

---

## Success Criteria

- [x] All 4 transcript functions implemented and tested
- [x] Uses Pydantic models (AvailableTranscripts, TranscriptPreview, FullTranscript, TranscriptChunk)
- [x] Handles errors: TranscriptsDisabled, NoTranscriptFound, invalid IDs
- [x] Smart language selection (manual > auto-generated when not specified)
- [x] All 26 unit tests pass (exceeded target of 15-20)
- [x] Linting passes (100% clean)
- [x] **User validation: Works in Zed with real API calls** ✅ Validated 2025-01-08
- [x] Caching configured (permanent, youtube.content namespace)
- [x] RefCache handles large transcripts gracefully (automatic)

---

## Reference Implementation Notes

From `.agent/youtube_toolset.py`:

**Key Patterns:**
- Use `YouTubeTranscriptApi.list_transcripts(video_id)` for listing
- Use `transcript.fetch()` to get actual transcript data
- Iterate through transcript items and extract text, start, duration
- Handle both dict-like and object-like transcript items
- Concatenate text with spaces for full_text
- Cache with `@content_cache.cached` (permanent)

**Error Handling:**
```python
except TranscriptsDisabled:
    raise ValueError("Transcripts are disabled for this video")
except NoTranscriptFound:
    raise ValueError("No transcript found for this video")
```

**Language Selection Logic:**
```python
if language_code:
    transcript = transcript_list.find_transcript([language_code])
else:
    # Get first available (YouTube API returns manual first if available)
    transcript = transcript_list.find_transcript([transcript_list[0].language_code])
```

**Entry Processing:**
```python
for item in transcript:
    if isinstance(item, dict):
        text = item["text"]
        segments.append(item)
    elif hasattr(item, "text"):
        text = getattr(item, "text")
        segment_dict = {
            "text": text,
            "start": getattr(item, "start", 0),
            "duration": getattr(item, "duration", 0),
        }
        segments.append(segment_dict)
```

---

## Testing Strategy

1. **Unit tests with mocks** - Mock YouTubeTranscriptApi to avoid API calls
2. **Edge cases** - Empty, disabled, missing, invalid IDs
3. **Language handling** - Explicit language vs auto-selection
4. **Chunking logic** - Boundary conditions, invalid indices
5. **Preview truncation** - Short vs long transcripts
6. **Integration test in Zed** - Real API call with caching validation

---

## Implementation Summary

**Files Created:**
- `app/tools/youtube/transcripts.py` (474 lines) - 4 transcript functions + helper
- `tests/test_youtube_transcripts.py` (568 lines) - 26 comprehensive tests

**Files Modified:**
- `app/tools/youtube/__init__.py` - Added transcript exports
- `app/server.py` - Registered 4 transcript tools with youtube.content namespace
  - **ALSO FIXED**: Removed underscore prefixes from all tool names (cleaner API)
  - Changed `_search_videos` → `search_videos`, etc.

**Test Results:**
- ✅ 26/26 transcript tests passing
- ✅ 145/145 total tests passing
- ✅ Linting clean (ruff)

**Key Implementation Details:**
1. **CRITICAL BUG FOUND & FIXED**: `YouTubeTranscriptApi.list()` is an **instance method**, not static
   - Original code: `YouTubeTranscriptApi.list(video_id)` ❌
   - Fixed code: `YouTubeTranscriptApi().list(video_id)` ✅
   - Fixed in both `transcripts.py` (lines 71, 376)
   - Fixed in all test mocks to patch the class correctly
2. **Internal helper pattern**: `_fetch_transcript_data()` shared by all 4 functions
3. **Mock testing**: Comprehensive fixtures for transcript_list with fetch() methods
4. **Error handling**: TranscriptsDisabled, NoTranscriptFound, invalid IDs, language mismatches
5. **Caching**: Permanent cache (youtube.content namespace, no TTL)
6. **Clean tool names**: Removed underscore prefixes from all MCP tools for better UX

**API Design:**
- `list_available_transcripts()` → Discovery (lists languages)
- `get_video_transcript_preview()` → Quick peek (first N chars)
- `get_full_transcript()` → Complete data (RefCache auto-handles large results)
- `get_transcript_chunk()` → Pagination (entry-by-entry iteration)

## Validation Status

**Implementation:** ✅ Complete
- Code written: 474 lines (transcripts.py)
- Tests written: 26 tests (all passing)
- Linting: Clean
- API bug fixed: YouTubeTranscriptApi().list() pattern

**User Validation:** ✅ Complete (2025-01-08)
- Zed restarted successfully
- All 4 transcript tools validated on video `a67Sv4Mbxmc` (Ultimate NixOS Guide):

### Validation Results:

**1. list_available_transcripts(a67Sv4Mbxmc)** ✅
- Returned available languages: `["en"]`
- Transcript info: English (auto-generated), translatable
- RefCache: 54 bytes, 3 items

**2. get_video_transcript_preview(a67Sv4Mbxmc, max_chars=500)** ✅
- Returned first 500 chars of transcript
- Total length: 16,368 chars
- Clean preview of NixOS video content
- RefCache: 156 bytes, 5 items

**3. get_full_transcript(a67Sv4Mbxmc, language="en")** ✅
- **Size: 13,448 TOKENS** (not bytes - mcp-refcache uses token-based mode!)
- RefCache automatically returned reference (exceeds 2,048 token preview limit)
- Caching verified: Second call returned same ref_id (instant cache hit)
- Original size: 13,448 tokens, preview: 7 bytes sample

**4. get_transcript_chunk(a67Sv4Mbxmc, start_index=0, chunk_size=10)** ✅
- Retrieved first 10 entries with timestamps
- Shows 434 total entries, has_more=true
- Perfect for pagination
- RefCache: 268 bytes, 7 items

**Critical Success:** YouTubeTranscriptApi().list() bug fix working correctly in production!

**Key Discovery:** mcp-refcache is configured with token-based mode (max_size=2048 tokens), not bytes. The full transcript is 13,448 tokens, which is why RefCache returns a reference for pagination.

## Next Steps

**Task 06 Complete** → Proceed to **Task 07: Comment Tools**
- All core YouTube data retrieval implemented (search, metadata, transcripts)
- Comment tools will be simpler than transcripts (no chunking complexity)
- 145/145 tests passing, all tools validated in production
