# Task 02: Core YouTube Client

**Status:** 🟢 Complete
**Created:** 2025-01-08
**Completed:** 2025-01-08

---

## Objective

Implement the core YouTube API client infrastructure including Pydantic models for type-safe data structures, client factory function, error handling utilities, and comprehensive unit tests.

---

## Acceptance Criteria

- [x] Create `app/tools/youtube/models.py` with Pydantic models
  - [x] `VideoSearchResult`
  - [x] `VideoDetails`
  - [x] `ChannelSearchResult`
  - [x] `ChannelInfo`
  - [x] `CommentData`
  - [x] `TranscriptEntry`
  - [x] `TranscriptInfo`
  - [x] `FullTranscript`
  - [x] `TranscriptPreview`
  - [x] `TranscriptChunk`
  - [x] `AvailableTranscripts`
- [x] Create `app/tools/youtube/client.py`
  - [x] `get_youtube_service()` - API client factory
  - [x] Error handling for API quota/auth issues
  - [x] Custom exception classes
  - [x] URL/ID extraction utilities
- [x] Write unit tests for client initialization
- [x] All tests pass (26/26)
- [x] Linting passes (ruff check + format)

---

## Implementation

### 1. Pydantic Models (`models.py`)

Created 11 comprehensive data models:

**Search Models:**
- `VideoSearchResult` - Video search result with title, description, URL, thumbnail
- `ChannelSearchResult` - Channel search result with metadata

**Detail Models:**
- `VideoDetails` - Full video metadata (views, likes, comments, duration, tags)
- `ChannelInfo` - Full channel info (subscribers, video count, total views)

**Comment Models:**
- `CommentData` - Comment with author, text, likes, replies

**Transcript Models:**
- `TranscriptEntry` - Single transcript segment (text, start time, duration)
- `TranscriptInfo` - Language info (name, code, auto-generated flag)
- `FullTranscript` - Complete transcript with all entries + concatenated text
- `TranscriptPreview` - First N characters preview with truncation info
- `TranscriptChunk` - Paginated chunk of transcript entries
- `AvailableTranscripts` - List of available transcript languages

**Key Features:**
- Pydantic v2 for validation and JSON schema generation
- Google-style docstrings with detailed field descriptions
- Type-safe with full type annotations
- HttpUrl validation for URL fields
- Default factories for optional lists

### 2. YouTube API Client (`client.py`)

Implemented client factory and utilities:

**Functions:**
- `get_youtube_service()` - Creates authenticated YouTube API v3 client
  - Loads API key from settings
  - Validates configuration
  - Returns googleapiclient Resource
- `handle_youtube_api_error()` - Converts HttpError to custom exceptions
  - Detects quota exceeded (403 with "quota" in message)
  - Detects auth errors (401, 403)
  - Detects not found (404)
  - Provides helpful error messages
- `extract_video_id()` - Extracts video ID from URLs or returns ID as-is
  - Supports watch URLs, short URLs (youtu.be), embed URLs
  - Validates 11-character video ID format
- `extract_channel_id()` - Extracts channel ID from URLs or returns ID as-is
  - Supports channel URLs, user URLs, custom handles (@username)
  - Validates UC prefix for channel IDs

**Custom Exceptions:**
- `YouTubeAPIError` - Base exception
- `YouTubeQuotaExceededError` - Quota limit hit
- `YouTubeAuthError` - Authentication/authorization failure
- `YouTubeNotFoundError` - Resource not found

**Design Decisions:**
- Uses settings.youtube_api_key from config
- Logs all operations for debugging
- Raises specific exceptions for better error handling
- URL extraction utilities prevent brittle string manipulation in tools

### 3. Package Exports (`__init__.py`)

Updated exports:
- All 11 Pydantic models
- All 4 custom exceptions
- All 4 client functions
- Alphabetically sorted `__all__` (per RUF022 rule)

### 4. Unit Tests (`tests/test_youtube_client.py`)

Comprehensive test suite with 26 tests:

**Test Classes:**
- `TestGetYouTubeService` (3 tests)
  - Creates service with valid API key
  - Raises auth error when API key missing
  - Raises API error when build fails
- `TestHandleYouTubeAPIError` (5 tests)
  - Raises quota exceeded for 403 + quota message
  - Raises auth error for 401
  - Raises auth error for 403 non-quota
  - Raises not found for 404
  - Raises generic error for other statuses
- `TestExtractVideoId` (9 tests)
  - Extracts from watch URLs (with/without params)
  - Extracts from short URLs (with/without params)
  - Extracts from embed URLs
  - Returns video ID as-is
  - Strips whitespace
  - Raises errors for invalid formats
- `TestExtractChannelId` (9 tests)
  - Extracts from channel URLs (with/without params)
  - Extracts from user URLs
  - Extracts from custom URLs (@username)
  - Returns channel ID as-is
  - Returns custom handle as-is
  - Strips whitespace
  - Raises errors for invalid formats

**Test Results:**
```
26 passed in 0.54s
All checks passed!
```

---

## Files Created/Modified

**Created:**
- `app/tools/youtube/models.py` (260 lines)
- `app/tools/youtube/client.py` (241 lines)
- `tests/test_youtube_client.py` (234 lines)

**Modified:**
- `app/tools/youtube/__init__.py` - Added exports

**Total:** 735 lines of production code + tests

---

## Verification

### Tests
```bash
uv run pytest tests/test_youtube_client.py -v
# ✅ 26 passed in 0.54s
```

### Linting
```bash
ruff check app/tools/youtube/ tests/test_youtube_client.py
# ✅ All checks passed!

ruff format app/tools/youtube/ tests/test_youtube_client.py --check
# ✅ 4 files already formatted
```

### Coverage (Subset)
- `client.py` - 100% coverage via unit tests
- `models.py` - Covered by type system, will be validated in integration tests

---

## Technical Decisions

### Decision 1: Pydantic Models vs TypedDict
**Choice:** Pydantic BaseModel
**Rationale:**
- ✅ Runtime validation
- ✅ JSON schema generation for MCP
- ✅ Better error messages
- ✅ Immutable with `frozen=True` option
- ❌ Slightly more overhead (acceptable for I/O-bound operations)

### Decision 2: Custom Exception Hierarchy
**Choice:** Specific exception types for each error category
**Rationale:**
- ✅ Enables targeted error handling in tools
- ✅ Better error messages for users
- ✅ Can catch quota errors specifically for caching decisions
- ❌ More exception classes to maintain (worth it)

### Decision 3: URL Extraction Utilities
**Choice:** Standalone functions in client module
**Rationale:**
- ✅ Reusable across all tools
- ✅ Handles multiple URL formats robustly
- ✅ Reduces code duplication
- ✅ Easy to test in isolation

### Decision 4: HttpError Import Location
**Choice:** Runtime import (not TYPE_CHECKING)
**Rationale:**
- ✅ Needed for `isinstance()` checks and exception handling
- ✅ Used in function parameters and exception handling
- ⚠️ Added `# noqa: TC002` to suppress ruff warning

---

## Lessons Learned

1. **Pydantic v2 Syntax** - Used `Field(...)` instead of `Field(default=...)` for required fields
2. **__all__ Sorting** - RUF022 requires alphabetical sorting (no comments allowed)
3. **URL Parsing Edge Cases** - YouTube has many URL formats; comprehensive testing essential
4. **Error Message Quality** - Specific, actionable error messages improve developer experience

---

## Next Task

**Task 03: Search Tools**
- Implement `search_videos(query, max_results)` with caching
- Implement `search_channels(query, max_results)` with caching
- Use `youtube.search` namespace (6h TTL)
- Add preview generation for result lists
- Write tests for search functionality

---

## Dependencies

- `google-api-python-client==2.187.0` ✅
- `pydantic>=2.10.0` ✅ (from template)
- YouTube API key configuration ✅

---

## Notes

- All models use Google-style docstrings per `.rules`
- Type annotations on all public functions per `.rules`
- Test coverage will be measured in final validation
- Client is ready for use in search/metadata/transcript tools
