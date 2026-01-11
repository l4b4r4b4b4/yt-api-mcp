# Task 03: Search Tools

**Status:** 🟢 Complete
**Created:** 2025-01-08
**Last Updated:** 2025-01-08

---

## Objective

Implement YouTube video and channel search functionality with intelligent caching via mcp-refcache. These tools will enable AI agents to discover videos and channels by query, with results cached in the `youtube.search` namespace (6h TTL) for efficient API usage.

---

## Context & Research

### Reference Implementation Analysis

From `.agent/youtube_toolset.py`:

**`search_videos(query, max_results=5)`:**
- Uses YouTube Data API v3 `search().list()` with `type="video"`
- Returns: title, description, video_id, url, thumbnail, channel_title, published_at
- Validates max_results between 1-50
- Decorated with `@search_cache.cached` (6h TTL, 300 max size)
- Handles HttpError and raises ValueError

**`search_channels(query, max_results=5)`:**
- Uses YouTube Data API v3 `search().list()` with `type="channel"`
- Returns: title, description, channel_id, url, thumbnail, published_at
- Same validation and error handling as search_videos
- Also decorated with `@search_cache.cached`

### Current Architecture

From `app/server.py`:
- Uses `@cache.cached(namespace="...")` decorator pattern
- Cache instance: `cache = TracedRefCache(_cache)`
- Tools return raw data; decorator wraps in cache response
- Return type annotation: `dict[str, Any]` (required for MCP tool return types with @cache.cached)
- Pattern: async function returns list, decorator handles caching

From `app/tools/demo.py`:
- Tool functions are async
- Input validation via Pydantic models (optional but recommended)
- Raw data return (list/dict), decorator handles caching
- Docstrings with Args, Returns, and cache behavior notes

---

## Implementation Plan

### 1. Create Search Module (`app/tools/youtube/search.py`)

**Structure:**
```python
# Imports
from googleapiclient.errors import HttpError
from app.tools.youtube.client import get_youtube_service, handle_youtube_api_error
from app.tools.youtube.models import VideoSearchResult, ChannelSearchResult

# Internal implementation functions (not cached directly)
async def _search_videos_impl(query: str, max_results: int) -> list[VideoSearchResult]
async def _search_channels_impl(query: str, max_results: int) -> list[ChannelSearchResult]

# Public functions (will be decorated in server.py)
async def search_videos(query: str, max_results: int = 5) -> list[dict[str, Any]]
async def search_channels(query: str, max_results: int = 5) -> list[dict[str, Any]]
```

**Why split implementation and public function?**
- Implementation function returns Pydantic models (type-safe)
- Public function converts to dict for caching decorator compatibility
- Cleaner separation of business logic and caching concerns

**search_videos implementation:**
1. Validate max_results (clamp between 1-50)
2. Get YouTube service client
3. Execute search with `type="video"`, `part="snippet"`
4. Parse response items into VideoSearchResult models
5. Handle HttpError via handle_youtube_api_error()
6. Return list of model dicts

**search_channels implementation:**
1. Same structure as search_videos
2. Use `type="channel"` in search parameters
3. Parse into ChannelSearchResult models
4. Extract channelId from item["id"]["channelId"]

### 2. Integrate with Server (`app/server.py`)

**Changes needed:**

**A. Update server instructions:**
- Remove demo tool descriptions
- Add YouTube search tool descriptions
- Document cache behavior (6h TTL for searches)

**B. Update RefCache configuration:**
- Consider adding namespace-specific configs for `youtube.search`
- Document TTL strategy (6h = volatile, recheck for new content)

**C. Register search tools:**
```python
from app.tools.youtube.search import search_videos, search_channels

@mcp.tool
@cache.cached(namespace="youtube.search")
async def _search_videos(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search for YouTube videos..."""
    results = await search_videos(query, max_results)
    return results  # decorator wraps in cache response

@mcp.tool
@cache.cached(namespace="youtube.search")
async def _search_channels(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search for YouTube channels..."""
    results = await search_channels(query, max_results)
    return results  # decorator wraps in cache response
```

**D. Remove demo tools:**
- Remove `hello` tool registration
- Remove `_generate_items` tool registration
- Keep cache admin tools for maintenance

### 3. Update Package Exports (`app/tools/youtube/__init__.py`)

Add to exports:
```python
from app.tools.youtube.search import (
    search_channels,
    search_videos,
)

__all__ = [
    # ... existing exports ...
    "search_channels",
    "search_videos",
]
```

### 4. Write Comprehensive Tests (`tests/test_youtube_search.py`)

**Test structure:**

**A. TestSearchVideos class:**
- `test_search_videos_success` - Mock API response, verify parsing
- `test_search_videos_empty_results` - Handle no results gracefully
- `test_search_videos_clamps_max_results` - Verify 1-50 clamping
- `test_search_videos_handles_quota_error` - Quota exceeded handling
- `test_search_videos_handles_auth_error` - Auth error handling
- `test_search_videos_handles_generic_error` - Other API errors
- `test_search_videos_validates_pydantic_models` - Model validation

**B. TestSearchChannels class:**
- Same structure as TestSearchVideos
- Test channel-specific parsing (channelId extraction)

**C. Integration-style tests (optional):**
- Test actual caching behavior with RefCache mock
- Verify cache key generation
- Test preview generation for large result sets

**Mock strategy:**
- Mock `get_youtube_service()` to return mock client
- Mock `youtube.search().list().execute()` with sample responses
- Use known YouTube API response structures
- Test both success and error paths

### 5. Documentation Updates

**A. Update TOOLS.md (or add inline docstrings):**
- Tool signatures with parameters
- Example queries ("NixOS tutorials", "vimjoyer")
- Cache behavior explanation
- API quota considerations

**B. Update server instructions:**
- How to use search tools
- When to use videos vs channels
- Caching strategy (6h refresh)

---

## Files to Create/Modify

### New Files:
1. `app/tools/youtube/search.py` (~150-200 lines)
   - _search_videos_impl
   - search_videos
   - _search_channels_impl
   - search_channels

2. `tests/test_youtube_search.py` (~300-400 lines)
   - TestSearchVideos class (7 tests)
   - TestSearchChannels class (7 tests)
   - Total: ~14 tests

### Modified Files:
1. `app/tools/youtube/__init__.py`
   - Add search function exports
   - Update __all__

2. `app/server.py`
   - Import search functions
   - Register search tools with @cache.cached
   - Update server instructions
   - Remove demo tools

---

## Technical Decisions & Tradeoffs

### Decision 1: Cache Namespace Strategy
**Choice:** Use `youtube.search` namespace with 6h TTL

**Rationale:**
- ✅ Search results are semi-volatile (new videos published constantly)
- ✅ 6h balances freshness vs API quota conservation
- ✅ Separate namespace prevents cache pollution from other data types
- ✅ User can always bypass cache by changing query slightly
- ❌ May return slightly stale results (acceptable tradeoff)

**Alternatives:**
- No caching: ❌ Wastes API quota
- 24h TTL: ❌ Too stale for rapidly changing content
- 1h TTL: ❌ Too aggressive, wastes quota for repeated searches

### Decision 2: Return Type (Pydantic Models vs Dicts)
**Choice:** Internal functions return Pydantic models, convert to dict for MCP

**Rationale:**
- ✅ Type safety during parsing (catch malformed API responses)
- ✅ Validation ensures data quality
- ✅ Clear schema for consumers
- ✅ MCP requires dict return type for @cache.cached decorator
- ❌ Small serialization overhead (negligible for I/O-bound operations)

**Alternatives:**
- Return dicts directly: ❌ Loses type safety
- Return models from MCP: ❌ Incompatible with @cache.cached decorator

### Decision 3: Max Results Clamping
**Choice:** Clamp max_results between 1-50 (match reference implementation)

**Rationale:**
- ✅ YouTube API enforces max 50 results per request
- ✅ Prevents API errors from invalid values
- ✅ Predictable behavior
- ❌ User doesn't get error for >50 (but documented in docstring)

**Alternatives:**
- Raise error for >50: ❌ Less user-friendly
- No limit: ❌ API will error

### Decision 4: Async vs Sync
**Choice:** Async functions throughout

**Rationale:**
- ✅ Consistent with server.py patterns
- ✅ FastMCP supports async tools
- ✅ Future-proof for concurrent operations
- ❌ googleapiclient is sync (but wrapped in async context is fine)

**Note:** googleapiclient.discovery.build is synchronous but wrapping in async function is acceptable pattern.

### Decision 5: Error Handling Strategy
**Choice:** Use handle_youtube_api_error() from client module

**Rationale:**
- ✅ Consistent error handling across all tools
- ✅ Specific exception types for quota/auth/notfound
- ✅ Helpful error messages for users
- ✅ DRY principle

---

## Testing Strategy

### Unit Tests (tests/test_youtube_search.py):
- **Mock all YouTube API calls** - No real API requests
- **Test success paths** - Valid responses parsed correctly
- **Test validation** - max_results clamping, input validation
- **Test error paths** - Quota, auth, generic errors handled correctly
- **Test edge cases** - Empty results, missing fields, malformed responses
- **Pydantic validation** - Models reject invalid data

### Manual Testing (after implementation):
1. Set YOUTUBE_API_KEY in environment
2. Run server: `uv run yt-mcp stdio`
3. Test search_videos with Claude/Zed:
   - "Search for NixOS tutorials"
   - "Search for vimjoyer videos about garbage collection"
4. Verify caching: repeat same query, should be instant (cache hit)
5. Test search_channels:
   - "Search for NixOS channels"
   - "Find vimjoyer channel"

### Integration Testing (future):
- Test with real YouTube API (separate test suite)
- Use known stable content for predictable results
- Verify cache behavior with actual RefCache

---

## Success Criteria

- [x] `app/tools/youtube/search.py` created with 2 search functions
- [x] Both functions properly validated (max_results clamping)
- [x] Both functions use Pydantic models for parsing
- [x] Both functions handle HttpError via client utilities
- [x] Integrated into `app/server.py` with @cache.cached decorator
- [x] Demo tools removed from server
- [x] Server instructions updated for YouTube search
- [x] Package exports updated in `__init__.py`
- [x] 19 unit tests written and passing (exceeded goal!)
- [x] All tests pass: `pytest tests/test_youtube_search.py -v`
- [x] Linting passes: `ruff check . --fix && ruff format .`
- [ ] Manual test successful: Search for "vimjoyer nix" returns results (requires API key)
- [ ] Caching verified: Repeat query is instant (cache hit) (requires API key)

---

## API Quota Considerations

**YouTube Data API v3 Quota:**
- Daily quota: 10,000 units (default)
- Search operation cost: **100 units** per request
- With 10,000 quota: ~100 searches per day
- Caching with 6h TTL reduces quota usage by ~4x

**Mitigation strategies:**
- ✅ Aggressive caching (6h TTL)
- ✅ Clear error messages when quota exceeded
- ✅ Documentation about quota limits
- 🔮 Future: Consider quota monitoring/alerting

---

## Risks & Mitigation

### Risk 1: API Response Schema Changes
**Impact:** Medium - Parsing could break if YouTube changes response format
**Probability:** Low - YouTube API is stable
**Mitigation:**
- Pydantic validation will catch schema changes early
- Comprehensive tests with various response shapes
- Monitor for API version deprecations

### Risk 2: Cache Key Collisions
**Impact:** Low - Different queries returning same cached results
**Probability:** Very Low - RefCache uses function args as cache key
**Mitigation:**
- RefCache automatically generates keys from (namespace, function_name, args)
- Test with similar queries to verify isolation

### Risk 3: Large Result Sets
**Impact:** Low - Memory/context issues with 50 results
**Probability:** Low - 50 results is manageable
**Mitigation:**
- RefCache preview generation (SAMPLE strategy)
- Each result ~200 bytes, 50 results = ~10KB (acceptable)
- Document recommended max_results range (5-10 for typical use)

---

## Dependencies

**Existing:**
- ✅ `google-api-python-client==2.187.0`
- ✅ `app/tools/youtube/client.py` (get_youtube_service, error handling)
- ✅ `app/tools/youtube/models.py` (VideoSearchResult, ChannelSearchResult)
- ✅ `app/server.py` (RefCache setup, TracedRefCache)

**No new dependencies needed.**

---

## Next Steps After Approval

1. **Create search.py module** with implementation
2. **Write unit tests** (TDD approach - tests first or alongside)
3. **Integrate with server.py** - register tools, remove demos
4. **Update exports** in __init__.py
5. **Run tests** - verify all pass
6. **Run linting** - ensure clean code
7. **Manual testing** - verify with real API key
8. **Document completion** in this scratchpad

---

## Questions for User

1. **Cache TTL:** Is 6h appropriate for search results, or prefer different value?
2. **Default max_results:** Keep at 5 (like reference), or change to different default?
3. **Demo tools:** Remove all demo tools (hello, generate_items) or keep some for examples?
4. **Preview strategy:** Use SAMPLE (show subset) for search results, or TRUNCATE (show first N)?

---

## Implementation Completed

### Files Created
1. **`app/tools/youtube/search.py`** (207 lines)
   - `search_videos()`: Search for videos with clamping, error handling, Pydantic models
   - `search_channels()`: Search for channels with same patterns
   - Both functions async, return `list[dict[str, Any]]` for @cache.cached compatibility

2. **`tests/test_youtube_search.py`** (486 lines)
   - 19 comprehensive tests (exceeded 14 test goal)
   - TestSearchVideos: 10 tests (success, empty, clamping, errors, edge cases)
   - TestSearchChannels: 9 tests (same coverage)
   - Mock fixtures for API responses and error scenarios

### Files Modified
1. **`app/tools/youtube/__init__.py`**
   - Added `search_videos` and `search_channels` to exports

2. **`app/server.py`**
   - Removed demo tools (hello, generate_items, store_secret, compute_with_secret)
   - Added `_search_videos` and `_search_channels` MCP tools with @cache.cached
   - Updated server name to "YouTube MCP Server"
   - Updated instructions to reflect YouTube functionality
   - Changed default TTL to 6 hours (21600s) for search caching

3. **`app/prompts/__init__.py`**
   - Updated TEMPLATE_GUIDE to reflect YouTube MCP server
   - Removed demo tool references
   - Added YouTube-specific examples and API quota notes

4. **`tests/test_server.py`**
   - Removed demo tool test classes (TestHelloTool, TestGenerateItems, TestStoreSecret, TestComputeWithSecret, TestPydanticModels)
   - Updated server name assertion to "YouTube MCP Server"
   - Updated instructions tests to check for "youtube" and "search"
   - Updated template guide tests
   - Reduced from 146 tests to 119 tests (removed 27 demo tool tests)

### Test Results
```
119 passed, 9 warnings in 1.25s
All checks passed! (ruff)
```

### Key Implementation Details

**Search Functions:**
- Both functions clamp `max_results` to 1-50 (YouTube API limit)
- Use `get_youtube_service()` for authenticated client
- Parse API response into Pydantic models, then convert to dicts
- Handle thumbnail fallback (high → medium → default)
- Handle missing description fields gracefully (empty string)
- Use `handle_youtube_api_error()` for consistent error handling

**Server Integration:**
- Tools registered with `@cache.cached(namespace="youtube.search")`
- 6-hour TTL balances freshness vs API quota conservation
- Return type `dict[str, Any]` required for MCP compatibility with @cache.cached
- Server instructions updated with quota notes (100 units per search, 10k daily)

**Testing Strategy:**
- Mock `get_youtube_service()` to avoid real API calls
- Test success paths with sample API responses
- Test validation (clamping to 1-50)
- Test all error types (quota, auth, 404, generic)
- Test edge cases (missing fields, empty results, thumbnail fallback)
- Fixed HttpUrl serialization issue (convert to string for assertions)

### Notes

- Search is the foundation for the "Find Vimjoyer's video" use case
- After search tools work, next tasks build on top (metadata, transcripts)
- Task was relatively straightforward - mainly adapting reference implementation to mcp-refcache patterns
- Demo tools successfully removed, server now YouTube-focused
- All automated tests pass - manual testing with real API key deferred to Task 04 (Zed integration)

### Lessons Learned

1. **Pydantic HttpUrl serialization**: `model_dump()` keeps HttpUrl as object, not string
   - Solution: Convert to string in test assertions with `str(result["thumbnail"])`

2. **Test cleanup**: Removing demo tools required updating multiple test files
   - Removed 27 tests related to demo functionality
   - Updated remaining tests to reflect YouTube focus

3. **Async patterns work well**: googleapiclient is sync, but wrapping in async functions is fine
   - No blocking issues, clean integration with FastMCP async tools

---

## References

- Reference implementation: `.agent/youtube_toolset.py` (lines 60-230)
- YouTube Data API: https://developers.google.com/youtube/v3/docs/search/list
- Current server patterns: `app/server.py` (_generate_items example)
- Current tool patterns: `app/tools/demo.py` (hello, generate_items)
