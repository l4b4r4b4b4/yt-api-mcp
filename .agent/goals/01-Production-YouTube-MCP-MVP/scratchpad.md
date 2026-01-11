# Goal 01: Production YouTube MCP MVP

**Status:** 🟡 In Progress (10/13 tasks complete - 77% done)
**Priority:** Critical
**Created:** 2025-01-08
**Last Updated:** 2025-01-09 (Task 10 Complete)

---

## Overview

Transform the reference YouTube toolset implementation (`.agent/youtube_toolset.py`) into a production-ready MCP server integrated with the mcp-refcache architecture. This server will provide comprehensive YouTube search, video details, transcript access, and semantic search capabilities for AI agents.

---

## Objectives

### Primary Goals
1. ✅ **Feature Parity** - Implement all tools from reference implementation
2. ✅ **RefCache Integration** - Leverage mcp-refcache for intelligent caching
3. ✅ **Production Quality** - Type-safe, tested, documented, secure
4. ✅ **Performance** - Efficient API usage, smart caching strategies
5. ✅ **Usability** - Clear tool descriptions, good error handling

### Success Criteria
- [x] All 16 YouTube tools implemented and working (12 original + 4 live streaming)
- [ ] 4-tier caching strategy (content/api/comments/search) operational
- [ ] ≥73% test coverage maintained
- [ ] All linting passes (ruff check + format)
- [ ] Type checking passes (mypy)
- [ ] README accurately reflects functionality
- [ ] Successfully tested with Zed/Claude Desktop
- [ ] Can find and analyze Vimjoyer's Nix GC video (practical validation)

---

## Reference Implementation Analysis

**Source:** `.agent/youtube_toolset.py` (744 lines)

### Existing Tools (13 total)
1. `search_videos` - Search YouTube by query
2. `get_video_details` - Detailed video metadata
3. `search_channels` - Find channels by query
4. `get_channel_info` - Channel statistics and metadata
5. `get_video_comments` - Top comments with engagement data
6. `get_video_transcript_preview` - First N chars of transcript
7. `get_full_transcript` - Complete transcript with timestamps
8. `_get_full_transcript` - Internal helper (cached)
9. `list_available_transcripts` - Available transcript languages
10. `get_transcript_chunk` - Paginated transcript access
11. `clear_cache` - Cache management
12. `get_cache_stats` - Cache statistics
13. `get_youtube_service` - API client factory

### Existing Cache Strategy
- **content_cache** - Permanent, 5000 entries (transcripts)
- **api_cache** - 24h TTL, 1000 entries (video/channel data)
- **comment_cache** - 12h TTL, 500 entries (comments)
- **search_cache** - 6h TTL, 300 entries (search results)

### Dependencies Used
- `googleapiclient` - YouTube Data API v3
- `youtube-transcript-api` - Transcript extraction
- `mcp.server.fastmcp.FastMCP` - MCP framework (old)
- Custom `ToolsetCache` implementation

---

## Architecture Design

### Project Structure
```
app/
├── __init__.py
├── __main__.py          # CLI entry point
├── server.py            # Main MCP server (integrate YouTube tools)
├── config.py            # Environment configuration
├── tracing.py           # Langfuse tracing wrapper
└── tools/
    ├── __init__.py
    ├── youtube/         # NEW: YouTube-specific tools
    │   ├── __init__.py
    │   ├── client.py    # YouTube API client with auth
    │   ├── search.py    # Search tools (videos, channels)
    │   ├── metadata.py  # Video/channel details
    │   ├── transcripts.py  # Transcript tools
    │   ├── comments.py  # Comment tools
    │   └── models.py    # Pydantic models for responses
    └── cache_admin.py   # Existing admin tools
```

### Cache Design (RefCache Integration)

**Namespaces:**
- `youtube.content` - Deterministic, permanent (transcripts)
- `youtube.api` - 24h TTL, 1000 max (video/channel metadata)
- `youtube.comments` - 12h TTL, 500 max (comments)
- `youtube.search` - 6h TTL, 300 max (search results)

**Preview Strategies:**
- **Transcripts** - TRUNCATE (show first 2000 chars)
- **Search Results** - SAMPLE (show subset of results)
- **Comments** - PAGINATE (page through results)
- **Metadata** - No preview (small enough)

### API Design

**Configuration (Environment Variables):**
```bash
YOUTUBE_API_KEY          # Required - YouTube Data API v3 key
LANGFUSE_PUBLIC_KEY      # Optional - Tracing
LANGFUSE_SECRET_KEY      # Optional - Tracing
LANGFUSE_HOST            # Optional - Tracing endpoint
```

**Tool Signatures:**
All tools follow FastMCP + mcp-refcache patterns:
- Type-annotated parameters and returns
- Pydantic models for structured data
- `@cache.cached(namespace=...)` decorator where appropriate
- Docstrings with Args/Returns/Raises sections
- Error handling with clear messages

---

## Implementation Plan

### Task Breakdown

#### Task 01: Project Setup & Dependencies ✅
- [x] Add YouTube dependencies to pyproject.toml
  - `google-api-python-client>=2.0.0` → installed v2.187.0
  - `youtube-transcript-api>=0.6.0` → installed v1.2.3
- [x] Update config.py with YOUTUBE_API_KEY
- [x] Create `app/tools/youtube/` module structure
- [x] Run `uv sync` to install dependencies
- [x] Verify linting passes

#### Task 02: Core YouTube Client ✅
- [x] Create `app/tools/youtube/models.py` with Pydantic models
  - `VideoSearchResult`
  - `VideoDetails`
  - `ChannelSearchResult`
  - `ChannelInfo`
  - `CommentData`
  - `TranscriptEntry`
  - `TranscriptInfo`
  - `FullTranscript`
  - `TranscriptPreview`
  - `TranscriptChunk`
  - `AvailableTranscripts`
- [x] Create `app/tools/youtube/client.py`
  - `get_youtube_service()` - API client factory
  - Error handling for API quota/auth issues
  - Custom exception classes
  - URL/ID extraction utilities
- [x] Write unit tests for client initialization
- [x] All tests pass (26/26)
- [x] Linting passes

#### Task 03: Search Tools ✅
- [x] Implement `app/tools/youtube/search.py`
  - `search_videos(query, max_results)` with caching
  - `search_channels(query, max_results)` with caching
- [x] Use `youtube.search` namespace (6h TTL)
- [x] Add preview generation for result lists
- [x] Write tests for search functionality (19 tests, all passing)
- [x] Remove demo tools from server
- [x] Update server instructions and prompts
- [x] All tests pass (119/119), linting passes

#### Task 04: Local Dev Testing & Integration ✅
- [x] Configure Zed to use local yt-mcp server
- [x] Set YOUTUBE_API_KEY in environment (.envrc.local)
- [x] Start local server: `uv run yt-mcp stdio`
- [x] Test search_videos in Zed chat (NixOS tutorials query)
- [x] Test search_channels in Zed chat
- [x] Verify caching behavior (instant cache hits confirmed)
- [ ] Practical test: "Search for vimjoyer nix garbage collection" (skipped for speed)
- [x] Document issues/improvements needed
  - Fixed: .envrc infinite loop (removed `use flake`)
  - Fixed: Environment variable loading in devShell
  - Created: ENV_SETUP.md, .env.example, .envrc.local.example
  - Validated: Cache working with ref_id reuse

#### Task 05: Metadata Tools ✅
- [x] Implement `app/tools/youtube/metadata.py`
  - `get_video_details(video_id)` with caching
  - `get_channel_info(channel_id)` with caching
- [x] Use `youtube.api` namespace (24h TTL)
- [x] Write tests for metadata retrieval (14 tests)
- [x] Local dev test in Zed (NixOS video nLwbNhSxLd4)
- [x] All tests pass (119/119), linting passes

#### Task 06: Transcript Tools ✅
- [x] Implement `app/tools/youtube/transcripts.py` (474 lines)
  - `list_available_transcripts(video_id)`
  - `get_video_transcript_preview(video_id, language, max_chars)`
  - `get_full_transcript(video_id, language)` with caching
  - `get_transcript_chunk(video_id, start_index, chunk_size, language)`
- [x] Use `youtube.content` namespace (permanent/deterministic)
- [x] Handle errors: no transcript, language not available
- [x] Write comprehensive tests with mock transcript data (26 tests)
- [x] Fixed YouTubeTranscriptApi to use instance method: `YouTubeTranscriptApi().list()`
- [x] Internal helper pattern (_fetch_transcript_data)
- [x] All tests pass (145/145), linting passes
- [x] Removed underscore prefixes from all tool names for cleaner API
- [x] Local dev test in Zed - All 4 tools validated successfully! ✅
  - Video: a67Sv4Mbxmc (Ultimate NixOS Guide)
  - list_available_transcripts: ✅ Returned ["en"]
  - get_video_transcript_preview: ✅ 500 chars preview (16,368 total)
  - get_full_transcript: ✅ 13,448 TOKENS (RefCache token mode working!)
  - get_transcript_chunk: ✅ First 10 entries with timestamps
  - Caching verified: ✅ Instant cache hits on repeat calls

#### Task 07: Comment Tools ✅
- [x] Implement `app/tools/youtube/comments.py` (99 lines)
  - `get_video_comments(video_id, max_results)` with caching
- [x] Use `youtube.comments` namespace (5 min TTL - changed for trending videos)
- [x] Handle disabled comments gracefully (returns empty list)
- [x] Write tests for comment retrieval (9 tests, all passing)
- [x] Local dev test in Zed - Works perfectly! ✅
  - Video: nLwbNhSxLd4 (NixOS Ultimate Guide)
  - Retrieved 10 comments with engagement metrics
  - 5-minute cache verified (instant cache hits)
  - Graceful handling of all edge cases

#### Task 08: Live Streaming Features ✅
- [x] Implement `app/tools/youtube/live.py` (371 lines)
  - `is_live(video_id)` - Check live status (30s cache)
  - `get_live_chat_id(video_id)` - Get chat ID (5m cache)
  - `get_live_chat_messages(video_id, max_results, page_token)` - Get chat with pagination (30s cache)
- [x] Add `search_live_videos(query, max_results)` to search.py
- [x] Add LiveStatus, LiveChatMessage, LiveChatResponse models
- [x] Use very short cache for real-time data (30s for status/chat)
- [x] Handle video not live, chat disabled gracefully
- [x] Write tests for live streaming (24 tests: 19 live + 5 search)
- [x] Document MCP polling limitations clearly in docstrings
- [x] All tests pass (178/178), linting passes
- [x] Local dev test in Zed - All 4 tools validated successfully! ✅
  - Video: e7AqeMm52LI (LiveNOW from FOX live stream)
  - search_live_videos: ✅ Found 5 currently broadcasting streams
  - is_live: ✅ Returned is_live=true, 2,160 viewers, active_live_chat_id
  - get_live_chat_messages: ✅ Retrieved 20 live chat messages with authors, timestamps
  - Pagination with page_token: ✅ Second call returned only NEW messages (no duplicates)
  - Cache hit verification: ✅ Instant responses on repeat calls (same ref_id)
  - Error handling: ✅ Gracefully handled regular video (is_live=false)

#### Task 09: Final Server Polish ✅
- [x] Review all registered tools in `app/server.py`
- [x] Update server instructions with all 16 YouTube tools
- [x] Verify all cache namespaces configured correctly
- [x] Remove any remaining demo/placeholder tools
- [x] Clean up any unused imports or code
- [x] Verify error handling consistency across all tools
- [x] Run full test suite: `pytest --cov` (178 tests passing, 76% coverage)
- [x] Run linting: `ruff check . --fix && ruff format .` (all checks passed)
- [x] Verify ≥73% coverage maintained (76% exceeds requirement)
- [x] All 16 YouTube tools verified with correct cache TTLs
- [x] Server instructions comprehensive and accurate
- [x] Code quality excellent, ready for Docker build

#### Task 10: Build & Test Local Docker ✅
- [x] Update Dockerfile if needed (ensure uv, dependencies correct)
- [x] Update docker-compose.yml with YouTube API key configuration
- [x] Build base image: `fastmcp-base:latest` (290MB)
- [x] Build production image: `yt-mcp:latest` (229MB total)
- [x] Fixed missing YouTube dependencies in base image (rebuilt with google-api-python-client)
- [x] Test Docker image locally with docker-compose
- [x] Verify all tools work in containerized environment (tested search, metadata, transcripts, live)
- [x] Test cache persistence in Docker (cache hits working correctly)
- [x] Document Docker-specific configuration
- [x] Fixed docker-compose.yml dev service port conflict (8001 vs 8000)
- [x] All 16 YouTube tools validated in Docker via Zed MCP client

#### Task 11: Documentation 🟠
- [x] Update README.md with:
  - Complete YouTube API key setup instructions
  - All 16 tool descriptions with examples
  - Caching strategy explanation (4-tier + live streaming)
  - Quota limits and best practices
  - Docker usage instructions
- [x] Update CHANGELOG.md for version 0.0.0
- [x] Add examples for common use cases:
  - Finding specific videos by creator
  - Analyzing transcripts
  - Monitoring live streams and chat
  - Checking channel activity
- [x] Verify all code examples are accurate and tested

#### Task 12: Publish to PyPI & GHCR
- [ ] Verify version is 0.0.0 in pyproject.toml and `__version__`
- [ ] Test PyPI publishing workflow locally (dry-run if possible)
- [ ] Publish to PyPI: Version 0.0.0
- [ ] Test GHCR publishing workflow
- [ ] Publish Docker image to GHCR: Version 0.0.0
- [ ] Verify both published artifacts are accessible
- [ ] Git tag release: `v0.0.0`
- [ ] Create GitHub release with CHANGELOG

#### Task 13: Test Published Versions
- [ ] Install from PyPI in clean environment: `uvx yt-mcp` or `pip install yt-mcp`
- [ ] Test published PyPI package with real YouTube API
- [ ] Pull Docker image from GHCR: `docker pull ghcr.io/[user]/yt-mcp:0.0.0`
- [ ] Test published Docker image with docker-compose
- [ ] Run **practical validation:** Find Vimjoyer's Nix GC video using published version
- [ ] Verify all 16 tools work end-to-end in published versions
- [ ] Document any issues found for 0.0.1 iteration
- [ ] Update project status to 🟢 Complete or 🟠 Needs 0.0.1 fixes
- [ ] If 0.0.0 works well: Plan 0.0.1 fixes, then 0.1.0 after stabilization



---

## Technical Decisions & Tradeoffs

### Decision 1: YouTube Data API vs Scraping
**Choice:** Use official YouTube Data API v3
**Rationale:**
- ✅ Stable, reliable, well-documented
- ✅ Official support, less likely to break
- ✅ Handles auth, rate limiting, pagination
- ❌ Requires API key (easy to get, free tier generous)
- ❌ Daily quota limits (10,000 units/day default)

**Alternative:** Web scraping (yt-dlp, youtube-search-python)
- ❌ Fragile, breaks when YouTube changes HTML
- ❌ Against YouTube TOS
- ❌ No official support
- ✅ No API key needed

### Decision 2: Transcript Library
**Choice:** `youtube-transcript-api`
**Rationale:**
- ✅ Actively maintained, 2.6k+ stars
- ✅ Handles auto-generated and manual captions
- ✅ Multi-language support
- ✅ No API key needed (uses YouTube's internal API)
- ❌ Can break if YouTube changes internals

**Alternative:** YouTube Data API captions endpoint
- ❌ Requires OAuth2 (complex for MCP servers)
- ❌ Only for video owners
- ✅ Official API

### Decision 3: Cache Namespace Strategy
**Choice:** 4 separate namespaces by data volatility
**Rationale:**
- ✅ Optimal TTL per data type
- ✅ Prevents cache pollution
- ✅ Clear separation of concerns
- ✅ Independent size limits
- ❌ Slightly more complex

**Alternative:** Single namespace with uniform TTL
- ✅ Simpler
- ❌ Suboptimal caching (either too aggressive or too conservative)

### Decision 4: Tool Granularity
**Choice:** Keep fine-grained tools (preview + full + chunk for transcripts)
**Rationale:**
- ✅ Agents can choose optimal data size
- ✅ Better context window management
- ✅ Faster iteration for exploration
- ❌ More tools to document/maintain

**Alternative:** Single "get_transcript" with optional parameters
- ✅ Simpler API surface
- ❌ Harder for agents to discover optimal usage

---

## Dependencies & Requirements

### Python Packages (Added)
```toml
dependencies = [
    "google-api-python-client>=2.0.0",
    "youtube-transcript-api>=0.6.0",
    # ... existing dependencies
]
```

### External Services
- **YouTube Data API v3** - Free tier: 10,000 quota units/day
  - Search: 100 units per request
  - Video details: 1 unit per request
  - Channel details: 1 unit per request
  - Comments: 1 unit per request
- **Langfuse** - Optional tracing (existing)

### Environment Setup
```bash
# Required
export YOUTUBE_API_KEY="AIza..."

# Optional (existing)
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
```

---

## Testing Strategy

### Unit Tests
- Mock YouTube API responses (don't hit real API in tests)
- Test error handling (quota exceeded, video not found, etc.)
- Test Pydantic model validation
- Test cache key generation

### Integration Tests
- Test with real API (separate test suite, requires API key)
- Use known stable videos for predictable results
- Test transcript retrieval with various languages
- Test pagination and chunking

### Manual Validation
- Install in Zed configuration
- Test conversation flow: search → details → transcript
- **Practical test:** "Find Vimjoyer's video about Nix garbage collection tool that keeps last N generations"
  - Should search videos
  - Should get transcripts
  - Should identify correct video from transcript content

---

## Risks & Mitigation

### Risk 1: API Quota Exhaustion
**Impact:** High - Server stops working after quota exceeded
**Probability:** Medium - Heavy usage can exhaust quota
**Mitigation:**
- Aggressive caching (reduces API calls)
- Clear documentation about quota limits
- Consider quota monitoring/alerting
- Fallback error messages with quota reset time

### Risk 2: Transcript API Breakage
**Impact:** High - Transcript features stop working
**Probability:** Low-Medium - Internal API can change
**Mitigation:**
- Graceful error handling
- Clear error messages to users
- Monitor `youtube-transcript-api` issues/updates
- Document alternative: use official API with OAuth2

### Risk 3: Complex Tool Discovery
**Impact:** Medium - Agents may not use tools optimally
**Probability:** Medium - 12+ tools is a lot
**Mitigation:**
- Excellent tool descriptions
- Clear parameter documentation
- Usage examples in README
- Server-level instructions guide tool selection

### Risk 4: Large Transcript Context Pollution
**Impact:** High - Large transcripts fill context window
**Probability:** Low - RefCache mitigates this
**Mitigation:**
- Reference-based returns for large data
- Preview generation (first 2k chars)
- Chunked access with pagination
- Clear documentation on when to use preview vs full

---

## Success Metrics

### Functional Metrics
- ✅ All 12 tools working without errors
- ✅ Can successfully find and analyze Vimjoyer video
- ✅ Transcripts retrieved in <2 seconds (cached)
- ✅ Search results returned in <1 second

### Quality Metrics
- ✅ Test coverage ≥73%
- ✅ Zero linting errors
- ✅ Zero type checking errors
- ✅ All public APIs documented

### Usability Metrics
- ✅ Setup takes <5 minutes (API key + install)
- ✅ Works in Zed and Claude Desktop
- ✅ Clear error messages for common issues
- ✅ README examples are copy-paste ready

---

## Next Steps

1. **Immediate:** Start with Task 01 (dependencies and project setup)
2. **Phase 1:** Core infrastructure (Tasks 01-02)
3. **Phase 2:** Feature implementation (Tasks 03-06)
4. **Phase 3:** Integration and polish (Tasks 07-08)
5. **Phase 4:** Validation and deployment (Tasks 09-10)

---

## Notes

- Reference implementation uses old FastMCP patterns - need to update to current API
- Consider adding semantic search capability in future (beyond MVP)
- YouTube API key is free and easy to get - not a major barrier
- RefCache previews will be crucial for large transcripts
- Test with real-world use case (Vimjoyer search) ensures practical value

---

## Related Goals

- None yet (this is Goal 01)

---

## References

- [YouTube Data API v3 Docs](https://developers.google.com/youtube/v3)
- [youtube-transcript-api GitHub](https://github.com/jdepoix/youtube-transcript-api)
- [mcp-refcache Documentation](https://github.com/l4b4r4b4b4/mcp-refcache)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- Reference implementation: `.agent/youtube_toolset.py`
