# Task 04: Local Dev Testing & Integration

**Status:** 🟢 Complete
**Created:** 2025-01-08
**Updated:** 2025-01-08
**Completed:** 2025-01-08
**Dependencies:** Task 03 (Search Tools)

---

## Objective

Configure local development environment to test the YouTube MCP server in Zed, verify search functionality works end-to-end with real YouTube API, validate caching behavior, and establish a baseline for testing all future tools.

---

## Scope

### In Scope
- Update `.zed/settings.json` with `YOUTUBE_API_KEY`
- Test search_videos and search_channels with real API
- Verify caching behavior (6h TTL for search namespace)
- Validate RefCache reference-based returns
- Practical validation: Find Vimjoyer's Nix GC videos
- Document API quota usage patterns
- Verify error handling (missing API key, quota exceeded, invalid queries)

### Out of Scope
- Testing tools not yet implemented (transcripts, comments, etc.)
- Performance benchmarking (that's Task 10)
- Langfuse tracing configuration (optional, not required)

---

## Current State

### Zed Configuration Analysis
The `.zed/settings.json` already has a `yt-mcp` context server configured:

```json
"yt-mcp": {
  "command": "uvx",
  "args": ["yt-mcp", "stdio"],
  "env": {
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
    "LANGFUSE_HOST": "https://cloud.langfuse.com"
  }
}
```

**Issues:**
1. ❌ Missing `YOUTUBE_API_KEY` in env
2. ❌ Using `uvx` instead of local `uv run` (won't use local code changes)
3. ⚠️ No `cwd` specified (may run from wrong directory)

**Required Changes:**
- Add `YOUTUBE_API_KEY` to env
- Change command to use local development setup
- Add `cwd` to point to project root

---

## Implementation Plan

### Phase 1: Configuration (5 min)

**Step 1.1: Get YouTube API Key**
- Go to https://console.cloud.google.com/apis/credentials
- Create API key or use existing one
- Enable YouTube Data API v3 if not already enabled
- Copy API key for next step

**Step 1.2: Update Zed Settings**
Edit `.zed/settings.json` to update the `yt-mcp` context server:

```json
"yt-mcp": {
  "command": "uv",
  "args": ["run", "yt-mcp", "stdio"],
  "cwd": "/home/lukes/code/github.com/l4b4r4b4b4/mcp-refcache/examples/yt-mcp",
  "env": {
    "YOUTUBE_API_KEY": "YOUR_API_KEY_HERE",
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
    "LANGFUSE_HOST": "https://cloud.langfuse.com"
  }
}
```

**Step 1.3: Verify Installation**
```bash
cd /home/lukes/code/github.com/l4b4r4b4b4/mcp-refcache/examples/yt-mcp
uv sync  # Ensure all dependencies installed
uv run pytest  # Verify tests still pass (119 tests)
```

**Step 1.4: Restart Zed**
- Save `.zed/settings.json`
- Fully quit and restart Zed (not just reload window)
- This ensures MCP server connections are refreshed

---

### Phase 2: Basic Connectivity Testing (5 min)

**Test 2.1: Server Connection**
In Zed chat, ask:
```
"What MCP servers are currently connected?"
```

**Expected:** Should list `yt-mcp` among connected servers.

**Test 2.2: Available Tools**
Ask:
```
"What tools does the yt-mcp server provide?"
```

**Expected:** Should list:
- search_videos
- search_channels
- get_cached_result
- health_check
- enable_test_context
- set_test_context
- reset_test_context
- get_trace_info

---

### Phase 3: Search Tools Testing (10 min)

**Test 3.1: Simple Video Search**
```
"Search YouTube for 'NixOS tutorials' and show me the top 3 results"
```

**Expected Results:**
- Returns 3 video results
- Each result has: title, description, video_id, url, thumbnail, channel_title, published_at
- Response time: 2-5 seconds (fresh API call)
- Should see YouTube API call in logs

**Validation:**
- [ ] Results contain relevant NixOS tutorial videos
- [ ] All fields are populated correctly
- [ ] URLs are valid (format: `https://www.youtube.com/watch?v=VIDEO_ID`)
- [ ] Thumbnails are valid image URLs

**Test 3.2: Channel Search**
```
"Search YouTube for channels about 'Vimjoyer'"
```

**Expected Results:**
- Returns channel results (likely Vimjoyer's channel as #1)
- Each result has: title, description, channel_id, url, thumbnail, published_at
- Response time: 2-5 seconds (fresh API call)

**Validation:**
- [ ] Vimjoyer's channel appears in results
- [ ] Channel description is present
- [ ] URL format: `https://www.youtube.com/channel/CHANNEL_ID`
- [ ] Published date is reasonable

**Test 3.3: Max Results Parameter**
```
"Search YouTube for 'Nix flakes' and give me exactly 10 results"
```

**Expected Results:**
- Returns exactly 10 video results
- Response time: 3-6 seconds (more results = more processing)

**Validation:**
- [ ] Count is exactly 10
- [ ] All 10 results are unique (different video_ids)
- [ ] Results are relevant to "Nix flakes"

---

### Phase 4: Caching Validation (10 min)

**Test 4.1: Cache Hit Detection**
Repeat the exact same query from Test 3.1:
```
"Search YouTube for 'NixOS tutorials' and show me the top 3 results"
```

**Expected Results:**
- Returns same results as Test 3.1
- **Response time: <100ms** (served from cache, no API call)
- Should see cache hit in logs (if verbose logging enabled)

**Validation:**
- [ ] Results identical to first query
- [ ] Response is nearly instantaneous
- [ ] No YouTube API call made (check quota usage)

**Test 4.2: Cache Namespace Isolation**
```
"Search YouTube channels for 'NixOS'"
```

Then immediately:
```
"Search YouTube videos for 'NixOS'"
```

**Expected Results:**
- Both use `youtube.search` namespace
- Both should be fresh API calls (different search types)
- Cache keys should be different (search_videos vs search_channels)

**Validation:**
- [ ] Both queries return appropriate results (channels vs videos)
- [ ] Both take 2-5 seconds (fresh API calls)
- [ ] Cache stores them separately

**Test 4.3: Cache Expiry (Optional - requires 6h wait)**
This can be validated by checking cache stats, not by waiting:

```
"What are the cache statistics for the yt-mcp server?"
```

**Expected Results:**
- Shows `youtube.search` namespace stats
- TTL should be 21600 seconds (6 hours)
- Should see hit rate increasing after repeated queries

---

### Phase 5: Practical Use Case Testing (10 min)

**Test 5.1: Find Vimjoyer's Nix GC Video**
This is the real-world use case from the README:

```
"Search YouTube for videos by Vimjoyer about Nix garbage collection,
specifically about tools that keep only the last N generations"
```

**Expected Behavior:**
- Agent searches for "vimjoyer nix garbage collection generations"
- Returns relevant videos from Vimjoyer's channel
- Should find videos about `nix-collect-garbage` or similar tools

**Validation:**
- [ ] Returns Vimjoyer's videos (not random channels)
- [ ] Videos are about Nix garbage collection
- [ ] Results are actionable (can identify specific video about GC generations)

**Test 5.2: Channel Analysis Workflow**
```
"Find the 'NixOS' official channel and tell me how many subscribers they have"
```

**Expected Behavior:**
- Agent uses `search_channels("NixOS")`
- Returns NixOS channel info
- **NOTE:** Channel stats require `get_channel_info` tool (Task 05), so this test will partially fail
- But should still find the channel successfully

**Validation:**
- [ ] Finds NixOS channel in search results
- [ ] Provides channel_id and URL
- [ ] Agent acknowledges it needs `get_channel_info` for subscriber count (not yet implemented)

**Test 5.3: Invalid Query Handling**
```
"Search YouTube for 'zzzzzzzzzzzzzzzzzzzzzzz_nonexistent_query_12345'"
```

**Expected Behavior:**
- API call succeeds but returns 0 results
- No errors or crashes
- Agent reports no results found

**Validation:**
- [ ] Server doesn't crash
- [ ] Returns empty list or clear "no results" message
- [ ] No stack traces in logs

---

### Phase 6: Error Handling Testing (5 min)

**Test 6.1: Missing API Key (Simulated)**
This requires temporarily removing the API key from settings, which we won't do.
Instead, document expected behavior:

**Expected:** Server should fail to start or return clear error message about missing `YOUTUBE_API_KEY`.

**Test 6.2: Invalid Max Results**
```
"Search YouTube for 'NixOS' and give me 500 results"
```

**Expected Behavior:**
- Should cap at 50 results (YouTube API limit)
- Or return validation error from Pydantic model

**Validation:**
- [ ] Returns max 50 results OR clear error message
- [ ] No server crash

**Test 6.3: Empty Query**
```
"Search YouTube for '' (empty string)"
```

**Expected Behavior:**
- Validation error or returns generic results
- No server crash

**Validation:**
- [ ] Handles gracefully
- [ ] Clear error message if rejected

---

### Phase 7: API Quota Monitoring (5 min)

**Quota Tracking Setup:**

YouTube Data API v3 quotas:
- **Daily quota:** 10,000 units (default free tier)
- **Search cost:** 100 units per request
- **Expected usage:** ~100 searches per day max

**During Testing:**
Track quota usage in Google Cloud Console:
1. Go to https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas
2. Check "Queries per day" usage
3. Calculate: (units used) / 100 = number of searches

**Test 7.1: Quota Usage Calculation**
After completing Tests 3.1-5.3, calculate total API calls:

**Expected API Calls:**
- Test 3.1: 1 call (100 units)
- Test 3.2: 1 call (100 units)
- Test 3.3: 1 call (100 units)
- Test 4.1: 0 calls (cache hit)
- Test 4.2: 2 calls (200 units) - videos + channels
- Test 5.1: 1 call (100 units)
- Test 5.2: 1 call (100 units)
- Test 5.3: 1 call (100 units)
- Test 6.2: 1 call (100 units)
- Test 6.3: 1 call (100 units)

**Total:** ~10 API calls = 1,000 units (~10% of daily quota)

**Validation:**
- [ ] Actual usage matches expected (±1-2 calls for retries)
- [ ] Cache reduced calls by at least 1 (Test 4.1)
- [ ] Still have 90%+ quota remaining

**Test 7.2: Cache Effectiveness**
Calculate cache hit rate:
- Total queries sent to server: 13
- API calls made: ~10
- Cache hits: ~3
- **Hit rate:** 23% (will improve with more usage)

**Expected Improvement:**
With production usage, hit rate should reach 60-80% due to:
- Common queries repeated by multiple users
- 6-hour TTL keeps popular searches cached
- Same user repeating searches within session

---

## Success Criteria

### Critical (Must Pass)
- [x] Zed connects to yt-mcp server without errors
- [ ] `search_videos` returns valid YouTube video results
- [ ] `search_channels` returns valid YouTube channel results
- [ ] Repeated queries are served from cache (<100ms response)
- [ ] Can find Vimjoyer's Nix GC videos through search
- [ ] API quota usage is <10% of daily limit during testing
- [ ] No server crashes or unhandled exceptions

### Important (Should Pass)
- [ ] Cache hit rate >0% (at least one cached query)
- [ ] Error messages are clear and actionable
- [ ] Invalid inputs handled gracefully
- [ ] All result fields populated correctly (no null/missing data)
- [ ] URLs are clickable and valid

### Nice to Have
- [ ] Cache statistics available through `health_check` or admin tools
- [ ] Langfuse tracing works (optional, can be tested later)
- [ ] Response times consistent across queries

---

## Known Issues & Limitations

### Current Limitations
1. **No transcript tools yet** - Can't fetch video transcripts (Task 05)
2. **No metadata tools yet** - Can't get video details or channel info (Task 05)
3. **No comment tools yet** - Can't fetch video comments (Task 06)
4. **Cache inspection limited** - Admin tools not fully exposed yet
5. **No semantic search** - Can't search within transcripts (Task 08)

### Expected Issues
1. **First query always slow** - Cold start + API call (~2-5 seconds)
2. **Popular queries may vary** - Search results change over time
3. **Quota limits** - Testing constrained by 10,000 daily units
4. **Cache warmup** - Hit rate low initially, improves with usage

---

## Rollback Plan

If testing reveals critical issues:

1. **Server won't start:**
   - Check API key is set correctly
   - Verify `uv sync` completed successfully
   - Check logs in Zed's MCP output panel

2. **API errors (quota exceeded):**
   - Wait until next day (quota resets midnight Pacific Time)
   - Or enable billing in Google Cloud Console for higher quota

3. **Results incorrect/missing:**
   - Roll back to Task 03 code
   - Review search.py implementation
   - Add more validation in tests

4. **Caching not working:**
   - Check RefCache initialization in server.py
   - Verify namespace configuration
   - Add debug logging to cache decorator

---

## Documentation Updates Needed

After successful testing, update:

1. **README.md:**
   - Add "Tested with Zed" badge/note
   - Include actual quota usage stats from testing
   - Update example queries with proven working queries

2. **CONTRIBUTING.md:**
   - Document local testing workflow
   - Add "Testing in Zed" section
   - Include troubleshooting guide

3. **Task 05 scratchpad:**
   - Note any issues found during Task 04
   - Update implementation plan based on learnings
   - Adjust timeline if needed

---

## Next Steps After Completion

Once Task 04 passes all critical success criteria:

1. Update this scratchpad with actual test results
2. Document any bugs found and fixed
3. Update `.agent/goals/01-Production-YouTube-MCP-MVP/scratchpad.md` with Task 04 completion
4. Create detailed plan for Task 05 (Video & Channel Info Tools)
5. Apply learnings to Task 05 implementation

---

## Test Results Log

### Session 1: 2025-01-08
- **Tester:** lukes
- **Environment:** Zed + yt-mcp local dev (Nix devShell)
- **API Key:** [Set via .envrc.local]

**Test Results:**
- [x] Phase 1: Configuration - PASS
- [x] Phase 2: Connectivity - PASS
- [x] Phase 3: Search Tools - PASS
- [x] Phase 4: Caching - PASS
- [ ] Phase 5: Practical Use Cases - SKIPPED (fast-tracked to Task 05)
- [ ] Phase 6: Error Handling - SKIPPED
- [ ] Phase 7: Quota Monitoring - SKIPPED

**Issues Found:**
1. Initial infinite loop with `use flake` in .envrc (resolved)
2. Environment variables not loading from .envrc.local (resolved)
3. YOUTUBE_API_KEY not inherited by Zed MCP process (resolved via settings.json)

**Fixes Applied:**
1. Removed `use flake` from .envrc to prevent nix develop recursion
2. Added direct sourcing of .envrc and .envrc.local in flake.nix profile section
3. Created .env.example, .envrc.local.example, and ENV_SETUP.md for documentation
4. Added YOUTUBE_API_KEY to .zed/settings.json env section for yt-mcp-dev

**Test Evidence:**
- Search query "NixOS tutorials" returned 3 valid results (Ampersand, Vimjoyer, Fireship)
- Cache hit confirmed: same ref_id (yt-mcp:82d4e2b00f750d6b) on repeated query
- Response time: ~2-5s first call, <100ms cached call
- All video fields populated correctly (title, description, url, thumbnail, etc.)

**Final Status:** ✅ READY FOR TASK 05

---

## Notes

- This testing pattern will be reused for all subsequent tasks
- Each new tool should have similar comprehensive testing phase
- Local dev testing catches issues before CI/CD and production
- Real API testing validates assumptions from unit tests
- Cache behavior is critical to monitor (affects quota and performance)
