# Zed Integration Test Script - Task 06 Transcript Tools

**Date:** 2025-01-08
**Task:** Task 06 - Transcript Tools
**Server:** yt-mcp-dev (local development server)

---

## Test Environment

- **MCP Server:** yt-mcp-dev (configured in `.zed/settings.json`)
- **Video ID:** nLwbNhSxLd4 (Full NixOS Guide by Vimjoyer)
- **Expected:** All 4 transcript tools functional with caching

---

## Test Cases

### Test 1: List Available Transcripts

**Prompt for Zed:**
```
Using the yt-mcp-dev server, list all available transcripts for YouTube video nLwbNhSxLd4
```

**Expected Result:**
- Tool call to `_list_available_transcripts` with video_id="nLwbNhSxLd4"
- Returns available_languages array (e.g., ["en", "de", "es"])
- Returns transcript_info with language names and generation status
- Response includes video_id confirmation

**Success Criteria:**
- ✅ No errors
- ✅ At least one language available
- ✅ Response structure matches AvailableTranscripts model

---

### Test 2: Get Transcript Preview

**Prompt for Zed:**
```
Get a 500-character preview of the transcript for YouTube video nLwbNhSxLd4 in English
```

**Expected Result:**
- Tool call to `_get_video_transcript_preview` with video_id="nLwbNhSxLd4", language="en", max_chars=500
- Returns preview text (exactly 500 chars or less)
- Returns total_length (full transcript length)
- Returns is_truncated=true if truncated
- Response includes language="en"

**Success Criteria:**
- ✅ Preview text is readable and makes sense
- ✅ Length is ≤ 500 characters
- ✅ total_length > 500 (confirms truncation)
- ✅ is_truncated flag is correct

---

### Test 3: Get Full Transcript

**Prompt for Zed:**
```
Get the complete transcript for YouTube video nLwbNhSxLd4 in English with all timestamps
```

**Expected Result:**
- Tool call to `_get_full_transcript` with video_id="nLwbNhSxLd4", language="en"
- Returns transcript array with entries (text, start, duration)
- Returns full_text (concatenated)
- May return RefCache reference if large (>2KB)

**Success Criteria:**
- ✅ Transcript entries have text, start, duration fields
- ✅ full_text is populated and coherent
- ✅ Multiple entries returned (should be 50+ for a tutorial video)
- ✅ Timestamps make sense (increasing start times)

**Note:** If response includes `ref_id`, use `get_cached_result` to retrieve full data.

---

### Test 4: Verify Caching (Repeat Query)

**Prompt for Zed:**
```
Get a 500-character preview of the transcript for YouTube video nLwbNhSxLd4 in English again
```

**Expected Result:**
- Same tool call as Test 2
- Response should be instant (cache hit)
- Identical data to Test 2
- May include cache metadata in response

**Success Criteria:**
- ✅ Response is immediate (< 100ms)
- ✅ Data matches Test 2 exactly
- ✅ No API quota used (cached)

---

### Test 5: Get Transcript Chunk (Pagination)

**Prompt for Zed:**
```
Get the first 10 entries of the transcript for YouTube video nLwbNhSxLd4 in English
```

**Expected Result:**
- Tool call to `_get_transcript_chunk` with video_id="nLwbNhSxLd4", start_index=0, chunk_size=10, language="en"
- Returns 10 transcript entries
- Returns total_entries count
- Returns has_more=true (more entries available)
- Response includes start_index=0, chunk_size=10

**Success Criteria:**
- ✅ Exactly 10 entries returned
- ✅ Entries are from the beginning of the video
- ✅ has_more=true indicates more data available
- ✅ total_entries > 10

---

### Test 6: Error Handling - Invalid Video ID

**Prompt for Zed:**
```
List available transcripts for YouTube video INVALID123
```

**Expected Result:**
- Tool call attempts to run
- Returns error: "Invalid video ID format: INVALID123. YouTube video IDs are
