"""Prompts module for FastMCP Template Server.

This module contains MCP prompts that provide guidance and documentation
for using the server features.
"""

from __future__ import annotations

from mcp_refcache.fastmcp import cache_guide_prompt

TEMPLATE_GUIDE = f"""# YouTube MCP Server Guide

## Overview

This MCP server provides YouTube search and discovery tools with intelligent caching
to minimize API quota usage. All operations are traced to Langfuse for observability.

## Quick Start

1. **Search for Videos**
   ```
   search_videos("NixOS tutorials", max_results=10)
   ```
   Returns video results with titles, descriptions, thumbnails, channels, and URLs.
   Results are cached for 6 hours.

2. **Search for Channels**
   ```
   search_channels("vimjoyer", max_results=5)
   ```
   Returns channel results with names, descriptions, thumbnails, and URLs.
   Results are cached for 6 hours.

3. **Retrieve Cached Results**
   When a tool returns a `ref_id` with a preview, you have three options:

   **Full retrieval** (recommended for transcripts and complete data):
   ```
   get_cached_result(ref_id, full=True)
   ```
   Returns the complete cached value with no preview truncation.

   **Larger preview** (override default preview size):
   ```
   get_cached_result(ref_id, max_size=100000)
   ```

   **Paginate** (navigate through large collections page by page):
   ```
   get_cached_result(ref_id, page=2, page_size=20)
   ```

## API Quota Management

- Search operations cost 100 quota units each
- Default daily quota: 10,000 units (~100 searches)
- Caching reduces usage by ~4x with 6-hour TTL
- Clear error messages when quota is exceeded

## Langfuse Tracing

All tool calls are traced to Langfuse with user/session attribution.

1. **Enable Test Mode**
   ```
   enable_test_context(True)
   ```

2. **Set User Context**
   ```
   set_test_context(user_id="alice", org_id="acme", session_id="chat-001")
   ```

3. **View Trace Info**
   ```
   get_trace_info()
   ```

4. **View in Langfuse Dashboard**
   - Filter by User: "alice"
   - Filter by Session: "chat-001"
   - Filter by Tags: "yt-mcp", "mcprefcache", "youtube.search"

## Working with Transcripts

Transcripts are cached permanently since content never changes.

1. **Check available languages first:**
   ```
   list_available_transcripts(video_id)
   ```

2. **Get the full transcript:**
   ```
   get_full_transcript(video_id, language="en")
   ```
   Small transcripts are returned directly. Large transcripts return a
   `ref_id` with a preview.

3. **Retrieve the complete transcript from cache:**
   ```
   get_cached_result(ref_id, full=True)
   ```
   This bypasses preview truncation and returns the entire transcript.

4. **Fallback — paginate entry by entry:**
   ```
   get_transcript_chunk(video_id, start_index=0, chunk_size=50)
   ```

---

{cache_guide_prompt()}
"""

LANGFUSE_GUIDE = """# Langfuse Tracing Guide

## Setup

Set environment variables before starting the server:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # Optional, defaults to cloud
```

## Context Propagation

All tool calls automatically propagate context to Langfuse traces:

1. **User Attribution**
   - `user_id`: Tracks which user made the request
   - `session_id`: Groups related requests into sessions
   - `metadata`: Additional context (org_id, agent_id, cache_namespace)

2. **Testing Context**
   Enable test mode to simulate different users:
   ```
   enable_test_context(True)
   set_test_context(user_id="alice", org_id="acme", session_id="chat-001")
   ```

3. **Cache Operations**
   Cache set/get/resolve operations create child spans that inherit
   user_id and session_id for complete attribution.

## Example Workflow

```python
# 1. Enable test mode and set user
enable_test_context(True)
set_test_context(user_id="alice", session_id="demo-session")

# 2. Search YouTube (traced with user attribution)
result = search_videos("NixOS tutorials", max_results=10)

# 3. Retrieve cached result (same user in trace)
cached = get_cached_result(result["ref_id"])

# 4. Check trace info
info = get_trace_info()
```

## Viewing Traces in Langfuse

1. Go to your Langfuse dashboard
2. Navigate to Traces
3. Filter by:
   - **User**: "alice" (or any user_id you set)
   - **Session**: "demo-session"
   - **Tags**: "yt-mcp", "mcprefcache", "cacheset", "cacheget", "youtube.search"
   - **Metadata**: orgid, agentid, cachenamespace

## Best Practices

- Enable test mode for demos and testing
- Use meaningful user_id and session_id values
- Check get_trace_info() to verify tracing is working
- Flush traces on server shutdown (handled automatically)
"""


def template_guide() -> str:
    """Guide for using this MCP server template."""
    return TEMPLATE_GUIDE


def langfuse_guide() -> str:
    """Guide for using Langfuse tracing with this server."""
    return LANGFUSE_GUIDE


__all__ = [
    "LANGFUSE_GUIDE",
    "TEMPLATE_GUIDE",
    "langfuse_guide",
    "template_guide",
]
