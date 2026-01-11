"""YouTube MCP Server with RefCache and Langfuse Tracing.

This module creates and configures the FastMCP server providing YouTube
integration tools with intelligent caching.

Features:
- YouTube video and channel search
- Reference-based caching for large results
- Intelligent cache namespaces (search: 6h, metadata: 24h+)
- Preview generation for large result sets
- Langfuse tracing integration for observability

Usage:
    # Run with typer CLI
    uvx yt-mcp stdio           # Local CLI mode
    uvx yt-mcp streamable-http # Remote/Docker mode

    # Or with uv
    uv run yt-mcp stdio
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP
from mcp_refcache import PreviewConfig, PreviewStrategy, RefCache
from mcp_refcache.fastmcp import cache_instructions, register_admin_tools

from app.prompts import langfuse_guide, template_guide
from app.tools import (
    create_get_cached_result,
    create_health_check,
    enable_test_context,
    get_trace_info,
    reset_test_context,
    set_test_context,
)
from app.tracing import TracedRefCache

# =============================================================================
# Initialize FastMCP Server
# =============================================================================

mcp = FastMCP(
    name="YouTube MCP Server",
    instructions=f"""YouTube integration MCP server with intelligent caching and Langfuse tracing.

This server provides tools for searching and discovering YouTube content with
intelligent caching to minimize API quota usage. All operations are traced to
Langfuse for observability.

Available YouTube Tools:
- search_videos: Search for YouTube videos by query (cached 6h)
- search_channels: Search for YouTube channels by query (cached 6h)
- get_video_details: Get detailed video metadata (views, likes, duration, etc.) (cached 24h)
- get_channel_info: Get channel statistics (subscribers, video count, etc.) (cached 24h)
- list_available_transcripts: List available transcript languages for a video (cached permanently)
- get_video_transcript_preview: Get preview of video transcript (first N chars) (cached permanently)
- get_full_transcript: Get complete video transcript with timestamps (cached permanently)
- get_transcript_chunk: Paginate through transcript entries (cached permanently)
- get_video_comments: Get top comments with engagement metrics (cached 5m)

Live Streaming Tools:
- search_live_videos: Search for currently live streams (cached 6h)
- is_live: Check if video is currently live (cached 30s)
- get_live_chat_id: Get chat ID for live video (cached 5m)
- get_live_chat_messages: Get recent live chat with pagination (cached 30s)

Cache Management:
- get_cached_result: Retrieve or paginate through cached results
- Admin tools available for cache inspection and management

Context & Tracing:
- enable_test_context: Enable/disable test context for Langfuse demos
- set_test_context: Set test context values (user_id, session_id, etc.)
- reset_test_context: Reset test context to defaults
- get_trace_info: Get current Langfuse tracing status

API Quota Notes:
- Search operations cost 100 units each
- Metadata operations cost 1 unit each
- Comment operations cost 1 unit each
- Transcript operations use no quota (third-party API)
- Default daily quota: 10,000 units (~100 searches)
- Caching reduces quota usage by ~4x with 6h TTL
- Clear error messages when quota is exceeded

Transcript Notes:
- Transcripts cached permanently (content never changes)
- Use list_available_transcripts first to discover available languages
- get_full_transcript may return RefCache preview for large transcripts
- Use get_transcript_chunk for entry-by-entry pagination if needed

Comment Notes:
- Returns empty list if comments disabled (not an error)
- Only top-level comments (no replies)
- Sorted by relevance
- Cached for 5 minutes (great for trending videos with active engagement)

Live Streaming Notes:
- Very short cache times for real-time data (30s for status/chat, 5m for chat ID)
- MCP is request/response (not true streaming) - agent must poll for new messages
- Use page_token from previous call to get only new messages (efficient polling)
- For continuous monitoring, poll every 30-60 seconds using returned next_page_token
- 30 second cache prevents excessive API usage during polling
- Same API quota as other tools (1 unit per request)
- For true real-time experience with instant updates, use YouTube web interface

{cache_instructions()}
""",
)

# =============================================================================
# Initialize RefCache with Langfuse Tracing
# =============================================================================

# Create the base RefCache instance with YouTube-optimized configuration
_cache = RefCache(
    name="yt-mcp",
    default_ttl=21600,  # 6 hours default TTL (good for search results)
    preview_config=PreviewConfig(
        max_size=2048,  # Max 2048 tokens in previews
        default_strategy=PreviewStrategy.SAMPLE,  # Sample large collections
    ),
)

# Wrap with TracedRefCache for Langfuse observability
cache = TracedRefCache(_cache)

# =============================================================================
# Create Bound Tool Functions
# =============================================================================

# These are created with factory functions and bound to the cache instance.
# We keep references for testing and re-export them as module attributes.
get_cached_result = create_get_cached_result(cache)
health_check = create_health_check(_cache)

# =============================================================================
# Register Tools
# =============================================================================


# YouTube search tools
@mcp.tool
@cache.cached(namespace="youtube.search")
async def search_videos(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search for YouTube videos by query string.

    Searches YouTube and returns video results with titles, descriptions,
    thumbnails, channels, and URLs. Results are cached for 6 hours to
    minimize API quota usage.

    Args:
        query: Search query (e.g., "NixOS tutorials", "vimjoyer nix").
        max_results: Maximum results to return (1-50, default 5).

    Returns:
        List of video results with video_id, title, description, url,
        thumbnail, channel_title, and published_at.

    Example:
        >>> results = _search_videos("vimjoyer garbage collection", 10)
        >>> print(results[0]["title"])

    Note:
        - Search costs 100 quota units per request
        - Results cached for 6 hours in youtube.search namespace
        - Use get_cached_result() to paginate large result sets
    """
    from app.tools.youtube import search_videos as search_videos_impl

    results = await search_videos_impl(query, max_results)
    return results  # type: ignore[return-value]


@mcp.tool
@cache.cached(namespace="youtube.search")
async def search_channels(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search for YouTube channels by query string.

    Searches YouTube and returns channel results with names, descriptions,
    thumbnails, and URLs. Results are cached for 6 hours to minimize
    API quota usage.

    Args:
        query: Search query (e.g., "Vimjoyer", "NixOS channels").
        max_results: Maximum results to return (1-50, default 5).

    Returns:
        List of channel results with channel_id, title, description,
        url, thumbnail, and published_at.

    Example:
        >>> results = _search_channels("vimjoyer", 5)
        >>> print(results[0]["title"])

    Note:
        - Search costs 100 quota units per request
        - Results cached for 6 hours in youtube.search namespace
        - Use get_cached_result() to paginate large result sets
    """
    from app.tools.youtube import search_channels as search_channels_impl

    results = await search_channels_impl(query, max_results)
    return results  # type: ignore[return-value]


# YouTube metadata tools
@mcp.tool
@cache.cached(namespace="youtube.api", ttl=86400)  # 24 hours
async def get_video_details(video_id: str) -> dict[str, Any]:
    """Get detailed information about a YouTube video.

    Retrieves comprehensive video metadata including title, description,
    statistics (views, likes, comments), duration, tags, and channel info.
    Cached for 24 hours to minimize API quota usage.

    Args:
        video_id: YouTube video ID (from URL or search results, e.g., "dQw4w9WgXcQ")

    Returns:
        Video details dictionary with:
        - title, description, video_id, url, thumbnail
        - view_count, like_count, comment_count
        - duration (ISO 8601 format like "PT15M30S")
        - tags, channel_title, published_at

    Example:
        >>> details = _get_video_details("nLwbNhSxLd4")
        >>> print(details["title"])
        "Full NixOS Guide"

    Note:
        - Costs 1 quota unit per request (100x cheaper than search)
        - Cached for 24h in youtube.api namespace
        - Use after search to get full details
    """
    from app.tools.youtube import get_video_details as get_video_details_impl

    result = await get_video_details_impl(video_id)
    return result  # type: ignore[return-value]


@mcp.tool
@cache.cached(namespace="youtube.api", ttl=86400)  # 24 hours
async def get_channel_info(channel_id: str) -> dict[str, Any]:
    """Get detailed information about a YouTube channel.

    Retrieves channel metadata including title, description, statistics
    (subscribers, videos, total views), and branding information.
    Cached for 24 hours to minimize API quota usage.

    Args:
        channel_id: YouTube channel ID (from search results, e.g., "UCuAXFkgsw1L7xaCfnd5JJOw")

    Returns:
        Channel info dictionary with:
        - title, description, channel_id, url, thumbnail
        - subscriber_count, video_count, view_count
        - published_at

    Example:
        >>> info = _get_channel_info("UCuAXFkgsw1L7xaCfnd5JJOw")
        >>> print(info["title"])
        "Vimjoyer"

    Note:
        - Costs 1 quota unit per request (100x cheaper than search)
        - Cached for 24h in youtube.api namespace
        - Use after channel search to get full details
    """
    from app.tools.youtube import get_channel_info as get_channel_info_impl

    result = await get_channel_info_impl(channel_id)
    return result  # type: ignore[return-value]


# YouTube transcript tools
@mcp.tool
@cache.cached(namespace="youtube.content")  # Permanent cache
async def list_available_transcripts(video_id: str) -> dict[str, Any]:
    """List all available transcript languages for a YouTube video.

    Discovers which transcript languages are available for a video,
    including both manual and auto-generated transcripts.
    Cached permanently as available transcripts don't change.

    Args:
        video_id: YouTube video ID (from URL or search, e.g., "dQw4w9WgXcQ")

    Returns:
        AvailableTranscripts dictionary with:
        - video_id: The video ID
        - available_languages: List of language codes (e.g., ["en", "es", "fr"])
        - transcript_info: Detailed info for each (language, is_generated, etc.)

    Example:
        >>> transcripts = list_available_transcripts("nLwbNhSxLd4")
        >>> print(transcripts["available_languages"])
        ["en", "de", "es"]

    Note:
        - Uses no YouTube API quota (third-party transcript API)
        - Cached permanently in youtube.content namespace
        - Call this first before requesting specific transcript
    """
    from app.tools.youtube import (
        list_available_transcripts as list_available_transcripts_impl,
    )

    result = await list_available_transcripts_impl(video_id)
    return result  # type: ignore[return-value]


@mcp.tool
@cache.cached(namespace="youtube.content")  # Permanent cache
async def get_video_transcript_preview(
    video_id: str,
    language: str = "",
    max_chars: int = 2000,
) -> dict[str, Any]:
    """Get a preview of a YouTube video transcript.

    Retrieves the first N characters of a video transcript for quick preview.
    Useful for deciding if you need the full transcript.
    Cached permanently as transcript content doesn't change.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")
        language: Language code (e.g., "en"). If empty, uses first available
        max_chars: Maximum characters to return (default: 2000)

    Returns:
        TranscriptPreview dictionary with:
        - video_id, language, preview text
        - total_length: Total characters in full transcript
        - is_truncated: Whether preview is truncated

    Example:
        >>> preview = get_video_transcript_preview("nLwbNhSxLd4", max_chars=500)
        >>> print(preview["preview"][:50])
        "Welcome to this NixOS tutorial..."

    Note:
        - Uses no YouTube API quota
        - Cached permanently in youtube.content namespace
        - Use list_available_transcripts first to see language options
    """
    from app.tools.youtube import (
        get_video_transcript_preview as get_video_transcript_preview_impl,
    )

    result = await get_video_transcript_preview_impl(video_id, language, max_chars)
    return result  # type: ignore[return-value]


@mcp.tool
@cache.cached(namespace="youtube.content")  # Permanent cache
async def get_full_transcript(
    video_id: str,
    language: str = "",
) -> dict[str, Any]:
    """Get the complete transcript for a YouTube video.

    Retrieves the full transcript with all entries and timestamps.
    For large transcripts (>2KB), RefCache automatically returns a preview
    with a reference that can be paginated using get_cached_result.
    Cached permanently as transcript content doesn't change.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")
        language: Language code (e.g., "en"). If empty, uses first available

    Returns:
        FullTranscript dictionary with:
        - video_id, language
        - transcript: List of entries with text, start time, duration
        - full_text: Complete transcript as plain text

    Example:
        >>> full = get_full_transcript("nLwbNhSxLd4", language="en")
        >>> print(len(full["transcript"]))
        150
        >>> print(full["full_text"][:100])

    Note:
        - Uses no YouTube API quota
        - Cached permanently in youtube.content namespace
        - RefCache may return preview + reference for large transcripts
        - Use get_transcript_chunk for entry-by-entry pagination
    """
    from app.tools.youtube import get_full_transcript as get_full_transcript_impl

    result = await get_full_transcript_impl(video_id, language)
    return result  # type: ignore[return-value]


@mcp.tool
@cache.cached(namespace="youtube.content")  # Permanent cache
async def get_transcript_chunk(
    video_id: str,
    start_index: int = 0,
    chunk_size: int = 50,
    language: str = "",
) -> dict[str, Any]:
    """Get a chunk of transcript entries for pagination.

    Retrieves a subset of transcript entries for large transcripts.
    Useful for iterating through transcripts entry-by-entry.
    Cached permanently as transcript content doesn't change.

    Args:
        video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")
        start_index: Starting entry index (0-based, default: 0)
        chunk_size: Number of entries to return (default: 50)
        language: Language code (e.g., "en"). If empty, uses first available

    Returns:
        TranscriptChunk dictionary with:
        - video_id, language, start_index, chunk_size
        - entries: List of transcript entries in this chunk
        - total_entries: Total entries in full transcript
        - has_more: Whether more entries available after this chunk

    Example:
        >>> chunk = get_transcript_chunk("nLwbNhSxLd4", start_index=0, chunk_size=10)
        >>> print(len(chunk["entries"]))
        10
        >>> print(chunk["has_more"])
        True
        >>> # Get next chunk
        >>> chunk2 = get_transcript_chunk("nLwbNhSxLd4", start_index=10, chunk_size=10)

    Note:
        - Uses no YouTube API quota
        - Cached permanently in youtube.content namespace
        - Use for iterating through large transcripts
    """
    from app.tools.youtube import get_transcript_chunk as get_transcript_chunk_impl

    result = await get_transcript_chunk_impl(
        video_id, start_index, chunk_size, language
    )
    return result  # type: ignore[return-value]


@mcp.tool
@cache.cached(namespace="youtube.comments", ttl=300)  # 5 min cache
async def get_video_comments(
    video_id: str,
    max_results: int = 20,
) -> dict[str, Any]:
    """Get top comments for a YouTube video with engagement metrics.

    Retrieves top-level comments (no replies) sorted by relevance.
    Comments are cached for 5 minutes. Returns empty list if comments
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
        - Cached for 5 minutes in youtube.comments namespace
        - Returns empty list if comments disabled (not an error)
    """
    from app.tools.youtube import get_video_comments as get_video_comments_impl

    result = await get_video_comments_impl(video_id, max_results)
    return result  # type: ignore[return-value]


# YouTube live streaming tools
@mcp.tool
@cache.cached(namespace="youtube.search")  # 6h cache (same as regular search)
async def search_live_videos(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search for currently live YouTube videos.

    Searches for videos that are currently streaming live, filtering results
    to only active broadcasts. Results are cached for 6 hours.

    Args:
        query: Search query (e.g., "gaming live", "news live now").
        max_results: Maximum results to return (1-50, default 5).

    Returns:
        List of live video results with video_id, title, description, url,
        thumbnail, channel_title, and published_at.

    Example:
        >>> results = search_live_videos("gaming", max_results=10)
        >>> print(results[0]["title"])
        'Live Gaming Stream - Fortnite'

    Note:
        - Search costs 100 quota units per request
        - Results cached for 6 hours in youtube.search namespace
        - Use is_live() to check if a specific video is currently live
        - Use get_live_chat_messages() to monitor chat
    """
    from app.tools.youtube import search_live_videos as search_live_videos_impl

    results = await search_live_videos_impl(query, max_results)
    return results  # type: ignore[return-value]


@mcp.tool
@cache.cached(namespace="youtube.api", ttl=30)  # 30 second cache
async def is_live(video_id: str) -> dict[str, Any]:
    """Check if a YouTube video is currently live.

    Queries the YouTube Data API to determine if a video is currently
    broadcasting live. Returns live status with viewer count and timing information.
    Cached for 30 seconds since live status changes quickly.

    Args:
        video_id: YouTube video ID to check (e.g., "dQw4w9WgXcQ").

    Returns:
        Dictionary with:
        - video_id: YouTube video ID
        - is_live: Boolean indicating if video is currently live
        - viewer_count: Current concurrent viewers (None if not live)
        - scheduled_start_time: ISO 8601 scheduled start time (None if not scheduled)
        - actual_start_time: ISO 8601 actual start time (None if not started)
        - active_live_chat_id: Live chat ID (None if no chat or not live)

    Example:
        >>> status = is_live("dQw4w9WgXcQ")
        >>> if status["is_live"]:
        ...     print(f"Live now with {status['viewer_count']} viewers!")

    Note:
        - Costs 1 quota unit per request
        - Cached for 30 seconds in youtube.api namespace
        - Use search_live_videos() to find live streams
    """
    from app.tools.youtube import is_live as is_live_impl

    result = await is_live_impl(video_id)
    return result  # type: ignore[return-value]


@mcp.tool
@cache.cached(namespace="youtube.api", ttl=300)  # 5 min cache
async def get_live_chat_id(video_id: str) -> dict[str, Any]:
    """Get the live chat ID for a currently streaming video.

    Retrieves the active live chat ID required for fetching chat messages.
    This ID remains constant throughout the stream's duration.
    Cached for 5 minutes since chat ID doesn't change during stream.

    Args:
        video_id: YouTube video ID of the live stream.

    Returns:
        Dictionary with:
        - video_id: YouTube video ID
        - live_chat_id: Active live chat ID
        - is_live: Boolean confirming video is live

    Example:
        >>> result = get_live_chat_id("dQw4w9WgXcQ")
        >>> chat_id = result["live_chat_id"]

    Note:
        - Costs 1 quota unit per request
        - Cached for 5 minutes in youtube.api namespace
        - Raises error if video is not live or chat disabled
        - Use is_live() first to check if video is broadcasting
    """
    from app.tools.youtube import get_live_chat_id as get_live_chat_id_impl

    result = await get_live_chat_id_impl(video_id)
    return result  # type: ignore[return-value]


@mcp.tool
@cache.cached(namespace="youtube.comments", ttl=30)  # 30 second cache
async def get_live_chat_messages(
    video_id: str,
    max_results: int = 200,
    page_token: str | None = None,
) -> dict[str, Any]:
    """Get recent live chat messages from a streaming video.

    Fetches live chat messages with pagination support for efficient polling.
    Use the returned next_page_token in subsequent calls to get only new messages.
    Cached for 30 seconds for near real-time monitoring.

    Args:
        video_id: YouTube video ID of the live stream.
        max_results: Maximum messages to return (1-2000, default 200).
        page_token: Pagination token from previous call (None for first call).

    Returns:
        Dictionary with:
        - video_id: YouTube video ID
        - messages: List of messages with author, text, published_at, author_channel_id
        - total_returned: Number of messages in this response
        - next_page_token: Token for next page (None if no more)
        - polling_interval_millis: YouTube's recommended polling interval

    Example:
        >>> # First call - get latest messages
        >>> result = get_live_chat_messages("dQw4w9WgXcQ", max_results=50)
        >>> print(f"Got {result['total_returned']} messages")
        >>>
        >>> # Second call - get only new messages since first call
        >>> result2 = get_live_chat_messages(
        ...     "dQw4w9WgXcQ",
        ...     max_results=50,
        ...     page_token=result["next_page_token"]
        ... )

    Note:
        - Costs 1 quota unit per request
        - Cached for 30 seconds in youtube.comments namespace
        - Polling Pattern:
          1. First call: No page_token → Get latest messages + next_page_token
          2. Store next_page_token
          3. Subsequent calls: Pass page_token → Get only NEW messages
          4. Repeat step 3 every 30-60 seconds for continuous monitoring
        - MCP Limitation: Agent must manually call this tool repeatedly to see new messages
    """
    from app.tools.youtube import get_live_chat_messages as get_live_chat_messages_impl

    result = await get_live_chat_messages_impl(video_id, max_results, page_token)
    return result  # type: ignore[return-value]


# Context management tools
mcp.tool(enable_test_context)
mcp.tool(set_test_context)
mcp.tool(reset_test_context)
mcp.tool(get_trace_info)

# Cache-bound tools (using pre-created module-level functions)
mcp.tool(get_cached_result)
mcp.tool(health_check)

# =============================================================================
# Admin Tools (Permission-Gated)
# =============================================================================


async def is_admin(ctx: Any) -> bool:
    """Check if the current context has admin privileges.

    Override this in your own server with proper auth logic.
    """
    # Demo: No admin access by default
    return False


# Register admin tools with the underlying cache (not the traced wrapper)
_admin_tools = register_admin_tools(
    mcp,
    _cache,
    admin_check=is_admin,
    prefix="admin_",
    include_dangerous=False,
)

# =============================================================================
# Register Prompts
# =============================================================================


@mcp.prompt
def _template_guide() -> str:
    """Guide for using this MCP server template."""
    return template_guide()


@mcp.prompt
def _langfuse_guide() -> str:
    """Guide for using Langfuse tracing with this server."""
    return langfuse_guide()
