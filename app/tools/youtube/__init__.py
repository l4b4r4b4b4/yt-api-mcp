"""YouTube tools package for MCP server.

This package provides YouTube integration tools including:
- Video and channel search
- Video and channel metadata retrieval
- Transcript access (full, preview, chunked)
- Comment retrieval
- Live streaming and live chat monitoring
- Cache management

All tools use mcp-refcache for intelligent caching with multiple
namespaces optimized for different data volatility levels.
"""

from __future__ import annotations

from app.tools.youtube.client import (
    YouTubeAPIError,
    YouTubeAuthError,
    YouTubeNotFoundError,
    YouTubeQuotaExceededError,
    extract_channel_id,
    extract_video_id,
    get_youtube_service,
    handle_youtube_api_error,
)
from app.tools.youtube.comments import (
    get_video_comments,
)
from app.tools.youtube.live import (
    get_live_chat_id,
    get_live_chat_messages,
    is_live,
)
from app.tools.youtube.metadata import (
    get_channel_info,
    get_video_details,
)
from app.tools.youtube.models import (
    AvailableTranscripts,
    ChannelInfo,
    ChannelSearchResult,
    CommentData,
    FullTranscript,
    LiveChatMessage,
    LiveChatResponse,
    LiveStatus,
    TranscriptChunk,
    TranscriptEntry,
    TranscriptInfo,
    TranscriptPreview,
    VideoDetails,
    VideoSearchResult,
)
from app.tools.youtube.search import (
    search_channels,
    search_live_videos,
    search_videos,
)
from app.tools.youtube.transcripts import (
    TranscriptError,
    get_full_transcript,
    get_transcript_chunk,
    get_video_transcript_preview,
    list_available_transcripts,
)

__all__ = [
    "AvailableTranscripts",
    "ChannelInfo",
    "ChannelSearchResult",
    "CommentData",
    "FullTranscript",
    "LiveChatMessage",
    "LiveChatResponse",
    "LiveStatus",
    "TranscriptChunk",
    "TranscriptEntry",
    "TranscriptError",
    "TranscriptInfo",
    "TranscriptPreview",
    "VideoDetails",
    "VideoSearchResult",
    "YouTubeAPIError",
    "YouTubeAuthError",
    "YouTubeNotFoundError",
    "YouTubeQuotaExceededError",
    "extract_channel_id",
    "extract_video_id",
    "get_channel_info",
    "get_full_transcript",
    "get_live_chat_id",
    "get_live_chat_messages",
    "get_transcript_chunk",
    "get_video_comments",
    "get_video_details",
    "get_video_transcript_preview",
    "get_youtube_service",
    "handle_youtube_api_error",
    "is_live",
    "list_available_transcripts",
    "search_channels",
    "search_live_videos",
    "search_videos",
]
