"""Pydantic models for YouTube API data structures.

This module defines type-safe models for all YouTube data returned by the MCP server.
All models use Pydantic v2 for validation, serialization, and JSON schema generation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class VideoSearchResult(BaseModel):
    """Result from a YouTube video search.

    Attributes:
        title: Video title.
        description: Video description snippet.
        video_id: YouTube video ID.
        url: Full YouTube video URL.
        thumbnail: URL to video thumbnail image.
        channel_title: Name of the channel that published the video.
        published_at: ISO 8601 timestamp when video was published.
    """

    title: str = Field(..., description="Video title")
    description: str = Field(..., description="Video description snippet")
    video_id: str = Field(..., description="YouTube video ID")
    url: str = Field(..., description="Full YouTube video URL")
    thumbnail: HttpUrl = Field(..., description="URL to video thumbnail image")
    channel_title: str = Field(..., description="Channel that published the video")
    published_at: str = Field(..., description="Publication timestamp (ISO 8601)")


class VideoDetails(BaseModel):
    """Detailed information about a YouTube video.

    Attributes:
        title: Video title.
        description: Full video description.
        video_id: YouTube video ID.
        url: Full YouTube video URL.
        thumbnail: URL to high-quality thumbnail image.
        channel_title: Name of the channel that published the video.
        published_at: ISO 8601 timestamp when video was published.
        view_count: Number of views (string representation).
        like_count: Number of likes (string representation).
        comment_count: Number of comments (string representation).
        duration: ISO 8601 duration format (e.g., "PT15M30S").
        tags: List of video tags/keywords.
    """

    title: str = Field(..., description="Video title")
    description: str = Field(..., description="Full video description")
    video_id: str = Field(..., description="YouTube video ID")
    url: str = Field(..., description="Full YouTube video URL")
    thumbnail: HttpUrl = Field(..., description="URL to high-quality thumbnail")
    channel_title: str = Field(..., description="Channel that published the video")
    published_at: str = Field(..., description="Publication timestamp (ISO 8601)")
    view_count: str = Field(..., description="Number of views")
    like_count: str = Field(..., description="Number of likes")
    comment_count: str = Field(..., description="Number of comments")
    duration: str = Field(..., description="Video duration (ISO 8601 format)")
    tags: list[str] = Field(default_factory=list, description="Video tags/keywords")


class ChannelSearchResult(BaseModel):
    """Result from a YouTube channel search.

    Attributes:
        title: Channel name/title.
        description: Channel description snippet.
        channel_id: YouTube channel ID.
        url: Full YouTube channel URL.
        thumbnail: URL to channel thumbnail/avatar image.
        published_at: ISO 8601 timestamp when channel was created.
    """

    title: str = Field(..., description="Channel name/title")
    description: str = Field(..., description="Channel description snippet")
    channel_id: str = Field(..., description="YouTube channel ID")
    url: str = Field(..., description="Full YouTube channel URL")
    thumbnail: HttpUrl = Field(..., description="URL to channel thumbnail/avatar")
    published_at: str = Field(..., description="Channel creation timestamp (ISO 8601)")


class ChannelInfo(BaseModel):
    """Detailed information about a YouTube channel.

    Attributes:
        title: Channel name/title.
        description: Full channel description.
        channel_id: YouTube channel ID.
        url: Full YouTube channel URL.
        thumbnail: URL to high-quality channel thumbnail/avatar.
        subscriber_count: Number of subscribers (string representation).
        video_count: Number of videos published (string representation).
        view_count: Total channel views (string representation).
        published_at: ISO 8601 timestamp when channel was created.
    """

    title: str = Field(..., description="Channel name/title")
    description: str = Field(..., description="Full channel description")
    channel_id: str = Field(..., description="YouTube channel ID")
    url: str = Field(..., description="Full YouTube channel URL")
    thumbnail: HttpUrl = Field(..., description="URL to high-quality channel thumbnail")
    subscriber_count: str = Field(..., description="Number of subscribers")
    video_count: str = Field(..., description="Number of videos published")
    view_count: str = Field(..., description="Total channel views")
    published_at: str = Field(..., description="Channel creation timestamp (ISO 8601)")


class CommentData(BaseModel):
    """YouTube video comment data.

    Attributes:
        author: Comment author username.
        text: Comment text content.
        like_count: Number of likes on the comment.
        published_at: ISO 8601 timestamp when comment was posted.
        reply_count: Number of replies to this comment.
    """

    author: str = Field(..., description="Comment author username")
    text: str = Field(..., description="Comment text content")
    like_count: int = Field(..., description="Number of likes on the comment")
    published_at: str = Field(..., description="Comment timestamp (ISO 8601)")
    reply_count: int = Field(..., description="Number of replies to this comment")


class TranscriptEntry(BaseModel):
    """Single entry in a video transcript.

    Attributes:
        text: Transcript text for this segment.
        start: Start time in seconds (float).
        duration: Duration of this segment in seconds (float).
    """

    text: str = Field(..., description="Transcript text for this segment")
    start: float = Field(..., description="Start time in seconds")
    duration: float = Field(..., description="Duration in seconds")


class TranscriptInfo(BaseModel):
    """Information about an available transcript language.

    Attributes:
        language: Human-readable language name (e.g., "English").
        language_code: ISO 639-1 language code (e.g., "en").
        is_generated: Whether this is auto-generated captions.
        is_translatable: Whether this transcript can be translated.
    """

    language: str = Field(..., description="Human-readable language name")
    language_code: str = Field(..., description="ISO 639-1 language code")
    is_generated: bool = Field(..., description="Whether auto-generated")
    is_translatable: bool = Field(..., description="Whether translatable")


class FullTranscript(BaseModel):
    """Complete transcript of a video with all entries.

    Attributes:
        video_id: YouTube video ID.
        language: Language code of the transcript.
        transcript: List of transcript entries with timestamps.
        full_text: Complete transcript as plain text (all entries concatenated).
    """

    video_id: str = Field(..., description="YouTube video ID")
    language: str = Field(..., description="Language code of transcript")
    transcript: list[TranscriptEntry] = Field(
        ..., description="List of transcript entries with timestamps"
    )
    full_text: str = Field(
        ..., description="Complete transcript as plain text (concatenated)"
    )


class TranscriptPreview(BaseModel):
    """Preview of a video transcript (first N characters).

    Attributes:
        video_id: YouTube video ID.
        language: Language code of the transcript.
        preview: First N characters of the transcript.
        total_length: Total character count of full transcript.
        is_truncated: Whether the preview is truncated.
    """

    video_id: str = Field(..., description="YouTube video ID")
    language: str = Field(..., description="Language code of transcript")
    preview: str = Field(..., description="First N characters of transcript")
    total_length: int = Field(
        ..., description="Total character count of full transcript"
    )
    is_truncated: bool = Field(..., description="Whether preview is truncated")


class TranscriptChunk(BaseModel):
    """Chunk of transcript entries for pagination.

    Attributes:
        video_id: YouTube video ID.
        language: Language code of the transcript.
        start_index: Starting index of this chunk.
        chunk_size: Number of entries in this chunk.
        entries: List of transcript entries in this chunk.
        total_entries: Total number of entries in full transcript.
        has_more: Whether more entries are available after this chunk.
    """

    video_id: str = Field(..., description="YouTube video ID")
    language: str = Field(..., description="Language code of transcript")
    start_index: int = Field(..., description="Starting index of this chunk")
    chunk_size: int = Field(..., description="Number of entries in this chunk")
    entries: list[TranscriptEntry] = Field(
        ..., description="Transcript entries in this chunk"
    )
    total_entries: int = Field(
        ..., description="Total number of entries in full transcript"
    )
    has_more: bool = Field(
        ..., description="Whether more entries are available after this chunk"
    )


class AvailableTranscripts(BaseModel):
    """List of available transcript languages for a video.

    Attributes:
        video_id: YouTube video ID.
        available_languages: List of available language codes.
        transcript_info: Detailed info for each available transcript.
    """

    video_id: str = Field(..., description="YouTube video ID")
    available_languages: list[str] = Field(
        ..., description="List of available language codes"
    )
    transcript_info: list[TranscriptInfo] = Field(
        ..., description="Detailed info for each available transcript"
    )


class LiveStatus(BaseModel):
    """Live broadcast status for a video.

    Attributes:
        video_id: YouTube video ID.
        is_live: Whether the video is currently live.
        viewer_count: Current number of concurrent viewers (None if not live).
        scheduled_start_time: ISO 8601 timestamp when stream was scheduled to start (None if not scheduled).
        actual_start_time: ISO 8601 timestamp when stream actually started (None if not live).
        active_live_chat_id: Live chat ID for the stream (None if no chat or not live).
    """

    video_id: str = Field(..., description="YouTube video ID")
    is_live: bool = Field(..., description="Whether video is currently live")
    viewer_count: int | None = Field(
        None, description="Current concurrent viewers (None if not live)"
    )
    scheduled_start_time: str | None = Field(
        None, description="Scheduled start time (ISO 8601, None if not scheduled)"
    )
    actual_start_time: str | None = Field(
        None, description="Actual start time (ISO 8601, None if not live)"
    )
    active_live_chat_id: str | None = Field(
        None, description="Live chat ID (None if no chat or not live)"
    )


class LiveChatMessage(BaseModel):
    """Single live chat message from a YouTube stream.

    Attributes:
        author: Display name of the message author.
        text: Text content of the message.
        published_at: ISO 8601 timestamp when message was posted.
        author_channel_id: YouTube channel ID of the author.
    """

    author: str = Field(..., description="Message author display name")
    text: str = Field(..., description="Message text content")
    published_at: str = Field(..., description="Message timestamp (ISO 8601)")
    author_channel_id: str = Field(..., description="Author's YouTube channel ID")


class LiveChatResponse(BaseModel):
    """Response from get_live_chat_messages with pagination support.

    Attributes:
        video_id: YouTube video ID.
        messages: List of live chat messages.
        total_returned: Number of messages in this response.
        next_page_token: Token for fetching next page of messages (None if no more).
        polling_interval_millis: YouTube's recommended polling interval in milliseconds.
    """

    video_id: str = Field(..., description="YouTube video ID")
    messages: list[LiveChatMessage] = Field(
        ..., description="List of live chat messages"
    )
    total_returned: int = Field(..., description="Number of messages in this response")
    next_page_token: str | None = Field(
        None, description="Token for next page (None if no more)"
    )
    polling_interval_millis: int = Field(
        ..., description="YouTube's recommended polling interval in milliseconds"
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
    "TranscriptInfo",
    "TranscriptPreview",
    "VideoDetails",
    "VideoSearchResult",
]
