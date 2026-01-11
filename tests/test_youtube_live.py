"""Tests for YouTube live streaming tools.

This module tests is_live, get_live_chat_id, and get_live_chat_messages functions,
including success paths, error handling, pagination, and edge cases.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from googleapiclient.errors import HttpError

from app.tools.youtube.client import (
    YouTubeAPIError,
    YouTubeNotFoundError,
    YouTubeQuotaExceededError,
)
from app.tools.youtube.live import (
    get_live_chat_id,
    get_live_chat_messages,
    is_live,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_youtube_service():
    """Create a mock YouTube service client."""
    service = Mock()
    videos_execute_mock = Mock()
    chat_execute_mock = Mock()

    # Setup videos().list() mock
    service.videos.return_value.list.return_value = videos_execute_mock
    videos_execute_mock.execute = Mock()

    # Setup liveChatMessages().list() mock
    service.liveChatMessages.return_value.list.return_value = chat_execute_mock
    chat_execute_mock.execute = Mock()

    return service, videos_execute_mock, chat_execute_mock


@pytest.fixture
def sample_live_video_response():
    """Sample response for a currently live video."""
    return {
        "items": [
            {
                "id": "video123",
                "liveStreamingDetails": {
                    "activeLiveChatId": "chat123",
                    "concurrentViewers": "1234",
                    "scheduledStartTime": "2025-01-08T10:00:00Z",
                    "actualStartTime": "2025-01-08T10:05:00Z",
                },
            }
        ]
    }


@pytest.fixture
def sample_not_live_video_response():
    """Sample response for a video that's not currently live."""
    return {
        "items": [
            {
                "id": "video123",
                "liveStreamingDetails": {
                    "scheduledStartTime": "2025-01-08T10:00:00Z",
                    "actualEndTime": "2025-01-08T11:00:00Z",
                },
            }
        ]
    }


@pytest.fixture
def sample_not_broadcast_response():
    """Sample response for a regular video (not a broadcast)."""
    return {
        "items": [
            {
                "id": "video123",
                "snippet": {
                    "title": "Regular Video",
                },
            }
        ]
    }


@pytest.fixture
def sample_chat_messages_response():
    """Sample response from liveChatMessages.list API."""
    return {
        "items": [
            {
                "id": "msg1",
                "snippet": {
                    "publishedAt": "2025-01-08T10:30:00Z",
                    "textMessageDetails": {
                        "messageText": "Hello from the stream!",
                    },
                },
                "authorDetails": {
                    "displayName": "TestUser1",
                    "channelId": "channel123",
                },
            },
            {
                "id": "msg2",
                "snippet": {
                    "publishedAt": "2025-01-08T10:31:00Z",
                    "textMessageDetails": {
                        "messageText": "Great content!",
                    },
                },
                "authorDetails": {
                    "displayName": "TestUser2",
                    "channelId": "channel456",
                },
            },
        ],
        "nextPageToken": "token123",
        "pollingIntervalMillis": 5000,
    }


# =============================================================================
# Tests for is_live
# =============================================================================


class TestIsLive:
    """Test suite for is_live function."""

    @pytest.mark.asyncio
    async def test_is_live_returns_true_for_live_video(
        self, mock_youtube_service, sample_live_video_response
    ):
        """Test that is_live correctly identifies a live video."""
        service, videos_execute_mock, _ = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_live_video_response

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await is_live("video123")

        # Verify API call
        service.videos.return_value.list.assert_called_once_with(
            part="liveStreamingDetails",
            id="video123",
        )

        # Verify result
        assert result["video_id"] == "video123"
        assert result["is_live"] is True
        assert result["viewer_count"] == 1234
        assert result["scheduled_start_time"] == "2025-01-08T10:00:00Z"
        assert result["actual_start_time"] == "2025-01-08T10:05:00Z"
        assert result["active_live_chat_id"] == "chat123"

    @pytest.mark.asyncio
    async def test_is_live_returns_false_for_not_live(
        self, mock_youtube_service, sample_not_live_video_response
    ):
        """Test that is_live returns False for ended stream."""
        service, videos_execute_mock, _ = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_not_live_video_response

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await is_live("video123")

        assert result["video_id"] == "video123"
        assert result["is_live"] is False
        assert result["viewer_count"] is None
        assert result["active_live_chat_id"] is None

    @pytest.mark.asyncio
    async def test_is_live_includes_scheduled_time(
        self, mock_youtube_service, sample_not_live_video_response
    ):
        """Test that scheduled time is included in response."""
        service, videos_execute_mock, _ = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_not_live_video_response

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await is_live("video123")

        assert result["scheduled_start_time"] == "2025-01-08T10:00:00Z"

    @pytest.mark.asyncio
    async def test_is_live_video_not_found(self, mock_youtube_service):
        """Test handling of video not found."""
        service, videos_execute_mock, _ = mock_youtube_service
        videos_execute_mock.execute.return_value = {"items": []}

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await is_live("nonexistent")

        # Should return graceful not live status
        assert result["video_id"] == "nonexistent"
        assert result["is_live"] is False
        assert result["viewer_count"] is None

    @pytest.mark.asyncio
    async def test_is_live_not_a_broadcast(
        self, mock_youtube_service, sample_not_broadcast_response
    ):
        """Test handling of regular video (not a broadcast)."""
        service, videos_execute_mock, _ = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_not_broadcast_response

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await is_live("video123")

        assert result["video_id"] == "video123"
        assert result["is_live"] is False
        assert result["active_live_chat_id"] is None


# =============================================================================
# Tests for get_live_chat_id
# =============================================================================


class TestGetLiveChatId:
    """Test suite for get_live_chat_id function."""

    @pytest.mark.asyncio
    async def test_get_chat_id_success(
        self, mock_youtube_service, sample_live_video_response
    ):
        """Test successful retrieval of live chat ID."""
        service, videos_execute_mock, _ = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_live_video_response

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await get_live_chat_id("video123")

        assert result["video_id"] == "video123"
        assert result["live_chat_id"] == "chat123"
        assert result["is_live"] is True

    @pytest.mark.asyncio
    async def test_get_chat_id_video_not_live(
        self, mock_youtube_service, sample_not_live_video_response
    ):
        """Test error when video is not currently live."""
        service, videos_execute_mock, _ = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_not_live_video_response

        with (
            patch("app.tools.youtube.live.get_youtube_service", return_value=service),
            pytest.raises(YouTubeAPIError) as exc_info,
        ):
            await get_live_chat_id("video123")

        assert "not currently live" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_chat_id_video_not_found(self, mock_youtube_service):
        """Test error when video not found."""
        service, videos_execute_mock, _ = mock_youtube_service
        videos_execute_mock.execute.return_value = {"items": []}

        with (
            patch("app.tools.youtube.live.get_youtube_service", return_value=service),
            pytest.raises(YouTubeNotFoundError) as exc_info,
        ):
            await get_live_chat_id("nonexistent")

        assert "Video not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_chat_id_no_chat_available(
        self, mock_youtube_service, sample_not_broadcast_response
    ):
        """Test error when video is not a broadcast."""
        service, videos_execute_mock, _ = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_not_broadcast_response

        with (
            patch("app.tools.youtube.live.get_youtube_service", return_value=service),
            pytest.raises(YouTubeAPIError) as exc_info,
        ):
            await get_live_chat_id("video123")

        assert "not a live broadcast" in str(exc_info.value)


# =============================================================================
# Tests for get_live_chat_messages
# =============================================================================


class TestGetLiveChatMessages:
    """Test suite for get_live_chat_messages function."""

    @pytest.mark.asyncio
    async def test_get_messages_success(
        self,
        mock_youtube_service,
        sample_live_video_response,
        sample_chat_messages_response,
    ):
        """Test successful retrieval of live chat messages."""
        service, videos_execute_mock, chat_execute_mock = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_live_video_response
        chat_execute_mock.execute.return_value = sample_chat_messages_response

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await get_live_chat_messages("video123", max_results=50)

        # Verify chat ID was retrieved first
        service.videos.return_value.list.assert_called_once()

        # Verify chat messages request
        service.liveChatMessages.return_value.list.assert_called_once()
        call_kwargs = service.liveChatMessages.return_value.list.call_args[1]
        assert call_kwargs["liveChatId"] == "chat123"
        assert call_kwargs["part"] == "snippet,authorDetails"
        assert call_kwargs["maxResults"] == 50
        assert "pageToken" not in call_kwargs  # First call without token

        # Verify result structure
        assert result["video_id"] == "video123"
        assert result["total_returned"] == 2
        assert result["next_page_token"] == "token123"
        assert result["polling_interval_millis"] == 5000

        # Verify messages
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0]["author"] == "TestUser1"
        assert messages[0]["text"] == "Hello from the stream!"
        assert messages[0]["author_channel_id"] == "channel123"

    @pytest.mark.asyncio
    async def test_get_messages_with_page_token(
        self,
        mock_youtube_service,
        sample_live_video_response,
        sample_chat_messages_response,
    ):
        """Test retrieval with pagination token."""
        service, videos_execute_mock, chat_execute_mock = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_live_video_response
        chat_execute_mock.execute.return_value = sample_chat_messages_response

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            await get_live_chat_messages(
                "video123", max_results=50, page_token="previous_token"
            )

        # Verify page token was passed
        call_kwargs = service.liveChatMessages.return_value.list.call_args[1]
        assert call_kwargs["pageToken"] == "previous_token"

    @pytest.mark.asyncio
    async def test_get_messages_returns_next_page_token(
        self,
        mock_youtube_service,
        sample_live_video_response,
        sample_chat_messages_response,
    ):
        """Test that next page token is returned for pagination."""
        service, videos_execute_mock, chat_execute_mock = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_live_video_response
        chat_execute_mock.execute.return_value = sample_chat_messages_response

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await get_live_chat_messages("video123")

        assert result["next_page_token"] == "token123"
        assert result["polling_interval_millis"] == 5000

    @pytest.mark.asyncio
    async def test_get_messages_clamps_max_results(
        self,
        mock_youtube_service,
        sample_live_video_response,
        sample_chat_messages_response,
    ):
        """Test max_results clamping to valid range."""
        service, videos_execute_mock, chat_execute_mock = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_live_video_response
        chat_execute_mock.execute.return_value = sample_chat_messages_response

        # Test too high
        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            await get_live_chat_messages("video123", max_results=5000)

        call_kwargs = service.liveChatMessages.return_value.list.call_args[1]
        assert call_kwargs["maxResults"] == 2000  # Clamped to max

        # Test too low
        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            await get_live_chat_messages("video123", max_results=-10)

        call_kwargs = service.liveChatMessages.return_value.list.call_args[1]
        assert call_kwargs["maxResults"] == 1  # Clamped to min

    @pytest.mark.asyncio
    async def test_get_messages_video_not_live(
        self, mock_youtube_service, sample_not_live_video_response
    ):
        """Test error when video is not currently live."""
        service, videos_execute_mock, _ = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_not_live_video_response

        with (
            patch("app.tools.youtube.live.get_youtube_service", return_value=service),
            pytest.raises(YouTubeAPIError) as exc_info,
        ):
            await get_live_chat_messages("video123")

        assert "not currently live" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_messages_empty_chat(
        self, mock_youtube_service, sample_live_video_response
    ):
        """Test handling of empty chat (no messages)."""
        service, videos_execute_mock, chat_execute_mock = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_live_video_response
        chat_execute_mock.execute.return_value = {
            "items": [],
            "nextPageToken": "token123",
            "pollingIntervalMillis": 5000,
        }

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await get_live_chat_messages("video123")

        assert result["total_returned"] == 0
        assert result["messages"] == []
        assert result["next_page_token"] == "token123"

    @pytest.mark.asyncio
    async def test_get_messages_parses_author_details(
        self,
        mock_youtube_service,
        sample_live_video_response,
        sample_chat_messages_response,
    ):
        """Test correct parsing of author details."""
        service, videos_execute_mock, chat_execute_mock = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_live_video_response
        chat_execute_mock.execute.return_value = sample_chat_messages_response

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await get_live_chat_messages("video123")

        messages = result["messages"]
        assert messages[0]["author"] == "TestUser1"
        assert messages[0]["author_channel_id"] == "channel123"
        assert messages[1]["author"] == "TestUser2"
        assert messages[1]["author_channel_id"] == "channel456"

    @pytest.mark.asyncio
    async def test_get_messages_includes_polling_interval(
        self,
        mock_youtube_service,
        sample_live_video_response,
        sample_chat_messages_response,
    ):
        """Test that polling interval is included in response."""
        service, videos_execute_mock, chat_execute_mock = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_live_video_response
        chat_execute_mock.execute.return_value = sample_chat_messages_response

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await get_live_chat_messages("video123")

        assert result["polling_interval_millis"] == 5000

    @pytest.mark.asyncio
    async def test_get_messages_handles_super_chat(
        self, mock_youtube_service, sample_live_video_response
    ):
        """Test handling of super chat messages."""
        service, videos_execute_mock, chat_execute_mock = mock_youtube_service
        videos_execute_mock.execute.return_value = sample_live_video_response

        # Response with super chat
        super_chat_response = {
            "items": [
                {
                    "id": "msg1",
                    "snippet": {
                        "publishedAt": "2025-01-08T10:30:00Z",
                        "superChatDetails": {
                            "userComment": "Great stream! Here's $5",
                            "amountMicros": "5000000",
                            "currency": "USD",
                        },
                    },
                    "authorDetails": {
                        "displayName": "SuperFan",
                        "channelId": "channel789",
                    },
                }
            ],
            "nextPageToken": "token123",
            "pollingIntervalMillis": 5000,
        }
        chat_execute_mock.execute.return_value = super_chat_response

        with patch("app.tools.youtube.live.get_youtube_service", return_value=service):
            result = await get_live_chat_messages("video123")

        messages = result["messages"]
        assert messages[0]["text"] == "Great stream! Here's $5"
        assert messages[0]["author"] == "SuperFan"

    @pytest.mark.asyncio
    async def test_get_messages_handles_quota_error(self, mock_youtube_service):
        """Test handling of quota exceeded error."""
        service, _, chat_execute_mock = mock_youtube_service

        # Mock quota exceeded error on chat messages request
        error_response = Mock()
        error_response.status = 403
        error_response.reason = "quotaExceeded"
        chat_execute_mock.execute.side_effect = HttpError(
            resp=error_response,
            content=b'{"error": {"errors": [{"reason": "quotaExceeded"}]}}',
        )

        # Need to mock the get_live_chat_id part to succeed

        async def mock_get_chat_id(video_id):
            return {
                "video_id": video_id,
                "live_chat_id": "chat123",
                "is_live": True,
            }

        with (
            patch("app.tools.youtube.live.get_youtube_service", return_value=service),
            patch(
                "app.tools.youtube.live.get_live_chat_id", side_effect=mock_get_chat_id
            ),
            pytest.raises(YouTubeQuotaExceededError),
        ):
            await get_live_chat_messages("video123")
