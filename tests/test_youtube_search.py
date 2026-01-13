"""Tests for YouTube search tools.

This module tests search_videos, search_channels, search_live_videos, and
get_channel_videos functions, including success paths, error handling,
validation, and edge cases.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from googleapiclient.errors import HttpError

from app.tools.youtube.client import (
    YouTubeAPIError,
    YouTubeAuthError,
    YouTubeNotFoundError,
    YouTubeQuotaExceededError,
)
from app.tools.youtube.search import (
    get_channel_videos,
    search_channels,
    search_live_videos,
    search_videos,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_youtube_service():
    """Create a mock YouTube service client."""
    service = Mock()
    execute_mock = Mock()

    service.search.return_value.list.return_value = execute_mock
    execute_mock.execute = Mock()

    return service, execute_mock


@pytest.fixture
def sample_video_response():
    """Sample response from YouTube API video search."""
    return {
        "items": [
            {
                "id": {"videoId": "video123"},
                "snippet": {
                    "title": "NixOS Garbage Collection Tutorial",
                    "description": "Learn about nix-collect-garbage",
                    "thumbnails": {
                        "default": {
                            "url": "https://i.ytimg.com/vi/video123/default.jpg"
                        }
                    },
                    "channelTitle": "Vimjoyer",
                    "publishedAt": "2024-01-15T10:30:00Z",
                },
            },
            {
                "id": {"videoId": "video456"},
                "snippet": {
                    "title": "Advanced Nix Flakes",
                    "description": "Deep dive into flakes",
                    "thumbnails": {
                        "high": {"url": "https://i.ytimg.com/vi/video456/high.jpg"}
                    },
                    "channelTitle": "NixOS Channel",
                    "publishedAt": "2024-01-10T08:15:00Z",
                },
            },
        ]
    }


@pytest.fixture
def sample_channel_response():
    """Sample response from YouTube API channel search."""
    return {
        "items": [
            {
                "id": {"channelId": "UCchannel123"},
                "snippet": {
                    "title": "Vimjoyer",
                    "description": "NixOS tutorials and guides",
                    "thumbnails": {
                        "default": {
                            "url": "https://yt3.ggpht.com/channel123/default.jpg"
                        }
                    },
                    "publishedAt": "2020-05-10T12:00:00Z",
                },
            },
        ]
    }


@pytest.fixture
def empty_response():
    """Empty response from YouTube API."""
    return {"items": []}


# =============================================================================
# Tests: search_videos
# =============================================================================


class TestSearchVideos:
    """Tests for search_videos function."""

    @pytest.mark.asyncio
    async def test_search_videos_success(
        self, mock_youtube_service, sample_video_response
    ):
        """Test successful video search with valid results."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await search_videos("NixOS tutorials", max_results=5)

        # Verify API call
        service.search.return_value.list.assert_called_once_with(
            part="snippet",
            q="NixOS tutorials",
            type="video",
            maxResults=5,
        )

        # Verify results structure
        assert len(results) == 2
        assert results[0]["video_id"] == "video123"
        assert results[0]["title"] == "NixOS Garbage Collection Tutorial"
        assert results[0]["description"] == "Learn about nix-collect-garbage"
        assert results[0]["channel_title"] == "Vimjoyer"
        assert results[0]["url"] == "https://www.youtube.com/watch?v=video123"
        assert "ytimg.com" in str(results[0]["thumbnail"])

    @pytest.mark.asyncio
    async def test_search_videos_empty_results(
        self, mock_youtube_service, empty_response
    ):
        """Test search with no results returns empty list."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = empty_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await search_videos("nonexistent query xyz", max_results=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_videos_clamps_max_results_upper(
        self, mock_youtube_service, sample_video_response
    ):
        """Test max_results is clamped to 50."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await search_videos("test", max_results=100)

        # Should clamp to 50
        service.search.return_value.list.assert_called_once()
        call_args = service.search.return_value.list.call_args[1]
        assert call_args["maxResults"] == 50

    @pytest.mark.asyncio
    async def test_search_videos_clamps_max_results_lower(
        self, mock_youtube_service, sample_video_response
    ):
        """Test max_results is clamped to 1."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await search_videos("test", max_results=0)

        # Should clamp to 1
        service.search.return_value.list.assert_called_once()
        call_args = service.search.return_value.list.call_args[1]
        assert call_args["maxResults"] == 1

    @pytest.mark.asyncio
    async def test_search_videos_handles_quota_error(self, mock_youtube_service):
        """Test quota exceeded error is handled correctly."""
        service, execute_mock = mock_youtube_service

        # Mock HttpError with quota exceeded
        error_response = Mock()
        error_response.status = 403
        error_content = b'{"error": {"errors": [{"reason": "quotaExceeded"}]}}'
        http_error = HttpError(error_response, error_content)
        execute_mock.execute.side_effect = http_error

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            with pytest.raises(YouTubeQuotaExceededError) as exc_info:
                await search_videos("test", max_results=5)

            assert "quota exceeded" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_videos_handles_auth_error(self, mock_youtube_service):
        """Test authentication error is handled correctly."""
        service, execute_mock = mock_youtube_service

        # Mock HttpError with auth error
        error_response = Mock()
        error_response.status = 401
        error_content = b'{"error": {"message": "Invalid API key"}}'
        http_error = HttpError(error_response, error_content)
        execute_mock.execute.side_effect = http_error

        with (
            patch("app.tools.youtube.search.get_youtube_service", return_value=service),
            pytest.raises(YouTubeAuthError),
        ):
            await search_videos("test", max_results=5)

    @pytest.mark.asyncio
    async def test_search_videos_handles_not_found_error(self, mock_youtube_service):
        """Test 404 error is handled correctly."""
        service, execute_mock = mock_youtube_service

        # Mock HttpError with 404
        error_response = Mock()
        error_response.status = 404
        error_content = b'{"error": {"message": "Not found"}}'
        http_error = HttpError(error_response, error_content)
        execute_mock.execute.side_effect = http_error

        with (
            patch("app.tools.youtube.search.get_youtube_service", return_value=service),
            pytest.raises(YouTubeNotFoundError),
        ):
            await search_videos("test", max_results=5)

    @pytest.mark.asyncio
    async def test_search_videos_handles_generic_error(self, mock_youtube_service):
        """Test generic API error is handled correctly."""
        service, execute_mock = mock_youtube_service

        # Mock HttpError with 500 server error
        error_response = Mock()
        error_response.status = 500
        error_content = b'{"error": {"message": "Internal server error"}}'
        http_error = HttpError(error_response, error_content)
        execute_mock.execute.side_effect = http_error

        with (
            patch("app.tools.youtube.search.get_youtube_service", return_value=service),
            pytest.raises(YouTubeAPIError),
        ):
            await search_videos("test", max_results=5)

    @pytest.mark.asyncio
    async def test_search_videos_uses_default_thumbnail(
        self, mock_youtube_service, sample_video_response
    ):
        """Test fallback to default thumbnail when high quality not available."""
        service, execute_mock = mock_youtube_service

        # Modify response to only have default thumbnail
        response = sample_video_response.copy()
        response["items"][1]["snippet"]["thumbnails"] = {
            "default": {"url": "https://i.ytimg.com/vi/video456/default.jpg"}
        }
        execute_mock.execute.return_value = response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await search_videos("test", max_results=5)

        # Should use default thumbnail for second result
        assert "default.jpg" in str(results[1]["thumbnail"])

    @pytest.mark.asyncio
    async def test_search_videos_handles_missing_description(
        self, mock_youtube_service, sample_video_response
    ):
        """Test handling of missing description field."""
        service, execute_mock = mock_youtube_service

        # Remove description from first result
        response = sample_video_response.copy()
        del response["items"][0]["snippet"]["description"]
        execute_mock.execute.return_value = response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await search_videos("test", max_results=5)

        # Should handle missing description gracefully (empty string)
        assert results[0]["description"] == ""


# =============================================================================
# Tests: search_channels
# =============================================================================


class TestSearchChannels:
    """Tests for search_channels function."""

    @pytest.mark.asyncio
    async def test_search_channels_success(
        self, mock_youtube_service, sample_channel_response
    ):
        """Test successful channel search with valid results."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_channel_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await search_channels("Vimjoyer", max_results=5)

        # Verify API call
        service.search.return_value.list.assert_called_once_with(
            part="snippet",
            q="Vimjoyer",
            type="channel",
            maxResults=5,
        )

        # Verify results structure
        assert len(results) == 1
        assert results[0]["channel_id"] == "UCchannel123"
        assert results[0]["title"] == "Vimjoyer"
        assert results[0]["description"] == "NixOS tutorials and guides"
        assert results[0]["url"] == "https://www.youtube.com/channel/UCchannel123"
        assert "yt3.ggpht.com" in str(results[0]["thumbnail"])

    @pytest.mark.asyncio
    async def test_search_channels_empty_results(
        self, mock_youtube_service, empty_response
    ):
        """Test channel search with no results returns empty list."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = empty_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await search_channels("nonexistent channel xyz", max_results=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_channels_clamps_max_results_upper(
        self, mock_youtube_service, sample_channel_response
    ):
        """Test max_results is clamped to 50."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_channel_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await search_channels("test", max_results=200)

        # Should clamp to 50
        service.search.return_value.list.assert_called_once()
        call_args = service.search.return_value.list.call_args[1]
        assert call_args["maxResults"] == 50

    @pytest.mark.asyncio
    async def test_search_channels_clamps_max_results_lower(
        self, mock_youtube_service, sample_channel_response
    ):
        """Test max_results is clamped to 1."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_channel_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await search_channels("test", max_results=-5)

        # Should clamp to 1
        service.search.return_value.list.assert_called_once()
        call_args = service.search.return_value.list.call_args[1]
        assert call_args["maxResults"] == 1

    @pytest.mark.asyncio
    async def test_search_channels_handles_quota_error(self, mock_youtube_service):
        """Test quota exceeded error is handled correctly."""
        service, execute_mock = mock_youtube_service

        # Mock HttpError with quota exceeded
        error_response = Mock()
        error_response.status = 403
        error_content = b'{"error": {"errors": [{"reason": "quotaExceeded"}]}}'
        http_error = HttpError(error_response, error_content)
        execute_mock.execute.side_effect = http_error

        with (
            patch("app.tools.youtube.search.get_youtube_service", return_value=service),
            pytest.raises(YouTubeQuotaExceededError),
        ):
            await search_channels("test", max_results=5)

    @pytest.mark.asyncio
    async def test_search_channels_handles_auth_error(self, mock_youtube_service):
        """Test authentication error is handled correctly."""
        service, execute_mock = mock_youtube_service

        # Mock HttpError with auth error
        error_response = Mock()
        error_response.status = 403
        error_content = b'{"error": {"message": "Invalid API key"}}'
        http_error = HttpError(error_response, error_content)
        execute_mock.execute.side_effect = http_error

        with (
            patch("app.tools.youtube.search.get_youtube_service", return_value=service),
            pytest.raises(YouTubeAuthError),
        ):
            await search_channels("test", max_results=5)

    @pytest.mark.asyncio
    async def test_search_channels_handles_generic_error(self, mock_youtube_service):
        """Test generic API error is handled correctly."""
        service, execute_mock = mock_youtube_service

        # Mock HttpError with server error
        error_response = Mock()
        error_response.status = 503
        error_content = b'{"error": {"message": "Service unavailable"}}'
        http_error = HttpError(error_response, error_content)
        execute_mock.execute.side_effect = http_error

        with (
            patch("app.tools.youtube.search.get_youtube_service", return_value=service),
            pytest.raises(YouTubeAPIError),
        ):
            await search_channels("test", max_results=5)

    @pytest.mark.asyncio
    async def test_search_channels_handles_missing_description(
        self, mock_youtube_service, sample_channel_response
    ):
        """Test handling of missing description field."""
        service, execute_mock = mock_youtube_service

        # Remove description
        response = sample_channel_response.copy()
        del response["items"][0]["snippet"]["description"]
        execute_mock.execute.return_value = response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await search_channels("test", max_results=5)

        # Should handle missing description gracefully
        assert results[0]["description"] == ""

    @pytest.mark.asyncio
    async def test_search_channels_uses_default_thumbnail(
        self, mock_youtube_service, sample_channel_response
    ):
        """Test fallback to default thumbnail when high quality not available."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_channel_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await search_channels("test", max_results=5)

        # Response only has default thumbnail
        assert "default.jpg" in str(results[0]["thumbnail"])


# =============================================================================
# Tests for search_live_videos
# =============================================================================


class TestSearchLiveVideos:
    """Test suite for search_live_videos function."""

    @pytest.mark.asyncio
    async def test_search_live_videos_success(
        self, mock_youtube_service, sample_video_response
    ):
        """Test successful live video search."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await search_live_videos("gaming live", max_results=5)

        # Verify search was called with eventType="live"
        service.search.return_value.list.assert_called_once()
        call_kwargs = service.search.return_value.list.call_args[1]
        assert call_kwargs["eventType"] == "live"
        assert call_kwargs["type"] == "video"
        assert call_kwargs["q"] == "gaming live"
        assert call_kwargs["maxResults"] == 5

        # Verify results structure
        assert len(results) == 2
        assert results[0]["video_id"] == "video123"
        assert results[0]["title"] == "NixOS Garbage Collection Tutorial"

    @pytest.mark.asyncio
    async def test_search_live_videos_empty_results(self, mock_youtube_service):
        """Test live search with no results."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = {"items": []}

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await search_live_videos("obscure query", max_results=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_live_videos_clamps_max_results(
        self, mock_youtube_service, sample_video_response
    ):
        """Test max_results clamping to valid range."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        # Test too high
        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await search_live_videos("test", max_results=100)

        call_kwargs = service.search.return_value.list.call_args[1]
        assert call_kwargs["maxResults"] == 50  # Clamped to max

        # Test too low
        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await search_live_videos("test", max_results=-5)

        call_kwargs = service.search.return_value.list.call_args[1]
        assert call_kwargs["maxResults"] == 1  # Clamped to min

    @pytest.mark.asyncio
    async def test_search_live_videos_handles_quota_error(self, mock_youtube_service):
        """Test handling of quota exceeded error."""
        service, execute_mock = mock_youtube_service

        # Mock quota exceeded error
        error_response = Mock()
        error_response.status = 403
        error_response.reason = "quotaExceeded"
        execute_mock.execute.side_effect = HttpError(
            resp=error_response,
            content=b'{"error": {"errors": [{"reason": "quotaExceeded"}]}}',
        )

        with (
            patch("app.tools.youtube.search.get_youtube_service", return_value=service),
            pytest.raises(YouTubeQuotaExceededError),
        ):
            await search_live_videos("test", max_results=5)

    @pytest.mark.asyncio
    async def test_search_live_videos_uses_event_type_filter(
        self, mock_youtube_service, sample_video_response
    ):
        """Test that eventType filter is correctly applied."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await search_live_videos("news", max_results=10)

        # Verify eventType="live" is passed
        call_kwargs = service.search.return_value.list.call_args[1]
        assert call_kwargs["eventType"] == "live"
        assert call_kwargs["part"] == "snippet"
        assert call_kwargs["type"] == "video"
        assert call_kwargs["q"] == "news"


# =============================================================================
# Tests: get_channel_videos
# =============================================================================


class TestGetChannelVideos:
    """Tests for get_channel_videos function."""

    @pytest.mark.asyncio
    async def test_get_channel_videos_success(
        self, mock_youtube_service, sample_video_response
    ):
        """Test successful retrieval of channel videos."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await get_channel_videos(
                "UCuAXFkgsw1L7xaCfnd5JJO1", max_results=10
            )

        assert len(results) == 2
        assert results[0]["video_id"] == "video123"
        assert results[0]["title"] == "NixOS Garbage Collection Tutorial"
        assert results[0]["channel_title"] == "Vimjoyer"
        assert results[0]["url"] == "https://www.youtube.com/watch?v=video123"

    @pytest.mark.asyncio
    async def test_get_channel_videos_empty_results(
        self, mock_youtube_service, empty_response
    ):
        """Test handling of empty results."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = empty_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            results = await get_channel_videos("UCuAXFkgsw1L7xaCfnd5JJO1")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_channel_videos_invalid_channel_id_empty(self):
        """Test validation rejects empty channel ID."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            await get_channel_videos("")

    @pytest.mark.asyncio
    async def test_get_channel_videos_invalid_channel_id_format(self):
        """Test validation rejects invalid channel ID format."""
        with pytest.raises(ValueError, match="Invalid channel ID format"):
            await get_channel_videos("invalid_id")

    @pytest.mark.asyncio
    async def test_get_channel_videos_invalid_channel_id_wrong_prefix(self):
        """Test validation rejects channel ID without UC prefix."""
        with pytest.raises(ValueError, match="start with 'UC'"):
            await get_channel_videos("ABuAXFkgsw1L7xaCfnd5JJO1")

    @pytest.mark.asyncio
    async def test_get_channel_videos_clamps_max_results_upper(
        self, mock_youtube_service, sample_video_response
    ):
        """Test max_results is clamped to 50."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await get_channel_videos("UCuAXFkgsw1L7xaCfnd5JJO1", max_results=100)

        call_kwargs = service.search.return_value.list.call_args[1]
        assert call_kwargs["maxResults"] == 50

    @pytest.mark.asyncio
    async def test_get_channel_videos_clamps_max_results_lower(
        self, mock_youtube_service, sample_video_response
    ):
        """Test max_results is clamped to minimum of 1."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await get_channel_videos("UCuAXFkgsw1L7xaCfnd5JJO1", max_results=0)

        call_kwargs = service.search.return_value.list.call_args[1]
        assert call_kwargs["maxResults"] == 1

    @pytest.mark.asyncio
    async def test_get_channel_videos_uses_channel_id_filter(
        self, mock_youtube_service, sample_video_response
    ):
        """Test that channelId filter is correctly applied."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await get_channel_videos("UCuAXFkgsw1L7xaCfnd5JJO1", max_results=10)

        call_kwargs = service.search.return_value.list.call_args[1]
        assert call_kwargs["channelId"] == "UCuAXFkgsw1L7xaCfnd5JJO1"
        assert call_kwargs["type"] == "video"
        assert call_kwargs["part"] == "snippet"

    @pytest.mark.asyncio
    async def test_get_channel_videos_respects_order(
        self, mock_youtube_service, sample_video_response
    ):
        """Test that order parameter is passed correctly."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await get_channel_videos(
                "UCuAXFkgsw1L7xaCfnd5JJO1", max_results=10, order="viewCount"
            )

        call_kwargs = service.search.return_value.list.call_args[1]
        assert call_kwargs["order"] == "viewCount"

    @pytest.mark.asyncio
    async def test_get_channel_videos_default_order_is_date(
        self, mock_youtube_service, sample_video_response
    ):
        """Test that default order is 'date'."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await get_channel_videos("UCuAXFkgsw1L7xaCfnd5JJO1")

        call_kwargs = service.search.return_value.list.call_args[1]
        assert call_kwargs["order"] == "date"

    @pytest.mark.asyncio
    async def test_get_channel_videos_invalid_order_defaults_to_date(
        self, mock_youtube_service, sample_video_response
    ):
        """Test that invalid order falls back to 'date'."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.return_value = sample_video_response

        with patch(
            "app.tools.youtube.search.get_youtube_service", return_value=service
        ):
            await get_channel_videos("UCuAXFkgsw1L7xaCfnd5JJO1", order="invalid_order")

        call_kwargs = service.search.return_value.list.call_args[1]
        assert call_kwargs["order"] == "date"

    @pytest.mark.asyncio
    async def test_get_channel_videos_handles_quota_error(self, mock_youtube_service):
        """Test handling of quota exceeded error."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.side_effect = HttpError(
            resp=Mock(status=403),
            content=b'{"error": {"message": "quota exceeded"}}',
        )

        with (
            patch("app.tools.youtube.search.get_youtube_service", return_value=service),
            pytest.raises(YouTubeQuotaExceededError),
        ):
            await get_channel_videos("UCuAXFkgsw1L7xaCfnd5JJO1")

    @pytest.mark.asyncio
    async def test_get_channel_videos_handles_auth_error(self, mock_youtube_service):
        """Test handling of authentication error."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.side_effect = HttpError(
            resp=Mock(status=401),
            content=b'{"error": {"message": "unauthorized"}}',
        )

        with (
            patch("app.tools.youtube.search.get_youtube_service", return_value=service),
            pytest.raises(YouTubeAuthError),
        ):
            await get_channel_videos("UCuAXFkgsw1L7xaCfnd5JJO1")

    @pytest.mark.asyncio
    async def test_get_channel_videos_handles_generic_error(self, mock_youtube_service):
        """Test handling of generic API error."""
        service, execute_mock = mock_youtube_service
        execute_mock.execute.side_effect = HttpError(
            resp=Mock(status=500),
            content=b'{"error": {"message": "internal server error"}}',
        )

        with (
            patch("app.tools.youtube.search.get_youtube_service", return_value=service),
            pytest.raises(YouTubeAPIError),
        ):
            await get_channel_videos("UCuAXFkgsw1L7xaCfnd5JJO1")
