"""Unit tests for YouTube API client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

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


class TestGetYouTubeService:
    """Tests for get_youtube_service function."""

    @patch("app.tools.youtube.client.build")
    @patch("app.tools.youtube.client.get_settings")
    def test_creates_service_with_valid_api_key(
        self, mock_get_settings: MagicMock, mock_build: MagicMock
    ) -> None:
        """Test that service is created when API key is configured."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.youtube_api_key = "test-api-key"
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_build.return_value = mock_service

        # Act
        result = get_youtube_service()

        # Assert
        assert result == mock_service
        mock_build.assert_called_once_with("youtube", "v3", developerKey="test-api-key")

    @patch("app.tools.youtube.client.get_settings")
    def test_raises_auth_error_when_api_key_missing(
        self, mock_get_settings: MagicMock
    ) -> None:
        """Test that YouTubeAuthError is raised when API key is not configured."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.youtube_api_key = None
        mock_get_settings.return_value = mock_settings

        # Act & Assert
        with pytest.raises(YouTubeAuthError, match="YouTube API key not configured"):
            get_youtube_service()

    @patch("app.tools.youtube.client.build")
    @patch("app.tools.youtube.client.get_settings")
    def test_raises_api_error_when_build_fails(
        self, mock_get_settings: MagicMock, mock_build: MagicMock
    ) -> None:
        """Test that YouTubeAPIError is raised when service build fails."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.youtube_api_key = "test-api-key"
        mock_get_settings.return_value = mock_settings

        mock_build.side_effect = Exception("Build failed")

        # Act & Assert
        with pytest.raises(YouTubeAPIError, match="Failed to build YouTube service"):
            get_youtube_service()


class TestHandleYouTubeAPIError:
    """Tests for handle_youtube_api_error function."""

    def test_raises_quota_exceeded_error_for_403_quota(self) -> None:
        """Test that quota exceeded error is raised for 403 with quota message."""
        # Arrange
        mock_resp = MagicMock()
        mock_resp.status = 403
        error = HttpError(mock_resp, b'{"error": {"message": "Quota exceeded"}}')

        # Act & Assert
        with pytest.raises(
            YouTubeQuotaExceededError, match="YouTube API quota exceeded"
        ):
            handle_youtube_api_error(error)

    def test_raises_auth_error_for_401(self) -> None:
        """Test that auth error is raised for 401 status."""
        # Arrange
        mock_resp = MagicMock()
        mock_resp.status = 401
        error = HttpError(mock_resp, b'{"error": {"message": "Unauthorized"}}')

        # Act & Assert
        with pytest.raises(YouTubeAuthError, match="authentication failed"):
            handle_youtube_api_error(error)

    def test_raises_auth_error_for_403_non_quota(self) -> None:
        """Test that auth error is raised for 403 without quota message."""
        # Arrange
        mock_resp = MagicMock()
        mock_resp.status = 403
        error = HttpError(mock_resp, b'{"error": {"message": "Forbidden"}}')

        # Act & Assert
        with pytest.raises(YouTubeAuthError, match="authentication failed"):
            handle_youtube_api_error(error)

    def test_raises_not_found_error_for_404(self) -> None:
        """Test that not found error is raised for 404 status."""
        # Arrange
        mock_resp = MagicMock()
        mock_resp.status = 404
        error = HttpError(mock_resp, b'{"error": {"message": "Not Found"}}')

        # Act & Assert
        with pytest.raises(YouTubeNotFoundError, match="not found"):
            handle_youtube_api_error(error)

    def test_raises_generic_api_error_for_other_status(self) -> None:
        """Test that generic API error is raised for other status codes."""
        # Arrange
        mock_resp = MagicMock()
        mock_resp.status = 500
        error = HttpError(mock_resp, b'{"error": {"message": "Internal Server Error"}}')

        # Act & Assert
        with pytest.raises(YouTubeAPIError, match=r"YouTube API error.*500"):
            handle_youtube_api_error(error)


class TestExtractVideoId:
    """Tests for extract_video_id function."""

    def test_extracts_from_watch_url(self) -> None:
        """Test extraction from standard watch URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extracts_from_watch_url_with_params(self) -> None:
        """Test extraction from watch URL with additional parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s&list=PLxyz"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extracts_from_short_url(self) -> None:
        """Test extraction from youtu.be short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extracts_from_short_url_with_params(self) -> None:
        """Test extraction from short URL with parameters."""
        url = "https://youtu.be/dQw4w9WgXcQ?t=30"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extracts_from_embed_url(self) -> None:
        """Test extraction from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_returns_video_id_as_is(self) -> None:
        """Test that valid video ID is returned unchanged."""
        video_id = "dQw4w9WgXcQ"
        assert extract_video_id(video_id) == "dQw4w9WgXcQ"

    def test_strips_whitespace(self) -> None:
        """Test that whitespace is stripped."""
        video_id = "  dQw4w9WgXcQ  "
        assert extract_video_id(video_id) == "dQw4w9WgXcQ"

    def test_raises_error_for_invalid_format(self) -> None:
        """Test that ValueError is raised for invalid format."""
        with pytest.raises(ValueError, match="Invalid YouTube URL"):
            extract_video_id("not-a-valid-url")

    def test_raises_error_for_empty_string(self) -> None:
        """Test that ValueError is raised for empty string."""
        with pytest.raises(ValueError, match="Invalid YouTube URL"):
            extract_video_id("")


class TestExtractChannelId:
    """Tests for extract_channel_id function."""

    def test_extracts_from_channel_url(self) -> None:
        """Test extraction from standard channel URL."""
        url = "https://www.youtube.com/channel/UCxyz123"
        assert extract_channel_id(url) == "UCxyz123"

    def test_extracts_from_channel_url_with_params(self) -> None:
        """Test extraction from channel URL with parameters."""
        url = "https://www.youtube.com/channel/UCxyz123?sub_confirmation=1"
        assert extract_channel_id(url) == "UCxyz123"

    def test_extracts_from_user_url(self) -> None:
        """Test extraction from legacy user URL."""
        url = "https://www.youtube.com/user/username"
        assert extract_channel_id(url) == "username"

    def test_extracts_from_custom_url(self) -> None:
        """Test extraction from custom URL with @."""
        url = "https://www.youtube.com/@vimjoyer"
        assert extract_channel_id(url) == "@vimjoyer"

    def test_returns_channel_id_as_is(self) -> None:
        """Test that valid channel ID is returned unchanged."""
        channel_id = "UCxyz123456789012345678"
        assert extract_channel_id(channel_id) == "UCxyz123456789012345678"

    def test_returns_custom_handle_as_is(self) -> None:
        """Test that custom handle with @ is returned unchanged."""
        handle = "@vimjoyer"
        assert extract_channel_id(handle) == "@vimjoyer"

    def test_strips_whitespace(self) -> None:
        """Test that whitespace is stripped."""
        channel_id = "  UCxyz123456789012345678  "
        assert extract_channel_id(channel_id) == "UCxyz123456789012345678"

    def test_raises_error_for_invalid_format(self) -> None:
        """Test that ValueError is raised for invalid format."""
        with pytest.raises(ValueError, match="Invalid YouTube channel"):
            extract_channel_id("not-a-valid-url")

    def test_raises_error_for_empty_string(self) -> None:
        """Test that ValueError is raised for empty string."""
        with pytest.raises(ValueError, match="Invalid YouTube channel"):
            extract_channel_id("")
