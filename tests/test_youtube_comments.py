"""Tests for YouTube comment retrieval tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from app.tools.youtube.comments import get_video_comments


@pytest.fixture
def mock_youtube_service():
    """Mock YouTube API service."""
    service = MagicMock()
    return service


@pytest.fixture
def sample_comments_response():
    """Sample comments API response."""
    return {
        "items": [
            {
                "snippet": {
                    "topLevelComment": {
                        "snippet": {
                            "authorDisplayName": "Alice",
                            "textDisplay": "Great video!",
                            "likeCount": 42,
                            "publishedAt": "2024-01-15T10:30:00Z",
                        }
                    },
                    "totalReplyCount": 3,
                }
            },
            {
                "snippet": {
                    "topLevelComment": {
                        "snippet": {
                            "authorDisplayName": "Bob",
                            "textDisplay": "Very helpful, thanks!",
                            "likeCount": 18,
                            "publishedAt": "2024-01-16T14:20:00Z",
                        }
                    },
                    "totalReplyCount": 1,
                }
            },
            {
                "snippet": {
                    "topLevelComment": {
                        "snippet": {
                            "authorDisplayName": "Charlie",
                            "textDisplay": "Could you do a follow-up?",
                            "likeCount": 5,
                            "publishedAt": "2024-01-17T09:15:00Z",
                        }
                    },
                    "totalReplyCount": 0,
                }
            },
        ]
    }


@pytest.fixture
def empty_comments_response():
    """Empty comments API response."""
    return {"items": []}


@pytest.fixture
def comments_disabled_error():
    """Mock HttpError for comments disabled."""
    # Create a real HttpError with commentsDisabled in the message
    resp = MagicMock()
    resp.status = 403
    content = b'{"error": {"errors": [{"reason": "commentsDisabled"}]}}'
    return HttpError(resp, content)


class TestGetVideoComments:
    """Tests for get_video_comments function."""

    @pytest.mark.asyncio
    async def test_get_comments_success(
        self, mock_youtube_service, sample_comments_response
    ):
        """Test successful comment retrieval."""
        # Setup mock
        mock_list = MagicMock()
        mock_list.execute.return_value = sample_comments_response
        mock_youtube_service.commentThreads.return_value.list.return_value = mock_list

        with patch(
            "app.tools.youtube.comments.get_youtube_service",
            return_value=mock_youtube_service,
        ):
            result = await get_video_comments("test_video_id", max_results=20)

        # Verify result structure
        assert result["video_id"] == "test_video_id"
        assert result["total_returned"] == 3
        assert len(result["comments"]) == 3

        # Verify first comment
        first_comment = result["comments"][0]
        assert first_comment["author"] == "Alice"
        assert first_comment["text"] == "Great video!"
        assert first_comment["like_count"] == 42
        assert first_comment["published_at"] == "2024-01-15T10:30:00Z"
        assert first_comment["reply_count"] == 3

    @pytest.mark.asyncio
    async def test_get_comments_with_max_results(
        self, mock_youtube_service, sample_comments_response
    ):
        """Test comment retrieval with custom max_results."""
        mock_list = MagicMock()
        mock_list.execute.return_value = sample_comments_response
        mock_youtube_service.commentThreads.return_value.list.return_value = mock_list

        with patch(
            "app.tools.youtube.comments.get_youtube_service",
            return_value=mock_youtube_service,
        ):
            await get_video_comments("test_video_id", max_results=50)

        # Verify API was called with correct max_results
        call_args = mock_youtube_service.commentThreads().list.call_args[1]
        assert call_args["maxResults"] == 50

    @pytest.mark.asyncio
    async def test_get_comments_clamps_min(
        self, mock_youtube_service, sample_comments_response
    ):
        """Test that max_results is clamped to minimum 1."""
        mock_list = MagicMock()
        mock_list.execute.return_value = sample_comments_response
        mock_youtube_service.commentThreads.return_value.list.return_value = mock_list

        with patch(
            "app.tools.youtube.comments.get_youtube_service",
            return_value=mock_youtube_service,
        ):
            await get_video_comments("test_video_id", max_results=0)

        # Verify max_results was clamped to 1
        call_args = mock_youtube_service.commentThreads().list.call_args[1]
        assert call_args["maxResults"] == 1

    @pytest.mark.asyncio
    async def test_get_comments_clamps_max(
        self, mock_youtube_service, sample_comments_response
    ):
        """Test that max_results is clamped to maximum 100."""
        mock_list = MagicMock()
        mock_list.execute.return_value = sample_comments_response
        mock_youtube_service.commentThreads.return_value.list.return_value = mock_list

        with patch(
            "app.tools.youtube.comments.get_youtube_service",
            return_value=mock_youtube_service,
        ):
            await get_video_comments("test_video_id", max_results=150)

        # Verify max_results was clamped to 100
        call_args = mock_youtube_service.commentThreads().list.call_args[1]
        assert call_args["maxResults"] == 100

    @pytest.mark.asyncio
    async def test_get_comments_disabled_returns_empty_list(
        self, mock_youtube_service, comments_disabled_error
    ):
        """Test that comments disabled returns empty list (not error)."""
        mock_list = MagicMock()
        mock_list.execute.side_effect = comments_disabled_error
        mock_youtube_service.commentThreads.return_value.list.return_value = mock_list

        with patch(
            "app.tools.youtube.comments.get_youtube_service",
            return_value=mock_youtube_service,
        ):
            result = await get_video_comments("test_video_id")

        # Verify empty result (not an error)
        assert result["video_id"] == "test_video_id"
        assert result["comments"] == []
        assert result["total_returned"] == 0

    @pytest.mark.asyncio
    async def test_get_comments_no_items_returns_empty(
        self, mock_youtube_service, empty_comments_response
    ):
        """Test that video with no comments returns empty list."""
        mock_list = MagicMock()
        mock_list.execute.return_value = empty_comments_response
        mock_youtube_service.commentThreads.return_value.list.return_value = mock_list

        with patch(
            "app.tools.youtube.comments.get_youtube_service",
            return_value=mock_youtube_service,
        ):
            result = await get_video_comments("test_video_id")

        # Verify empty result
        assert result["video_id"] == "test_video_id"
        assert result["comments"] == []
        assert result["total_returned"] == 0

    @pytest.mark.asyncio
    async def test_get_comments_network_error(self, mock_youtube_service):
        """Test that network errors are raised as ValueError."""
        mock_list = MagicMock()
        # Create a real HttpError for network error
        resp = MagicMock()
        resp.status = 500
        content = b'{"error": {"message": "Network error"}}'
        error = HttpError(resp, content)
        mock_list.execute.side_effect = error
        mock_youtube_service.commentThreads.return_value.list.return_value = mock_list

        with (
            patch(
                "app.tools.youtube.comments.get_youtube_service",
                return_value=mock_youtube_service,
            ),
            pytest.raises(ValueError, match="Failed to get video comments"),
        ):
            await get_video_comments("test_video_id")

    @pytest.mark.asyncio
    async def test_get_comments_api_called_correctly(
        self, mock_youtube_service, sample_comments_response
    ):
        """Test that YouTube API is called with correct parameters."""
        mock_list = MagicMock()
        mock_list.execute.return_value = sample_comments_response
        mock_youtube_service.commentThreads.return_value.list.return_value = mock_list

        with patch(
            "app.tools.youtube.comments.get_youtube_service",
            return_value=mock_youtube_service,
        ):
            await get_video_comments("test_video_id", max_results=25)

        # Verify API call parameters
        call_args = mock_youtube_service.commentThreads().list.call_args[1]
        assert call_args["part"] == "snippet"
        assert call_args["videoId"] == "test_video_id"
        assert call_args["maxResults"] == 25
        assert call_args["textFormat"] == "plainText"
        assert call_args["order"] == "relevance"

    @pytest.mark.asyncio
    async def test_get_comments_handles_missing_like_count(self, mock_youtube_service):
        """Test that missing like_count is handled gracefully."""
        response = {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {
                                "authorDisplayName": "Alice",
                                "textDisplay": "Great video!",
                                # likeCount missing
                                "publishedAt": "2024-01-15T10:30:00Z",
                            }
                        },
                        # totalReplyCount missing
                    }
                }
            ]
        }

        mock_list = MagicMock()
        mock_list.execute.return_value = response
        mock_youtube_service.commentThreads.return_value.list.return_value = mock_list

        with patch(
            "app.tools.youtube.comments.get_youtube_service",
            return_value=mock_youtube_service,
        ):
            result = await get_video_comments("test_video_id")

        # Verify defaults are used for missing fields
        comment = result["comments"][0]
        assert comment["like_count"] == 0
        assert comment["reply_count"] == 0
