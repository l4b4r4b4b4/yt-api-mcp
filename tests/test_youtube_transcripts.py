"""Tests for YouTube transcript tools.

This module tests the transcript retrieval functionality including
language discovery, preview, full transcript, and chunked access.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from app.tools.youtube.transcripts import (
    TranscriptError,
    get_full_transcript,
    get_transcript_chunk,
    get_video_transcript_preview,
    list_available_transcripts,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_transcript_list(sample_transcript_entries):
    """Mock YouTubeTranscriptApi().list_transcripts response."""
    # Create mock transcript objects
    en_transcript = Mock()
    en_transcript.language = "English"
    en_transcript.language_code = "en"
    en_transcript.is_generated = False
    en_transcript.is_translatable = True
    en_transcript.fetch = Mock(return_value=sample_transcript_entries)

    es_transcript = Mock()
    es_transcript.language = "Spanish"
    es_transcript.language_code = "es"
    es_transcript.is_generated = True
    es_transcript.is_translatable = True
    es_transcript.fetch = Mock(return_value=sample_transcript_entries)

    de_transcript = Mock()
    de_transcript.language = "German"
    de_transcript.language_code = "de"
    de_transcript.is_generated = False
    de_transcript.is_translatable = False
    de_transcript.fetch = Mock(return_value=sample_transcript_entries)

    # Create mock transcript list that supports iteration and find_transcript
    mock_list = Mock()
    mock_list.__iter__ = Mock(
        return_value=iter([en_transcript, es_transcript, de_transcript])
    )

    def find_transcript_mock(language_codes):
        code = language_codes[0]
        if code == "en":
            return en_transcript
        elif code == "es":
            return es_transcript
        elif code == "de":
            return de_transcript
        else:
            raise NoTranscriptFound(
                video_id="test",
                requested_language_codes=language_codes,
                transcript_data={},
            )

    mock_list.find_transcript = Mock(side_effect=find_transcript_mock)

    return mock_list


@pytest.fixture
def sample_transcript_entries():
    """Sample transcript entries for testing."""
    return [
        {"text": "Welcome to this video", "start": 0.0, "duration": 2.5},
        {"text": "Today we'll learn about NixOS", "start": 2.5, "duration": 3.0},
        {"text": "Let's get started", "start": 5.5, "duration": 2.0},
        {"text": "First, install Nix", "start": 7.5, "duration": 2.5},
        {"text": "Then configure your system", "start": 10.0, "duration": 3.0},
    ]


@pytest.fixture
def mock_transcript_fetch(sample_transcript_entries):
    """Mock transcript.fetch() to return sample entries."""
    return sample_transcript_entries


# =============================================================================
# Tests for list_available_transcripts
# =============================================================================


class TestListAvailableTranscripts:
    """Tests for list_available_transcripts function."""

    @pytest.mark.asyncio
    async def test_list_available_transcripts_success(self, mock_transcript_list):
        """Test successful listing of available transcripts."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            result = await list_available_transcripts("dQw4w9WgXcQ")

            assert result["video_id"] == "dQw4w9WgXcQ"
            assert result["available_languages"] == ["en", "es", "de"]
            assert len(result["transcript_info"]) == 3

            # Check first transcript info
            assert result["transcript_info"][0]["language"] == "English"
            assert result["transcript_info"][0]["language_code"] == "en"
            assert result["transcript_info"][0]["is_generated"] is False
            assert result["transcript_info"][0]["is_translatable"] is True

            # Check auto-generated transcript
            assert result["transcript_info"][1]["is_generated"] is True

            mock_list.assert_called_once_with("dQw4w9WgXcQ")

    @pytest.mark.asyncio
    async def test_list_available_transcripts_empty_video_id(self):
        """Test error when video_id is empty."""
        with pytest.raises(ValueError, match="video_id must be a non-empty string"):
            await list_available_transcripts("")

    @pytest.mark.asyncio
    async def test_list_available_transcripts_invalid_id_length(self):
        """Test error when video_id has invalid length."""
        with pytest.raises(ValueError, match="Invalid video ID format"):
            await list_available_transcripts("short")

    @pytest.mark.asyncio
    async def test_list_available_transcripts_disabled(self):
        """Test handling when transcripts are disabled."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.side_effect = TranscriptsDisabled(video_id="dQw4w9WgXcQ")

            with pytest.raises(TranscriptError, match="Transcripts are disabled"):
                await list_available_transcripts("dQw4w9WgXcQ")

    @pytest.mark.asyncio
    async def test_list_available_transcripts_not_found(self):
        """Test handling when no transcripts found."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.side_effect = NoTranscriptFound(
                video_id="dQw4w9WgXcQ",
                requested_language_codes=["en"],
                transcript_data={},
            )

            with pytest.raises(TranscriptError, match="No transcripts found"):
                await list_available_transcripts("dQw4w9WgXcQ")

    @pytest.mark.asyncio
    async def test_list_available_transcripts_single_language(self):
        """Test listing when only one language is available."""
        en_transcript = Mock()
        en_transcript.language = "English"
        en_transcript.language_code = "en"
        en_transcript.is_generated = False
        en_transcript.is_translatable = False

        mock_list = Mock()
        mock_list.__iter__ = Mock(return_value=iter([en_transcript]))

        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_fn = mock_api.return_value.list
            mock_fn.return_value = mock_list

            result = await list_available_transcripts("dQw4w9WgXcQ")

            assert result["available_languages"] == ["en"]
            assert len(result["transcript_info"]) == 1


# =============================================================================
# Tests for get_video_transcript_preview
# =============================================================================


class TestGetVideoTranscriptPreview:
    """Tests for get_video_transcript_preview function."""

    @pytest.mark.asyncio
    async def test_get_preview_success(
        self, mock_transcript_list, mock_transcript_fetch
    ):
        """Test successful transcript preview retrieval."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            result = await get_video_transcript_preview(
                "dQw4w9WgXcQ", language="en", max_chars=50
            )

            assert result["video_id"] == "dQw4w9WgXcQ"
            assert result["language"] == "en"
            assert result["is_truncated"] is True
            assert len(result["preview"]) == 50
            assert result["total_length"] > 50

    @pytest.mark.asyncio
    async def test_get_preview_short_text_not_truncated(
        self, mock_transcript_list, mock_transcript_fetch
    ):
        """Test preview when full text is shorter than max_chars."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            result = await get_video_transcript_preview(
                "dQw4w9WgXcQ", language="en", max_chars=5000
            )

            assert result["is_truncated"] is False
            assert len(result["preview"]) == result["total_length"]

    @pytest.mark.asyncio
    async def test_get_preview_no_language_uses_first(
        self, mock_transcript_list, mock_transcript_fetch
    ):
        """Test that preview uses first available language when not specified."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            # Setup first transcript to be returned
            first_transcript = Mock()
            first_transcript.language_code = "en"
            first_transcript.is_generated = False
            first_transcript.fetch = Mock(return_value=mock_transcript_fetch)
            mock_list.return_value.__iter__ = Mock(
                return_value=iter([first_transcript])
            )

            result = await get_video_transcript_preview("dQw4w9WgXcQ", max_chars=100)

            assert result["language"] == "en"

    @pytest.mark.asyncio
    async def test_get_preview_invalid_max_chars(self):
        """Test error when max_chars is invalid."""
        with pytest.raises(ValueError, match="max_chars must be greater than 0"):
            await get_video_transcript_preview("dQw4w9WgXcQ", max_chars=0)

        with pytest.raises(ValueError, match="max_chars must be greater than 0"):
            await get_video_transcript_preview("dQw4w9WgXcQ", max_chars=-10)

    @pytest.mark.asyncio
    async def test_get_preview_transcripts_disabled(self):
        """Test preview when transcripts are disabled."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.side_effect = TranscriptsDisabled(video_id="dQw4w9WgXcQ")

            with pytest.raises(TranscriptError, match="Transcripts are disabled"):
                await get_video_transcript_preview("dQw4w9WgXcQ")

    @pytest.mark.asyncio
    async def test_get_preview_language_not_found(self, mock_transcript_list):
        """Test preview when requested language not available."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            with pytest.raises(
                TranscriptError, match="No transcript found for language 'fr'"
            ):
                await get_video_transcript_preview("dQw4w9WgXcQ", language="fr")


# =============================================================================
# Tests for get_full_transcript
# =============================================================================


class TestGetFullTranscript:
    """Tests for get_full_transcript function."""

    @pytest.mark.asyncio
    async def test_get_full_transcript_success(
        self, mock_transcript_list, mock_transcript_fetch
    ):
        """Test successful full transcript retrieval."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            result = await get_full_transcript("dQw4w9WgXcQ", language="en")

            assert result["video_id"] == "dQw4w9WgXcQ"
            assert result["language"] == "en"
            assert len(result["transcript"]) == 5
            assert "full_text" in result

            # Check first entry
            assert result["transcript"][0]["text"] == "Welcome to this video"
            assert result["transcript"][0]["start"] == 0.0
            assert result["transcript"][0]["duration"] == 2.5

            # Check that full_text is concatenated
            assert "Welcome to this video" in result["full_text"]
            assert "NixOS" in result["full_text"]

    @pytest.mark.asyncio
    async def test_get_full_transcript_with_specific_language(
        self, mock_transcript_list, mock_transcript_fetch
    ):
        """Test full transcript with specific language code."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            result = await get_full_transcript("dQw4w9WgXcQ", language="es")

            assert result["language"] == "es"
            mock_transcript_list.find_transcript.assert_called_with(["es"])

    @pytest.mark.asyncio
    async def test_get_full_transcript_no_language_uses_first(
        self, mock_transcript_list, mock_transcript_fetch
    ):
        """Test that full transcript uses first available when language not specified."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            first_transcript = Mock()
            first_transcript.language_code = "en"
            first_transcript.is_generated = False
            first_transcript.fetch = Mock(return_value=mock_transcript_fetch)
            mock_list.return_value.__iter__ = Mock(
                return_value=iter([first_transcript])
            )

            result = await get_full_transcript("dQw4w9WgXcQ")

            assert result["language"] == "en"

    @pytest.mark.asyncio
    async def test_get_full_transcript_disabled(self):
        """Test full transcript when transcripts disabled."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.side_effect = TranscriptsDisabled(video_id="dQw4w9WgXcQ")

            with pytest.raises(TranscriptError, match="Transcripts are disabled"):
                await get_full_transcript("dQw4w9WgXcQ")

    @pytest.mark.asyncio
    async def test_get_full_transcript_not_found(self):
        """Test full transcript when no transcripts exist."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.side_effect = NoTranscriptFound(
                video_id="dQw4w9WgXcQ",
                requested_language_codes=["en"],
                transcript_data={},
            )

            with pytest.raises(TranscriptError, match="No transcripts found"):
                await get_full_transcript("dQw4w9WgXcQ")

    @pytest.mark.asyncio
    async def test_get_full_transcript_empty_video_id(self):
        """Test error when video_id is empty."""
        with pytest.raises(ValueError, match="video_id must be a non-empty string"):
            await get_full_transcript("")


# =============================================================================
# Tests for get_transcript_chunk
# =============================================================================


class TestGetTranscriptChunk:
    """Tests for get_transcript_chunk function."""

    @pytest.mark.asyncio
    async def test_get_chunk_success(self, mock_transcript_list, mock_transcript_fetch):
        """Test successful transcript chunk retrieval."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            first_transcript = Mock()
            first_transcript.language_code = "en"
            first_transcript.is_generated = False
            first_transcript.fetch = Mock(return_value=mock_transcript_fetch)
            mock_list.return_value.__iter__ = Mock(
                return_value=iter([first_transcript])
            )

            result = await get_transcript_chunk(
                "dQw4w9WgXcQ", start_index=0, chunk_size=2
            )

            assert result["video_id"] == "dQw4w9WgXcQ"
            assert result["language"] == "en"
            assert result["start_index"] == 0
            assert result["chunk_size"] == 2
            assert len(result["entries"]) == 2
            assert result["total_entries"] == 5
            assert result["has_more"] is True

    @pytest.mark.asyncio
    async def test_get_chunk_first_chunk(
        self, mock_transcript_list, mock_transcript_fetch
    ):
        """Test getting the first chunk of entries."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            first_transcript = Mock()
            first_transcript.language_code = "en"
            first_transcript.is_generated = False
            first_transcript.fetch = Mock(return_value=mock_transcript_fetch)
            mock_list.return_value.__iter__ = Mock(
                return_value=iter([first_transcript])
            )

            result = await get_transcript_chunk(
                "dQw4w9WgXcQ", start_index=0, chunk_size=3
            )

            assert result["entries"][0]["text"] == "Welcome to this video"
            assert result["entries"][2]["text"] == "Let's get started"
            assert result["has_more"] is True

    @pytest.mark.asyncio
    async def test_get_chunk_middle_chunk(
        self, mock_transcript_list, mock_transcript_fetch
    ):
        """Test getting a middle chunk of entries."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            first_transcript = Mock()
            first_transcript.language_code = "en"
            first_transcript.is_generated = False
            first_transcript.fetch = Mock(return_value=mock_transcript_fetch)
            mock_list.return_value.__iter__ = Mock(
                return_value=iter([first_transcript])
            )

            result = await get_transcript_chunk(
                "dQw4w9WgXcQ", start_index=2, chunk_size=2
            )

            assert len(result["entries"]) == 2
            assert result["entries"][0]["text"] == "Let's get started"
            assert result["has_more"] is True

    @pytest.mark.asyncio
    async def test_get_chunk_last_chunk(
        self, mock_transcript_list, mock_transcript_fetch
    ):
        """Test getting the last chunk of entries."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            first_transcript = Mock()
            first_transcript.language_code = "en"
            first_transcript.is_generated = False
            first_transcript.fetch = Mock(return_value=mock_transcript_fetch)
            mock_list.return_value.__iter__ = Mock(
                return_value=iter([first_transcript])
            )

            result = await get_transcript_chunk(
                "dQw4w9WgXcQ", start_index=4, chunk_size=2
            )

            assert len(result["entries"]) == 1  # Only 1 entry left
            assert result["entries"][0]["text"] == "Then configure your system"
            assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_get_chunk_invalid_index_negative(self):
        """Test error when start_index is negative."""
        with pytest.raises(ValueError, match="start_index must be >= 0"):
            await get_transcript_chunk("dQw4w9WgXcQ", start_index=-1, chunk_size=10)

    @pytest.mark.asyncio
    async def test_get_chunk_invalid_index_out_of_bounds(
        self, mock_transcript_list, mock_transcript_fetch
    ):
        """Test error when start_index is out of bounds."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            first_transcript = Mock()
            first_transcript.language_code = "en"
            first_transcript.is_generated = False
            first_transcript.fetch = Mock(return_value=mock_transcript_fetch)
            mock_list.return_value.__iter__ = Mock(
                return_value=iter([first_transcript])
            )

            with pytest.raises(ValueError, match=r"start_index .* out of bounds"):
                await get_transcript_chunk("dQw4w9WgXcQ", start_index=10, chunk_size=2)

    @pytest.mark.asyncio
    async def test_get_chunk_invalid_chunk_size(self):
        """Test error when chunk_size is invalid."""
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            await get_transcript_chunk("dQw4w9WgXcQ", start_index=0, chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            await get_transcript_chunk("dQw4w9WgXcQ", start_index=0, chunk_size=-5)

    @pytest.mark.asyncio
    async def test_get_chunk_with_language(
        self, mock_transcript_list, mock_transcript_fetch
    ):
        """Test chunk retrieval with specific language."""
        with patch("app.tools.youtube.transcripts.YouTubeTranscriptApi") as mock_api:
            mock_list = mock_api.return_value.list
            mock_list.return_value = mock_transcript_list

            result = await get_transcript_chunk(
                "dQw4w9WgXcQ", start_index=0, chunk_size=2, language="es"
            )

            assert result["language"] == "es"
            mock_transcript_list.find_transcript.assert_called_with(["es"])
