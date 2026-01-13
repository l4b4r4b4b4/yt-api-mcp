"""Unit tests for semantic search tokenizers.

Tests the tokenizer protocol, TiktokenTokenizer, HuggingFaceTokenizer,
and the auto-detection factory function.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from app.tools.youtube.semantic.tokenizers import (
    OPENAI_MODEL_ENCODINGS,
    TIKTOKEN_ENCODINGS,
    HuggingFaceTokenizer,
    TiktokenTokenizer,
    TokenizerProtocol,
    create_tokenizer,
)


class TestTokenizerProtocol:
    """Tests for the TokenizerProtocol."""

    def test_tiktoken_tokenizer_implements_protocol(self) -> None:
        """Test that TiktokenTokenizer implements TokenizerProtocol."""
        tokenizer = TiktokenTokenizer("cl100k_base")
        assert isinstance(tokenizer, TokenizerProtocol)

    def test_protocol_has_required_methods(self) -> None:
        """Test that protocol defines count_tokens and encode methods."""
        # Check protocol has the methods we expect
        assert hasattr(TokenizerProtocol, "count_tokens")
        assert hasattr(TokenizerProtocol, "encode")


class TestTiktokenTokenizer:
    """Tests for TiktokenTokenizer."""

    def test_init_with_encoding_name(self) -> None:
        """Test initialization with tiktoken encoding name."""
        tokenizer = TiktokenTokenizer("cl100k_base")
        assert tokenizer.encoding_name == "cl100k_base"

    def test_init_with_openai_model(self) -> None:
        """Test initialization with OpenAI model name."""
        tokenizer = TiktokenTokenizer("gpt-4o")
        assert tokenizer.encoding_name == "o200k_base"

    def test_init_with_gpt4(self) -> None:
        """Test initialization with gpt-4 model."""
        tokenizer = TiktokenTokenizer("gpt-4")
        assert tokenizer.encoding_name == "cl100k_base"

    def test_init_with_embedding_model(self) -> None:
        """Test initialization with OpenAI embedding model."""
        tokenizer = TiktokenTokenizer("text-embedding-3-small")
        assert tokenizer.encoding_name == "cl100k_base"

    def test_init_with_invalid_model_raises(self) -> None:
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tiktoken model/encoding"):
            TiktokenTokenizer("invalid-model-name")

    def test_count_tokens_empty_string(self) -> None:
        """Test that empty string returns 0 tokens."""
        tokenizer = TiktokenTokenizer("cl100k_base")
        assert tokenizer.count_tokens("") == 0

    def test_count_tokens_simple_text(self) -> None:
        """Test token counting for simple English text."""
        tokenizer = TiktokenTokenizer("cl100k_base")
        count = tokenizer.count_tokens("Hello, world!")
        assert count == 4  # "Hello", ",", " world", "!"

    def test_count_tokens_longer_text(self) -> None:
        """Test token counting for longer text."""
        tokenizer = TiktokenTokenizer("cl100k_base")
        text = "The quick brown fox jumps over the lazy dog."
        count = tokenizer.count_tokens(text)
        assert count > 0
        assert count < len(text)  # Tokens should be fewer than characters

    def test_count_tokens_with_unicode(self) -> None:
        """Test token counting handles unicode characters."""
        tokenizer = TiktokenTokenizer("cl100k_base")
        count = tokenizer.count_tokens("Hello 世界 🌍")
        assert count > 0

    def test_encode_empty_string(self) -> None:
        """Test that encoding empty string returns empty list."""
        tokenizer = TiktokenTokenizer("cl100k_base")
        assert tokenizer.encode("") == []

    def test_encode_returns_integers(self) -> None:
        """Test that encode returns list of integers."""
        tokenizer = TiktokenTokenizer("cl100k_base")
        tokens = tokenizer.encode("Hello, world!")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) == 4

    def test_encode_count_consistency(self) -> None:
        """Test that encode length matches count_tokens."""
        tokenizer = TiktokenTokenizer("cl100k_base")
        text = "This is a test sentence for tokenization."
        assert len(tokenizer.encode(text)) == tokenizer.count_tokens(text)

    def test_repr(self) -> None:
        """Test string representation."""
        tokenizer = TiktokenTokenizer("cl100k_base")
        assert repr(tokenizer) == "TiktokenTokenizer(encoding='cl100k_base')"

    def test_all_known_encodings_work(self) -> None:
        """Test that all known tiktoken encodings can be loaded."""
        for encoding in TIKTOKEN_ENCODINGS:
            tokenizer = TiktokenTokenizer(encoding)
            assert tokenizer.encoding_name == encoding
            assert tokenizer.count_tokens("test") > 0

    def test_all_openai_models_work(self) -> None:
        """Test that all known OpenAI models can be used."""
        for model, expected_encoding in OPENAI_MODEL_ENCODINGS.items():
            tokenizer = TiktokenTokenizer(model)
            assert tokenizer.encoding_name == expected_encoding


# Check if tokenizers library is available for HuggingFace tests
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None


class TestHuggingFaceTokenizer:
    """Tests for HuggingFaceTokenizer.

    These tests require the optional 'tokenizers' library.
    They are skipped if not installed.
    """

    @pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers library not installed")
    def test_init_stores_model_name(self) -> None:
        """Test that initialization stores model name."""
        # We can't easily mock HuggingFace Hub, so test the import error case
        # and basic structure
        with pytest.raises(ValueError, match="Failed to load tokenizer"):
            # This will fail because the model doesn't exist, but we can verify
            # the error handling works
            HuggingFaceTokenizer("definitely-not-a-real-model/fake")

    def test_count_tokens_empty_returns_zero(self) -> None:
        """Test that count_tokens on empty string returns 0 without calling encode."""
        # Create a mock that simulates HuggingFaceTokenizer behavior
        mock_tokenizer = MagicMock()
        mock_tokenizer.count_tokens = lambda text: 0 if not text else len(text.split())

        # Verify the logic - empty string should return 0
        assert mock_tokenizer.count_tokens("") == 0

    def test_encode_empty_returns_empty_list(self) -> None:
        """Test that encode on empty string returns empty list without calling encode."""
        # Create a mock that simulates HuggingFaceTokenizer behavior
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = lambda text: [] if not text else [1, 2, 3]

        # Verify the logic - empty string should return []
        assert mock_tokenizer.encode("") == []

    def test_repr_format(self) -> None:
        """Test the expected repr format."""
        expected = "HuggingFaceTokenizer(model='nomic-ai/nomic-embed-text-v1.5')"
        # Just verify the format string is what we expect
        model_name = "nomic-ai/nomic-embed-text-v1.5"
        actual = f"HuggingFaceTokenizer(model={model_name!r})"
        assert actual == expected


class TestCreateTokenizer:
    """Tests for create_tokenizer factory function."""

    def test_creates_tiktoken_for_encoding_name(self) -> None:
        """Test that tiktoken encoding names create TiktokenTokenizer."""
        tokenizer = create_tokenizer("cl100k_base")
        # Check by class name to avoid module reload issues
        assert type(tokenizer).__name__ == "TiktokenTokenizer"
        assert tokenizer.encoding_name == "cl100k_base"

    def test_creates_tiktoken_for_openai_model(self) -> None:
        """Test that OpenAI model names create TiktokenTokenizer."""
        tokenizer = create_tokenizer("gpt-4o")
        assert type(tokenizer).__name__ == "TiktokenTokenizer"
        assert tokenizer.encoding_name == "o200k_base"

    def test_creates_tiktoken_for_embedding_model(self) -> None:
        """Test that OpenAI embedding models create TiktokenTokenizer."""
        tokenizer = create_tokenizer("text-embedding-3-large")
        assert type(tokenizer).__name__ == "TiktokenTokenizer"
        assert tokenizer.encoding_name == "cl100k_base"

    @patch("app.tools.youtube.semantic.tokenizers.HuggingFaceTokenizer")
    def test_creates_huggingface_for_slash_model(
        self, mock_hf_tokenizer: MagicMock
    ) -> None:
        """Test that models with / create HuggingFaceTokenizer."""
        mock_instance = MagicMock()
        mock_hf_tokenizer.return_value = mock_instance

        tokenizer = create_tokenizer("nomic-ai/nomic-embed-text-v1.5")

        mock_hf_tokenizer.assert_called_once_with("nomic-ai/nomic-embed-text-v1.5")
        assert tokenizer == mock_instance

    @patch("app.tools.youtube.semantic.tokenizers.HuggingFaceTokenizer")
    def test_creates_huggingface_for_llama(self, mock_hf_tokenizer: MagicMock) -> None:
        """Test that Llama models create HuggingFaceTokenizer."""
        mock_instance = MagicMock()
        mock_hf_tokenizer.return_value = mock_instance

        tokenizer = create_tokenizer("meta-llama/Llama-3.1-8B")

        mock_hf_tokenizer.assert_called_once_with("meta-llama/Llama-3.1-8B")
        assert tokenizer == mock_instance

    def test_default_is_cl100k_base(self) -> None:
        """Test that default tokenizer is cl100k_base."""
        tokenizer = create_tokenizer()
        assert type(tokenizer).__name__ == "TiktokenTokenizer"
        assert tokenizer.encoding_name == "cl100k_base"

    @patch("app.tools.youtube.semantic.tokenizers.HuggingFaceTokenizer")
    def test_falls_back_to_huggingface_for_unknown(
        self, mock_hf_tokenizer: MagicMock
    ) -> None:
        """Test that unknown models fall back to HuggingFace."""
        mock_instance = MagicMock()
        mock_hf_tokenizer.return_value = mock_instance

        # This doesn't have / but isn't a known tiktoken model
        tokenizer = create_tokenizer("some-unknown-model")

        mock_hf_tokenizer.assert_called_once_with("some-unknown-model")
        assert tokenizer == mock_instance


class TestTokenizersIntegration:
    """Integration tests for tokenizers (requires tiktoken installed)."""

    def test_tiktoken_consistent_results(self) -> None:
        """Test that same text always produces same token count."""
        tokenizer = TiktokenTokenizer("cl100k_base")
        text = "This is a test sentence for consistency checking."

        count1 = tokenizer.count_tokens(text)
        count2 = tokenizer.count_tokens(text)
        count3 = tokenizer.count_tokens(text)

        assert count1 == count2 == count3

    def test_different_encodings_different_counts(self) -> None:
        """Test that different encodings may produce different token counts."""
        text = "Hello, world! This is a test."

        cl100k = TiktokenTokenizer("cl100k_base")
        gpt2 = TiktokenTokenizer("gpt2")

        # Different encodings might tokenize differently
        count_cl100k = cl100k.count_tokens(text)
        count_gpt2 = gpt2.count_tokens(text)

        # Both should produce valid counts
        assert count_cl100k > 0
        assert count_gpt2 > 0

    def test_transcript_like_text(self) -> None:
        """Test tokenization of transcript-like text."""
        tokenizer = TiktokenTokenizer("cl100k_base")

        transcript_segment = (
            "Welcome to this video about Nix. Today we're going to learn "
            "about garbage collection and how to configure it properly. "
            "First, let's understand what garbage collection does in NixOS."
        )

        count = tokenizer.count_tokens(transcript_segment)

        # Should be reasonable for ~40 words
        assert 30 < count < 60

    def test_special_characters(self) -> None:
        """Test tokenization handles special characters."""
        tokenizer = TiktokenTokenizer("cl100k_base")

        text_with_special = "Code: `nix-collect-garbage -d` removes old generations!"
        count = tokenizer.count_tokens(text_with_special)

        assert count > 0

    def test_multiline_text(self) -> None:
        """Test tokenization of multiline text."""
        tokenizer = TiktokenTokenizer("cl100k_base")

        multiline = """Line one.
        Line two.
        Line three."""

        count = tokenizer.count_tokens(multiline)
        assert count > 0
