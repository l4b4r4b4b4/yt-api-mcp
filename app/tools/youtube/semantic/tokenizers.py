"""Tokenizer abstractions for semantic transcript chunking.

Provides a protocol-based tokenizer interface with implementations for:
- tiktoken (OpenAI encodings and models)
- HuggingFace tokenizers (open source embedding and LLM models)

The factory function auto-detects the appropriate backend based on model name.

Example:
    >>> from app.tools.youtube.semantic.tokenizers import create_tokenizer
    >>> # tiktoken (default)
    >>> tokenizer = create_tokenizer("cl100k_base")
    >>> tokenizer.count_tokens("Hello, world!")
    4
    >>> # OpenAI model name
    >>> tokenizer = create_tokenizer("gpt-4o")
    >>> tokenizer.count_tokens("Hello, world!")
    4
    >>> # HuggingFace model (requires optional deps)
    >>> tokenizer = create_tokenizer("nomic-ai/nomic-embed-text-v1.5")
    >>> tokenizer.count_tokens("Hello, world!")
    5
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Known tiktoken encodings
TIKTOKEN_ENCODINGS = frozenset(
    {
        "cl100k_base",  # GPT-3.5/4, text-embedding-3-*
        "o200k_base",  # GPT-4o
        "p50k_base",  # Codex models
        "p50k_edit",  # Edit models
        "r50k_base",  # GPT-3 (davinci)
        "gpt2",  # GPT-2
    }
)

# OpenAI models that use tiktoken (maps to encoding)
# Reference: https://github.com/openai/tiktoken/blob/main/tiktoken/model.py
OPENAI_MODEL_ENCODINGS: dict[str, str] = {
    # GPT-4o family
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4o-2024-05-13": "o200k_base",
    "gpt-4o-2024-08-06": "o200k_base",
    "gpt-4o-mini-2024-07-18": "o200k_base",
    # GPT-4 family
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4-32k-0613": "cl100k_base",
    # GPT-3.5 family
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    # Embedding models
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
}


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol for text tokenizers.

    Defines the interface for tokenizers used in transcript chunking.
    Implementations must provide token counting functionality.
    """

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text: The text to tokenize.

        Returns:
            Number of tokens in the text.
        """
        ...

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs.

        Args:
            text: The text to encode.

        Returns:
            List of token IDs.
        """
        ...


class TiktokenTokenizer:
    """Tokenizer using OpenAI's tiktoken library.

    Supports tiktoken encoding names (e.g., 'cl100k_base') and
    OpenAI model names (e.g., 'gpt-4o', 'text-embedding-3-small').

    Attributes:
        encoding_name: The tiktoken encoding being used.
    """

    def __init__(self, model: str = "cl100k_base") -> None:
        """Initialize tiktoken tokenizer.

        Args:
            model: Either a tiktoken encoding name or an OpenAI model name.
                Encoding names: 'cl100k_base', 'o200k_base', 'p50k_base', 'gpt2'
                Model names: 'gpt-4o', 'gpt-4', 'text-embedding-3-small', etc.

        Raises:
            ValueError: If the model/encoding is not recognized.
            ImportError: If tiktoken is not installed.
        """
        try:
            import tiktoken
        except ImportError as exc:
            msg = (
                "tiktoken is required for TiktokenTokenizer. "
                "Install with: uv add tiktoken"
            )
            raise ImportError(msg) from exc

        # Determine encoding name from model
        if model in TIKTOKEN_ENCODINGS:
            self.encoding_name = model
        elif model in OPENAI_MODEL_ENCODINGS:
            self.encoding_name = OPENAI_MODEL_ENCODINGS[model]
        else:
            # Try tiktoken's model-to-encoding lookup
            try:
                encoding = tiktoken.encoding_for_model(model)
                self.encoding_name = encoding.name
            except KeyError as exc:
                msg = (
                    f"Unknown tiktoken model/encoding: {model}. "
                    f"Valid encodings: {sorted(TIKTOKEN_ENCODINGS)}. "
                    f"Valid models: {sorted(OPENAI_MODEL_ENCODINGS.keys())}"
                )
                raise ValueError(msg) from exc

        self._encoding = tiktoken.get_encoding(self.encoding_name)
        logger.debug(
            "Initialized TiktokenTokenizer with encoding: %s", self.encoding_name
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: The text to tokenize.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: The text to encode.

        Returns:
            List of token IDs.
        """
        if not text:
            return []
        return self._encoding.encode(text)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TiktokenTokenizer(encoding={self.encoding_name!r})"


class HuggingFaceTokenizer:
    """Tokenizer using HuggingFace tokenizers library.

    Supports any model available on HuggingFace Hub, including:
    - Embedding models: 'nomic-ai/nomic-embed-text-v1.5', 'sentence-transformers/*'
    - LLMs: 'meta-llama/Llama-3.1-8B', 'mistralai/Mistral-7B-v0.1'

    Requires optional dependencies: tokenizers, huggingface_hub

    Attributes:
        model_name: The HuggingFace model name.
    """

    def __init__(self, model: str) -> None:
        """Initialize HuggingFace tokenizer.

        Args:
            model: HuggingFace model name (e.g., 'nomic-ai/nomic-embed-text-v1.5').

        Raises:
            ImportError: If tokenizers/huggingface_hub are not installed.
            ValueError: If the model tokenizer cannot be loaded.
        """
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            msg = (
                "tokenizers is required for HuggingFaceTokenizer. "
                "Install with: uv add tokenizers huggingface_hub"
            )
            raise ImportError(msg) from exc

        self.model_name = model

        try:
            # Try to load from HuggingFace Hub
            self._tokenizer = Tokenizer.from_pretrained(model)
            logger.debug("Initialized HuggingFaceTokenizer with model: %s", model)
        except Exception as exc:
            msg = (
                f"Failed to load tokenizer for model: {model}. "
                f"Ensure the model exists on HuggingFace Hub and has a tokenizer. "
                f"Error: {exc}"
            )
            raise ValueError(msg) from exc

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using HuggingFace tokenizer.

        Args:
            text: The text to tokenize.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        encoding = self._tokenizer.encode(text)
        return len(encoding.ids)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: The text to encode.

        Returns:
            List of token IDs.
        """
        if not text:
            return []
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def __repr__(self) -> str:
        """Return string representation."""
        return f"HuggingFaceTokenizer(model={self.model_name!r})"


def _is_huggingface_model(model: str) -> bool:
    """Check if model string looks like a HuggingFace model name.

    HuggingFace models typically have format: 'organization/model-name'

    Args:
        model: The model string to check.

    Returns:
        True if it looks like a HuggingFace model name.
    """
    return "/" in model


def _is_tiktoken_model(model: str) -> bool:
    """Check if model string is a known tiktoken encoding or OpenAI model.

    Args:
        model: The model string to check.

    Returns:
        True if it's a known tiktoken encoding or OpenAI model.
    """
    return model in TIKTOKEN_ENCODINGS or model in OPENAI_MODEL_ENCODINGS


def create_tokenizer(model: str = "cl100k_base") -> TokenizerProtocol:
    """Create a tokenizer with auto-detected backend.

    Automatically selects the appropriate tokenizer backend based on
    the model name format:
    - Contains "/" → HuggingFace (e.g., 'nomic-ai/nomic-embed-text-v1.5')
    - Known tiktoken encoding → tiktoken (e.g., 'cl100k_base')
    - Known OpenAI model → tiktoken (e.g., 'gpt-4o', 'text-embedding-3-small')
    - Otherwise → tries HuggingFace as fallback

    Args:
        model: Model or encoding name. Examples:
            - 'cl100k_base' - tiktoken encoding (default)
            - 'gpt-4o' - OpenAI model → tiktoken
            - 'text-embedding-3-small' - OpenAI embeddings → tiktoken
            - 'nomic-ai/nomic-embed-text-v1.5' - HuggingFace
            - 'meta-llama/Llama-3.1-8B' - HuggingFace

    Returns:
        Tokenizer instance implementing TokenizerProtocol.

    Raises:
        ValueError: If the model is not recognized by any backend.
        ImportError: If required dependencies are not installed.

    Example:
        >>> tokenizer = create_tokenizer("cl100k_base")
        >>> tokenizer.count_tokens("Hello, world!")
        4
        >>> tokenizer = create_tokenizer("gpt-4o")
        >>> tokenizer.count_tokens("Hello, world!")
        4
    """
    # Check for HuggingFace model format first (contains /)
    if _is_huggingface_model(model):
        logger.info("Auto-detected HuggingFace model: %s", model)
        return HuggingFaceTokenizer(model)

    # Check for known tiktoken encoding or OpenAI model
    if _is_tiktoken_model(model):
        logger.info("Auto-detected tiktoken model: %s", model)
        return TiktokenTokenizer(model)

    # Try tiktoken first (might be an unknown but valid OpenAI model)
    try:
        return TiktokenTokenizer(model)
    except ValueError:
        pass

    # Fall back to HuggingFace
    logger.info("Falling back to HuggingFace for unknown model: %s", model)
    return HuggingFaceTokenizer(model)


# Convenience exports
__all__ = [
    "OPENAI_MODEL_ENCODINGS",
    "TIKTOKEN_ENCODINGS",
    "HuggingFaceTokenizer",
    "TiktokenTokenizer",
    "TokenizerProtocol",
    "create_tokenizer",
]
