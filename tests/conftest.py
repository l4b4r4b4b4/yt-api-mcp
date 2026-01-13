"""Pytest configuration and fixtures for yt-mcp tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp_refcache import PreviewConfig, PreviewStrategy, RefCache, SizeMode

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def clear_semantic_caches() -> Generator[None, None, None]:
    """Clear lru_cache on semantic search singletons between tests.

    This fixture runs automatically before each test to prevent test pollution
    from cached vector stores, embeddings, and config objects.

    The lru_cache on these functions causes issues when:
    1. A test calls the real function (caches the real instance)
    2. A later test tries to mock the function (gets cached real instance instead)

    By clearing caches before each test, we ensure a clean slate.
    """
    # Import the cached functions
    from app.tools.youtube.semantic.config import get_semantic_config
    from app.tools.youtube.semantic.embeddings import get_embeddings
    from app.tools.youtube.semantic.store import get_vector_store

    # Clear all semantic search related caches before the test
    get_vector_store.cache_clear()
    get_embeddings.cache_clear()
    get_semantic_config.cache_clear()

    yield

    # Also clear after the test to be safe
    get_vector_store.cache_clear()
    get_embeddings.cache_clear()
    get_semantic_config.cache_clear()


@pytest.fixture
def cache() -> Generator[RefCache, None, None]:
    """Create a fresh RefCache instance for testing."""
    test_cache = RefCache(
        name="test_fastmcp_template",
        default_ttl=3600,
        preview_config=PreviewConfig(
            size_mode=SizeMode.CHARACTER,
            max_size=500,
            default_strategy=PreviewStrategy.SAMPLE,
        ),
    )
    yield test_cache
    # Cleanup after test
    test_cache.clear()


@pytest.fixture
def sample_items() -> list[dict[str, int | str]]:
    """Generate sample items for testing."""
    return [{"id": i, "name": f"item_{i}", "value": i * 10} for i in range(100)]
