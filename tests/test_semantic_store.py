"""Unit tests for semantic search vector store."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.tools.youtube.semantic.config import SemanticSearchConfig
from app.tools.youtube.semantic.store import create_vector_store, get_vector_store


class TestCreateVectorStore:
    """Tests for create_vector_store factory function."""

    @patch("app.tools.youtube.semantic.store.Chroma")
    @patch("app.tools.youtube.semantic.store.create_embeddings")
    def test_creates_store_with_config(
        self,
        mock_create_embeddings: MagicMock,
        mock_chroma: MagicMock,
    ) -> None:
        """Test that vector store is created with correct config values."""
        config = SemanticSearchConfig(
            collection_name="test_collection",
            persist_directory="/tmp/test",
        )

        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings

        mock_store = MagicMock()
        mock_chroma.return_value = mock_store

        result = create_vector_store(config)

        assert result == mock_store
        mock_chroma.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=mock_embeddings,
            collection_configuration=config.collection_configuration,
            persist_directory="/tmp/test",
        )

    @patch("app.tools.youtube.semantic.store.Chroma")
    @patch("app.tools.youtube.semantic.store.create_embeddings")
    def test_creates_store_with_custom_embeddings(
        self,
        mock_create_embeddings: MagicMock,
        mock_chroma: MagicMock,
    ) -> None:
        """Test that custom embeddings instance can be passed."""
        config = SemanticSearchConfig()
        custom_embeddings = MagicMock()

        mock_store = MagicMock()
        mock_chroma.return_value = mock_store

        result = create_vector_store(config, embeddings=custom_embeddings)

        assert result == mock_store
        # Should not create embeddings since custom one was provided
        mock_create_embeddings.assert_not_called()
        # Should use the custom embeddings
        call_kwargs = mock_chroma.call_args.kwargs
        assert call_kwargs["embedding_function"] == custom_embeddings

    @patch("app.tools.youtube.semantic.store.Chroma")
    @patch("app.tools.youtube.semantic.store.create_embeddings")
    def test_passes_hnsw_configuration(
        self,
        mock_create_embeddings: MagicMock,
        mock_chroma: MagicMock,
    ) -> None:
        """Test that HNSW configuration is passed to Chroma."""
        config = SemanticSearchConfig(
            hnsw_space="cosine",
            hnsw_max_neighbors=48,
            hnsw_ef_construction=200,
            hnsw_ef_search=128,
        )

        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        mock_chroma.return_value = MagicMock()

        create_vector_store(config)

        call_kwargs = mock_chroma.call_args.kwargs
        expected_config = {
            "hnsw": {
                "space": "cosine",
                "max_neighbors": 48,
                "ef_construction": 200,
                "ef_search": 128,
            }
        }
        assert call_kwargs["collection_configuration"] == expected_config

    @patch("app.tools.youtube.semantic.store.Chroma")
    @patch("app.tools.youtube.semantic.store.create_embeddings")
    def test_handles_none_persist_directory(
        self,
        mock_create_embeddings: MagicMock,
        mock_chroma: MagicMock,
    ) -> None:
        """Test that None persist_directory creates in-memory store."""
        config = SemanticSearchConfig(persist_directory=None)

        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        mock_chroma.return_value = MagicMock()

        create_vector_store(config)

        call_kwargs = mock_chroma.call_args.kwargs
        assert call_kwargs["persist_directory"] is None


class TestGetVectorStore:
    """Tests for get_vector_store cached factory function."""

    @patch("app.tools.youtube.semantic.store.create_vector_store")
    @patch("app.tools.youtube.semantic.store.get_semantic_config")
    def test_uses_global_config(
        self,
        mock_get_config: MagicMock,
        mock_create_store: MagicMock,
    ) -> None:
        """Test that get_vector_store uses global semantic config."""
        # Clear the lru_cache
        get_vector_store.cache_clear()

        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        result = get_vector_store()

        assert result == mock_store
        mock_get_config.assert_called_once()
        mock_create_store.assert_called_once_with(mock_config)

    @patch("app.tools.youtube.semantic.store.create_vector_store")
    @patch("app.tools.youtube.semantic.store.get_semantic_config")
    def test_caches_store_instance(
        self,
        mock_get_config: MagicMock,
        mock_create_store: MagicMock,
    ) -> None:
        """Test that get_vector_store caches the result."""
        # Clear the lru_cache
        get_vector_store.cache_clear()

        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_store = MagicMock()
        mock_create_store.return_value = mock_store

        # Call twice
        result1 = get_vector_store()
        result2 = get_vector_store()

        # Should return same instance
        assert result1 is result2

        # create_vector_store should only be called once due to caching
        assert mock_create_store.call_count == 1


class TestVectorStoreIntegration:
    """Integration tests for vector store (requires chromadb installed).

    These tests are marked as slow and can be skipped in CI with: pytest -m "not slow"
    """

    @pytest.mark.slow
    def test_in_memory_store_operations(self) -> None:
        """Test basic operations with in-memory store."""
        config = SemanticSearchConfig(persist_directory=None)
        store = create_vector_store(config)

        # Add documents
        texts = [
            "How to configure Nix garbage collection",
            "NixOS home-manager setup guide",
            "Flakes tutorial for beginners",
        ]
        metadatas = [
            {"video_id": "v1", "channel_id": "c1"},
            {"video_id": "v2", "channel_id": "c1"},
            {"video_id": "v3", "channel_id": "c2"},
        ]

        store.add_texts(texts, metadatas=metadatas)

        # Search
        results = store.similarity_search("garbage collection nix", k=2)

        assert len(results) == 2
        # First result should be about garbage collection
        assert "garbage" in results[0].page_content.lower()

    @pytest.mark.slow
    def test_metadata_filtering(self) -> None:
        """Test that metadata filtering works correctly."""
        config = SemanticSearchConfig(persist_directory=None)
        store = create_vector_store(config)

        texts = [
            "Document from channel 1",
            "Another from channel 1",
            "Document from channel 2",
        ]
        metadatas = [
            {"video_id": "v1", "channel_id": "c1"},
            {"video_id": "v2", "channel_id": "c1"},
            {"video_id": "v3", "channel_id": "c2"},
        ]

        store.add_texts(texts, metadatas=metadatas)

        # Filter by channel_id
        results = store.similarity_search(
            "document",
            k=10,
            filter={"channel_id": "c1"},
        )

        assert len(results) == 2
        for r in results:
            assert r.metadata["channel_id"] == "c1"

    @pytest.mark.slow
    def test_persistence(self) -> None:
        """Test that documents persist across store instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = str(Path(tmpdir) / "test_chroma")

            # Create store and add documents
            config = SemanticSearchConfig(persist_directory=persist_path)
            store1 = create_vector_store(config)

            store1.add_texts(
                ["Test document for persistence"],
                metadatas=[{"test_id": "1"}],
            )

            # Create new store instance pointing to same directory
            store2 = create_vector_store(config)

            # Should find the document
            results = store2.similarity_search("persistence", k=1)

            assert len(results) == 1
            assert "persistence" in results[0].page_content.lower()

    @pytest.mark.slow
    def test_similarity_search_with_score(self) -> None:
        """Test similarity search returns scores."""
        config = SemanticSearchConfig(persist_directory=None)
        store = create_vector_store(config)

        texts = [
            "Nix garbage collection tutorial",
            "Python programming basics",
        ]

        store.add_texts(texts)

        # Search with scores
        results = store.similarity_search_with_score("nix garbage", k=2)

        assert len(results) == 2
        # Results are tuples of (Document, score)
        doc1, score1 = results[0]
        _doc2, score2 = results[1]

        # First result should be more relevant (lower distance = higher similarity)
        assert (
            "nix" in doc1.page_content.lower() or "garbage" in doc1.page_content.lower()
        )
        # Scores should be numeric
        assert isinstance(score1, float)
        assert isinstance(score2, float)

    @pytest.mark.slow
    def test_collection_name_config(self) -> None:
        """Test that custom collection name is used."""
        config = SemanticSearchConfig(
            persist_directory=None,
            collection_name="custom_test_collection",
        )
        store = create_vector_store(config)

        # Add a document to ensure collection is created
        store.add_texts(["Test document"])

        # The collection should exist with the custom name
        # Access underlying Chroma client to verify
        assert store._collection.name == "custom_test_collection"
