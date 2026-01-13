"""Unit tests for semantic search configuration and embeddings."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tools.youtube.semantic.config import (
    SemanticSearchConfig,
    get_semantic_config,
)
from app.tools.youtube.semantic.embeddings import create_embeddings, get_embeddings


class TestSemanticSearchConfig:
    """Tests for SemanticSearchConfig settings model."""

    def test_default_values(self) -> None:
        """Test that default configuration values are set correctly."""
        config = SemanticSearchConfig()

        # Embedding defaults
        assert config.embedding_model == "nomic-embed-text-v1.5"
        assert config.embedding_dimensionality == 512
        assert config.embedding_inference_mode == "local"

        # HNSW defaults
        assert config.hnsw_space == "cosine"
        assert config.hnsw_max_neighbors == 48
        assert config.hnsw_ef_construction == 200
        assert config.hnsw_ef_search == 128

        # Chunking defaults
        assert config.chunk_size == 800
        assert config.chunk_overlap == 100

        # Collection defaults
        assert config.collection_name == "youtube_transcripts"

    def test_custom_values(self) -> None:
        """Test that custom configuration values override defaults."""
        config = SemanticSearchConfig(
            embedding_dimensionality=256,
            hnsw_max_neighbors=64,
            chunk_size=1000,
            collection_name="custom_collection",
        )

        assert config.embedding_dimensionality == 256
        assert config.hnsw_max_neighbors == 64
        assert config.chunk_size == 1000
        assert config.collection_name == "custom_collection"

    def test_matryoshka_dimensionality_options(self) -> None:
        """Test that all valid Matryoshka dimensionality values are accepted."""
        valid_dims = [64, 128, 256, 512, 768]

        for dim in valid_dims:
            config = SemanticSearchConfig(embedding_dimensionality=dim)
            assert config.embedding_dimensionality == dim

    def test_invalid_dimensionality_rejected(self) -> None:
        """Test that invalid dimensionality values are rejected."""
        with pytest.raises(ValueError, match="must be one of"):
            SemanticSearchConfig(embedding_dimensionality=100)

    def test_hnsw_config_property(self) -> None:
        """Test that hnsw_config property returns correct dictionary."""
        config = SemanticSearchConfig(
            hnsw_space="cosine",
            hnsw_max_neighbors=48,
            hnsw_ef_construction=200,
            hnsw_ef_search=128,
        )

        hnsw_config = config.hnsw_config

        assert hnsw_config == {
            "space": "cosine",
            "max_neighbors": 48,
            "ef_construction": 200,
            "ef_search": 128,
        }

    def test_collection_configuration_property(self) -> None:
        """Test that collection_configuration wraps hnsw_config correctly."""
        config = SemanticSearchConfig()

        collection_config = config.collection_configuration

        assert "hnsw" in collection_config
        assert collection_config["hnsw"] == config.hnsw_config

    def test_inference_mode_options(self) -> None:
        """Test that all valid inference modes are accepted."""
        valid_modes = ["local", "remote", "dynamic"]

        for mode in valid_modes:
            config = SemanticSearchConfig(embedding_inference_mode=mode)  # type: ignore[arg-type]
            assert config.embedding_inference_mode == mode

    def test_persist_directory_expansion(self) -> None:
        """Test that ~ is expanded in persist_directory."""
        config = SemanticSearchConfig(persist_directory="~/test/path")

        assert "~" not in config.persist_directory  # type: ignore[operator]
        assert "/test/path" in config.persist_directory  # type: ignore[operator]

    def test_persist_directory_none(self) -> None:
        """Test that persist_directory can be None for in-memory only."""
        config = SemanticSearchConfig(persist_directory=None)

        assert config.persist_directory is None

    @patch.dict(
        "os.environ",
        {
            "SEMANTIC_EMBEDDING_DIMENSIONALITY": "256",
            "SEMANTIC_HNSW_MAX_NEIGHBORS": "64",
            "SEMANTIC_CHUNK_SIZE": "1200",
        },
    )
    def test_environment_variable_loading(self) -> None:
        """Test that configuration loads from environment variables."""
        # Clear the lru_cache to pick up new env vars
        get_semantic_config.cache_clear()

        config = SemanticSearchConfig()

        assert config.embedding_dimensionality == 256
        assert config.hnsw_max_neighbors == 64
        assert config.chunk_size == 1200

    def test_hnsw_max_neighbors_bounds(self) -> None:
        """Test that hnsw_max_neighbors respects min/max bounds."""
        # Valid within bounds
        config = SemanticSearchConfig(hnsw_max_neighbors=4)
        assert config.hnsw_max_neighbors == 4

        config = SemanticSearchConfig(hnsw_max_neighbors=128)
        assert config.hnsw_max_neighbors == 128

        # Invalid below minimum
        with pytest.raises(ValueError, match="greater than or equal to"):
            SemanticSearchConfig(hnsw_max_neighbors=3)

        # Invalid above maximum
        with pytest.raises(ValueError, match="less than or equal to"):
            SemanticSearchConfig(hnsw_max_neighbors=129)

    def test_chunk_size_bounds(self) -> None:
        """Test that chunk_size respects min/max bounds."""
        # Valid within bounds
        config = SemanticSearchConfig(chunk_size=100)
        assert config.chunk_size == 100

        config = SemanticSearchConfig(chunk_size=4000)
        assert config.chunk_size == 4000

        # Invalid below minimum
        with pytest.raises(ValueError, match="greater than or equal to"):
            SemanticSearchConfig(chunk_size=99)

        # Invalid above maximum
        with pytest.raises(ValueError, match="less than or equal to"):
            SemanticSearchConfig(chunk_size=4001)


class TestCreateEmbeddings:
    """Tests for create_embeddings factory function."""

    @patch("app.tools.youtube.semantic.embeddings.NomicEmbeddings")
    def test_creates_embeddings_with_config(
        self, mock_nomic_embeddings: MagicMock
    ) -> None:
        """Test that embeddings are created with correct config values."""
        config = SemanticSearchConfig(
            embedding_model="nomic-embed-text-v1.5",
            embedding_dimensionality=512,
            embedding_inference_mode="local",
        )

        mock_instance = MagicMock()
        mock_nomic_embeddings.return_value = mock_instance

        result = create_embeddings(config)

        assert result == mock_instance
        mock_nomic_embeddings.assert_called_once_with(
            model="nomic-embed-text-v1.5",
            dimensionality=512,
            inference_mode="local",
        )

    @patch("app.tools.youtube.semantic.embeddings.NomicEmbeddings")
    def test_creates_embeddings_with_custom_dimensionality(
        self, mock_nomic_embeddings: MagicMock
    ) -> None:
        """Test that custom dimensionality is passed to NomicEmbeddings."""
        config = SemanticSearchConfig(embedding_dimensionality=256)

        mock_instance = MagicMock()
        mock_nomic_embeddings.return_value = mock_instance

        create_embeddings(config)

        call_kwargs = mock_nomic_embeddings.call_args.kwargs
        assert call_kwargs["dimensionality"] == 256


class TestGetEmbeddings:
    """Tests for get_embeddings cached factory function."""

    @patch("app.tools.youtube.semantic.embeddings.create_embeddings")
    @patch("app.tools.youtube.semantic.embeddings.get_semantic_config")
    def test_uses_global_config(
        self,
        mock_get_config: MagicMock,
        mock_create_embeddings: MagicMock,
    ) -> None:
        """Test that get_embeddings uses global semantic config."""
        # Clear the lru_cache
        get_embeddings.cache_clear()

        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings

        result = get_embeddings()

        assert result == mock_embeddings
        mock_get_config.assert_called_once()
        mock_create_embeddings.assert_called_once_with(mock_config)

    @patch("app.tools.youtube.semantic.embeddings.create_embeddings")
    @patch("app.tools.youtube.semantic.embeddings.get_semantic_config")
    def test_caches_embeddings_instance(
        self,
        mock_get_config: MagicMock,
        mock_create_embeddings: MagicMock,
    ) -> None:
        """Test that get_embeddings caches the result."""
        # Clear the lru_cache
        get_embeddings.cache_clear()

        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings

        # Call twice
        result1 = get_embeddings()
        result2 = get_embeddings()

        # Should return same instance
        assert result1 is result2

        # create_embeddings should only be called once due to caching
        assert mock_create_embeddings.call_count == 1


class TestEmbeddingsIntegration:
    """Integration tests for embeddings (requires nomic[local] installed).

    These tests are marked as slow and can be skipped in CI with: pytest -m "not slow"
    """

    @pytest.mark.slow
    def test_embed_query_returns_correct_dimensions(self) -> None:
        """Test that embed_query returns vector with correct dimensionality."""
        config = SemanticSearchConfig(embedding_dimensionality=512)
        embeddings = create_embeddings(config)

        vector = embeddings.embed_query("test query about Nix garbage collection")

        assert isinstance(vector, list)
        assert len(vector) == 512
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.slow
    def test_embed_documents_batch(self) -> None:
        """Test that embed_documents handles batch embedding correctly."""
        config = SemanticSearchConfig(embedding_dimensionality=512)
        embeddings = create_embeddings(config)

        docs = [
            "How to configure Nix garbage collection",
            "NixOS home-manager setup guide",
            "Flakes tutorial for beginners",
        ]

        vectors = embeddings.embed_documents(docs)

        assert isinstance(vectors, list)
        assert len(vectors) == 3
        assert all(len(v) == 512 for v in vectors)

    @pytest.mark.slow
    def test_matryoshka_256_dimensions(self) -> None:
        """Test that Matryoshka 256-dim embeddings work correctly."""
        config = SemanticSearchConfig(embedding_dimensionality=256)
        embeddings = create_embeddings(config)

        vector = embeddings.embed_query("test query")

        assert len(vector) == 256

    @pytest.mark.slow
    def test_semantic_similarity(self) -> None:
        """Test that similar texts have higher similarity than dissimilar texts."""
        config = SemanticSearchConfig(embedding_dimensionality=512)
        embeddings = create_embeddings(config)

        # Similar texts
        query = "How to configure Nix garbage collection"
        similar = "Nix garbage collector configuration options"
        dissimilar = "Recipe for chocolate chip cookies"

        query_vec = embeddings.embed_query(query)
        similar_vec = embeddings.embed_query(similar)
        dissimilar_vec = embeddings.embed_query(dissimilar)

        # Calculate cosine similarity
        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot_product = sum(x * y for x, y in zip(a, b, strict=True))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot_product / (norm_a * norm_b)

        sim_to_similar = cosine_similarity(query_vec, similar_vec)
        sim_to_dissimilar = cosine_similarity(query_vec, dissimilar_vec)

        # Similar text should have higher similarity
        assert sim_to_similar > sim_to_dissimilar
