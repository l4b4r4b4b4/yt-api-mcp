# Task-01: Setup Dependencies and Module Structure

**Status:** 🟠 Implemented
**Priority:** High
**Created:** 2025-01-12
**Parent:** [Goal 02: Semantic Transcript Search](../scratchpad.md)

---

## Objective

Add LangChain dependencies to the project and create the module structure for semantic transcript search.

---

## Dependencies to Add

Run with `uv add`:

```bash
uv add "langchain-chroma>=1.1.0" "langchain-nomic>=1.0.1" "langchain-text-splitters>=0.3.0" "langchain-core>=1.0.0"
```

**Package Details:**

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain-chroma` | >=1.1.0 | ChromaDB vector store integration (requires chromadb >= 1.3.5) |
| `langchain-nomic` | >=1.0.1 | NomicEmbeddings with Matryoshka dimensionality support |
| `langchain-text-splitters` | >=0.3.0 | RecursiveCharacterTextSplitter for chunking |
| `langchain-core` | >=1.0.0 | Base abstractions (Document, VectorStore) |

**Note:** `langchain-nomic` requires `nomic>=3.5.3` and `pillow>=10.3.0` as transitive deps.

---

## Module Structure to Create

```
app/tools/youtube/semantic/
├── __init__.py           # Public exports
├── config.py             # SemanticSearchConfig (Pydantic settings)
├── embeddings.py         # Nomic Matryoshka embedding setup
├── store.py              # Chroma vector store initialization
├── chunker.py            # Transcript-aware text splitter
├── indexer.py            # Batch indexing logic
└── tools.py              # MCP tools (index_channel, semantic_search)
```

---

## Files to Create

### 1. `__init__.py`
```python
"""Semantic transcript search module.

Provides semantic search capabilities over YouTube video transcripts
using LangChain, ChromaDB, and Matryoshka embeddings.
"""

from app.tools.youtube.semantic.config import SemanticSearchConfig
from app.tools.youtube.semantic.embeddings import get_embeddings
from app.tools.youtube.semantic.store import get_vector_store

__all__ = [
    "SemanticSearchConfig",
    "get_embeddings",
    "get_vector_store",
]
```

### 2. `config.py`
Pydantic settings for semantic search configuration:
- `embedding_model`: Default "nomic-embed-text-v1.5"
- `embedding_dimensionality`: Default 512
- `embedding_inference_mode`: Default "local"
- `hnsw_space`: Default "cosine"
- `hnsw_max_neighbors`: Default 48
- `hnsw_ef_construction`: Default 200
- `hnsw_ef_search`: Default 128
- `chunk_size`: Default 800
- `chunk_overlap`: Default 100
- `persist_directory`: Optional path for persistence

### 3. `embeddings.py`
Factory function for NomicEmbeddings with Matryoshka config.

### 4. `store.py`
Factory function for Chroma vector store with HNSW settings.

### 5. `chunker.py`
Transcript-aware text splitter (placeholder for Task-04).

### 6. `indexer.py`
Batch indexing logic (placeholder for Task-05).

### 7. `tools.py`
MCP tools (placeholder for Task-06/07).

---

## Implementation Plan

1. [x] Run `uv add` to add dependencies
2. [x] Verify dependencies installed correctly with `uv sync`
3. [x] Create `app/tools/youtube/semantic/` directory
4. [x] Create `__init__.py` with public exports
5. [x] Create `config.py` with Pydantic settings model
6. [x] Create placeholder files for other modules
7. [x] Run `ruff check . --fix && ruff format .`
8. [x] Run tests to verify no regressions
9. [ ] Commit changes

---

## Acceptance Criteria

- [x] All 4 LangChain packages added to `pyproject.toml`
- [x] `uv.lock` updated with new dependencies
- [x] Module directory structure created
- [x] `config.py` with `SemanticSearchConfig` Pydantic model
- [x] All files pass linting (`ruff check`)
- [x] Existing tests still pass (178 passed)
- [x] No import errors when loading the module

---

## Notes

- This task focuses on setup only, not implementation
- Placeholder files should have docstrings explaining future purpose
- Config should load from environment variables where appropriate
- Follow existing patterns in `app/config.py` for settings

---

## References

- [Goal 02 Scratchpad](../scratchpad.md) - Full architecture details
- [langchain-nomic PyPI](https://pypi.org/project/langchain-nomic/)
- [langchain-chroma PyPI](https://pypi.org/project/langchain-chroma/)

---

## Completion Notes (2025-01-12)

### What Was Done

1. **Dependencies added via `uv add`:**
   - `langchain-chroma>=1.1.0`
   - `langchain-core>=1.2.7`
   - `langchain-nomic>=1.0.1`
   - `langchain-text-splitters>=1.1.0`

2. **Module structure created:**
   ```
   app/tools/youtube/semantic/
   ├── __init__.py      # Public exports (12 items)
   ├── config.py        # SemanticSearchConfig with HNSW/embedding/chunk settings
   ├── embeddings.py    # NomicEmbeddings factory with Matryoshka support
   ├── store.py         # Chroma vector store factory with collection_configuration
   ├── chunker.py       # TranscriptChunker placeholder (Task-04)
   ├── indexer.py       # TranscriptIndexer + IndexingResult (Task-05)
   └── tools.py         # MCP tools placeholders (Task-06/07)
   ```

3. **SemanticSearchConfig features:**
   - Environment variable configuration with `SEMANTIC_` prefix
   - Embedding settings: model, dimensionality (Matryoshka), inference_mode
   - HNSW settings: space, max_neighbors, ef_construction, ef_search
   - Chunking settings: chunk_size, chunk_overlap
   - Persistence: persist_directory, collection_name
   - Helper properties: `hnsw_config`, `collection_configuration`

4. **Verification:**
   - All imports work: `from app.tools.youtube.semantic import ...`
   - Linting passes: `ruff check` - All checks passed
   - Tests pass: 178 passed in 1.63s

### Awaiting User Validation

Task is marked 🟠 Implemented. Awaiting user validation before marking 🟢 Complete.
