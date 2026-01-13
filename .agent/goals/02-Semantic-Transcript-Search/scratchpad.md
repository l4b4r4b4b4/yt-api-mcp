# Goal 02: Semantic Transcript Search

**Status:** 🟡 In Progress
**Priority:** High
**Created:** 2025-01-11
**Updated:** 2025-01-12
**Parent:** [Goals Index](../scratchpad.md)

---

## Objective

Implement semantic search over YouTube video transcripts to enable queries like "find the Vimjoyer video about Nix garbage collection keeping N generations" without knowing exact video titles or keywords.

## Use Case

Given a channel/set of videos, be able to:
1. Fetch all transcripts (batch operation)
2. Index them for semantic search
3. Query with natural language to find relevant video segments

---

## Architecture (Revised: LangChain-Based)

### Design Decisions

**User Request:** Use LangChain for all parts of the retrieval pipeline (ingestion + retrieval), include available filters, set good HNSW defaults, use Matryoshka embeddings.

**Rationale for LangChain:**
- Comprehensive abstractions for the entire RAG pipeline
- Built-in VectorStore protocol (no need for custom adapter pattern)
- Easy swapping between vector stores (Chroma → Pinecone → FAISS)
- RecursiveCharacterTextSplitter with configurable chunking
- Native metadata filtering support
- HuggingFaceEmbeddings with Matryoshka model support

### Component Structure

```
app/tools/youtube/
├── semantic/                    # NEW module
│   ├── __init__.py
│   ├── config.py               # SemanticSearchConfig (HNSW, embedding settings)
│   ├── embeddings.py           # Matryoshka embedding setup
│   ├── chunker.py              # Transcript-aware text splitter
│   ├── store.py                # Chroma vector store initialization
│   ├── indexer.py              # Transcript indexing logic
│   └── tools.py                # MCP tools: index_channel, semantic_search
```

### Embedding Model: Matryoshka

**Selected Model:** `nomic-embed-text-v1.5` via `langchain-nomic`

**Why Nomic:**
- Native Matryoshka support (can truncate to 64/128/256/512/768 dims)
- Runs locally via Embed4All (`inference_mode="local"`) - no API key needed
- High quality on MTEB benchmarks
- First-class LangChain integration with `dimensionality` parameter

**Configuration (via langchain-nomic):**
```python
from langchain_nomic import NomicEmbeddings

embeddings = NomicEmbeddings(
    model="nomic-embed-text-v1.5",
    dimensionality=512,          # Matryoshka: 64, 128, 256, 512, or 768
    inference_mode="local",      # "local" (Embed4All), "remote", or "dynamic"
    # device="cpu",              # Optional: "cpu", "gpu", "nvidia", "amd"
)
```

**Matryoshka Dimension Options:**
| Dims | Quality | Speed | Use Case |
|------|---------|-------|----------|
| 768 | Best | Slowest | Production, high accuracy needed |
| 512 | Good | Fast | **Default** - balanced tradeoff |
| 256 | Acceptable | Faster | Large corpora, memory constrained |
| 128 | Lower | Very Fast | Prototyping, quick experiments |
| 64 | Lowest | Fastest | Extreme constraints only |

We'll use **512 dimensions** as default (good quality/speed tradeoff).

**Inference Modes:**
- `"local"` - Uses Embed4All, runs entirely on device (recommended)
- `"remote"` - Uses Nomic API (requires `NOMIC_API_KEY`)
- `"dynamic"` - Automatic selection based on availability

### HNSW Settings for ChromaDB (v1.4.0+ via LangChain)

**Using `collection_configuration` (new API, chromadb >= 1.3.5):**
```python
from langchain_chroma import Chroma
from chromadb.api import CreateCollectionConfiguration

# Optimized HNSW settings
collection_configuration: CreateCollectionConfiguration = {
    "hnsw": {
        "space": "cosine",           # Best for normalized embeddings
        "max_neighbors": 48,         # M parameter - connections per node (default 16)
        "ef_construction": 200,      # Build-time accuracy (default 100)
        "ef_search": 128,            # Search-time accuracy (default 100)
    }
}

vector_store = Chroma(
    collection_name="youtube_transcripts",
    embedding_function=embeddings,
    collection_configuration=collection_configuration,
    persist_directory="./chroma_db",  # Optional persistence
)
```

**Parameter Reference:**
| Parameter | Default | Our Value | Description |
|-----------|---------|-----------|-------------|
| `space` | `"l2"` | `"cosine"` | Distance metric for similarity |
| `max_neighbors` | 16 | 48 | HNSW M parameter (graph connectivity) |
| `ef_construction` | 100 | 200 | Build-time accuracy (higher = better index) |
| `ef_search` | 100 | 128 | Search-time accuracy (higher = better recall) |

**Rationale:**
- `cosine` space: Standard for semantic similarity with normalized embeddings
- `max_neighbors=48`: Better recall than default 16, reasonable memory overhead
- `ef_construction=200`: One-time build cost, worth the accuracy
- `ef_search=128`: Good recall/latency tradeoff for search

**Note:** The old `collection_metadata` dict with `hnsw:space`, `hnsw:M` keys is deprecated.
Use `collection_configuration` with the new typed dict structure.

### Chunking Strategy (Transcript-Aware)

**Approach:** Custom transcript chunker that preserves timestamp mappings

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Base splitter for text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

**Transcript-Specific Logic:**
1. Group transcript entries by target chunk size (~800 chars)
2. Preserve timestamp boundaries (don't split mid-word)
3. Calculate `start_time` and `end_time` for each chunk
4. Store rich metadata for filtering and display

### Metadata Schema

Each indexed chunk stores:
```python
metadata = {
    # Identifiers
    "video_id": str,           # YouTube video ID
    "channel_id": str,         # YouTube channel ID
    "channel_title": str,      # Channel name for display

    # Video info
    "video_title": str,        # Video title for display
    "video_url": str,          # Direct link to video
    "published_at": str,       # ISO 8601 timestamp

    # Timestamp info
    "start_time": float,       # Start time in seconds
    "end_time": float,         # End time in seconds
    "timestamp_url": str,      # URL with timestamp (video?t=123)

    # Transcript info
    "language": str,           # Transcript language code
    "chunk_index": int,        # Position in video transcript
}
```

### Available Filters

LangChain + Chroma support these filter operations:

```python
# Filter by channel
filter={"channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw"}

# Filter by multiple videos
filter={"video_id": {"$in": ["video1", "video2", "video3"]}}

# Filter by language
filter={"language": "en"}

# Compound filters (AND)
filter={
    "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw",
    "language": "en",
}

# Complex filters with operators
filter={
    "$and": [
        {"channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw"},
        {"start_time": {"$gte": 0}},
        {"start_time": {"$lte": 600}},  # First 10 minutes
    ]
}
```

**Supported Operators:** `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`

---

## MCP Tools

### 1. `index_channel_transcripts`

Index all video transcripts from a YouTube channel for semantic search.

**Parameters:**
- `channel_id` (required): YouTube channel ID
- `max_videos` (optional): Maximum videos to index (default: 50)
- `language` (optional): Preferred transcript language (default: "en")
- `force_reindex` (optional): Re-index even if already indexed (default: false)

**Returns:**
- `indexed_count`: Number of videos indexed
- `chunk_count`: Total chunks created
- `skipped_count`: Videos skipped (no transcript)
- `errors`: List of errors encountered

### 2. `semantic_search_transcripts`

Search indexed transcripts using natural language queries.

**Parameters:**
- `query` (required): Natural language search query
- `k` (optional): Number of results (default: 10)
- `channel_id` (optional): Filter by channel
- `video_ids` (optional): Filter by specific videos
- `language` (optional): Filter by transcript language
- `min_score` (optional): Minimum similarity score (0-1)

**Returns:**
- `results`: List of matches with:
  - `video_id`, `video_title`, `video_url`
  - `text`: Matched transcript segment
  - `start_time`, `end_time`, `timestamp_url`
  - `score`: Similarity score
  - `channel_title`

### 3. `get_all_channel_transcripts`

Batch fetch all transcripts from a channel (helper for indexing).

**Parameters:**
- `channel_id` (required): YouTube channel ID
- `max_videos` (optional): Maximum videos to fetch (default: 50)
- `language` (optional): Preferred transcript language

**Returns:**
- RefCache reference (large result set)
- Preview with video count and sample titles

---

## Dependencies

```toml
# Add to pyproject.toml [project.dependencies]
langchain-chroma = ">=1.1.0"       # Requires chromadb >= 1.3.5
langchain-nomic = ">=1.0.1"        # NomicEmbeddings with Matryoshka support
langchain-text-splitters = ">=0.3.0"
langchain-core = ">=1.0.0"
```

**Why these packages:**
- `langchain-chroma`: ChromaDB integration with `collection_configuration` support
- `langchain-nomic`: NomicEmbeddings with native Matryoshka dimensionality parameter
- `langchain-text-splitters`: RecursiveCharacterTextSplitter for chunking
- `langchain-core`: Base abstractions (Document, VectorStore protocol)

**Note:** We use `langchain-nomic` instead of `langchain-huggingface` because:
1. Native Matryoshka `dimensionality` parameter support
2. Can run locally via Embed4All (`inference_mode="local"`)
3. No separate HuggingFace model download required
4. Better integration with Nomic's optimizations

---

## Task Breakdown (Revised)

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| Task-01 | Setup: Add LangChain dependencies, create module structure | 🟠 | Done, awaiting validation |
| Task-02 | Embeddings: Configure Nomic Matryoshka embeddings | ⚪ | 512 dims default |
| Task-03 | Vector Store: Initialize Chroma with HNSW settings | ⚪ | Persistence options |
| Task-04 | Chunker: Transcript-aware splitter with timestamps | ⚪ | Preserve metadata |
| Task-05 | Indexer: Batch indexing logic with progress | ⚪ | Error handling |
| Task-06 | Tool: `index_channel_transcripts` MCP tool | ⚪ | RefCache integration |
| Task-07 | Tool: `semantic_search_transcripts` MCP tool | ⚪ | Filters, scoring |
| Task-08 | Helper: `get_all_channel_transcripts` | ⚪ | Batch fetch |
| Task-09 | Testing and validation | ⚪ | Unit + integration |
| Task-10 | Documentation and release (0.0.2) | ⚪ | README, CHANGELOG |

---

## Success Criteria

- [ ] Can index all transcripts from a YouTube channel
- [ ] Semantic search returns relevant video segments with timestamps
- [ ] Results include clickable timestamp URLs
- [ ] Metadata filters work (channel, video, language)
- [ ] HNSW settings provide good recall/latency tradeoff
- [ ] Matryoshka embeddings enable dimension flexibility
- [ ] Vector store is swappable via LangChain abstractions
- [ ] Works with existing RefCache caching strategy
- [ ] Tests cover core functionality (≥73% coverage)
- [ ] Documentation updated with usage examples

---

## Future Considerations

### mcp-refcache Integration
This feature could eventually become part of `mcp-refcache`:
- Decorator option: `@cache.cached(semantic_index=True)`
- Automatic semantic indexing of cached content
- Cross-tool semantic search ("find all cached results about X")

### Client/Server Vector Store
When scaling beyond in-memory:
- ChromaDB client/server mode
- Replace with `langchain-qdrant`, `langchain-pinecone`, etc.
- Same code, different VectorStore class

### Advanced Features
- Hybrid search (semantic + keyword)
- Reranking with cross-encoders
- Multi-language support with translation
- Incremental indexing (new videos only)

---

## Technical Notes

### Quota Considerations
- **Search operations:** 100 quota units per search call
- **Video details:** 1 quota unit per video
- **Transcripts:** 0 quota units (uses youtube-transcript-api)

For indexing 50 videos:
- ~50 quota units for video metadata
- 0 for transcripts
- Total: ~50 units (well within 10,000 daily limit)

### Performance Estimates
- Embedding generation: ~100ms per chunk (CPU)
- Indexing 50 videos (~500 chunks): ~1-2 minutes
- Search latency: ~50-100ms per query

### Storage Requirements
- 512-dim embeddings: ~2KB per chunk
- 500 chunks (50 videos): ~1MB
- With metadata: ~2-3MB total

---

## References

- [LangChain ChromaDB Integration](https://python.langchain.com/docs/integrations/vectorstores/chroma)
- [LangChain HuggingFace Embeddings](https://python.langchain.com/docs/integrations/text_embedding/sentence_transformers)
- [LangChain Nomic Embeddings](https://python.langchain.com/docs/integrations/text_embedding/nomic)
- [Nomic Embed Text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [ChromaDB Collection Configuration](https://github.com/chroma-core/chroma/blob/main/chromadb/api/collection_configuration.py)
- [langchain-chroma source](https://github.com/langchain-ai/langchain/blob/master/libs/partners/chroma/langchain_chroma/vectorstores.py)

---

## Session Log

### 2025-01-12: Architecture Revision
- Switched from custom VectorStoreProtocol to LangChain abstractions
- Selected Nomic Embed Text v1.5 for Matryoshka embeddings
- Defined HNSW settings (max_neighbors=48, ef_construction=200, ef_search=128)
- Updated task breakdown to 10 tasks
- Added detailed metadata schema and filter documentation

### 2025-01-12: API Verification
- Verified latest LangChain docs and source code
- Confirmed `collection_configuration` is the new API (not `collection_metadata`)
- ChromaDB 1.4.0 uses `max_neighbors` instead of `M` parameter
- `langchain-chroma >= 1.1.0` requires `chromadb >= 1.3.5`
- `langchain-nomic >= 1.0.1` supports Matryoshka dimensionality parameter
- Updated package versions to latest stable releases
