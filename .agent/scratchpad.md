# YouTube MCP Server Development Scratchpad

> **High-level session state only.** See goal scratchpads for details.

---

## Current Status: 🟡 Goal 02 Planning Complete

**Version:** 0.0.1 (published)
**Next Version:** 0.0.2 (semantic search)
**Last Updated:** 2025-01-12

---

## Goals

### Goal 01: Production YouTube MCP MVP
**Status:** 🟠 Implemented (awaiting broader user validation)
**Link:** [Goal 01 Scratchpad](goals/01-Production-YouTube-MCP-MVP/scratchpad.md)

- ✅ 16 YouTube tools (search, metadata, transcripts, comments, live)
- ✅ 178 tests, 76% coverage
- ✅ v0.0.0 + v0.0.1 published to PyPI
- ✅ GitHub releases created
- ⚠️ GHCR failed (base image issue - fix later)

### Goal 02: Semantic Transcript Search
**Status:** 🟡 In Progress (Planning Complete, Ready for Implementation)
**Link:** [Goal 02 Scratchpad](goals/02-Semantic-Transcript-Search/scratchpad.md)

- **Architecture verified** against latest LangChain/ChromaDB APIs
- LangChain-based pipeline (not raw ChromaDB)
- Matryoshka embeddings via `langchain-nomic` (512 dims default)
- HNSW settings via `collection_configuration` (new API)
- 10 tasks planned, 0 completed
- Version target: 0.0.2

---

## Session 2025-01-12 Summary

**Goal 02 Planning:**
- Defined LangChain-based architecture for semantic transcript search
- Verified latest APIs from source code:
  - `langchain-chroma` v1.1.0 (requires chromadb >= 1.3.5)
  - `langchain-nomic` v1.0.1 (Matryoshka embeddings)
  - `chromadb` v1.4.0 (new `collection_configuration` API)
- Key decisions documented:
  - Use `NomicEmbeddings` with `dimensionality=512` and `inference_mode="local"`
  - HNSW: `max_neighbors=48`, `ef_construction=200`, `ef_search=128`, `space="cosine"`
  - Transcript-aware chunking with timestamp preservation
- 10 tasks defined, ready for implementation

**Next Steps:**
1. Task-01: Add dependencies, create module structure
2. Task-02: Configure Nomic Matryoshka embeddings
3. Continue through remaining tasks

---

## Session 2025-01-11 Summary

**Published:**
- PyPI: `yt-api-mcp` v0.0.0 and v0.0.1
- GitHub Release: https://github.com/l4b4r4b4b4/yt-api-mcp

**v0.0.1 Fixes:**
- stdio mode writes to stderr (fixes MCP handshake)
- Added `yt-api-mcp` executable

**Security:**
- Rotated exposed API key
- Force-pushed to remove from history
- Purged local git objects
