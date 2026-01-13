# Goals Index & Tracking Scratchpad

> Central hub for tracking all active goals in the nix-configs repository.

---

## Active Goals

| ID | Goal Name | Status | Priority | Last Updated |
|----|-----------|--------|----------|--------------|
| 01 | [Production YouTube MCP MVP](./01-Production-YouTube-MCP-MVP/scratchpad.md) | 🟠 Implemented | Critical | 2025-01-11 |
| 02 | [Semantic Transcript Search](./02-Semantic-Transcript-Search/scratchpad.md) | 🟢 Complete | High | 2025-01-14 |
| 03 | [Context Limit Research](./03-Context-Limit-Research/scratchpad.md) | 🟡 In Progress | Low | 2025-01-13 |
| 04 | [Semantic Search Comments & Mixed](./04-Semantic-Search-Comments-Mixed/scratchpad.md) | 🟢 Complete | High | 2025-01-15 |
| 05 | (Reserved) | ⚪ Not Started | - | - |
| 06 | (Reserved) | ⚪ Not Started | - | - |
| 07 | (Reserved) | ⚪ Not Started | - | - |
| 08 | (Reserved) | ⚪ Not Started | - | - |
| 09 | (Reserved) | ⚪ Not Started | - | - |
| 10 | (Reserved) | ⚪ Not Started | - | - |

---

## Status Legend

- 🟢 **Complete** — Goal achieved AND validated by user in real environment
- 🟠 **Implemented** — Code complete, tests passing, NOT yet user-validated
- 🟡 **In Progress** — Actively being worked on
- 🔴 **Blocked** — Waiting on external dependency or decision
- ⚪ **Not Started** — Planned but not yet begun
- ⚫ **Archived** — Abandoned or superseded

---

## Priority Levels

- **Critical** — Blocking other work or system stability
- **High** — Important for near-term objectives
- **Medium** — Should be addressed when time permits
- **Low** — Nice to have, no urgency

---

## Quick Links

- [00-Template-Goal](./00-Template-Goal/scratchpad.md) — Template for new goals

---

## Notes

- Each goal has its own directory under `.agent/goals/`
- Goals contain a `scratchpad.md` and one or more `Task-XX/` subdirectories
- Tasks are atomic, actionable units of work within a goal
- Use the template in `00-Template-Goal/` when creating new goals

---

## Recent Activity

### 2025-01-15
- **Goal 04 Complete**: Semantic Search for Comments & Mixed Content
  - All 5 new MCP tools validated live:
    - `semantic_search_comments` - search comments with auto-indexing
    - `semantic_search_all` - unified transcript + comment search
    - `get_indexed_videos` - list indexed video IDs with filters
    - `delete_indexed_video` - remove content from index
  - 406 tests passing, 104 new tests for this goal
  - Ready for v0.0.3 release

### 2025-01-14
- **Goal 02 Complete**: Semantic Transcript Search v0.0.2 released
  - PR #1 merged, published to PyPI
  - 343 tests passing, semantic search validated live
- **Goal 04 Created**: Semantic Search for Comments & Mixed Content
  - Extend semantic search to YouTube comments
  - Add unified search across transcripts + comments
  - Add utility tools (get_indexed_videos, delete_indexed_video)
  - Target version: 0.0.3

### 2025-01-13
- **Goal 03 Created**: Context Limit Research
  - Track Zed vs Claude context reporting discrepancies
  - Understand soft/hard limit behavior
  - Inform .rules handoff guidelines
- **Goal 02 Task-06**: Semantic Search with auto-indexing
  - Revised approach: agent doesn't need to call "index" explicitly
  - `semantic_search_transcripts` auto-indexes missing videos
  - Supports multiple channels + specific videos

### 2025-01-12
- **Goal 02 Planning**: Semantic Transcript Search
  - Architecture defined: LangChain-based with Matryoshka embeddings
  - Verified latest APIs: langchain-chroma 1.1.0, langchain-nomic 1.0.1, chromadb 1.4.0
  - 10 tasks planned, ready for implementation
  - Target version: 0.0.2

### 2025-01-11
- **Goal 01 Implemented**: Production YouTube MCP MVP
  - v0.0.0 and v0.0.1 published to PyPI as `yt-api-mcp`
  - 16 YouTube tools working (search, metadata, transcripts, comments, live)
  - 178 tests, 76% coverage
  - Awaiting broader user validation

### 2025-01-08
- **Goal 01 Created**: Production YouTube MCP MVP
  - Migrating from reference implementation in `.agent/youtube_toolset.py`
  - Integrating with mcp-refcache architecture
  - Target: Feature-complete, production-ready YouTube search and transcript MCP server
