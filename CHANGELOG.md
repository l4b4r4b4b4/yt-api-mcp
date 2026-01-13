# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for Future Releases
- Video playlist analysis tools
- Channel statistics trends over time
- Batch operations for multiple videos
- Advanced filtering options for search results
- Community post integration
- Video category and tag analysis

---

## [0.0.3] - 2025-01-15

### Added

#### Semantic Comment Search (2 new tools)
- `semantic_search_comments` - Search comments using natural language
  - **Auto-indexing**: Automatically indexes video comments before searching
  - Supports `channel_ids` and/or `video_ids` for flexible scoping
  - Returns author, like_count, reply_count with each result
  - Includes similarity scores and indexing statistics
  - Example: "find comments discussing flake configuration issues"

- `semantic_search_all` - Unified search across transcripts AND comments
  - Search both content types in a single query
  - Optional `content_types` parameter to filter (e.g., `["transcript"]` only)
  - Results include `content_type` field ("transcript" or "comment")
  - Transcript results include timestamp URLs; comment results include author
  - Combined indexing stats for both content types

#### Index Management Utilities (2 new tools)
- `get_indexed_videos` - List videos in the semantic search index
  - Optional `channel_id` filter to scope by channel
  - Optional `content_type` filter ("transcript" or "comment")
  - Returns video IDs and total count

- `delete_indexed_video` - Remove content from the semantic search index
  - Delete by video ID
  - Optional `content_type` to delete only transcripts or comments
  - Returns count of deleted chunks by type

#### Infrastructure Improvements
- Added `content_type` metadata field to all indexed documents
  - Enables filtering by content type in ChromaDB queries
  - Values: `"transcript"` or `"comment"`
- New `CommentChunker` class for processing YouTube comments
  - One Document per comment (no chunking needed for short content)
  - Preserves author attribution and engagement metrics
- Extended `TranscriptIndexer` with comment indexing methods
  - `index_video_comments()` - Index comments from a video
  - `is_video_comments_indexed()` - Check if comments are indexed
  - `delete_video_comments()` - Remove comment index for a video

### Changed
- Server instructions updated with new tool documentation
- Test count increased from 334 to 406 tests (104 new tests)
- Code coverage maintained at 73%+

### Technical Details

#### New Module
```
app/tools/youtube/semantic/
└── comment_chunker.py   # CommentChunker class
```

#### Comment Metadata Schema
```python
{
    "content_type": "comment",
    "video_id": str,
    "channel_id": str,
    "channel_title": str,
    "video_title": str,
    "video_url": str,
    "published_at": str,      # When comment was posted
    "author": str,            # Comment author display name
    "like_count": int,        # Engagement metric
    "reply_count": int,       # Number of replies
    "chunk_index": int,       # Position in comment list
}
```

#### API Quota for Comments
- `get_video_comments`: 1 quota unit per call (max 100 comments)
- For 50 videos × 100 comments = 50 quota units
- Well within 10,000 daily limit

### Known Limitations
- Existing indexed transcripts (from v0.0.2) lack `content_type` metadata
  - They won't appear when filtering by content_type
  - Re-indexing will add the field automatically
- YouTube may rate-limit transcript fetching from cloud IPs

---

## [0.0.2] - 2025-01-14

### Added

#### Semantic Transcript Search (3 new tools)
- `warmup_semantic_search` - Pre-load embedding model for faster first search
  - Downloads Nomic Embed Text v1.5 (~270MB on first run)
  - Initializes ChromaDB vector store
  - Returns ready status with model info
  - Subsequent calls are instant (model cached)

- `semantic_search_transcripts` - Search transcripts with natural language
  - **Auto-indexing**: Automatically indexes missing videos before searching
  - Supports `channel_ids` and/or `video_ids` for flexible scoping
  - Returns timestamped URLs for direct playback at matching segments
  - Includes similarity scores and indexing statistics
  - Example: "find the video about nix garbage collection keeping N generations"

- `index_channel_transcripts` - Pre-index channel transcripts (optional)
  - Bulk indexing for faster subsequent searches
  - Useful for pre-warming cache before users search
  - Returns detailed indexing statistics

#### Embedding & Vector Store Infrastructure
- **Nomic Embed Text v1.5** with Matryoshka dimensionality (512 dims default)
  - Runs locally via Embed4All (no API key needed)
  - Supports 64/128/256/512/768 dimension options
- **ChromaDB vector store** with optimized HNSW settings
  - `max_neighbors=48`, `ef_construction=200`, `ef_search=128`
  - Cosine similarity for semantic matching
  - Persistent storage in `./chroma_db`
- **Token-based chunking** with timestamp preservation
  - 256 tokens per chunk with 10% overlap
  - Preserves transcript entry boundaries
  - Maintains start/end timestamps for each chunk

#### Test Infrastructure Improvements
- Added `clear_semantic_caches` autouse fixture in `conftest.py`
  - Clears `lru_cache` on singleton functions between tests
  - Prevents test pollution from cached vector stores/embeddings
- Fixed set ordering issues in semantic search tests
  - Tests now compare sets instead of lists where order is non-deterministic

### Changed
- Server instructions updated with semantic search documentation
- Test count increased from 178 to 334 tests
- Code coverage maintained at 73%+

### Technical Details

#### New Dependencies
```toml
langchain-chroma = ">=1.1.0"
langchain-core = ">=1.2.7"
langchain-nomic = ">=1.0.1"
langchain-text-splitters = ">=1.1.0"
nomic[local] = ">=3.9.0"
```

#### New Module Structure
```
app/tools/youtube/semantic/
├── __init__.py
├── config.py       # SemanticSearchConfig with HNSW settings
├── embeddings.py   # Nomic Matryoshka embeddings
├── chunker.py      # Token-based transcript chunker
├── store.py        # ChromaDB vector store
├── indexer.py      # Batch indexing logic
├── tokenizers.py   # Tiktoken/HuggingFace tokenizers
└── tools.py        # MCP tool implementations
```

#### Performance Characteristics
- **First search on new channel**: 1-2 minutes (indexing 50 videos)
- **Subsequent searches**: ~100ms (already indexed)
- **Embedding generation**: ~100ms per chunk (CPU)
- **Storage**: ~2KB per chunk (512-dim embeddings)
- **API quota**: ~1 unit per video (transcripts are free)

### Known Limitations
- First semantic search requires warmup (call `warmup_semantic_search` first)
- Some videos lack transcripts (auto-generated or manual)
- Indexing is CPU-bound (no GPU acceleration in local mode)

---

## [0.0.1] - 2025-01-11

### Fixed
- **stdio mode protocol fix**: Startup info now writes to stderr instead of stdout, fixing MCP client handshake issues with Zed and Claude Desktop
- Package now works correctly with `uvx yt-api-mcp stdio`

### Added
- **Aligned executable name**: Added `yt-api-mcp` as primary executable (matching package name)
- `yt-mcp` remains as legacy alias for backwards compatibility

### Changed
- Package name is `yt-api-mcp` on PyPI (was `yt-mcp` but that conflicted with existing package)

### Installation
```bash
# PyPI (recommended)
pip install yt-api-mcp

# Or with uvx
uvx yt-api-mcp stdio

# Legacy alias still works
uvx --from yt-api-mcp yt-mcp stdio
```

---

## [0.0.0] - 2025-01-11

### Overview
Initial experimental release of the YouTube MCP server. This version tests both the core implementation and the release workflow. Version 0.0.0 signals "experimental test release - expect issues" which is honest for a first release.

### Added

#### YouTube Search Tools (3 tools)
- `search_videos` - Search for YouTube videos by query string
  - Returns video metadata (title, description, thumbnail, etc.)
  - Cached for 6 hours in `youtube.search` namespace
  - Costs 100 API quota units per request
- `search_channels` - Search for YouTube channels by query
  - Returns channel information and URLs
  - Cached for 6 hours in `youtube.search` namespace
  - Costs 100 API quota units per request
- `search_live_videos` - Find currently broadcasting live streams
  - Filters results to only active live videos
  - Cached for 6 hours in `youtube.search` namespace
  - Costs 100 API quota units per request

#### YouTube Metadata Tools (3 tools)
- `get_video_details` - Retrieve comprehensive video metadata
  - Includes views, likes, comments, duration, tags
  - Cached for 24 hours in `youtube.api` namespace
  - Costs 1 API quota unit per request
- `get_channel_info` - Get detailed channel statistics
  - Includes subscriber count, video count, total views
  - Cached for 24 hours in `youtube.api` namespace
  - Costs 1 API quota unit per request
- `is_live` - Check if a video is currently streaming
  - Returns live status, viewer count, chat ID
  - Cached for 30 seconds in `youtube.api` namespace
  - Costs 1 API quota unit per request

#### YouTube Transcript Tools (4 tools)
- `list_available_transcripts` - List all transcript languages for a video
  - Shows manual and auto-generated transcripts
  - Cached permanently in `youtube.content` namespace
  - Costs 0 API quota (uses youtube-transcript-api)
- `get_video_transcript_preview` - Get preview of video transcript
  - Returns first N characters (default 2000)
  - Cached permanently in `youtube.content` namespace
  - Costs 0 API quota
- `get_full_transcript` - Download complete video transcript with timestamps
  - Returns all transcript entries with start time and duration
  - Large transcripts return RefCache reference for pagination
  - Cached permanently in `youtube.content` namespace
  - Costs 0 API quota
- `get_transcript_chunk` - Paginate through transcript entries
  - Efficient access to specific transcript sections
  - Cached permanently in `youtube.content` namespace
  - Costs 0 API quota

#### Engagement & Live Chat Tools (3 tools)
- `get_video_comments` - Fetch top comments with engagement metrics
  - Returns author, text, like count, reply count
  - Returns empty list if comments disabled (not an error)
  - Cached for 5 minutes in `youtube.comments` namespace
  - Costs 1 API quota unit per request
- `get_live_chat_id` - Get chat ID for live streaming video
  - Required for accessing live chat messages
  - Cached for 5 minutes in `youtube.api` namespace
  - Costs 1 API quota unit per request
- `get_live_chat_messages` - Access live chat messages with pagination
  - Supports efficient polling with page tokens
  - Returns only new messages when using page_token
  - Cached for 30 seconds in `youtube.comments` namespace
  - Costs 1 API quota unit per request

#### Cache Management Tools (1 tool + admin)
- `get_cached_result` - Retrieve and paginate cached results
  - Access full data from RefCache references
  - Supports pagination for large datasets
  - No API quota cost
- Admin tools available (optional, disabled by default)

#### Multi-Tier Caching Strategy
- **Tier 1: Search Cache** - 6 hours, 300 entries (most volatile)
- **Tier 2: API Metadata Cache** - 24 hours, 1000 entries (semi-stable)
- **Tier 3: Comments Cache** - 5 minutes, 500 entries (high volatility)
- **Tier 4: Content Cache** - Permanent, 5000 entries (immutable transcripts)
- **Tier 5: Live Streaming** - 30s-5m, for real-time data
- RefCache integration for large results (automatic reference-based returns)
- Preview generation for transcript data
- Reduces API quota usage by ~75% through intelligent caching

#### Docker Support
- Production Docker image: 229MB (optimized for size)
- Base image: 290MB (shared across FastMCP projects)
- Docker Compose configuration with production and development profiles
- Non-root user (appuser) for security
- Built-in health checks
- Environment-based configuration
- Multi-architecture support (amd64, arm64)

#### Development Infrastructure
- Comprehensive test suite: 178 tests, 76% code coverage
- Pytest with async support
- Type checking with mypy (strict mode)
- Linting and formatting with ruff
- Pre-commit hooks for code quality
- Nix flake for reproducible development environment
- UV for fast dependency management

#### Documentation
- Complete README.md with all 16 tools documented
- YouTube API key setup guide
- Docker usage instructions (compose and direct)
- Caching strategy explanation
- Practical use case examples
- Troubleshooting guide
- API quota management guide
- TOOLS.md template reference
- This CHANGELOG.md

#### Integrations
- Claude Desktop configuration example
- Zed editor configuration example
- Langfuse tracing for observability (optional)
- Environment variable configuration
- Supports both stdio and HTTP transports

### Technical Details

#### Dependencies
- Python 3.12+
- FastMCP >=2.14.0 (MCP server framework)
- mcp-refcache >=0.1.0 (reference-based caching)
- google-api-python-client >=2.187.0 (YouTube Data API v3)
- youtube-transcript-api >=1.2.3 (transcript access, no quota)
- Pydantic >=2.10.0 (data validation)
- Langfuse >=3.10.0 (tracing, optional)

#### Architecture
- FastMCP for MCP server framework
- RefCache for intelligent caching and large result handling
- Multi-namespace caching with different TTLs
- Async/await throughout for performance
- Type-safe with comprehensive type annotations
- Modular tool organization

#### Build System
- UV for dependency management (10-100x faster than pip)
- Hatchling for Python packaging
- Docker multi-stage builds for size optimization
- Nix for reproducible development environments

### Known Limitations

#### Version 0.0.0 Disclaimer
- This is an **experimental first release**
- Published to test both implementation and release workflow
- Limited real-world validation beyond development testing
- Documentation may have gaps or inaccuracies
- Docker images published but not battle-tested in production

#### YouTube API Limitations
- Daily API quota: 10,000 units (free tier)
  - Search operations: 100 units each (~100 searches/day)
  - Metadata operations: 1 unit each (~10,000 requests/day)
- Transcript availability varies by video (creator-controlled)
- Language support depends on video creator's captions
- Some videos have comments disabled
- Live chat only available during active streams

#### MCP Protocol Limitations
- Request/response model (not true streaming for live chat)
- Agent must manually poll for new live chat messages
- No push notifications for new content
- Each operation is a separate request

### Breaking Changes
None (initial release)

### Deprecated
None (initial release)

### Security
- Non-root Docker user for production containers
- API key passed via environment variables (not hardcoded)
- No secrets committed to repository
- Optional Langfuse tracing with secure credential handling

### Performance
- Intelligent caching reduces API calls by ~75%
- RefCache minimizes context window pollution for agents
- Permanent caching for immutable content (transcripts)
- Preview generation for large datasets
- Docker images optimized for size and startup speed

---

## Version History

- **0.0.0** (2025-01-08) - Initial experimental release
  - 16 YouTube tools implemented
  - Multi-tier caching strategy
  - Docker support
  - 178 tests, 76% coverage

---

## Next Steps After 0.0.0

### Version 0.0.1 (Planned)
- Bug fixes based on 0.0.0 user feedback
- Documentation improvements
- Performance optimizations
- Error message refinements

### Version 0.0.x (Iterations)
- Continue bug fixes and improvements
- Add missing features from user feedback
- Improve test coverage (target: 85%+)
- Refine caching strategies based on usage patterns

### Version 0.1.0 (Future)
- After 5-10 patch releases (0.0.x)
- Proven stability in real-world usage
- Comprehensive integration testing
- Performance benchmarks
- Production-ready designation

### Version 1.0.0 (Long-term)
- After 6+ months of stable 0.x usage
- Battle-tested in production environments
- Stable public API
- Complete documentation
- Comprehensive error handling
- Full feature set

---

## Feedback Welcome

This is version 0.0.0 - we expect issues! Please report:
- Bugs and unexpected behavior
- Documentation gaps or inaccuracies
- Performance problems
- Feature requests
- API quota concerns
- Docker deployment issues

Open issues on [GitHub](https://github.com/l4b4r4b4b4/yt-mcp/issues)

---

## Links

- [GitHub Repository](https://github.com/l4b4r4b4b4/yt-mcp)
- [PyPI Package](https://pypi.org/project/yt-mcp/)
- [Docker Image](https://ghcr.io/l4b4r4b4b4/yt-mcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [mcp-refcache](https://github.com/l4b4r4b4b4/mcp-refcache)
- [FastMCP](https://github.com/jlowin/fastmcp)
