# YouTube MCP Server

A production-ready MCP (Model Context Protocol) server for YouTube integration with intelligent caching via [mcp-refcache](https://github.com/l4b4r4b4b4/mcp-refcache). Search videos, retrieve transcripts, analyze channels, monitor live streams, and more - all optimized for AI agents with smart caching to minimize API quota usage.

**Version:** 0.0.0 (Experimental First Release)

## Features

### 🔍 Search & Discovery
- **Video Search** - Find videos by keywords with metadata (title, description, views, etc.)
- **Channel Search** - Discover channels by query
- **Live Stream Search** - Find currently broadcasting live videos

### 📊 Metadata & Analytics
- **Video Details** - Complete metadata including statistics (views, likes, comments)
- **Channel Info** - Detailed channel statistics and subscriber counts
- **Live Status** - Check if a video is currently streaming with viewer counts

### 📝 Transcript Management
- **Full Transcripts** - Download complete video transcripts with timestamps
- **Transcript Previews** - Get summarized transcript snippets for quick context
- **Chunked Access** - Navigate large transcripts in manageable pieces
- **Multi-Language Support** - List and retrieve transcripts in available languages

### 💬 Engagement & Live Chat
- **Video Comments** - Fetch top comments with engagement metrics
- **Live Chat Monitoring** - Real-time access to live stream chat messages
- **Live Chat Pagination** - Efficient polling for new chat messages

### ⚡ Performance & Caching
- **Intelligent Multi-Tier Caching** - Optimized for different data volatility:
  - `youtube.content` - Permanent caching for immutable content (transcripts)
  - `youtube.api` - 24h cache for general API data (video/channel metadata)
  - `youtube.comments` - 5m cache for rapidly changing comment data
  - `youtube.search` - 6h cache for search results
  - Live streaming - 30s-5m cache for real-time data
- **Reference-Based Results** - Large datasets returned as references to minimize context usage
- **Preview Generation** - Automatic previews for transcript and large data
- **Smart Quota Management** - Caching reduces API quota usage by ~75%

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- **YouTube Data API v3 Key** - [Get one here](https://console.cloud.google.com/apis/credentials)

## Getting Your YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **YouTube Data API v3**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "YouTube Data API v3"
   - Click "Enable"
4. Create credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy your API key
5. (Optional) Restrict your API key:
   - Click on the key to edit
   - Under "API restrictions", select "Restrict key"
   - Choose "YouTube Data API v3"
   - Save

**Default Quota:** 10,000 units/day (~100 searches or ~10,000 metadata requests)

## Quick Start

### Installation (Local)

```bash
# Clone the repository
git clone https://github.com/l4b4r4b4b4/yt-mcp
cd yt-mcp

# Install dependencies
uv sync

# Set your API key
export YOUTUBE_API_KEY="your-api-key-here"

# Run the server (stdio mode for Claude Desktop)
uv run yt-mcp stdio
```

### Installation (Docker)

```bash
# Clone the repository
git clone https://github.com/l4b4r4b4b4/yt-mcp
cd yt-mcp

# Set your API key in .env file
echo "YOUTUBE_API_KEY=your-api-key-here" > .env

# Build and run with docker-compose
docker compose up
```

The server will be available at `http://localhost:8000` in HTTP mode.

## Configuration

### Environment Variables

Set your YouTube API key via environment variable:

```bash
export YOUTUBE_API_KEY="your-youtube-api-key"
```

Or add to your shell profile (`~/.zshrc`, `~/.bashrc`):

```bash
echo 'export YOUTUBE_API_KEY="your-key"' >> ~/.zshrc
```

**Optional Langfuse Tracing:**
```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

### Using with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "youtube": {
      "command": "uv",
      "args": ["--directory", "/path/to/yt-mcp", "run", "yt-mcp", "stdio"],
      "env": {
        "YOUTUBE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Using with Zed

Add to your Zed settings (`.zed/settings.json` or global settings):

```json
{
  "context_servers": {
    "youtube-mcp": {
      "command": {
        "path": "uv",
        "args": ["--directory", "/path/to/yt-mcp", "run", "yt-mcp", "stdio"],
        "env": {
          "YOUTUBE_API_KEY": "your-api-key-here"
        }
      }
    }
  }
}
```

### Using with Docker

#### Production (docker-compose)

```bash
# Create .env file with your API key
echo "YOUTUBE_API_KEY=your-api-key" > .env

# Run production server
docker compose up

# Run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop server
docker compose down
```

#### Development (with hot reload)

```bash
# Run development server with code volume mount
docker compose --profile dev up
```

#### Direct Docker Run

```bash
# Build the image
docker build -f docker/Dockerfile -t yt-mcp:latest .

# Run the container
docker run -p 8000:8000 \
  -e YOUTUBE_API_KEY="your-api-key" \
  yt-mcp:latest

# With Langfuse tracing
docker run -p 8000:8000 \
  -e YOUTUBE_API_KEY="your-api-key" \
  -e LANGFUSE_PUBLIC_KEY="pk-lf-..." \
  -e LANGFUSE_SECRET_KEY="sk-lf-..." \
  yt-mcp:latest
```

## Available Tools

### 🔍 Search Tools

#### `search_videos(query: str, max_results: int = 5)`

Search for YouTube videos matching a query.

**Parameters:**
- `query` (string, required) - Search term (e.g., "NixOS tutorials", "vimjoyer nix")
- `max_results` (integer, optional) - Number of results, 1-50, default: 5

**Returns:**
```json
[
  {
    "video_id": "abc123",
    "title": "Video Title",
    "description": "Video description...",
    "url": "https://www.youtube.com/watch?v=abc123",
    "thumbnail": "https://i.ytimg.com/vi/abc123/default.jpg",
    "channel_title": "Channel Name",
    "published_at": "2024-01-15T10:30:00Z"
  }
]
```

**Caching:** 6 hours (youtube.search namespace)
**Quota Cost:** 100 units per request

**Example:**
```
Search for videos about "Nix flakes tutorial"
```

---

#### `search_channels(query: str, max_results: int = 5)`

Search for YouTube channels matching a query.

**Parameters:**
- `query` (string, required) - Channel search term
- `max_results` (integer, optional) - Number of results, 1-50, default: 5

**Returns:**
```json
[
  {
    "channel_id": "UCxyz123",
    "title": "Channel Name",
    "description": "Channel description...",
    "url": "https://www.youtube.com/channel/UCxyz123",
    "thumbnail": "https://yt3.ggpht.com/...",
    "published_at": "2020-05-10T08:00:00Z"
  }
]
```

**Caching:** 6 hours (youtube.search namespace)
**Quota Cost:** 100 units per request

---

#### `search_live_videos(query: str, max_results: int = 5)`

Search for currently live YouTube videos.

**Parameters:**
- `query` (string, required) - Search query (e.g., "gaming live", "news live")
- `max_results` (integer, optional) - Number of results, 1-50, default: 5

**Returns:**
```json
[
  {
    "video_id": "live123",
    "title": "Live Stream Title",
    "description": "Stream description...",
    "url": "https://www.youtube.com/watch?v=live123",
    "thumbnail": "https://i.ytimg.com/vi/live123/default.jpg",
    "channel_title": "Streamer Name",
    "published_at": "2024-01-20T15:00:00Z"
  }
]
```

**Caching:** 6 hours (youtube.search namespace)
**Quota Cost:** 100 units per request

---

### 📊 Metadata & Status Tools

#### `get_video_details(video_id: str)`

Get detailed information about a specific video.

**Parameters:**
- `video_id` (string, required) - YouTube video ID (e.g., "dQw4w9WgXcQ")

**Returns:**
```json
{
  "video_id": "abc123",
  "title": "Video Title",
  "description": "Full description...",
  "url": "https://www.youtube.com/watch?v=abc123",
  "thumbnail": "https://i.ytimg.com/vi/abc123/maxresdefault.jpg",
  "channel_title": "Channel Name",
  "published_at": "2024-01-15T10:30:00Z",
  "view_count": "150000",
  "like_count": "5000",
  "comment_count": "300",
  "duration": "PT15M30S",
  "tags": ["nix", "linux", "tutorial"]
}
```

**Caching:** 24 hours (youtube.api namespace)
**Quota Cost:** 1 unit per request

---

#### `get_channel_info(channel_id: str)`

Get detailed information about a YouTube channel.

**Parameters:**
- `channel_id` (string, required) - YouTube channel ID (e.g., "UCuAXFkgsw1L7xaCfnd5JJOw")

**Returns:**
```json
{
  "channel_id": "UCxyz123",
  "title": "Channel Name",
  "description": "Channel description...",
  "url": "https://www.youtube.com/channel/UCxyz123",
  "thumbnail": "https://yt3.ggpht.com/...",
  "subscriber_count": "50000",
  "video_count": "200",
  "view_count": "5000000",
  "published_at": "2020-05-10T08:00:00Z"
}
```

**Caching:** 24 hours (youtube.api namespace)
**Quota Cost:** 1 unit per request

---

#### `is_live(video_id: str)`

Check if a YouTube video is currently live streaming.

**Parameters:**
- `video_id` (string, required) - YouTube video ID to check

**Returns:**
```json
{
  "video_id": "live123",
  "is_live": true,
  "viewer_count": 1234,
  "scheduled_start_time": "2024-01-20T15:00:00Z",
  "actual_start_time": "2024-01-20T15:02:00Z",
  "active_live_chat_id": "Cg0KC2xpdmUxMjM..."
}
```

**Caching:** 30 seconds (youtube.api namespace)
**Quota Cost:** 1 unit per request

**Note:** Use this to check status before accessing live chat.

---

### 📝 Transcript Tools

#### `list_available_transcripts(video_id: str)`

List all available transcript languages for a video.

**Parameters:**
- `video_id` (string, required) - YouTube video ID

**Returns:**
```json
{
  "video_id": "abc123",
  "available_languages": ["en", "es", "fr", "de"],
  "transcript_info": [
    {
      "language": "en",
      "language_code": "en",
      "is_generated": false,
      "is_translatable": true
    },
    {
      "language": "es",
      "language_code": "es",
      "is_generated": true,
      "is_translatable": false
    }
  ]
}
```

**Caching:** Permanent (youtube.content namespace)
**Quota Cost:** 0 (uses youtube-transcript-api, not YouTube Data API)

**Note:** Always check this first before requesting transcripts.

---

#### `get_video_transcript_preview(video_id: str, language: str = "en", max_chars: int = 2000)`

Get a preview of the video transcript (first N characters).

**Parameters:**
- `video_id` (string, required) - YouTube video ID
- `language` (string, optional) - Language code (default: "en")
- `max_chars` (integer, optional) - Maximum characters to return (default: 2000)

**Returns:**
```json
{
  "video_id": "abc123",
  "language": "en",
  "preview": "First 2000 characters of transcript...",
  "total_length": 50000,
  "is_truncated": true
}
```

**Caching:** Permanent (youtube.content namespace)
**Quota Cost:** 0

**Note:** Use this for quick context before fetching full transcript.

---

#### `get_full_transcript(video_id: str, language: str = "en")`

Get the complete transcript of a video with timestamps.

**Parameters:**
- `video_id` (string, required) - YouTube video ID
- `language` (string, optional) - Language code (default: "en")

**Returns:**
```json
{
  "video_id": "abc123",
  "language": "en",
  "transcript": [
    {
      "text": "Hello everyone, welcome to this tutorial...",
      "start": 0.0,
      "duration": 3.5
    },
    {
      "text": "Today we're going to learn about...",
      "start": 3.5,
      "duration": 4.2
    }
  ],
  "full_text": "Hello everyone, welcome to this tutorial. Today we're going to learn about..."
}
```

**Caching:** Permanent (youtube.content namespace)
**Quota Cost:** 0

**Note:** Large transcripts return a RefCache reference. Use `get_cached_result` to paginate or retrieve full data.

---

#### `get_transcript_chunk(video_id: str, start_index: int = 0, chunk_size: int = 50, language: str = "en")`

Get a specific chunk of transcript entries (for pagination).

**Parameters:**
- `video_id` (string, required) - YouTube video ID
- `start_index` (integer, optional) - Starting entry index, 0-based (default: 0)
- `chunk_size` (integer, optional) - Number of entries to return (default: 50)
- `language` (string, optional) - Language code (default: "en")

**Returns:**
```json
{
  "video_id": "abc123",
  "language": "en",
  "start_index": 0,
  "chunk_size": 50,
  "entries": [
    {"text": "...", "start": 0.0, "duration": 3.5}
  ],
  "total_entries": 250,
  "has_more": true
}
```

**Caching:** Permanent (youtube.content namespace)
**Quota Cost:** 0

---

### 💬 Engagement & Live Chat Tools

#### `get_video_comments(video_id: str, max_results: int = 20)`

Get top comments from a video with engagement metrics.

**Parameters:**
- `video_id` (string, required) - YouTube video ID
- `max_results` (integer, optional) - Number of comments, 1-100 (default: 20)

**Returns:**
```json
{
  "video_id": "abc123",
  "comments": [
    {
      "author": "Username",
      "text": "Great video! This really helped me understand...",
      "like_count": 42,
      "published_at": "2024-01-20T15:30:00Z",
      "reply_count": 3
    }
  ],
  "total_returned": 20
}
```

**Caching:** 5 minutes (youtube.comments namespace)
**Quota Cost:** 1 unit per request

**Note:** Returns empty list if comments are disabled (not an error). Only top-level comments, no replies.

---

#### `get_live_chat_id(video_id: str)`

Get the live chat ID for a currently streaming video.

**Parameters:**
- `video_id` (string, required) - YouTube video ID of the live stream

**Returns:**
```json
{
  "video_id": "live123",
  "live_chat_id": "Cg0KC2xpdmUxMjM...",
  "is_live": true
}
```

**Caching:** 5 minutes (youtube.api namespace)
**Quota Cost:** 1 unit per request

**Note:** Use `is_live` first to verify video is streaming. Chat ID remains constant during stream.

---

#### `get_live_chat_messages(video_id: str, max_results: int = 200, page_token: str | None = None)`

Get recent live chat messages from a streaming video with pagination.

**Parameters:**
- `video_id` (string, required) - YouTube video ID of the live stream
- `max_results` (integer, optional) - Maximum messages to return, 1-2000 (default: 200)
- `page_token` (string, optional) - Pagination token from previous call (None for first call)

**Returns:**
```json
{
  "video_id": "live123",
  "messages": [
    {
      "author": "ViewerName",
      "text": "Great stream!",
      "published_at": "2024-01-20T16:45:30Z",
      "author_channel_id": "UCxyz..."
    }
  ],
  "total_returned": 50,
  "next_page_token": "GgkKBxIFMTIzNDU",
  "polling_interval_millis": 30000
}
```

**Caching:** 30 seconds (youtube.comments namespace)
**Quota Cost:** 1 unit per request

**Polling Pattern:**
1. First call: No `page_token` → Get latest messages + `next_page_token`
2. Store `next_page_token`
3. Wait 30-60 seconds (respect `polling_interval_millis`)
4. Subsequent calls: Pass `page_token` → Get only NEW messages
5. Repeat steps 2-4 for continuous monitoring

**Note:** MCP is request/response (not true streaming). Agent must manually poll this tool repeatedly to see new messages.

---

### 🗂️ Cache Management Tools

#### `get_cached_result(ref_id: str, page: int | None = None, page_size: int | None = None, max_size: int | None = None)`

Retrieve and paginate through cached results.

**Parameters:**
- `ref_id` (string, required) - Reference ID from cached tool (e.g., from large transcript)
- `page` (integer, optional) - Page number, 1-indexed
- `page_size` (integer, optional) - Items per page, 1-100
- `max_size` (integer, optional) - Maximum preview size in tokens

**Returns:**
```json
{
  "ref_id": "youtube.content:transcript_abc123_en",
  "preview": [...],
  "total_items": 250,
  "page": 2,
  "total_pages": 5
}
```

**Note:** Use this when a tool returns a `ref_id` instead of full data (for large results).

---

## Example Use Cases

### Finding a Specific Video

**Goal:** Find Vimjoyer's video about Nix garbage collection that keeps only the last N generations

**Workflow:**
```
1. Search: "Search for videos by Vimjoyer about Nix garbage collection generations"
   → Returns list of videos with IDs

2. Preview: "Get transcript preview for video abc123"
   → Returns first 2000 characters to check relevance

3. Analyze: "Get full transcript for video abc123 and find the section about keeping last N generations"
   → Returns complete transcript with timestamps

4. Extract: AI analyzes transcript and returns relevant section with timestamp
```

---

### Channel Analysis

**Goal:** Analyze a channel's recent content and engagement

**Workflow:**
```
1. Search: "Find the NixOS channel"
   → Returns channel ID

2. Info: "Get channel info for UC[channel-id]"
   → Returns subscriber count, video count, total views

3. Videos: "Search for recent videos from NixOS channel"
   → Returns latest video list

4. Engagement: "Get comments for video abc123"
   → Returns top comments with like counts
```

---

### Live Stream Monitoring

**Goal:** Monitor a live stream and track chat activity

**Workflow:**
```
1. Find: "Search for live videos about Python programming"
   → Returns currently live streams

2. Check: "Is video live123 currently streaming?"
   → Confirms live status and viewer count

3. Connect: "Get live chat ID for video live123"
   → Returns chat ID needed for messages

4. Monitor: "Get live chat messages for video live123"
   → Returns recent messages + next_page_token

5. Poll: "Get live chat messages for video live123 with page_token=XYZ"
   → Returns only new messages since last call

6. Repeat: Wait 30-60 seconds, then repeat step 5
```

---

### Transcript Analysis Across Languages

**Goal:** Find and compare transcripts in multiple languages

**Workflow:**
```
1. Search: "Search for videos about 'machine learning basics'"
   → Returns video IDs

2. Check: "List available transcripts for video abc123"
   → Returns ["en", "es", "fr", "de", "auto-generated"]

3. Compare: "Get transcript preview for abc123 in English"
   → Preview English version

4. Compare: "Get transcript preview for abc123 in Spanish"
   → Preview Spanish version

5. Analyze: AI compares content across languages
```

---

## Caching Strategy

The server uses a **4-tier caching architecture** optimized for different data volatility levels:

### Tier 1: Search Results (6 hours)
- **Namespace:** `youtube.search`
- **TTL:** 6 hours (21,600 seconds)
- **Size:** 300 entries
- **Use:** Video search, channel search, live video search
- **Rationale:** Search rankings change throughout the day; 6h balances freshness with quota savings

### Tier 2: API Metadata (24 hours)
- **Namespace:** `youtube.api`
- **TTL:** 24 hours (86,400 seconds)
- **Size:** 1000 entries
- **Use:** Video details, channel info
- **Rationale:** Video stats change daily but not hourly; 24h cache reduces quota by 24x

### Tier 3: Comments & Engagement (5 minutes)
- **Namespace:** `youtube.comments`
- **TTL:** 5 minutes (300 seconds)
- **Size:** 500 entries
- **Use:** Video comments
- **Rationale:** Comments can change rapidly on viral videos; 5m balances real-time with quota

### Tier 4: Immutable Content (Permanent)
- **Namespace:** `youtube.content`
- **TTL:** Permanent (no expiration)
- **Size:** 5000 entries
- **Use:** Video transcripts (all transcript tools)
- **Rationale:** Transcripts never change once published; permanent cache eliminates redundant fetches

### Tier 5: Live Streaming (30 seconds - 5 minutes)
- **Namespaces:** `youtube.api` (live status), `youtube.comments` (chat messages)
- **TTL:** 30 seconds (live status/chat), 5 minutes (chat ID)
- **Use:** Live stream status, chat messages, chat IDs
- **Rationale:** Real-time data needs frequent updates but excessive polling wastes quota

### RefCache Integration

Large results (transcripts, long comment lists) are automatically handled by RefCache:

1. **Small Results (≤2048 tokens):** Returned inline directly to agent
2. **Large Results (>2048 tokens):** Cached with `ref_id` + preview returned
3. **Pagination:** Use `get_cached_result(ref_id, page=N)` to access specific pages
4. **Sample Previews:** Large lists show representative samples in preview

**Benefits:**
- Minimizes context window pollution for agents
- Enables efficient pagination without re-fetching
- Preserves full data for detailed analysis when needed

---

## API Quota Management

### Understanding Quotas

YouTube Data API v3 has daily quotas measured in "units":

- **Default Quota:** 10,000 units/day (free tier)
- **Search Operation:** 100 units each
- **Metadata Operation:** 1 unit each (video details, channel info, comments)
- **Live Chat Messages:** 1 unit per request
- **Transcript Operations:** 0 units (uses youtube-transcript-api, not YouTube Data API)

### Quota Calculation Examples

**Without Caching:**
- 100 video searches = 10,000 units = **entire daily quota**
- 10,000 video detail requests = 10,000 units = **entire daily quota**

**With Caching (6h TTL for search, 24h for metadata):**
- Same 100 searches (6h cache) = 400 units/day (~75% savings)
- Same 10,000 metadata requests (24h cache) = ~420 units/day (~96% savings)

### Best Practices

1. **Use transcript tools first** - They cost 0 quota
2. **Search broadly, then get details** - Search costs 100x more than metadata
3. **Cache effectively** - Let the built-in caching do its job
4. **Batch operations** - Group related requests in single session
5. **Monitor usage** - Server returns quota errors with clear messages

### Increasing Quota

If you need higher quota:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to your project → APIs & Services → YouTube Data API v3
3. Click "Quotas" tab
4. Request quota increase (requires billing account, but API is still free)
5. Typical increases: 50,000 to 1,000,000 units/day

---

## Docker Details

### Image Sizes

- **Base Image:** 290MB (`ghcr.io/l4b4r4b4b4/fastmcp-base:latest`)
  - Python 3.12-slim + uv + dependencies
  - Shared across all FastMCP projects

- **Production Image:** 229MB (`ghcr.io/l4b4r4b4b4/yt-mcp:latest`)
  - Base + application code
  - Optimized for size and startup speed

### Container Features

- **Non-root user:** Runs as `appuser` for security
- **Health checks:** Built-in health endpoint at `/health`
- **Environment config:** All settings via environment variables
- **Multi-arch:** Supports amd64 and arm64 (M1/M2 Macs)
- **Streamable HTTP:** Uses HTTP transport (recommended for Docker/remote)

### Docker Compose Configuration

The `docker-compose.yml` includes three profiles:

1. **Production (default):** `docker compose up`
   - Port 8000 exposed
   - Optimized production image
   - Auto-restart on failure

2. **Development:** `docker compose --profile dev up`
   - Port 8000 exposed
   - Volume mount for hot reload
   - Development dependencies included

3. **Build:** `docker compose --profile build up base`
   - Builds base image for publishing
   - Only used for releases

---

## Troubleshooting

### "Invalid API Key" Error

**Symptoms:**
```
Error: API key not valid. Please pass a valid API key.
```

**Solutions:**
1. Verify key is set: `echo $YOUTUBE_API_KEY`
2. Check for typos or extra spaces in key
3. Verify key has YouTube Data API v3 enabled in Google Cloud Console
4. Make sure key restrictions (if any) allow YouTube Data API v3

---

### "Quota Exceeded" Error

**Symptoms:**
```
Error: The request cannot be completed because you have exceeded your quota.
```

**Solutions:**
1. Wait until quota resets (midnight Pacific Time)
2. Enable billing in Google Cloud Console for higher quota
3. Use caching effectively (it's automatic, but check `get_cached_result` for large operations)
4. Use transcript tools (0 quota cost) instead of search when possible
5. Request quota increase from Google Cloud Console

---

### "No Transcript Available" Error

**Symptoms:**
```
Error: No transcript found for this video
```

**Solutions:**
1. Use `list_available_transcripts` first to check availability
2. Try auto-generated transcripts: often available even without manual captions
3. Some videos genuinely don't have transcripts (creator didn't enable)
4. Check if video is age-restricted or private

---

### "Comments Disabled" (Empty Result)

**Symptoms:**
```json
{"video_id": "abc123", "comments": [], "total_returned": 0}
```

**This is NOT an error** - the video has comments disabled by the creator. The tool returns an empty list as expected behavior.

---

### Docker: "Cannot connect to server"

**Symptoms:**
```
Error: Failed to connect to localhost:8000
```

**Solutions:**
1. Verify container is running: `docker compose ps`
2. Check container logs: `docker compose logs -f`
3. Ensure port 8000 is not in use: `lsof -i :8000` (macOS/Linux)
4. Verify API key is set in `.env` file or docker-compose environment
5. Check health: `curl http://localhost:8000/health`

---

### Docker: "Rate limiting" or slow responses

**Symptoms:**
- Slow API responses
- Timeout errors

**Solutions:**
1. YouTube API has rate limits - this is normal behavior
2. Caching will improve performance after first requests
3. For local development, use stdio mode instead of HTTP: `uv run yt-mcp stdio`
4. Check your network connection
5. Verify Docker has sufficient resources (memory, CPU)

---

## Development

### Setup Development Environment

```bash
# Using Nix (recommended)
nix develop

# Or install dependencies manually with uv
uv sync
```

### Running Tests

```bash
# Run all tests
uv run pytest

# With coverage report
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_server.py

# Watch mode (requires pytest-watch)
uv run ptw
```

**Current Test Status:** 178 tests passing, 76% code coverage

### Linting and Formatting

```bash
# Check and fix linting issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Type checking
uv run mypy app
```

### Project Structure

```
yt-mcp/
├── app/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── server.py            # Main MCP server with all tools
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── youtube.py       # YouTube API integration
│   │   └── ...              # Other tool modules
│   ├── tracing.py           # Langfuse tracing integration
│   └── prompts.py           # MCP prompts
├── tests/
│   ├── conftest.py          # Pytest configuration
│   ├── test_server.py       # Server tests
│   └── test_youtube.py      # YouTube tool tests
├── docker/
│   ├── Dockerfile           # Production image
│   ├── Dockerfile.base      # Base image with dependencies
│   └── Dockerfile.dev       # Development image
├── .agent/                  # Development notes and planning
├── pyproject.toml           # Dependencies and configuration
├── docker-compose.yml       # Container orchestration
├── flake.nix                # Nix development environment
└── README.md                # This file
```

---

## Version 0.0.0 Release Notes

This is the **first experimental release** of the YouTube MCP server. It's published to test both the implementation and the release workflow.

### What Works
- ✅ All 16 YouTube tools implemented and tested
- ✅ Comprehensive test suite (178 tests, 76% coverage)
- ✅ Multi-tier caching with RefCache integration
- ✅ Docker support (production + development)
- ✅ Langfuse tracing for observability
- ✅ Claude Desktop and Zed integration

### Known Limitations
- This is version 0.0.0 - **expect issues**
- Limited real-world validation (this tests the release process)
- Documentation may have gaps or inaccuracies
- Docker images published but not battle-tested

### Next Steps
- 0.0.1: Bug fixes and improvements from 0.0.0 feedback
- 0.0.x: Continued iteration and refinement
- 0.1.0: After 5-10 patch releases and proven stability
- 1.0.0: Production-ready after 6+ months of 0.x usage

**We encourage feedback!** Open issues on GitHub with any problems or suggestions.

---

## Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `YOUTUBE_API_KEY` | YouTube Data API v3 key | Yes | None |
| `LANGFUSE_PUBLIC_KEY` | Langfuse tracing public key | No | None |
| `LANGFUSE_SECRET_KEY` | Langfuse tracing secret key | No | None |
| `LANGFUSE_HOST` | Langfuse host URL | No | https://cloud.langfuse.com |
| `FASTMCP_PORT` | Server port (HTTP mode) | No | 8000 |
| `FASTMCP_HOST` | Server host (HTTP mode) | No | 0.0.0.0 |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and how to submit pull requests.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Related Projects

- [mcp-refcache](https://github.com/l4b4r4b4b4/mcp-refcache) - Reference-based caching for MCP servers
- [FastMCP](https://github.com/jlowin/fastmcp) - High-performance MCP server framework
- [Model Context Protocol](https://modelcontextprotocol.io/) - Official MCP specification
- [YouTube Data API v3](https://developers.google.com/youtube/v3) - YouTube API documentation
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) - Transcript library

---

## Acknowledgments

- Built on FastMCP and mcp-refcache libraries
- Uses Google's YouTube Data API v3
- Uses youtube-transcript-api for quota-free transcript access
- Langfuse for observability and tracing
- Docker for containerization

---

**Questions or Issues?** Open an issue on [GitHub](https://github.com/l4b4r4b4b4/yt-mcp/issues)
