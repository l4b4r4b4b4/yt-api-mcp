import logging
import time
from typing import List, Dict, Optional, Any
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from mcp.server.fastmcp import FastMCP
from toolsets.servers.cache import ToolsetCache

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("YouTube_API")

# You should store this in environment variables or a secure configuration
YOUTUBE_API_KEY = "super_safe_youtube_key"

mcp = FastMCP("YouTube")

# Initialize separate caches for different types of data
content_cache = ToolsetCache(
    name="youtube.content",
    deterministic=True,  # Content like transcripts doesn't change
    max_size=5000,  # Larger size for permanent content
)

api_cache = ToolsetCache(
    name="youtube.api",
    deterministic=False,
    expiry_seconds=3600 * 24,  # 24 hours for general API data
    max_size=1000,
)

comment_cache = ToolsetCache(
    name="youtube.comments",
    deterministic=False,
    expiry_seconds=3600 * 12,  # 12 hours for comments (more volatile)
    max_size=500,
)

search_cache = ToolsetCache(
    name="youtube.search",
    deterministic=False,
    expiry_seconds=3600 * 6,  # 6 hours for search results (most volatile)
    max_size=300,
)


def get_youtube_service():
    """Create and return the YouTube API service."""
    try:
        return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to build YouTube service: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to build YouTube service: {str(e)}")


@mcp.tool()
@search_cache.cached
def search_videos(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for YouTube videos matching the given query.

    Parameters:
    - query: The search term
    - max_results: Maximum number of results to return (default: 5, max: 50)

    Returns a list of dictionaries with video information.
    """
    logger.debug(f"Searching YouTube for: '{query}' with limit {max_results}")

    max_results = min(max(1, max_results), 50)  # Limit between 1 and 50

    try:
        youtube = get_youtube_service()

        # Execute the search
        search_response = (
            youtube.search()
            .list(
                q=query,
                part="snippet",
                maxResults=max_results,
                type="video",  # Only return videos, not playlists or channels
            )
            .execute()
        )

        results = []

        for item in search_response.get("items", []):
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]

            result = {
                "title": snippet["title"],
                "description": snippet["description"],
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": snippet["thumbnails"]["medium"]["url"],
                "channel_title": snippet["channelTitle"],
                "published_at": snippet["publishedAt"],
            }

            results.append(result)
            logger.debug(f"Added result: {result['title']} (ID: {result['video_id']})")

        logger.info(f"Returning {len(results)} search results")
        return results

    except HttpError as e:
        logger.error(f"YouTube search failed: {str(e)}", exc_info=True)
        raise ValueError(f"YouTube search failed: {str(e)}")


@mcp.tool()
@api_cache.cached
def get_video_details(video_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific YouTube video.

    Parameters:
    - video_id: The YouTube video ID

    Returns a dictionary with detailed video information.
    """
    logger.debug(f"Getting details for YouTube video ID: {video_id}")

    try:
        youtube = get_youtube_service()

        # Get video details
        video_response = (
            youtube.videos()
            .list(part="snippet,contentDetails,statistics", id=video_id)
            .execute()
        )

        if not video_response.get("items"):
            logger.warning(f"No video found with ID: {video_id}")
            raise ValueError(f"No video found with ID: {video_id}")

        video = video_response["items"][0]
        snippet = video["snippet"]
        statistics = video.get("statistics", {})
        content_details = video.get("contentDetails", {})

        result = {
            "title": snippet["title"],
            "description": snippet["description"],
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": snippet["thumbnails"]["high"]["url"],
            "channel_title": snippet["channelTitle"],
            "published_at": snippet["publishedAt"],
            "view_count": statistics.get("viewCount", "0"),
            "like_count": statistics.get("likeCount", "0"),
            "comment_count": statistics.get("commentCount", "0"),
            "duration": content_details.get("duration", ""),
            "tags": snippet.get("tags", []),
        }

        logger.info(f"Retrieved details for video: '{result['title']}'")
        return result

    except HttpError as e:
        logger.error(f"Failed to get video details: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to get video details: {str(e)}")


@mcp.tool()
@search_cache.cached
def search_channels(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for YouTube channels matching the given query.

    Parameters:
    - query: The search term
    - max_results: Maximum number of results to return (default: 5, max: 50)

    Returns a list of dictionaries with channel information.
    """
    logger.debug(f"Searching YouTube channels for: '{query}' with limit {max_results}")

    max_results = min(max(1, max_results), 50)  # Limit between 1 and 50

    try:
        youtube = get_youtube_service()

        # Execute the search for channels
        search_response = (
            youtube.search()
            .list(
                q=query,
                part="snippet",
                maxResults=max_results,
                type="channel",  # Only return channels
            )
            .execute()
        )

        results = []

        for item in search_response.get("items", []):
            channel_id = item["id"]["channelId"]
            snippet = item["snippet"]

            result = {
                "title": snippet["title"],
                "description": snippet["description"],
                "channel_id": channel_id,
                "url": f"https://www.youtube.com/channel/{channel_id}",
                "thumbnail": snippet["thumbnails"]["medium"]["url"],
                "published_at": snippet["publishedAt"],
            }

            results.append(result)
            logger.debug(
                f"Added result: {result['title']} (ID: {result['channel_id']})"
            )

        logger.info(f"Returning {len(results)} channel search results")
        return results

    except HttpError as e:
        logger.error(f"YouTube channel search failed: {str(e)}", exc_info=True)
        raise ValueError(f"YouTube channel search failed: {str(e)}")


@mcp.tool()
@api_cache.cached
def get_channel_info(channel_id: str) -> Dict[str, Any]:
    """
    Get information about a YouTube channel.

    Parameters:
    - channel_id: The YouTube channel ID

    Returns a dictionary with channel information.
    """
    logger.debug(f"Getting info for YouTube channel ID: {channel_id}")

    try:
        youtube = get_youtube_service()

        # Get channel details
        channel_response = (
            youtube.channels()
            .list(part="snippet,statistics,contentDetails", id=channel_id)
            .execute()
        )

        if not channel_response.get("items"):
            logger.warning(f"No channel found with ID: {channel_id}")
            raise ValueError(f"No channel found with ID: {channel_id}")

        channel = channel_response["items"][0]
        snippet = channel["snippet"]
        statistics = channel.get("statistics", {})

        result = {
            "title": snippet["title"],
            "description": snippet["description"],
            "channel_id": channel_id,
            "url": f"https://www.youtube.com/channel/{channel_id}",
            "thumbnail": snippet["thumbnails"]["high"]["url"],
            "subscriber_count": statistics.get("subscriberCount", "0"),
            "video_count": statistics.get("videoCount", "0"),
            "view_count": statistics.get("viewCount", "0"),
            "published_at": snippet["publishedAt"],
        }

        logger.info(f"Retrieved info for channel: '{result['title']}'")
        return result

    except HttpError as e:
        logger.error(f"Failed to get channel info: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to get channel info: {str(e)}")


@mcp.tool()
@comment_cache.cached
def get_video_comments(video_id: str, max_results: int = 20) -> List[Dict[str, str]]:
    """
    Get comments for a YouTube video.

    Parameters:
    - video_id: The YouTube video ID
    - max_results: Maximum number of comments to return (default: 20, max: 100)

    Returns a list of dictionaries with comment information.
    """
    logger.debug(f"Getting comments for YouTube video ID: {video_id}")

    max_results = min(max(1, max_results), 100)  # Limit between 1 and 100

    try:
        youtube = get_youtube_service()

        # Get video comments
        comments_response = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                textFormat="plainText",
            )
            .execute()
        )

        results = []

        for item in comments_response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]

            result = {
                "author": comment["authorDisplayName"],
                "text": comment["textDisplay"],
                "like_count": comment["likeCount"],
                "published_at": comment["publishedAt"],
            }

            results.append(result)

        logger.info(f"Retrieved {len(results)} comments for video ID: {video_id}")
        return results

    except HttpError as e:
        if "commentsDisabled" in str(e):
            logger.warning(f"Comments are disabled for video ID: {video_id}")
            return []
        logger.error(f"Failed to get video comments: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to get video comments: {str(e)}")


@mcp.tool()
@content_cache.cached
def get_video_transcript_preview(
    video_id: str, language_code: Optional[str] = None, max_length: int = 4000
) -> Dict[str, Any]:
    """
    Get a preview of the transcript for a YouTube video.

    Parameters:
    - video_id: The YouTube video ID
    - language_code: Optional ISO language code (e.g., 'en', 'es', 'fr')
    - max_length: Maximum length of the transcript preview (default: 4000 chars)

    Returns a dictionary with a preview of the transcript and metadata.
    """
    logger.debug(f"Getting transcript preview for YouTube video ID: {video_id}")

    try:
        # Get the full transcript data (reusing the implementation)
        full_transcript = _get_full_transcript(video_id, language_code)

        # The cache decorator will automatically handle any CacheReference objects

        # Extract transcript text safely
        transcript_text = ""
        language_code_result = None
        is_generated = True
        segment_count = 0

        if isinstance(full_transcript, dict):
            transcript_text = full_transcript.get("full_text", "")
            language_code_result = full_transcript.get("language_code")
            is_generated = full_transcript.get("is_generated", True)
            segments = full_transcript.get("segments", [])
            segment_count = len(segments)
        else:
            # Handle case where full_transcript might be a string or other type
            transcript_text = str(full_transcript)

        # Create a preview version with limited text
        if len(transcript_text) > max_length:
            preview_text = (
                transcript_text[:max_length]
                + "... [preview truncated, use get_full_transcript for complete text]"
            )
        else:
            preview_text = transcript_text

        # Calculate approximate statistics about the full transcript
        word_count = len(transcript_text.split())
        total_length = len(transcript_text)

        result = {
            "video_id": video_id,
            "preview_text": preview_text,
            "language_code": language_code_result,
            "is_generated": is_generated,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "full_transcript_stats": {
                "total_length": total_length,
                "word_count": word_count,
                "segment_count": segment_count,
            },
        }

        logger.info(f"Retrieved transcript preview for video ID: {video_id}")
        return result

    except Exception as e:
        logger.error(f"Failed to get video transcript preview: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to get video transcript preview: {str(e)}")


@mcp.tool()
@content_cache.cached
def get_full_transcript(
    video_id: str, language_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get the complete transcript for a YouTube video.

    Parameters:
    - video_id: The YouTube video ID
    - language_code: Optional ISO language code (e.g., 'en', 'es', 'fr')

    Returns a dictionary with the full transcript text and metadata.
    """
    logger.debug(f"Getting full transcript for YouTube video ID: {video_id}")

    try:
        # Use the internal function to get the full transcript
        # The cache decorator will handle any CacheReference resolution automatically
        return _get_full_transcript(video_id, language_code)
    except Exception as e:
        logger.error(f"Failed to get full video transcript: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to get full video transcript: {str(e)}")


# Internal helper function for transcript retrieval
@content_cache.cached
def _get_full_transcript(
    video_id: str, language_code: Optional[str] = None
) -> Dict[str, Any]:
    """Internal function to get the full transcript data"""
    try:
        # Get all available transcripts
        if language_code:
            logger.debug(f"Requesting transcript in language: {language_code}")
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            try:
                # Try to get the specific language
                transcript = transcript_list.find_transcript([language_code])
            except NoTranscriptFound:
                # If requested language isn't available, try to translate to it
                logger.info(
                    f"No {language_code} transcript found, attempting to translate"
                )
                try:
                    transcript = transcript_list.find_transcript(["en"]).translate(
                        language_code
                    )
                except Exception as e:
                    logger.warning(f"Translation failed: {str(e)}")
                    # If translation fails, just get the default transcript
                    # Provide a default language array parameter
                    transcript = transcript_list.find_generated_transcript(["en"])

            # Get the transcript data
            transcript_data = transcript.fetch()

            language_code_result = transcript.language_code
            is_generated = transcript.is_generated

        else:
            # Get the default transcript (usually auto-generated in video's language)
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            language_code_result = (
                None  # We don't know the language code when getting default
            )
            is_generated = True

        # Combine all text segments
        full_text = ""
        segments = []

        for item in transcript_data:
            try:
                # Handle different types of transcript items
                text = ""
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                    segments.append(item)
                elif hasattr(item, "text"):
                    text = getattr(item, "text")
                    # Convert object to dictionary for consistency
                    segment_dict = {
                        "text": text,
                        "start": getattr(item, "start", 0),
                        "duration": getattr(item, "duration", 0),
                    }
                    segments.append(segment_dict)
                else:
                    # Skip if we can't get text
                    logger.warning(
                        f"Couldn't extract text from transcript item: {item}"
                    )
                    continue

                full_text += text + " "
            except Exception as e:
                logger.warning(f"Error processing transcript segment: {str(e)}")
                continue

        full_text = full_text.strip()

        result = {
            "video_id": video_id,
            "full_text": full_text,
            "segments": segments,
            "language_code": language_code_result,
            "is_generated": is_generated,
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

        text_preview = full_text[:100] + "..." if len(full_text) > 100 else full_text
        logger.info(
            f"Retrieved transcript for video ID: {video_id} in language: {language_code_result}"
        )
        logger.debug(f"Transcript preview: {text_preview}")

        return result

    except TranscriptsDisabled:
        logger.warning(f"Transcripts are disabled for video ID: {video_id}")
        raise ValueError(f"Transcripts are disabled for this video")

    except NoTranscriptFound:
        logger.warning(f"No transcript found for video ID: {video_id}")
        raise ValueError(f"No transcript found for this video")

    except Exception as e:
        logger.error(f"Failed to get video transcript: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to get video transcript: {str(e)}")


@mcp.tool()
@content_cache.cached
def list_available_transcripts(video_id: str) -> List[Dict[str, str]]:
    """
    List all available transcript languages for a YouTube video.

    Parameters:
    - video_id: The YouTube video ID

    Returns a list of dictionaries with language information.
    """
    logger.debug(f"Listing available transcripts for YouTube video ID: {video_id}")

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        results = []
        for transcript in transcript_list:
            result = {
                "language_code": transcript.language_code,
                "language": transcript.language,
                "is_generated": transcript.is_generated,
                "video_id": video_id,
            }
            results.append(result)
            logger.debug(
                f"Found transcript: {result['language']} ({result['language_code']})"
            )

        logger.info(
            f"Found {len(results)} available transcripts for video ID: {video_id}"
        )
        return results

    except TranscriptsDisabled:
        logger.warning(f"Transcripts are disabled for video ID: {video_id}")
        raise ValueError(f"Transcripts are disabled for this video")

    except Exception as e:
        logger.error(f"Failed to list available transcripts: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to list available transcripts: {str(e)}")


@mcp.tool()
@content_cache.cached
def get_transcript_chunk(
    video_id: str,
    chunk_index: int = 0,
    chunk_size: int = 4000,
    language_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a specific chunk of a transcript for a YouTube video.

    Parameters:
    - video_id: The YouTube video ID
    - chunk_index: Which chunk to retrieve (0-based index)
    - chunk_size: Maximum size of each chunk in characters (default: 4000)
    - language_code: Optional ISO language code (e.g., 'en', 'es', 'fr')

    Returns a dictionary with the chunk text and metadata.
    """
    logger.debug(f"Getting transcript chunk {chunk_index} for video ID: {video_id}")

    try:
        # Get full transcript - cache decorator will handle references automatically
        full_transcript = _get_full_transcript(video_id, language_code)

        # Extract the full text safely
        full_text = ""
        language_code_result = None
        is_generated = True

        if isinstance(full_transcript, dict):
            full_text = full_transcript.get("full_text", "")
            language_code_result = full_transcript.get("language_code")
            is_generated = full_transcript.get("is_generated", True)
        else:
            # Handle non-dictionary case
            full_text = str(full_transcript)

        # Calculate total chunks
        total_chunks = (len(full_text) + chunk_size - 1) // chunk_size

        # Validate chunk index
        if chunk_index < 0 or chunk_index >= total_chunks:
            raise ValueError(
                f"Invalid chunk index {chunk_index}. Available chunks: 0-{total_chunks-1}"
            )

        # Extract the requested chunk
        start_pos = chunk_index * chunk_size
        end_pos = min(start_pos + chunk_size, len(full_text))
        chunk_text = full_text[start_pos:end_pos]

        result = {
            "video_id": video_id,
            "chunk_text": chunk_text,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "language_code": language_code_result,
            "is_generated": is_generated,
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

        logger.info(
            f"Retrieved transcript chunk {chunk_index}/{total_chunks-1} for video ID: {video_id}"
        )
        return result

    except Exception as e:
        logger.error(f"Failed to get transcript chunk: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to get transcript chunk: {str(e)}")


def clear_cache() -> Dict[str, Any]:
    """
    Developer utility to clear all YouTube API caches.

    This is NOT exposed as a tool, but can be called programmatically
    by developers for maintenance or testing.

    Returns a dictionary with information about the clearing operation.
    """
    api_count = api_cache.clear()
    content_count = content_cache.clear()
    comment_count = comment_cache.clear()
    search_count = search_cache.clear()

    total_count = api_count + content_count + comment_count + search_count

    print(f"Cleared {total_count} items from YouTube caches")

    return {
        "cleared_items": {
            "api_cache": api_count,
            "content_cache": content_count,
            "comment_cache": comment_count,
            "search_cache": search_count,
            "total": total_count,
        },
        "timestamp": time.time(),
        "status": "All YouTube caches cleared successfully",
    }


@mcp.tool()
def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about all YouTube API caches.

    This tool provides information about cache usage including hits, misses,
    and cache sizes for each of the specialized YouTube API caches.

    Returns a dictionary with cache statistics.
    """
    api_stats = api_cache.get_stats()
    content_stats = content_cache.get_stats()
    comment_stats = comment_cache.get_stats()
    search_stats = search_cache.get_stats()

    total_hits = (
        api_stats["hits"]
        + content_stats["hits"]
        + comment_stats["hits"]
        + search_stats["hits"]
    )

    total_misses = (
        api_stats["misses"]
        + content_stats["misses"]
        + comment_stats["misses"]
        + search_stats["misses"]
    )

    total_entries = (
        api_stats["total_entries"]
        + content_stats["total_entries"]
        + comment_stats["total_entries"]
        + search_stats["total_entries"]
    )

    # Calculate cache efficiency
    cache_efficiency = 0
    if total_hits + total_misses > 0:
        cache_efficiency = (total_hits / (total_hits + total_misses)) * 100

    return {
        "api_cache": api_stats,
        "content_cache": content_stats,
        "comment_cache": comment_stats,
        "search_cache": search_stats,
        "overall_stats": {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_entries": total_entries,
            "cache_efficiency_percentage": cache_efficiency,
        },
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    if not YOUTUBE_API_KEY:
        logger.error(
            "YouTube API key not found. Set the YOUTUBE_API_KEY environment variable."
        )
        exit(1)

    logger.info("Starting YouTube API server")
    mcp.run(transport="stdio")
