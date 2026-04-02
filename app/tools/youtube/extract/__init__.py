"""Extract layer for YouTube MCP ELT pipeline.

This module provides raw data extraction functions that wrap YouTube API calls.
The Extract layer is responsible for fetching unprocessed data from the API,
which is then cached by the Load layer and analyzed by the Transform layer.

Key principles:
1. Return raw data exactly as received from API
2. No transformation or analysis at this layer
3. Designed for caching - consistent output formats
4. Quota-consuming operations that benefit from caching

Extract modules:
- videos: Video search and details extraction
- channels: Channel search and info extraction
- comments: Video comment extraction
- trending: Trending video extraction
- batch: Utilities for batch extraction operations

Example:
    >>> from app.tools.youtube.extract import (
    ...     extract_videos_raw,
    ...     extract_channel_info_batch,
    ...     batch_extract,
    ... )
    >>>
    >>> # Extract raw video search results
    >>> videos = await extract_videos_raw("kubernetes", max_results=50)
    >>>
    >>> # Batch extract channel info
    >>> channel_ids = [v["channel_id"] for v in channels]
    >>> info = await extract_channel_info_batch(channel_ids)
"""

from __future__ import annotations

from app.tools.youtube.extract.batch import (
    QUOTA_COSTS,
    BatchResult,
    QuotaTracker,
    batch_extract,
    batch_extract_chunked,
    can_complete_extraction,
    estimate_quota_cost,
)
from app.tools.youtube.extract.channels import (
    extract_channel_info_batch,
    extract_channel_info_single,
    extract_channels_from_videos,
    extract_channels_raw,
)
from app.tools.youtube.extract.comments import (
    extract_comments_batch,
    extract_comments_raw,
    extract_top_comments,
)
from app.tools.youtube.extract.trending import (
    CATEGORY_IDS,
    extract_trending_by_category,
    extract_trending_raw,
)
from app.tools.youtube.extract.videos import (
    extract_channel_videos_raw,
    extract_video_details_batch,
    extract_video_details_single,
    extract_videos_raw,
)

__all__ = [
    # Trending extraction
    "CATEGORY_IDS",
    "QUOTA_COSTS",
    # Batch utilities
    "BatchResult",
    "QuotaTracker",
    "batch_extract",
    "batch_extract_chunked",
    "can_complete_extraction",
    "estimate_quota_cost",
    # Channel extraction
    "extract_channel_info_batch",
    "extract_channel_info_single",
    # Video extraction
    "extract_channel_videos_raw",
    "extract_channels_from_videos",
    "extract_channels_raw",
    # Comment extraction
    "extract_comments_batch",
    "extract_comments_raw",
    "extract_top_comments",
    "extract_trending_by_category",
    "extract_trending_raw",
    "extract_video_details_batch",
    "extract_video_details_single",
    "extract_videos_raw",
]
