"""Batch extraction utilities for YouTube MCP ELT pipeline.

This module provides utilities for efficient batch data extraction,
coordinating multiple extraction operations while managing quota
and rate limits.

Key features:
- Parallel extraction with concurrency limits
- Quota-aware batching
- Progress tracking for large extractions
- Error handling with partial results
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class BatchResult:
    """Result of a batch extraction operation.

    Attributes:
        successful: List of successfully extracted items.
        failed: Dictionary mapping failed keys to error messages.
        total_requested: Total number of items requested.
        total_successful: Number of successfully extracted items.
        total_failed: Number of failed extractions.
    """

    successful: list[Any] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    total_requested: int = 0
    total_successful: int = 0
    total_failed: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_requested == 0:
            return 100.0
        return (self.total_successful / self.total_requested) * 100


@dataclass
class QuotaTracker:
    """Track quota usage across batch operations.

    Attributes:
        daily_limit: Daily quota limit (default: 10,000).
        used: Quota units used so far.
        remaining: Quota units remaining.
    """

    daily_limit: int = 10000
    used: int = 0

    @property
    def remaining(self) -> int:
        """Calculate remaining quota."""
        return max(0, self.daily_limit - self.used)

    def can_afford(self, cost: int) -> bool:
        """Check if we can afford a quota cost."""
        return self.remaining >= cost

    def charge(self, cost: int) -> None:
        """Charge quota units."""
        self.used += cost
        logger.debug(f"Quota charged: {cost} units, remaining: {self.remaining}")

    def reset(self) -> None:
        """Reset quota counter (e.g., at midnight)."""
        self.used = 0


# Quota costs for different operations
QUOTA_COSTS = {
    "search_videos": 100,
    "search_channels": 100,
    "video_details": 1,  # Per video in batch (up to 50)
    "channel_info": 1,  # Per channel in batch (up to 50)
    "comments": 1,
    "trending": 1,
    "transcripts": 0,  # Free (uses youtube-transcript-api)
}


async def batch_extract[T](
    items: list[str],
    extractor: Callable[[str], Awaitable[T]],
    max_concurrency: int = 5,
    delay_between_batches: float = 0.1,
    stop_on_error: bool = False,
) -> BatchResult:
    """Extract data for multiple items with controlled concurrency.

    Executes the extractor function for each item in parallel,
    with configurable concurrency limits to avoid overwhelming
    the API or hitting rate limits.

    Args:
        items: List of item identifiers (e.g., video IDs, channel IDs).
        extractor: Async function that takes an item ID and returns data.
        max_concurrency: Maximum parallel extractions (default: 5).
        delay_between_batches: Delay in seconds between batches (default: 0.1).
        stop_on_error: If True, stop on first error. If False, continue.

    Returns:
        BatchResult with successful extractions and failures.

    Example:
        >>> async def fetch_video(video_id: str):
        ...     return await get_video_details(video_id)
        ...
        >>> result = await batch_extract(
        ...     items=["id1", "id2", "id3", "id4", "id5"],
        ...     extractor=fetch_video,
        ...     max_concurrency=3,
        ... )
        >>> print(f"Success: {result.total_successful}/{result.total_requested}")
        Success: 5/5
        >>> print(result.successful[0]["title"])
        "My Video Title"

    Note:
        Failed extractions are recorded but don't stop the batch
        (unless stop_on_error=True). Use result.failed to inspect errors.
    """
    if not items:
        return BatchResult()

    logger.info(
        f"Starting batch extraction: {len(items)} items, "
        f"max_concurrency={max_concurrency}"
    )

    result = BatchResult(total_requested=len(items))
    semaphore = asyncio.Semaphore(max_concurrency)

    async def extract_with_semaphore(item: str) -> tuple[str, Any | None, str | None]:
        """Extract single item with semaphore control."""
        async with semaphore:
            try:
                data = await extractor(item)
                return (item, data, None)
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Extraction failed for {item}: {error_msg}")
                return (item, None, error_msg)

    # Process all items concurrently (semaphore limits actual concurrency)
    tasks = [extract_with_semaphore(item) for item in items]

    if stop_on_error:
        # Stop on first error
        for coro in asyncio.as_completed(tasks):
            item, data, error = await coro
            if error:
                result.failed[item] = error
                result.total_failed += 1
                logger.error(f"Stopping batch due to error: {error}")
                break
            else:
                result.successful.append(data)
                result.total_successful += 1
    else:
        # Continue on errors
        results_list = await asyncio.gather(*tasks)
        for item, data, error in results_list:
            if error:
                result.failed[item] = error
                result.total_failed += 1
            else:
                result.successful.append(data)
                result.total_successful += 1

    logger.info(
        f"Batch extraction complete: {result.total_successful} succeeded, "
        f"{result.total_failed} failed ({result.success_rate:.1f}% success rate)"
    )

    return result


async def batch_extract_chunked[T](
    items: list[str],
    extractor: Callable[[str], Awaitable[T]],
    chunk_size: int = 10,
    max_concurrency: int = 5,
    delay_between_chunks: float = 1.0,
) -> BatchResult:
    """Extract data in chunks with delays between chunks.

    Similar to batch_extract but processes items in smaller chunks
    with configurable delays between chunks. Useful for very large
    extractions where you want to avoid sustained high request rates.

    Args:
        items: List of item identifiers.
        extractor: Async function that takes an item ID and returns data.
        chunk_size: Number of items per chunk (default: 10).
        max_concurrency: Max parallel extractions per chunk (default: 5).
        delay_between_chunks: Delay in seconds between chunks (default: 1.0).

    Returns:
        Combined BatchResult from all chunks.

    Example:
        >>> result = await batch_extract_chunked(
        ...     items=video_ids,  # 100 video IDs
        ...     extractor=get_video_details,
        ...     chunk_size=20,
        ...     delay_between_chunks=2.0,
        ... )
        >>> # Processes 20 at a time with 2 second breaks
    """
    if not items:
        return BatchResult()

    # Split into chunks
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    logger.info(
        f"Starting chunked batch extraction: {len(items)} items in {len(chunks)} chunks"
    )

    combined_result = BatchResult(total_requested=len(items))

    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} items)")

        chunk_result = await batch_extract(
            items=chunk,
            extractor=extractor,
            max_concurrency=max_concurrency,
        )

        # Merge results
        combined_result.successful.extend(chunk_result.successful)
        combined_result.failed.update(chunk_result.failed)
        combined_result.total_successful += chunk_result.total_successful
        combined_result.total_failed += chunk_result.total_failed

        # Delay between chunks (except after last chunk)
        if i < len(chunks) - 1 and delay_between_chunks > 0:
            logger.debug(f"Waiting {delay_between_chunks}s before next chunk...")
            await asyncio.sleep(delay_between_chunks)

    logger.info(
        f"Chunked extraction complete: {combined_result.total_successful} succeeded, "
        f"{combined_result.total_failed} failed"
    )

    return combined_result


def estimate_quota_cost(
    operation: str,
    count: int = 1,
) -> int:
    """Estimate quota cost for an operation.

    Args:
        operation: Operation type (key from QUOTA_COSTS).
        count: Number of items (for batch operations).

    Returns:
        Estimated quota units.

    Example:
        >>> estimate_quota_cost("search_videos")
        100
        >>> estimate_quota_cost("video_details", count=50)
        1  # Batch of 50 is still 1 unit
    """
    base_cost = QUOTA_COSTS.get(operation, 1)

    # Some operations are batched (1 API call for multiple items)
    if operation in ("video_details", "channel_info"):
        # YouTube allows 50 IDs per request
        return base_cost * ((count + 49) // 50)  # Ceiling division

    # Most operations cost the same regardless of count
    return base_cost * count


def can_complete_extraction(
    quota_tracker: QuotaTracker,
    planned_operations: list[tuple[str, int]],
) -> tuple[bool, int]:
    """Check if planned operations fit within remaining quota.

    Args:
        quota_tracker: Current quota state.
        planned_operations: List of (operation, count) tuples.

    Returns:
        Tuple of (can_complete, total_cost).

    Example:
        >>> tracker = QuotaTracker(used=9500)
        >>> can_complete, cost = can_complete_extraction(
        ...     tracker,
        ...     [("search_videos", 2), ("video_details", 100)]
        ... )
        >>> print(f"Can complete: {can_complete}, Cost: {cost}")
        Can complete: False, Cost: 202
    """
    total_cost = sum(
        estimate_quota_cost(operation, count) for operation, count in planned_operations
    )

    can_complete = quota_tracker.remaining >= total_cost
    return (can_complete, total_cost)


__all__ = [
    "QUOTA_COSTS",
    "BatchResult",
    "QuotaTracker",
    "batch_extract",
    "batch_extract_chunked",
    "can_complete_extraction",
    "estimate_quota_cost",
]
