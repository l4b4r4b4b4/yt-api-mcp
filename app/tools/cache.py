"""Cache query and retrieval tools.

This module provides tools for querying and retrieving cached results,
with support for pagination, preview customization, and full-value retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.tracing import traced_tool

if TYPE_CHECKING:
    from mcp_refcache import RefCache


class CacheQueryInput(BaseModel):
    """Input model for cache queries."""

    ref_id: str = Field(
        description="Reference ID to look up",
    )
    page: int | None = Field(
        default=None,
        ge=1,
        description="Page number for pagination (1-indexed)",
    )
    page_size: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of items per page",
    )
    max_size: int | None = Field(
        default=None,
        ge=1,
        description="Maximum preview size (tokens/chars). Overrides server defaults. "
        "Use larger values (e.g. 100000) to retrieve more content.",
    )
    full: bool = Field(
        default=False,
        description="If True, return the complete cached value without any preview "
        "truncation. Bypasses preview generation entirely. Use with caution "
        "for large values.",
    )


def create_get_cached_result(cache: RefCache) -> Any:
    """Create a get_cached_result tool function bound to the given cache.

    Args:
        cache: The RefCache instance to use for cache lookups.

    Returns:
        The get_cached_result tool function.
    """

    @traced_tool("get_cached_result")
    async def get_cached_result(
        ref_id: str,
        page: int | None = None,
        page_size: int | None = None,
        max_size: int | None = None,
        full: bool = False,
    ) -> dict[str, Any]:
        """Retrieve a cached result, optionally with pagination.

        Use this to:
        - Get a preview of a cached value
        - Paginate through large lists
        - Access the full value of a cached result

        All cache operations are traced to Langfuse with hit/miss status.

        Args:
            ref_id: Reference ID to look up.
            page: Page number (1-indexed).
            page_size: Items per page.
            max_size: Maximum preview size (overrides defaults).
            full: If True, return the complete cached value without preview
                truncation. Bypasses all preview generation. Use when you
                need the entire value (e.g. full transcripts).

        Returns:
            The cached value or a preview with pagination info.

        **Caching:** Large results are returned as references with previews.

        **Pagination:** Use `page` and `page_size` to navigate results.

        **Full retrieval:** Use `full=True` to get the complete value.

        **References:** This tool accepts `ref_id` from previous tool calls.
        """
        validated = CacheQueryInput(
            ref_id=ref_id,
            page=page,
            page_size=page_size,
            max_size=max_size,
            full=full,
        )

        try:
            # Full-value retrieval: bypass preview generation entirely
            if validated.full:
                value = cache.resolve(validated.ref_id, actor="agent")
                return {
                    "ref_id": validated.ref_id,
                    "value": value,
                    "is_complete": True,
                    "retrieval_mode": "full",
                }

            # Preview retrieval with max_size forwarded correctly
            response = cache.get(
                validated.ref_id,
                page=validated.page,
                page_size=validated.page_size,
                max_size=validated.max_size,
                actor="agent",
            )

            # Handle AsyncTaskResponse from mcp-refcache 0.2.x
            # (returned when ref_id corresponds to an in-flight async task)
            from mcp_refcache.models import AsyncTaskResponse

            if isinstance(response, AsyncTaskResponse):
                result: dict[str, Any] = {
                    "ref_id": validated.ref_id,
                    "is_async": True,
                    "status": response.status.value
                    if hasattr(response.status, "value")
                    else str(response.status),
                    "message": getattr(response, "message", None)
                    or "Task is still processing. Poll again shortly.",
                }
                if hasattr(response, "progress") and response.progress is not None:
                    result["progress"] = {
                        "current": response.progress.current,
                        "total": response.progress.total,
                        "percentage": response.progress.percentage,
                    }
                if (
                    hasattr(response, "eta_seconds")
                    and response.eta_seconds is not None
                ):
                    result["eta_seconds"] = response.eta_seconds
                return result

            # Standard CacheResponse handling
            result = {
                "ref_id": validated.ref_id,
                "preview": response.preview,
                "preview_strategy": response.preview_strategy.value,
                "total_items": response.total_items,
                "retrieval_mode": "preview",
            }

            if response.page is not None:
                result["page"] = response.page
                result["total_pages"] = response.total_pages

            if response.original_size:
                result["original_size"] = response.original_size
                result["preview_size"] = response.preview_size

            return result

        except (PermissionError, KeyError):
            return {
                "error": "Invalid or inaccessible reference",
                "message": "Reference not found, expired, or access denied",
                "ref_id": validated.ref_id,
            }

    return get_cached_result


__all__ = [
    "CacheQueryInput",
    "create_get_cached_result",
]
