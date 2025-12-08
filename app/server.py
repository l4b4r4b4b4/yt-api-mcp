#!/usr/bin/env python3
"""FastMCP Template Server with RefCache Integration.

This template demonstrates how to use mcp-refcache with FastMCP to build
an MCP server that handles large results efficiently.

Features demonstrated:
- Reference-based caching for large results
- Preview generation (sample, truncate, paginate strategies)
- Pagination for accessing large datasets
- Access control (user vs agent permissions)
- Private computation (EXECUTE without READ)
- Both sync and async tool implementations
- Optional Langfuse tracing integration

Usage:
    # Install dependencies
    uv sync

    # Run with stdio (for Claude Desktop / Zed)
    uv run fastmcp-template

    # Run with SSE (for web clients / debugging)
    uv run fastmcp-template --transport sse --port 8000

Claude Desktop Configuration:
    Add to your claude_desktop_config.json:
    {
        "mcpServers": {
            "fastmcp-template": {
                "command": "uv",
                "args": ["run", "fastmcp-template"]
            }
        }
    }
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Check for FastMCP availability
# =============================================================================

try:
    from fastmcp import FastMCP
except ImportError:
    print(
        "Error: FastMCP is not installed. Install with:\n  uv sync\n",
        file=sys.stderr,
    )
    sys.exit(1)

# =============================================================================
# Import mcp-refcache components
# =============================================================================

from mcp_refcache import (
    AccessPolicy,
    CacheResponse,
    DefaultActor,
    Permission,
    PreviewConfig,
    PreviewStrategy,
    RefCache,
)
from mcp_refcache.fastmcp import (
    cache_guide_prompt,
    cache_instructions,
    register_admin_tools,
    with_cache_docs,
)

# =============================================================================
# Initialize FastMCP Server
# =============================================================================

mcp = FastMCP(
    name="FastMCP Template",
    instructions=f"""A template MCP server with reference-based caching.

Available tools:
- hello: Simple greeting tool (no caching)
- generate_items: Generate a list of items (cached in public namespace)
- store_secret: Store a secret value for private computation
- compute_with_secret: Use a secret in computation without revealing it
- get_cached_result: Retrieve or paginate through cached results

{cache_instructions()}
""",
)

# =============================================================================
# Initialize RefCache
# =============================================================================

# Create a RefCache instance with sensible defaults
# Uses token-based sizing (default) for accurate LLM context management
cache = RefCache(
    name="fastmcp-template",
    default_ttl=3600,  # 1 hour TTL
    preview_config=PreviewConfig(
        max_size=64,  # Max 64 tokens in previews
        default_strategy=PreviewStrategy.SAMPLE,  # Sample large collections
    ),
)

# =============================================================================
# Pydantic Models for Tool Inputs
# =============================================================================


class ItemGenerationInput(BaseModel):
    """Input model for item generation."""

    count: int = Field(
        default=10,
        ge=1,
        le=10000,
        description="Number of items to generate",
    )
    prefix: str = Field(
        default="item",
        description="Prefix for item names",
    )


class SecretInput(BaseModel):
    """Input model for storing secret values."""

    name: str = Field(
        description="Name for the secret (used as key)",
        min_length=1,
        max_length=100,
    )
    value: float = Field(
        description="The secret numeric value",
    )


class SecretComputeInput(BaseModel):
    """Input model for computing with secrets."""

    secret_ref: str = Field(
        description="Reference ID of the secret value",
    )
    multiplier: float = Field(
        default=1.0,
        description="Multiplier to apply to the secret value",
    )


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
        description="Maximum preview size (tokens/chars). Overrides defaults.",
    )


# =============================================================================
# Tool Implementations
# =============================================================================


@mcp.tool
def hello(name: str = "World") -> dict[str, Any]:
    """Say hello to someone.

    A simple example tool that doesn't use caching.

    Args:
        name: The name to greet.

    Returns:
        A greeting message.
    """
    return {
        "message": f"Hello, {name}!",
        "server": "fastmcp-template",
    }


@mcp.tool
@cache.cached(namespace="public")
async def generate_items(
    count: int = 10,
    prefix: str = "item",
) -> list[dict[str, Any]]:
    """Generate a list of items.

    Demonstrates caching of large results in the PUBLIC namespace.
    For large counts, returns a reference with a preview instead of the full data.

    Use get_cached_result to paginate through large results.

    Args:
        count: Number of items to generate.
        prefix: Prefix for item names.

    Returns:
        List of items with id, name, and value.

    **Caching:** Large results are cached in the public namespace.

    **Pagination:** Use `page` and `page_size` to navigate results.
    """
    validated = ItemGenerationInput(count=count, prefix=prefix)

    items = [
        {
            "id": i,
            "name": f"{validated.prefix}_{i}",
            "value": i * 10,
        }
        for i in range(validated.count)
    ]

    # Return raw data - decorator handles caching and structured response
    return items


@mcp.tool
def store_secret(name: str, value: float) -> dict[str, Any]:
    """Store a secret value that agents cannot read, only use in computations.

    This demonstrates the EXECUTE permission - agents can use the value
    in compute_with_secret without ever seeing what it is.

    Args:
        name: Name for the secret.
        value: The secret numeric value.

    Returns:
        Reference ID and confirmation message.
    """
    validated = SecretInput(name=name, value=value)

    # Create a policy where agents can EXECUTE but not READ
    secret_policy = AccessPolicy(
        user_permissions=Permission.FULL,  # Users can see everything
        agent_permissions=Permission.EXECUTE,  # Agents can only use in computation
    )

    ref = cache.set(
        key=f"secret_{validated.name}",
        value=validated.value,
        namespace="user:secrets",
        policy=secret_policy,
        tool_name="store_secret",
    )

    return {
        "ref_id": ref.ref_id,
        "name": validated.name,
        "message": f"Secret '{validated.name}' stored. Use compute_with_secret to use it.",
        "permissions": {
            "user": "FULL (can read, write, execute)",
            "agent": "EXECUTE only (can use in computation, cannot read)",
        },
    }


@mcp.tool
@with_cache_docs(accepts_references=True, private_computation=True)
def compute_with_secret(secret_ref: str, multiplier: float = 1.0) -> dict[str, Any]:
    """Compute using a secret value without revealing it.

    The secret is multiplied by the provided multiplier.
    This demonstrates private computation - the agent orchestrates
    the computation but never sees the actual secret value.

    Args:
        secret_ref: Reference ID of the secret value.
        multiplier: Value to multiply the secret by.

    Returns:
        The computation result (without revealing the secret).

    **References:** This tool accepts `ref_id` from previous tool calls.

    **Private Compute:** Values are processed server-side without exposure.
    """
    validated = SecretComputeInput(secret_ref=secret_ref, multiplier=multiplier)

    # Create a system actor to resolve the secret (bypasses agent restrictions)
    system_actor = DefaultActor.system()

    try:
        # Resolve the secret value as system (has full access)
        secret_value = cache.resolve(validated.secret_ref, actor=system_actor)
    except KeyError as e:
        raise ValueError(f"Secret reference '{validated.secret_ref}' not found") from e

    result = secret_value * validated.multiplier

    return {
        "result": result,
        "multiplier": validated.multiplier,
        "secret_ref": validated.secret_ref,
        "message": "Computed using secret value (value not revealed)",
    }


@mcp.tool
@with_cache_docs(accepts_references=True, supports_pagination=True)
async def get_cached_result(
    ref_id: str,
    page: int | None = None,
    page_size: int | None = None,
    max_size: int | None = None,
) -> dict[str, Any]:
    """Retrieve a cached result, optionally with pagination.

    Use this to:
    - Get a preview of a cached value
    - Paginate through large lists
    - Access the full value of a cached result

    Args:
        ref_id: Reference ID to look up.
        page: Page number (1-indexed).
        page_size: Items per page.
        max_size: Maximum preview size (overrides defaults).

    Returns:
        The cached value or a preview with pagination info.

    **Caching:** Large results are returned as references with previews.

    **Pagination:** Use `page` and `page_size` to navigate results.

    **References:** This tool accepts `ref_id` from previous tool calls.
    """
    validated = CacheQueryInput(
        ref_id=ref_id, page=page, page_size=page_size, max_size=max_size
    )

    try:
        response: CacheResponse = cache.get(
            validated.ref_id,
            page=validated.page,
            page_size=validated.page_size,
            actor="agent",
        )

        result: dict[str, Any] = {
            "ref_id": validated.ref_id,
            "preview": response.preview,
            "preview_strategy": response.preview_strategy.value,
            "total_items": response.total_items,
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


# =============================================================================
# Health Check
# =============================================================================


@mcp.tool
def health_check() -> dict[str, Any]:
    """Check server health status.

    Returns:
        Health status information.
    """
    return {
        "status": "healthy",
        "server": "fastmcp-template",
        "cache": cache.name,
    }


# =============================================================================
# Admin Tools (Permission-Gated)
# =============================================================================


async def is_admin(ctx: Any) -> bool:
    """Check if the current context has admin privileges.

    Override this in your own server with proper auth logic.
    """
    # Demo: No admin access by default
    return False


# Register admin tools with the cache
_admin_tools = register_admin_tools(
    mcp,
    cache,
    admin_check=is_admin,
    prefix="admin_",
    include_dangerous=False,
)


# =============================================================================
# Prompts for Guidance
# =============================================================================


@mcp.prompt
def template_guide() -> str:
    """Guide for using this MCP server template."""
    return f"""# FastMCP Template Guide

## Quick Start

1. **Simple Tool**
   Use `hello` for a basic greeting:
   - `hello("World")` → "Hello, World!"

2. **Generate Items (Caching Demo)**
   Use `generate_items` to create a list:
   - `generate_items(count=100, prefix="widget")`
   - Returns ref_id + preview for large results
   - Cached in the PUBLIC namespace (shared)

3. **Paginate Results**
   Use `get_cached_result` to navigate large results:
   - `get_cached_result(ref_id, page=2, page_size=20)`

## Private Computation

Store values that agents can use but not see:

```
# Store a secret
store_secret("api_key_hash", 12345.0)
# Returns ref_id for the secret

# Use in computation (agent never sees the value)
compute_with_secret(ref_id, multiplier=2.0)
# Returns the result
```

---

{cache_guide_prompt()}
"""


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the MCP server."""
    parser = argparse.ArgumentParser(
        description="FastMCP Template Server with RefCache",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode (default: stdio for Claude Desktop)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for SSE transport (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport="sse",
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
