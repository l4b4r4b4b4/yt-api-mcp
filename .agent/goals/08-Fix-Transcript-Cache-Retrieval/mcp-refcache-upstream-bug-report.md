# Bug Report: `@cache.cached()` Injected Docstrings Missing `full=True` Retrieval Documentation

## Summary

The `@cache.cached()` decorator auto-injects caching behavior docs into every decorated tool's docstring, but the injected text has no mention of `full=True` on `get_cached_result` — the primary mechanism for agents to retrieve complete cached values without preview truncation.

This means every tool that uses `@cache.cached()` tells agents to "use get_cached_result to paginate" but never tells them they can bypass preview truncation entirely with `full=True`.

Similarly, `cache_instructions()` and `cache_guide_prompt()` in the FastMCP integration module are missing `full=True` documentation.

---

## Environment

- **mcp-refcache version:** 0.2.0
- **Discovered in:** yt-api-mcp (examples/yt-mcp)
- **Affects:** All downstream MCP servers using `@cache.cached()` decorator

---

## Affected Code Locations

### 1. `src/mcp_refcache/cache.py` — `RefCache.cached()` decorator

The `cache_doc` string injected into every decorated tool's `__doc__`:

```python
# Around line 440-449 in RefCache.cached()
cache_doc = f"""

**Caching Behavior:**
- Any input parameter can accept a ref_id from a previous tool call
- Large results return ref_id + preview; use get_cached_result to paginate
- All responses include ref_id for future reference

**Preview Size:** {max_size_doc}. Override per-call with `get_cached_result(ref_id, max_size=...)`."""
func.__doc__ = original_doc + cache_doc
```

**Problem:** No mention of `full=True` for complete value retrieval. Agents only learn about pagination and `max_size`, never about the escape hatch from preview truncation.

### 2. `src/mcp_refcache/fastmcp/__init__.py` — `cache_instructions()`

The server-level instructions text:

```
**Working with References:**
- Paginate: `get_cached_result(ref_id, page=2, page_size=20)`
- Pass to tools: use `ref_id` as input parameter
- Some refs are execute-only (use in computation, can't read)
```

**Problem:** Lists pagination and ref passing, but not `full=True`.

### 3. `src/mcp_refcache/fastmcp/__init__.py` — `cache_guide_prompt()`

The detailed prompt guide mentions `"resolve_full"` in `available_actions` of a sample JSON response, but never explains how to actually do it. The Quick Reference table at the bottom is also missing the `full=True` action.

Current Quick Reference:

```
| Action | How |
|--------|-----|
| View preview | Included in response |
| Get page N | `get_cached_result(ref_id, page=N)` |
| Pass to tool | Use `ref_id` as parameter |
| Private compute | Tools resolve references server-side |
```

**Problem:** No row for "Get full value".

---

## What Agents Currently See (on every cached tool)

When an agent calls any `@cache.cached()` tool and checks its description, the injected tail reads:

```
**Caching Behavior:**
- Any input parameter can accept a ref_id from a previous tool call
- Large results return ref_id + preview; use get_cached_result to paginate
- All responses include ref_id for future reference

**Preview Size:** server default. Override per-call with `get_cached_result(ref_id, max_size=...)`.
```

The agent has **no way to discover** that `get_cached_result(ref_id, full=True)` exists unless the downstream server author manually documents it in their server instructions (which is what yt-mcp had to do as a workaround).

---

## Expected Behavior

The auto-injected docstring should inform agents about all three retrieval modes:

1. **Preview** (default) — sampled/truncated for context efficiency
2. **Larger preview** (`max_size=N`) — override preview size limit
3. **Full retrieval** (`full=True`) — complete cached value, no truncation

---

## Proposed Fix

### 1. Update `cache_doc` in `RefCache.cached()`:

```python
cache_doc = f"""

**Caching Behavior:**
- Any input parameter can accept a ref_id from a previous tool call
- Large results return ref_id + preview; use get_cached_result to paginate
- All responses include ref_id for future reference

**Full retrieval:** Use `full=True` to get the complete value.

**Preview Size:** {max_size_doc}. Override per-call with `get_cached_result(ref_id, max_size=...)`."""
```

### 2. Update `cache_instructions()`:

```
**Working with References:**
- Full value: `get_cached_result(ref_id, full=True)`
- Paginate: `get_cached_result(ref_id, page=2, page_size=20)`
- Larger preview: `get_cached_result(ref_id, max_size=100000)`
- Pass to tools: use `ref_id` as input parameter
- Some refs are execute-only (use in computation, can't read)
```

### 3. Update Quick Reference in `cache_guide_prompt()`:

```
| Action | How |
|--------|-----|
| View preview | Included in response |
| Get full value | `get_cached_result(ref_id, full=True)` |
| Get page N | `get_cached_result(ref_id, page=N)` |
| Larger preview | `get_cached_result(ref_id, max_size=100000)` |
| Pass to tool | Use `ref_id` as parameter |
| Private compute | Tools resolve references server-side |
```

---

## Impact

- **All MCP servers** using `@cache.cached()` are affected — agents interacting with any mcp-refcache-powered server won't discover `full=True` unless the server author documents it separately.
- **Transcript and large-payload use cases** are most impacted — agents get stuck in a loop of pagination/chunking when `full=True` would give them everything in one call.
- The yt-mcp project hit this as a user-reported bug: agents couldn't retrieve full transcripts from cache refs despite the capability existing.

---

## Context: How This Was Discovered

A user reported that `get_full_transcript` + `get_cached_result(ref_id, max_size=100000)` still returned preview-only payloads. Root cause analysis revealed two bugs in yt-mcp:

1. **`max_size` was accepted but never forwarded** to `cache.get()` — fixed in yt-mcp
2. **No full-value retrieval exposed** — added `full=True` param calling `cache.resolve()` in yt-mcp

After fixing both bugs and live-validating on the dev server, we noticed that the `@cache.cached()` decorator's auto-injected docs don't mention `full=True`, so agents using any other mcp-refcache server would never discover the feature.

See: `examples/yt-mcp/.agent/goals/08-Fix-Transcript-Cache-Retrieval/scratchpad.md` for full investigation notes.

---

## Related Files

- `src/mcp_refcache/cache.py` — `RefCache.cached()` method, `cache_doc` string (~line 440)
- `src/mcp_refcache/fastmcp/__init__.py` — `cache_instructions()` and `cache_guide_prompt()`
- `examples/yt-mcp/app/tools/cache.py` — downstream `get_cached_result` implementation with `full=True`
- `examples/yt-mcp/app/server.py` — workaround: manually documented `full=True` in server instructions
