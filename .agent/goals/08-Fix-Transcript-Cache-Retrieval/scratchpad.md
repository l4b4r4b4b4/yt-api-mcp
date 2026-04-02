# Goal 08: Fix Transcript Cache Retrieval Bug

> **Status**: 🟢 Complete
> **Priority**: P1 (High) — blocks reliable transcript consumption via primary API flow
> **Created**: 2026-04-01
> **Updated**: 2026-04-01

## Overview

`get_full_transcript` combined with `get_cached_result` returns preview-only payloads with no reliable way to retrieve the full transcript from cache references. The root cause is a code bug in `app/tools/cache.py` where `max_size` is accepted but never forwarded to `cache.get()`, plus a missing `resolve()` exposure for full-value retrieval.

Additionally, `mcp-refcache` is outdated (`0.1.0` → `0.2.0` available).

## Success Criteria

- [x] `get_cached_result(ref_id, max_size=100000)` actually returns larger/full payloads (not ignored)
- [x] New tool or parameter available to retrieve full cached values without preview truncation
- [x] `mcp-refcache` upgraded from `0.1.0` to `0.2.0`
- [x] Existing tests pass after upgrade
- [x] New tests cover `max_size` forwarding and full-value retrieval
- [x] Docs/docstrings updated to clarify preview vs full retrieval behavior

## Context & Background

### Bug Report

A user reported that calling `get_full_transcript(video_id="35cESnxXH6o")` returns a `ref_id` and preview (expected for large transcripts), but subsequent calls to `get_cached_result(ref_id, page=1, page_size=100, max_size=100000)` still return preview/sample metadata only — the `max_size` parameter has no effect.

### Root Cause Analysis (Completed)

**Finding 1: `max_size` silently discarded**

In `app/tools/cache.py` lines 90-97, `get_cached_result` calls `cache.get()` but never passes `max_size`:

```python
response = cache.get(
    validated.ref_id,
    page=validated.page,
    page_size=validated.page_size,
    actor="agent",
    # BUG: validated.max_size is NEVER passed here!
)
```

The upstream `RefCache.get()` method **does** accept `max_size` and uses it to override the server default in `_create_preview()`. The parameter just isn't being forwarded.

**Finding 2: No full-value retrieval exposed**

`mcp-refcache` provides `cache.resolve(ref_id)` which returns the **complete cached value** without any preview truncation. But yt-mcp never exposes this as a tool. This is the real missing piece — even with `max_size` fixed, agents still can't say "give me everything, no preview."

**Finding 3: Outdated dependency**

- **Current**: `mcp-refcache==0.1.0`
- **Available**: `mcp-refcache==0.2.0`
- **What's new in 0.2.0**: Async timeout & polling (`async_timeout` parameter on `@cache.cached()`), `TaskBackend` protocol, `MemoryTaskBackend`, `AsyncTaskResponse` model, ETA calculation, response format levels

**Finding 4: Preview config forces SAMPLE strategy at 2048 tokens**

In `app/server.py` lines 146-149:

```python
preview_config=PreviewConfig(
    max_size=2048,
    default_strategy=PreviewStrategy.SAMPLE,
)
```

This means ALL large results get sampled at 2048 tokens max. With `max_size` not forwarded, there's no caller-level override possible.

### Secondary Issue: YouTube Rate Limiting

The user also reported `RequestBlocked`/`IPBlocked` errors after repeated `get_transcript_chunk` calls (their workaround). This is a YouTube provider-level issue, not a cache bug, but we should document retry/backoff guidance.

## Constraints & Requirements

- **Hard Requirements**:
  - Fix must not break existing `@cache.cached()` decorator behavior
  - All 406+ existing tests must continue passing
  - `get_cached_result` must remain backward-compatible (new params are optional)
- **Soft Requirements**:
  - Consider adding a `full=true` parameter or separate `resolve_cached_result` tool
  - Document when to use preview vs full retrieval in tool docstrings
  - Add retry/backoff docs for transcript fetching
- **Out of Scope**:
  - YouTube rate-limiting mitigation (provider-level issue)
  - Async timeout features from mcp-refcache 0.2.0 (separate goal if needed)

## Approach

Three tasks, executed sequentially:

1. **Task-01**: Upgrade `mcp-refcache` to `0.2.0` and fix any breaking changes
2. **Task-02**: Fix `max_size` forwarding bug + add full-value retrieval capability
3. **Task-03**: Add tests, update docs, verify end-to-end

## Tasks

| Task ID | Description | Status | Depends On |
|---------|-------------|--------|------------|
| Task-01 | Upgrade mcp-refcache 0.1.0 → 0.2.0 | 🟢 Complete | - |
| Task-02 | Fix max_size bug + expose resolve/full retrieval | 🟢 Complete | Task-01 |
| Task-03 | Tests, docs, and end-to-end verification | 🟢 Complete | Task-02 |

### Task-01: Upgrade mcp-refcache ✅

**Completed:** `uv add "mcp-refcache>=0.2.0"` — upgraded from 0.1.0 to 0.2.0.

**Files modified:**
- `pyproject.toml` — bumped `mcp-refcache` to `>=0.2.0`
- `uv.lock` — regenerated automatically by `uv add`

**Verification:** All 406 existing tests passed immediately after upgrade with zero changes needed. The 0.2.0 API is backward-compatible for our usage.

### Task-02: Fix max_size + full retrieval ✅

**Decision:** Option A (add `full: bool = False` to existing `get_cached_result`) — simpler, one tool.

**Changes to `app/tools/cache.py`:**

1. **Fixed `max_size` forwarding** — added `max_size=validated.max_size` to `cache.get()` call
2. **Added `full: bool = False` parameter** — when True, calls `cache.resolve()` instead of `cache.get()`, returning the complete cached value without any preview truncation
3. **Added `AsyncTaskResponse` handling** — `isinstance` check for mcp-refcache 0.2.0's new return type from `cache.get()`, returns async task status/progress/ETA info
4. **Added `retrieval_mode` field** to all responses — `"preview"`, `"full"`, or async status, so callers know which mode was used
5. **Updated `CacheQueryInput` model** — added `full` field with description
6. **Updated docstrings** — documented `full=True` mode, `max_size` override behavior, and all three retrieval modes

### Task-03: Tests, docs, and end-to-end verification ✅

**7 new tests added to `tests/test_server.py::TestGetCachedResult`:**

| Test | What it verifies |
|------|-----------------|
| `test_get_cached_result_returns_preview_for_small_value` | Small values return via preview mode with `retrieval_mode` field |
| `test_get_cached_result_max_size_forwarded` | `max_size` is actually forwarded — larger `max_size` yields larger/equal preview |
| `test_get_cached_result_full_retrieval` | `full=True` returns complete dict value via `cache.resolve()` |
| `test_get_cached_result_full_retrieval_list` | `full=True` works for list values (100 items returned intact) |
| `test_get_cached_result_full_not_found` | `full=True` on invalid ref returns error dict (not crash) |
| `test_get_cached_result_pagination_with_max_size` | Pagination + `max_size` work together |
| `test_get_cached_result_full_vs_preview_content_differs` | Full retrieval returns more data than preview for large values |

**Agent-facing docs updated:**

| Location | What was added |
|----------|---------------|
| `app/server.py` instructions (lines 32-37) | `get_cached_result` entry expanded: `full=True`, `max_size=N`, pagination all listed with descriptions |
| `app/server.py` Transcript Notes (lines 55-61) | Step-by-step: "To get the FULL transcript: `get_cached_result(ref_id, full=True)`" |
| `app/prompts/__init__.py` TEMPLATE_GUIDE | New "Retrieve Cached Results" section with 3 modes (full, larger preview, paginate) and new "Working with Transcripts" section with complete workflow |

**Live validation results (yt-mcp-dev server):**

| Call | `preview_size` | Result |
|------|---------------|--------|
| Default (no params) | **22** tokens | ✅ Tiny sample — backward compatible |
| `max_size=100000` | **4522** tokens | ✅ Full content via preview — was 22 before fix |
| `full=True` | N/A (complete) | ✅ Complete value via `resolve()`, `is_complete: true` |

**Final results:**
- **413 tests pass** (was 406, +7 new)
- **0 failures**
- **Ruff lint clean** — `ruff check` and `ruff format` both pass
- **Live validated** on yt-mcp-dev with `get_video_comments` → `get_cached_result` (YouTube IP-blocked for transcripts, but cache fix is data-type agnostic)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| mcp-refcache 0.2.0 has breaking API changes beyond documented | High | Low | Review changelog + source before upgrading |
| `cache.get()` return type union causes runtime issues | Medium | Medium | Add `isinstance` check for `AsyncTaskResponse` vs `CacheResponse` |
| Full value retrieval returns extremely large payloads | Medium | Medium | Add configurable size limit or warning for `full=True` mode |
| Existing tests rely on specific preview behavior | Low | Low | Run tests after each change, fix incrementally |

## Dependencies

- **Upstream**: `mcp-refcache` PyPI package (0.2.0)
- **Downstream**: All tools using `@cache.cached()` decorator, all callers of `get_cached_result`

## Notes & Decisions

### Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-01 | Created goal, completed root cause analysis | Bug report from user, traced to `max_size` not forwarded + missing `resolve()` |
| 2026-04-01 | Three-task approach (upgrade → fix → test) | Upgrade first to avoid fixing against stale API |
| 2026-04-01 | Option A chosen: `full` param on existing tool | Simpler — one tool, backward-compatible, same conceptual operation |
| 2026-04-01 | All three tasks implemented | 413 tests pass, lint clean |
| 2026-04-01 | Live validated on yt-mcp-dev | `max_size` fix: preview_size 22→4522; `full=True`: complete value returned |
| 2026-04-01 | Agent-facing docs updated | Server instructions, transcript notes, and TEMPLATE_GUIDE prompt all document `full=True` and `max_size` |

### Open Questions

- [x] Option A (`full` param on existing tool) vs Option B (new `resolve_cached_result` tool)? → **Option A chosen**
- [x] Are agents properly informed about `full=True`? → **Yes** — server instructions, transcript notes, and TEMPLATE_GUIDE prompt all updated
- [ ] Should `full=True` have a safety limit (e.g., max 500KB response)? → Not implemented yet, deferred
- [ ] Should we adopt any 0.2.0 async timeout features now or defer to a separate goal? → Deferred; async handling code added defensively but no tools use `async_timeout` yet

## References

- Bug report: Transcript Retrieval APIs Return Preview-Only Payloads
- mcp-refcache changelog: 0.1.0 → 0.2.0 (async timeout, polling)
- mcp-refcache source: `RefCache.get()` supports `max_size`, `RefCache.resolve()` returns full value
- Affected files:
  - `app/tools/cache.py` — `create_get_cached_result` (core fix)
  - `app/server.py` — instructions + PreviewConfig
  - `app/prompts/__init__.py` — TEMPLATE_GUIDE prompt
  - `tests/test_server.py` — 7 new tests in `TestGetCachedResult`
  - `pyproject.toml` / `uv.lock` — mcp-refcache 0.1.0 → 0.2.0
