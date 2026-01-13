# Task-04: Transcript-Aware Text Chunker

**Status:** 🟠 Implemented
**Priority:** High
**Created:** 2025-01-12
**Updated:** 2025-01-13
**Parent:** [Goal 02: Semantic Transcript Search](../scratchpad.md)

## Completion Summary

**What was done:**
- Created `tokenizers.py` with `TokenizerProtocol`, `TiktokenTokenizer`, and `HuggingFaceTokenizer`
- Auto-detection factory supports tiktoken encodings, OpenAI models, and HuggingFace models
- Refactored `chunker.py` to use token-based chunking (not character-based)
- Added chapter-aware chunking with `chapters` parameter
- Updated config defaults: `chunk_size=256`, `chunk_overlap=50` (tokens)
- Added `tokenizer_model` config field with auto-detection
- Added optional `[huggingface]` dependency group in pyproject.toml
- Created 68 new tests in `test_semantic_tokenizers.py` and `test_semantic_chunker.py`
- All 268 tests passing, linting clean

**Awaiting:** User validation in real usage

---

## Objective

Implement a transcript-aware text chunker that:
1. Uses **token-based** chunking (not character-based) for accurate embedding model limits
2. Supports **chapter-aware** chunking when YouTube chapter markers are available
3. Preserves timestamp boundaries and generates rich metadata for each chunk

---

## Implementation Plan

### Phase 1: Token-Based Chunking (Rework)

1. [x] Add tiktoken dependency
2. [x] Create tokenizer helper with cl100k_base encoding
3. [x] Refactor chunking logic to count tokens instead of characters
4. [x] Update config defaults (chunk_size=256 tokens, chunk_overlap=50 tokens)
5. [x] Update config descriptions to clarify token-based sizing

### Phase 2: Chapter Awareness

6. [x] Add optional `chapters` parameter to chunk_transcript()
7. [x] Implement chapter grouping logic
8. [x] Split large chapters that exceed chunk_size
9. [x] Add chapter_title and chapter_index to metadata
10. [x] Fall back to token-based when no chapters provided

### Phase 3: Testing

11. [x] Add unit tests for token counting
12. [x] Add unit tests for basic token-based chunking
13. [x] Add unit tests for chapter-aware chunking
14. [x] Add edge case tests (empty, single entry, large entries)
15. [x] Verify all 200+ tests still pass (268 passing)

---

## Design Decisions

### Tokenizer Choice: tiktoken with cl100k_base

**Decision:** Use `tiktoken` with `cl100k_base` encoding.

**Rationale:**
- cl100k_base is the modern encoding used by GPT-3.5/4 and most recent models
- tiktoken is fast (Rust-based) and well-maintained
- Nomic doesn't publish their exact tokenizer, but cl100k_base provides a reasonable approximation
- Already a transitive dependency via langchain-text-splitters

### Chunk Size: 256 Tokens

**Decision:** Default chunk_size = 256 tokens (was 800 characters).

**Rationale:**
- For 512-dim Matryoshka embeddings, 256-512 tokens is typical for semantic search
- Too small = loss of context for understanding
- Too large = diluted semantic signal, harder to pinpoint relevant passages
- 256 tokens ≈ 1000 characters ≈ 150-200 words
- NomicEmbeddings has 8192 token context, so 256 gives plenty of headroom

### Chunk Overlap: 50 Tokens

**Decision:** Default chunk_overlap = 50 tokens (was 100 characters).

**Rationale:**
- ~20% of chunk_size provides good continuity
- Helps maintain context across chunk boundaries
- Not too large to cause excessive duplication

### Chapter Handling Strategy

**Decision:** Chapters are semantic boundaries; never merge across chapters.

**Behavior:**
- Each chapter starts a new chunk (no overlap from previous chapter)
- If chapter content > chunk_size, split into multiple chunks WITH overlap
- If chapter content < chunk_size, keep as single chunk (don't pad or merge)
- Include `chapter_title` and `chapter_index` in metadata

**Rationale:**
- YouTube chapters represent intentional content boundaries set by creators
- Merging across chapters would mix semantically distinct content
- Small chapters are fine; they represent focused, discrete topics

---

## API Changes

### Config Changes (config.py)

```python
# Before
chunk_size: int = Field(
    default=800,
    ge=100,
    le=4000,
    description="Target chunk size in characters.",
)
chunk_overlap: int = Field(
    default=100,
    ge=0,
    le=500,
    description="Overlap between consecutive chunks in characters.",
)

# After
chunk_size: int = Field(
    default=256,
    ge=50,
    le=2000,
    description="Target chunk size in tokens (using cl100k_base encoding).",
)
chunk_overlap: int = Field(
    default=50,
    ge=0,
    le=500,
    description="Overlap between consecutive chunks in tokens.",
)
```

### Chunker API Changes (chunker.py)

```python
def chunk_transcript(
    self,
    transcript_entries: list[dict[str, Any]],
    video_metadata: dict[str, str],
    chapters: list[dict[str, Any]] | None = None,  # NEW PARAMETER
) -> list[Document]:
    """Chunk a transcript into Documents with timestamp metadata.

    Args:
        transcript_entries: List of transcript entries (text, start, duration)
        video_metadata: Video information for metadata
        chapters: Optional list of chapter markers, each with:
            - start_time: Start time in seconds (float)
            - title: Chapter title (str)
            When provided, chunks respect chapter boundaries.
    """
```

### Chapter Format

```python
chapters = [
    {"start_time": 0.0, "title": "Introduction"},
    {"start_time": 120.5, "title": "Getting Started"},
    {"start_time": 300.0, "title": "Advanced Topics"},
]
```

---

## Metadata Schema

Each Document's metadata will include:

```python
metadata = {
    # Video identifiers
    "video_id": "abc123",
    "channel_id": "UC123",

    # Display info
    "video_title": "Tutorial Video",
    "channel_title": "Tech Channel",
    "video_url": "https://youtube.com/watch?v=abc123",

    # Timestamps
    "start_time": 120.5,      # seconds
    "end_time": 180.3,        # seconds
    "timestamp_url": "https://youtube.com/watch?v=abc123&t=120",

    # Chunk info
    "chunk_index": 3,         # 0-indexed within video
    "token_count": 245,       # NEW: actual token count

    # Chapter info (when chapters provided)
    "chapter_title": "Getting Started",  # NEW
    "chapter_index": 1,                   # NEW: 0-indexed

    # Optional
    "published_at": "2024-01-01T00:00:00Z",
    "language": "en",
}
```

---

## Test Cases

### Token Counting Tests
1. `test_count_tokens_empty_string` - Returns 0 for empty
2. `test_count_tokens_simple_text` - Correct count for English text
3. `test_count_tokens_with_punctuation` - Handles punctuation
4. `test_count_tokens_unicode` - Handles non-ASCII characters

### Basic Chunking Tests
5. `test_chunk_single_entry` - One entry becomes one chunk
6. `test_chunk_multiple_entries_within_limit` - Grouped correctly
7. `test_chunk_entries_exceed_limit` - Splits at entry boundary
8. `test_chunk_overlap_applied` - Overlap includes trailing entries
9. `test_chunk_metadata_correct` - All metadata fields present

### Chapter-Aware Tests
10. `test_chunk_with_chapters` - Respects chapter boundaries
11. `test_large_chapter_splits` - Splits chapters exceeding chunk_size
12. `test_small_chapter_preserved` - Small chapters stay intact
13. `test_chapter_metadata_included` - chapter_title/index in metadata
14. `test_no_overlap_across_chapters` - New chapter starts fresh

### Edge Cases
15. `test_empty_transcript` - Returns empty list
16. `test_empty_chapters_list` - Treats as no chapters
17. `test_single_large_entry` - Entry larger than chunk_size (kept whole)
18. `test_chapter_with_no_entries` - Empty chapter skipped

---

## Files to Modify

1. **app/tools/youtube/semantic/config.py**
   - Update chunk_size default: 800 → 256
   - Update chunk_overlap default: 100 → 50
   - Update descriptions to say "tokens"
   - Adjust bounds (ge/le values)

2. **app/tools/youtube/semantic/chunker.py**
   - Add tiktoken import and encoder initialization
   - Add `_count_tokens()` helper method
   - Refactor `chunk_transcript()` for token-based logic
   - Add `chapters` parameter
   - Add chapter grouping logic
   - Update `_apply_overlap()` for tokens
   - Add `token_count` to metadata

3. **tests/test_semantic_chunker.py** (NEW)
   - All test cases listed above

---

## Acceptance Criteria

- [ ] Chunking uses token count, not character count
- [ ] Config defaults are token-based (256 chunk_size, 50 overlap)
- [ ] `chapters` parameter is optional and works when provided
- [ ] Chapter boundaries are respected (no cross-chapter merging)
- [ ] Large chapters are split with overlap
- [ ] Small chapters are preserved as single chunks
- [ ] `chapter_title` and `chapter_index` in metadata when chapters provided
- [ ] `token_count` in metadata for all chunks
- [ ] All new tests pass
- [ ] All 200+ existing tests still pass
- [ ] Linting passes (ruff check/format)

---

## References

- [tiktoken documentation](https://github.com/openai/tiktoken)
- [LangChain TokenTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/token_text_splitter)
- [Goal 02 Scratchpad](../scratchpad.md)
