# Task-04: Transcript-Aware Text Chunker

**Status:** ⚪ Not Started
**Priority:** High
**Created:** 2025-01-12
**Parent:** [Goal 02: Semantic Transcript Search](../scratchpad.md)

---

## Objective

Implement a transcript-aware text chunker that groups transcript entries by target character count while preserving timestamp boundaries and generating rich metadata for each chunk.

---

## Implementation Plan

1. [ ] Implement TranscriptChunker.chunk_transcript() method
2. [ ] Group entries by chunk_size, respecting entry boundaries
3. [ ] Calculate start_time and end_time for each chunk
4. [ ] Generate timestamp URLs for playback
5. [ ] Store rich metadata (video_id, channel_id, etc.)
6. [ ] Handle chunk_overlap by including trailing entries
7. [ ] Add unit tests

---

## Key Requirements

### Input: Transcript Entries
```python
transcript_entries = [
    {"text": "Hello world", "start": 0.0, "duration": 2.5},
    {"text": "This is a test", "start": 2.5, "duration": 3.0},
    ...
]
```

### Input: Video Metadata
```python
video_metadata = {
    "video_id": "abc123",
    "video_title": "Test Video",
    "channel_id": "UC123",
    "channel_title": "Test Channel",
    "video_url": "https://youtube.com/watch?v=abc123",
    "published_at": "2024-01-01T00:00:00Z",
    "language": "en",
}
```

### Output: LangChain Documents
```python
Document(
    page_content="Hello world This is a test...",
    metadata={
        "video_id": "abc123",
        "video_title": "Test Video",
        "channel_id": "UC123",
        "channel_title": "Test Channel",
        "video_url": "https://youtube.com/watch?v=abc123",
        "start_time": 0.0,
        "end_time": 5.5,
        "timestamp_url": "https://www.youtube.com/watch?v=abc123&t=0",
        "language": "en",
        "chunk_index": 0,
    }
)
```

---

## Algorithm

1. Initialize empty current_chunk (text list, entries list)
2. For each transcript entry:
   a. Calculate if adding entry exceeds chunk_size
   b. If yes, finalize current chunk and start new one
   c. Add entry to current chunk
3. Finalize last chunk
4. For overlap: include last N chars from previous chunk in next

### Chunk Finalization
- Concatenate all entry texts with spaces
- start_time = first entry's start
- end_time = last entry's start + duration
- Generate timestamp_url from video_id + start_time
- Create Document with text and metadata

---

## Test Cases

1. **Basic chunking:**
   - 10 short entries, chunk_size=500
   - Verify chunks respect size limit
   - Verify no entry is split

2. **Timestamp preservation:**
   - Verify start_time is first entry's start
   - Verify end_time is last entry's start + duration

3. **Metadata propagation:**
   - All video metadata present in each chunk
   - chunk_index increments correctly

4. **Overlap handling:**
   - chunk_overlap=100 includes trailing text
   - Overlap doesn't cause duplicate entries

5. **Edge cases:**
   - Empty transcript
   - Single entry larger than chunk_size
   - Very short entries

---

## Acceptance Criteria

- [ ] TranscriptChunker.chunk_transcript() returns list of Documents
- [ ] Chunks respect chunk_size limit (at entry boundaries)
- [ ] Timestamps correctly calculated for each chunk
- [ ] timestamp_url generated with correct format
- [ ] All video metadata propagated to chunk metadata
- [ ] chunk_overlap works correctly
- [ ] Unit tests cover all cases

---

## Notes

- Never split mid-entry (preserve transcript structure)
- Use space as separator when concatenating entries
- Timestamps in seconds (float)
- chunk_index is 0-indexed

### Required Enhancement: Token-Based Chunking

**IMPORTANT:** Current implementation uses character count, but embedding models have **token limits**, not character limits. Character count is inconsistent across languages and content.

**TODO:**
- Use tiktoken or model's tokenizer to count tokens instead of characters
- `chunk_size` should be in tokens (e.g., 256 tokens for 512-dim embeddings)
- Consider using langchain's `TokenTextSplitter` as reference
- NomicEmbeddings has 8192 token context window

### Future Enhancement: Chapter Awareness

**TODO:** YouTube videos can have chapter markers (from video descriptions) that would be natural chunk boundaries. Consider:
- Accept optional `chapters` parameter with start times and titles
- Prefer chapter boundaries over arbitrary chunk_size splits
- Include chapter title in chunk metadata
- Fall back to current algorithm when no chapters available

---

## References

- [LangChain Document](https://python.langchain.com/docs/concepts/documents/)
- [Goal 02 Scratchpad](../scratchpad.md)
