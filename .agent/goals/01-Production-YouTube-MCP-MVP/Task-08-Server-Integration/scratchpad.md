# Task 07: Comment Tools

**Status:** ⚪ Not Started
**Created:** 2025-01-08
**Dependencies:** Task 06 (Transcript Tools)

---

## Objective

Implement YouTube video comment retrieval with caching. Enable agents to access top comments for sentiment analysis and engagement metrics.

---

## Scope

- Implement `get_video_comments(video_id, max_results)` - Get top comments
- Use `youtube.comments` namespace (12h TTL - semi-volatile)
- Handle disabled comments gracefully
- Write comprehensive unit tests
- Test locally in Zed

---

## Tasks

- [ ] Gather context - Read reference implementation
- [ ] Document detailed plan in this scratchpad
- [ ] Get approval before implementation
- [ ] Create `app/tools/youtube/comments.py`
- [ ] Implement `get_video_comments(video_id, max_results)`
- [ ] Write unit tests (~8-10 tests)
- [ ] Register tool in `app/server.py` with @cache.cached
- [ ] Run tests and linting
- [ ] Local dev test in Zed
- [ ] Document completion

---

## Success Criteria

- [ ] Comment retrieval function implemented and tested
- [ ] Uses CommentData Pydantic model
- [ ] Handles disabled comments gracefully (clear error message)
- [ ] All unit tests pass
- [ ] Linting passes
- [ ] Works in Zed: "Get comments for video XYZ"
- [ ] Caching verified (12h TTL)

---

## Notes

After local testing validates this works, proceed to Task 08 (Final Server Polish).
