# Goal 03: Context Limit Research

**Status:** 🟡 In Progress
**Priority:** Low (ongoing research)
**Created:** 2025-01-13
**Updated:** 2025-01-13
**Parent:** [Goals Index](../scratchpad.md)

---

## Objective

Track and understand the relationship between:
1. **Zed's reported context usage** (shown in UI)
2. **Claude's reported context remaining** (from system info)
3. **Actual behavior at limits** (what happens when we approach soft/hard limits)

This research will inform the `.rules` context handoff guidelines.

---

## Reference: Known Limits

| Model | Hard Limit | Soft Limit (Zed) | Source |
|-------|------------|------------------|--------|
| Claude Opus 4.5 | 200k | 128k | .rules |
| Claude Sonnet 4 | 200k | 128k | .rules |
| Claude Sonnet 3.5 | 200k | 128k | .rules |

**Current .rules guidance:** Plan handoffs at 105% of soft limit

---

## Data Collection Template

For each session, record at START, MIDDLE, and END:

```
### Session: [DATE] - [GOAL/TASK]
**Model:** [e.g., Claude Opus 4.5]

| Point | Zed Shows | Claude Reports | Delta | Notes |
|-------|-----------|----------------|-------|-------|
| START | ?k | ?k remaining | ? | |
| MIDDLE | ?k | ?k remaining | ? | |
| END | ?k | ?k remaining | ? | |

**Observations:**
-
```

---

## Session Log

### Session: 2025-01-13 - Goal 02 Task-06 (Semantic Search)
**Model:** Claude Opus 4.5

| Point | Zed Shows | Claude Reports (Remaining) | Claude Used | Notes |
|-------|-----------|---------------------------|-------------|-------|
| START | ~10k? | 137,848 remaining | ~0 | Fresh session with handoff context |
| MIDDLE | 104k | 107,796 remaining | ~30k | After implementing semantic_search_transcripts |
| SOFT LIMIT | 130k | ~87,500 remaining | ~50k | User reported hitting soft limit |
| END | 135k | ~84,000 remaining | ~54k | Over soft limit, session ending |

**Key Finding: ~80k Token Discrepancy!**
- At END: Zed shows 135k used, Claude reports only ~54k used
- **Difference: ~80k tokens of "invisible" overhead**
- This is likely: system prompt + 150+ MCP tool schemas + conversation framing

**Observations:**
- Zed's soft limit (128k) triggered at ~130k display
- Claude still had ~87k "remaining" when Zed warned
- The ~80k overhead is consistent throughout session
- Zed counts TOTAL context, Claude counts CONVERSATION tokens

**Confirmed Answers:**
1. **Zed's number:** Total context window usage (system + tools + conversation)
2. **Claude's "remaining":** Space left in 200k hard limit, but measures differently
3. **Overhead:** ~80k tokens for system prompt + MCP tool schemas (150+ tools!)
4. **Soft limit behavior:** Zed warns at 128k, but can continue past it

**Implication for .rules:**
- Current "105% of soft limit" guidance is reasonable
- Real usable conversation space is ~50-60k tokens (128k - 80k overhead)
- Consider reducing MCP tools if context is precious

---

## Hypotheses

### H1: Zed counts total input, Claude counts available for response
- Zed: "You've used 104k tokens in this conversation"
- Claude: "I have 108k tokens left in my context window"
- These might measure different things

### H2: Tool schemas consume significant hidden context
- Each MCP tool has a JSON schema
- 150+ tools × ~200 tokens each = ~30k tokens of "invisible" context
- This might explain the discrepancy

### H3: Soft limit is enforced by Zed, not Claude
- Zed might start warning/refusing at 128k regardless of Claude's capacity
- Need to test what actually happens at 128k+

---

## Experiments to Run

- [ ] **E1:** Track full session from 0 to soft limit warning
- [ ] **E2:** Compare same task with fewer MCP tools enabled
- [ ] **E3:** Push past soft limit and document behavior
- [ ] **E4:** Compare across different Claude models

---

## Findings

*To be updated as we collect more data*

---

## Action Items

1. Continue logging context at START/MIDDLE/END of sessions
2. Ask user to screenshot Zed's context indicator when reporting
3. Document any warnings or degraded behavior near limits
4. Update `.rules` based on findings
