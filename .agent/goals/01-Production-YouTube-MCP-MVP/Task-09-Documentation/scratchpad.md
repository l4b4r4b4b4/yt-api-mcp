# Task 08: Final Server Polish

**Status:** ⚪ Not Started
**Created:** 2025-01-08
**Dependencies:** Task 07 (Comment Tools)

---

## Objective

Polish the MCP server for production readiness: update instructions, verify all tools are registered correctly, ensure cache namespaces are configured, and perform final integration testing with all YouTube tools working together.

---

## Scope

- Update server instructions with all YouTube tool descriptions
- Ensure all tools properly registered in server.py
- Remove any remaining demo tools (hello, generate_items, etc.)
- Verify cache namespace configuration (search, api, content, comments)
- Final end-to-end integration test with all tools
- Verify server starts without errors
- Test tool discovery in Zed

---

## Tasks

- [ ] Review server.py for completeness
- [ ] Update FastMCP server instructions with:
  - List of all YouTube tools
  - Brief description of each tool
  - Cache behavior explanation
  - API key requirements
- [ ] Verify all tools registered with @mcp.tool
- [ ] Verify all cache decorators have correct namespaces
- [ ] Remove demo tools (hello, generate_items, store_secret, compute_with_secret)
- [ ] Keep cache admin tools (get_cached_result, health_check, admin tools)
- [ ] Update cache configuration if needed (TTLs, sizes)
- [ ] Test server startup: `uv run yt-mcp stdio`
- [ ] Verify all tools appear in Zed tool list
- [ ] End-to-end test: Search → Details → Transcript → Comments
- [ ] Document completion

---

## Success Criteria

- [ ] Server instructions accurate and helpful
- [ ] All YouTube tools registered (search, metadata, transcripts, comments)
- [ ] No demo tools in production server
- [ ] Cache namespaces correctly configured
- [ ] Server starts without errors
- [ ] All tools discoverable in Zed
- [ ] End-to-end workflow works: "Find vimjoyer video, get transcript"

---

## Notes

This task ensures the server is production-ready before documentation and final validation.

After completion, proceed to Task 09 (Documentation).
