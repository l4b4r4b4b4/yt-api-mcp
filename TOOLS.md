# MCP Server - Available Tools

> **Note:** This file serves as a **template reference** for MCP server development patterns. For the actual YouTube MCP server tools, see [README.md](README.md#available-tools).

This document describes MCP tool patterns and best practices. Update this file as you add, modify, or remove tools in your own MCP servers.

## Quick Reference

| Tool | Description | Caching | Namespace |
|------|-------------|---------|-----------|
| `example_tool` | Example tool demonstrating basic patterns | No | - |
| `cached_tool` | Example tool with caching enabled | Yes | public |

---

## Tool Documentation Template

When adding new tools, use this template:

### `tool_name`

Brief description of what the tool does and when to use it.

**Parameters:**
- `param_name` (type, required/optional): Description. Range/constraints if applicable. Default: `value`

**Returns:**
```json
{
  "field": "value",
  "description": "Example return structure"
}
```

**Example:**
```
tool_name(param1="value")
→ {"field": "result"}
```

**Notes:**
- Any special considerations
- Permission requirements
- Rate limits or quotas
- Related tools

---

## Core MCP Patterns

### Basic Tools (No Caching)

Simple tools that return small results directly:

```python
@mcp.tool
def simple_tool(param: str) -> dict[str, Any]:
    """Description of what the tool does."""
    return {"result": f"Processed: {param}"}
```

---

### Cached Tools (Reference-Based)

Tools that cache large results and return references:

```python
@mcp.tool
@cache.cached(namespace="public")
async def cached_tool(count: int = 10) -> dict[str, Any]:
    """Tool with caching - MUST return dict[str, Any]."""
    large_data = generate_large_dataset(count)
    return large_data  # Decorator wraps in cache response
```

**Important:**
- Return type MUST be `dict[str, Any]` (decorator wraps the response)
- Small results (≤64 tokens) return full data
- Large results return `ref_id` + preview
- Use `get_cached_result` to paginate

---

### Private Computation Tools

Tools that work with sensitive data agents cannot read:

```python
@mcp.tool
def store_sensitive(name: str, value: Any) -> dict[str, Any]:
    """Store sensitive data with EXECUTE-only permission for agents."""
    policy = AccessPolicy(
        user_permissions=Permission.FULL,
        agent_permissions=Permission.EXECUTE  # Agent can use, not read
    )
    ref = cache.set(key=name, value=value, policy=policy)
    return {"ref_id": ref.ref_id}

@mcp.tool
def compute_with_sensitive(ref_id: str, operation: str) -> dict[str, Any]:
    """Compute using sensitive data without revealing it."""
    value = cache.resolve(ref_id, actor=DefaultActor.system())
    result = perform_operation(value, operation)
    return {"result": result}  # Agent sees result, not original value
```

---

## Cache Management Tools

### `get_cached_result`

Retrieve and paginate cached results.

**Parameters:**
- `ref_id` (string, required): Reference ID from cached tool
- `page` (integer, optional): Page number (1-indexed)
- `page_size` (integer, optional): Items per page (1-100)
- `max_size` (integer, optional): Maximum preview size in tokens

**Returns:**
```json
{
  "ref_id": "namespace:key",
  "preview": [...],
  "total_items": 100,
  "page": 2,
  "total_pages": 5
}
```

---

## Health & Status Tools

### `health_check`

Check server health and configuration.

**Parameters:** None

**Returns:**
```json
{
  "status": "healthy",
  "server": "mcp-server-name",
  "cache": "cache-name",
  "features_enabled": ["feature1", "feature2"]
}
```

---

## Admin Tools (Optional)

Admin tools typically require authentication and elevated privileges:

| Tool | Description |
|------|-------------|
| `admin_list_references` | List cached references with filtering |
| `admin_get_cache_stats` | Get cache statistics by namespace |
| `admin_clear_cache` | Clear cache for specific namespace |
| `admin_delete_reference` | Delete a specific cached reference |

**Implementation Pattern:**
```python
def is_admin(context: Context) -> bool:
    """Override with your authentication logic."""
    return False  # Default: admin tools disabled

@mcp.tool
def admin_tool() -> dict[str, Any]:
    """Admin tool with permission check."""
    if not is_admin(get_current_context()):
        raise PermissionError("Admin access required")
    return perform_admin_operation()
```

---

## MCP Prompts (Optional)

Prompts provide guidance and documentation to LLMs:

```python
@mcp.prompt()
def server_guide() -> list[PromptMessage]:
    """Comprehensive guide for using this MCP server."""
    return [
        UserMessage(content="Guide content here..."),
        AssistantMessage(content="Example responses...")
    ]
```

---

## Best Practices

### Tool Naming
- Use clear, descriptive names: `search_videos` not `search`
- Action verbs: `get_`, `list_`, `create_`, `update_`, `delete_`
- Consistent prefixes for related tools

### Descriptions
- First sentence: What the tool does
- When to use it vs alternatives
- Any side effects or state changes
- Permission requirements

### Error Handling
```python
@mcp.tool
def tool_with_errors(param: str) -> dict[str, Any]:
    """Tool with proper error handling."""
    if not param:
        raise ValueError("param is required")

    try:
        result = external_api_call(param)
    except ExternalAPIError as e:
        raise RuntimeError(f"API call failed: {e}")

    return {"result": result}
```

### Return Structures
- Always return `dict[str, Any]` for consistency
- Include helpful metadata: timestamps, counts, status
- Use clear field names: `total_items` not `cnt`
- Include hints for pagination/continuation

---

## Testing Your Tools

Test each tool with:

1. **Valid inputs** - Happy path
2. **Edge cases** - Empty, null, boundary values
3. **Invalid inputs** - Wrong types, out of range
4. **Large datasets** - Test caching behavior
5. **Permissions** - Test access control

Example test structure:
```python
def test_tool_happy_path():
    result = tool(valid_param="test")
    assert result["status"] == "success"

def test_tool_with_invalid_input():
    with pytest.raises(ValueError):
        tool(invalid_param=None)

def test_cached_tool_returns_reference():
    result = cached_tool(count=1000)
    assert "ref_id" in result
    assert "preview" in result
```

---

## Documentation Updates

When you add/modify tools:

1. ✅ Update this TOOLS.md file
2. ✅ Update README.md with examples
3. ✅ Add docstrings to tool functions
4. ✅ Add tests in `tests/` directory
5. ✅ Update CHANGELOG.md if significant change

---

## Additional Resources

- [MCP Specification](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [mcp-refcache Documentation](https://github.com/l4b4r4b4b4/mcp-refcache)
- Project README.md
- CONTRIBUTING.md
