"""Tools package for fastmcp-template MCP server.

Add your custom tools in this package and register them with the MCP server.

Example:
    # In tools/my_tools.py
    from fastmcp_template.server import cache, mcp

    @mcp.tool
    @cache.cached(namespace="my-namespace")
    async def my_tool(param: str) -> dict:
        '''My custom tool.'''
        return {"result": param}
"""
