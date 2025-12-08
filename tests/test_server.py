"""Tests for the fastmcp-template server module."""

from __future__ import annotations

from app.server import cache, mcp


class TestServerInitialization:
    """Tests for server initialization."""

    def test_mcp_instance_exists(self) -> None:
        """Test that FastMCP instance is created."""
        assert mcp is not None
        assert mcp.name == "FastMCP Template"

    def test_cache_instance_exists(self) -> None:
        """Test that RefCache instance is created."""
        assert cache is not None
        assert cache.name == "fastmcp-template"


class TestHelloTool:
    """Tests for the hello tool."""

    def _call_hello(self, name: str = "World") -> dict:
        """Helper to call hello, handling FunctionTool wrapper."""
        from app import server

        hello_fn = server.hello
        if hasattr(hello_fn, "fn"):
            return hello_fn.fn(name)
        return hello_fn(name)

    def test_hello_default(self) -> None:
        """Test hello with default name."""
        result = self._call_hello()
        assert result["message"] == "Hello, World!"
        assert result["server"] == "fastmcp-template"

    def test_hello_custom_name(self) -> None:
        """Test hello with custom name."""
        result = self._call_hello("Alice")
        assert result["message"] == "Hello, Alice!"


class TestHealthCheck:
    """Tests for health_check tool."""

    def _call_health_check(self) -> dict:
        """Helper to call health_check, handling FunctionTool wrapper."""
        from app import server

        health_fn = server.health_check
        if hasattr(health_fn, "fn"):
            return health_fn.fn()
        return health_fn()

    def test_health_check_returns_status(self) -> None:
        """Test that health check returns healthy status."""
        result = self._call_health_check()

        assert "status" in result
        assert result["status"] == "healthy"

    def test_health_check_returns_server_name(self) -> None:
        """Test that health check returns server name."""
        result = self._call_health_check()

        assert "server" in result
        assert result["server"] == "fastmcp-template"

    def test_health_check_returns_cache_name(self) -> None:
        """Test that health check returns cache name."""
        result = self._call_health_check()

        assert "cache" in result
        assert result["cache"] == "fastmcp-template"


class TestMCPConfiguration:
    """Tests for MCP server configuration."""

    def test_mcp_has_instructions(self) -> None:
        """Test that MCP has instructions configured."""
        assert mcp.instructions is not None
        assert len(mcp.instructions) > 0

    def test_instructions_mention_caching(self) -> None:
        """Test that instructions mention caching."""
        assert "cach" in mcp.instructions.lower()

    def test_instructions_mention_secret(self) -> None:
        """Test that instructions mention secret computation."""
        assert "secret" in mcp.instructions.lower()
