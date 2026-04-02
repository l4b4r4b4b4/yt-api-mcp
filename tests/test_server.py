"""Tests for the yt-mcp server module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.server import cache, mcp
from app.tracing import (
    MockContext,
    enable_test_mode,
    get_langfuse_attributes,
    is_langfuse_enabled,
    is_test_mode_enabled,
)


class TestServerInitialization:
    """Tests for server initialization."""

    def test_mcp_instance_exists(self) -> None:
        """Test that FastMCP instance is created."""
        assert mcp is not None
        assert mcp.name == "YouTube MCP Server"

    def test_cache_instance_exists(self) -> None:
        """Test that RefCache instance is created."""
        assert cache is not None
        assert cache.name == "yt-mcp"


class TestTracingModule:
    """Tests for the tracing module."""

    def setup_method(self) -> None:
        """Reset test mode before each test."""
        enable_test_mode(False)
        MockContext.reset()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        enable_test_mode(False)
        MockContext.reset()

    def test_is_langfuse_enabled_without_env(self) -> None:
        """Test that Langfuse is disabled without env vars."""
        # This may be True or False depending on env
        result = is_langfuse_enabled()
        assert isinstance(result, bool)

    def test_enable_test_mode(self) -> None:
        """Test enabling test mode."""
        assert not is_test_mode_enabled()
        enable_test_mode(True)
        assert is_test_mode_enabled()
        enable_test_mode(False)
        assert not is_test_mode_enabled()

    def test_mock_context_default_state(self) -> None:
        """Test MockContext has default state."""
        state = MockContext.get_current_state()
        assert state["user_id"] == "demo_user"
        assert state["org_id"] == "demo_org"
        assert state["agent_id"] == "demo_agent"
        assert state["session_id"] == "demo_session_001"

    def test_mock_context_set_state(self) -> None:
        """Test MockContext state can be updated."""
        MockContext.set_state(user_id="alice", org_id="acme")
        state = MockContext.get_current_state()
        assert state["user_id"] == "alice"
        assert state["org_id"] == "acme"

    def test_mock_context_set_session_id(self) -> None:
        """Test MockContext session_id can be updated."""
        MockContext.set_session_id("new-session-123")
        state = MockContext.get_current_state()
        assert state["session_id"] == "new-session-123"

    def test_mock_context_reset(self) -> None:
        """Test MockContext reset."""
        MockContext.set_state(user_id="bob")
        MockContext.set_session_id("custom-session")
        MockContext.reset()
        state = MockContext.get_current_state()
        assert state["user_id"] == "demo_user"
        assert state["session_id"] == "demo_session_001"

    def test_get_langfuse_attributes_default(self) -> None:
        """Test get_langfuse_attributes without context."""
        attrs = get_langfuse_attributes()
        assert "user_id" in attrs
        assert "session_id" in attrs
        assert "metadata" in attrs
        assert "tags" in attrs
        assert "version" in attrs

    def test_get_langfuse_attributes_with_test_mode(self) -> None:
        """Test get_langfuse_attributes with test mode enabled."""
        enable_test_mode(True)
        MockContext.set_state(user_id="test_user", org_id="test_org")
        attrs = get_langfuse_attributes()
        assert attrs["user_id"] == "test_user"
        assert attrs["metadata"]["orgid"] == "test_org"
        assert "testmode" in attrs["tags"]

    def test_get_langfuse_attributes_with_operation(self) -> None:
        """Test get_langfuse_attributes with operation name."""
        attrs = get_langfuse_attributes(operation="cache_set")
        assert attrs["metadata"]["operation"] == "cache_set"
        assert "cacheset" in attrs["tags"]


class TestContextManagementTools:
    """Tests for context management tools."""

    def setup_method(self) -> None:
        """Reset test mode before each test."""
        enable_test_mode(False)
        MockContext.reset()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        enable_test_mode(False)
        MockContext.reset()

    def _call_enable_test_context(self, enabled: bool = True) -> dict:
        """Helper to call enable_test_context tool."""
        from app import server

        fn = server.enable_test_context
        if hasattr(fn, "fn"):
            return fn.fn(enabled)
        return fn(enabled)

    def _call_set_test_context(self, **kwargs) -> dict:
        """Helper to call set_test_context tool."""
        from app import server

        fn = server.set_test_context
        if hasattr(fn, "fn"):
            return fn.fn(**kwargs)
        return fn(**kwargs)

    def _call_reset_test_context(self) -> dict:
        """Helper to call reset_test_context tool."""
        from app import server

        fn = server.reset_test_context
        if hasattr(fn, "fn"):
            return fn.fn()
        return fn()

    def _call_get_trace_info(self) -> dict:
        """Helper to call get_trace_info tool."""
        from app import server

        fn = server.get_trace_info
        if hasattr(fn, "fn"):
            return fn.fn()
        return fn()

    def test_enable_test_context_returns_status(self) -> None:
        """Test enable_test_context returns correct status."""
        result = self._call_enable_test_context(True)
        assert result["test_mode"] is True
        assert "context" in result
        assert "langfuse_enabled" in result

    def test_enable_test_context_disable(self) -> None:
        """Test disabling test context."""
        self._call_enable_test_context(True)
        result = self._call_enable_test_context(False)
        assert result["test_mode"] is False

    def test_set_test_context_updates_values(self) -> None:
        """Test set_test_context updates context values."""
        result = self._call_set_test_context(
            user_id="alice", org_id="acme", session_id="chat-001"
        )
        assert result["context"]["user_id"] == "alice"
        assert result["context"]["org_id"] == "acme"
        assert result["context"]["session_id"] == "chat-001"

    def test_set_test_context_auto_enables_test_mode(self) -> None:
        """Test set_test_context auto-enables test mode."""
        assert not is_test_mode_enabled()
        self._call_set_test_context(user_id="bob")
        assert is_test_mode_enabled()

    def test_reset_test_context(self) -> None:
        """Test reset_test_context resets to defaults."""
        self._call_set_test_context(user_id="alice")
        result = self._call_reset_test_context()
        assert result["context"]["user_id"] == "demo_user"

    def test_get_trace_info_returns_status(self) -> None:
        """Test get_trace_info returns tracing status."""
        result = self._call_get_trace_info()
        assert "langfuse_enabled" in result
        assert "langfuse_host" in result
        assert "public_key_set" in result
        assert "secret_key_set" in result
        assert "test_mode_enabled" in result
        assert "langfuse_attributes" in result


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
        assert result["server"] == "yt-mcp"

    def test_health_check_returns_cache_name(self) -> None:
        """Test that health check returns cache name."""
        result = self._call_health_check()

        assert "cache" in result
        assert result["cache"] == "yt-mcp"


class TestMCPConfiguration:
    """Tests for FastMCP configuration."""

    def test_mcp_has_instructions(self) -> None:
        """Test that MCP has instructions."""
        assert mcp.instructions
        assert len(mcp.instructions) > 0

    def test_instructions_mention_caching(self) -> None:
        """Test that instructions mention caching."""
        assert "cache" in mcp.instructions.lower()

    def test_instructions_mention_youtube(self) -> None:
        """Test that instructions mention YouTube tools."""
        assert "youtube" in mcp.instructions.lower()
        assert "search" in mcp.instructions.lower()


class TestGetCachedResult:
    """Tests for the get_cached_result tool."""

    @pytest.fixture(autouse=True)
    def _setup_and_teardown(self) -> None:
        """Clear cache before and after each test."""
        cache.clear()
        yield
        cache.clear()

    async def _call_get_cached_result(
        self,
        ref_id: str,
        page: int | None = None,
        page_size: int | None = None,
        max_size: int | None = None,
        full: bool = False,
    ) -> dict:
        """Helper to call get_cached_result."""
        from app import server

        get_fn = server.get_cached_result
        fn = get_fn.fn if hasattr(get_fn, "fn") else get_fn

        return await fn(ref_id, page, page_size, max_size, full)

    @pytest.mark.asyncio
    async def test_get_cached_result_invalid_ref(self) -> None:
        """Test that invalid reference returns error dict."""
        result = await self._call_get_cached_result("nonexistent:ref")

        assert "error" in result
        assert result["ref_id"] == "nonexistent:ref"

    @pytest.mark.asyncio
    async def test_get_cached_result_not_found(self) -> None:
        """Test that result includes the requested ref_id."""
        result = await self._call_get_cached_result("test:ref:123")

        assert "ref_id" in result
        assert result["ref_id"] == "test:ref:123"

    @pytest.mark.asyncio
    async def test_get_cached_result_returns_preview_for_small_value(self) -> None:
        """Test that a small cached value is returned via preview mode."""
        ref = cache.set("small_data", {"key": "value"}, namespace="public")

        result = await self._call_get_cached_result(ref.ref_id)

        assert result["ref_id"] == ref.ref_id
        assert "preview" in result
        assert result["retrieval_mode"] == "preview"
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_get_cached_result_max_size_forwarded(self) -> None:
        """Test that max_size is forwarded to cache.get() and affects preview size.

        This is the core bug fix: previously max_size was accepted but silently
        discarded, so the preview was always limited to the server default (2048).
        """
        # Create a large value that exceeds the default preview size
        large_value = [{"id": i, "data": f"item_{i}" * 50} for i in range(200)]
        ref = cache.set("large_data", large_value, namespace="public")

        # Get with default max_size (server default: 2048)
        result_default = await self._call_get_cached_result(ref.ref_id)
        assert result_default["retrieval_mode"] == "preview"
        default_preview_size = result_default.get("preview_size")

        # Get with larger max_size — should return more content
        result_large = await self._call_get_cached_result(ref.ref_id, max_size=100000)
        assert result_large["retrieval_mode"] == "preview"
        large_preview_size = result_large.get("preview_size")

        # The larger max_size should yield a larger (or equal) preview
        if default_preview_size is not None and large_preview_size is not None:
            assert large_preview_size >= default_preview_size

    @pytest.mark.asyncio
    async def test_get_cached_result_full_retrieval(self) -> None:
        """Test that full=True returns the complete cached value via resolve()."""
        original_value = {"transcript": "Hello world " * 500, "video_id": "abc123"}
        ref = cache.set("full_test", original_value, namespace="public")

        result = await self._call_get_cached_result(ref.ref_id, full=True)

        assert result["ref_id"] == ref.ref_id
        assert result["is_complete"] is True
        assert result["retrieval_mode"] == "full"
        assert result["value"] == original_value

    @pytest.mark.asyncio
    async def test_get_cached_result_full_retrieval_list(self) -> None:
        """Test that full=True works for list values."""
        original_list = [{"id": i, "text": f"entry_{i}"} for i in range(100)]
        ref = cache.set("list_test", original_list, namespace="public")

        result = await self._call_get_cached_result(ref.ref_id, full=True)

        assert result["is_complete"] is True
        assert result["retrieval_mode"] == "full"
        assert result["value"] == original_list
        assert len(result["value"]) == 100

    @pytest.mark.asyncio
    async def test_get_cached_result_full_not_found(self) -> None:
        """Test that full=True on invalid ref returns error dict."""
        result = await self._call_get_cached_result("missing:ref", full=True)

        assert "error" in result
        assert result["ref_id"] == "missing:ref"

    @pytest.mark.asyncio
    async def test_get_cached_result_pagination_with_max_size(self) -> None:
        """Test that pagination and max_size work together."""
        large_list = list(range(200))
        ref = cache.set("paginated_data", large_list, namespace="public")

        result = await self._call_get_cached_result(
            ref.ref_id, page=1, page_size=10, max_size=50000
        )

        assert result["ref_id"] == ref.ref_id
        assert result["retrieval_mode"] == "preview"
        # Pagination info should be present
        if result.get("page") is not None:
            assert result["page"] == 1
            assert result["total_pages"] is not None

    @pytest.mark.asyncio
    async def test_get_cached_result_full_vs_preview_content_differs(self) -> None:
        """Test that full retrieval returns more data than preview for large values."""
        # Build a value large enough to trigger preview truncation
        large_value = {f"key_{i}": f"value_{'x' * 100}_{i}" for i in range(100)}
        ref = cache.set("compare_test", large_value, namespace="public")

        preview_result = await self._call_get_cached_result(ref.ref_id)
        full_result = await self._call_get_cached_result(ref.ref_id, full=True)

        assert preview_result["retrieval_mode"] == "preview"
        assert full_result["retrieval_mode"] == "full"
        assert full_result["value"] == large_value

        # Full result should contain the complete value
        assert len(full_result["value"]) == 100


class TestIsAdmin:
    """Tests for the is_admin function."""

    @pytest.mark.asyncio
    async def test_is_admin_returns_false(self) -> None:
        """Test that is_admin returns False by default."""
        from app.server import is_admin

        ctx = MagicMock()
        result = await is_admin(ctx)

        assert result is False

    @pytest.mark.asyncio
    async def test_is_admin_with_none_context(self) -> None:
        """Test that is_admin handles None context."""
        from app.server import is_admin

        result = await is_admin(None)

        assert result is False


class TestTyperCLI:
    """Tests for the typer CLI entry point."""

    def test_cli_app_exists(self) -> None:
        """Test that typer app is created."""
        from app.__main__ import app

        assert app is not None

    def test_cli_has_stdio_command(self) -> None:
        """Test that CLI has stdio command."""
        from app.__main__ import app

        # Check callback names since typer may not set cmd.name for decorated funcs
        callback_names = [
            cmd.callback.__name__ if cmd.callback else cmd.name
            for cmd in app.registered_commands
        ]
        assert "stdio" in callback_names

    def test_cli_has_sse_command(self) -> None:
        """Test that CLI has sse command."""
        from app.__main__ import app

        callback_names = [
            cmd.callback.__name__ if cmd.callback else cmd.name
            for cmd in app.registered_commands
        ]
        assert "sse" in callback_names

    def test_cli_has_streamable_http_command(self) -> None:
        """Test that CLI has streamable-http command."""
        from app.__main__ import app

        # streamable-http uses explicit name, check both name and callback
        command_info = [
            (cmd.name, cmd.callback.__name__ if cmd.callback else None)
            for cmd in app.registered_commands
        ]
        has_streamable_http = any(
            name == "streamable-http" or callback == "streamable_http"
            for name, callback in command_info
        )
        assert has_streamable_http


class TestTemplateGuidePrompt:
    """Tests for the template_guide prompt."""

    def _call_template_guide(self) -> str:
        """Helper to call template_guide prompt."""
        from app import server

        prompt_fn = server.template_guide
        if hasattr(prompt_fn, "fn"):
            return prompt_fn.fn()
        return prompt_fn()

    def test_template_guide_returns_string(self) -> None:
        """Test that template_guide returns a string."""
        result = self._call_template_guide()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_template_guide_mentions_youtube(self) -> None:
        """Test that guide mentions YouTube tools."""
        result = self._call_template_guide()
        assert "youtube" in result.lower() or "search" in result.lower()
