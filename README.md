# FastMCP Template

A production-ready FastMCP server template with [mcp-refcache](https://github.com/l4b4r4b4b4/mcp-refcache) integration for building AI agent tools that handle large data efficiently.

## Features

- **Reference-Based Caching** - Return references instead of large data, reducing context window usage
- **Preview Generation** - Automatic previews for large results (sample, truncate, paginate strategies)
- **Pagination** - Navigate large datasets without loading everything at once
- **Access Control** - Separate user and agent permissions for sensitive data
- **Private Computation** - Let agents compute with values they cannot see
- **Docker Ready** - Production-ready containers with Chainguard secure base image
- **Optional Langfuse Tracing** - Built-in observability integration

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the template
git clone https://github.com/l4b4r4b4b4/fastmcp-template
cd fastmcp-template

# Install dependencies
uv sync

# Run the server (stdio mode for Claude Desktop)
uv run fastmcp-template

# Run the server (SSE/HTTP mode for deployment)
uv run fastmcp-template --transport sse --port 8000
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker compose up

# Or build images manually
docker build -f docker/Dockerfile.base -t fastmcp-base:latest .
docker build -f docker/Dockerfile -t fastmcp-template:latest .
docker run -p 8000:8000 fastmcp-template:latest
```

### Using with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fastmcp-template": {
      "command": "uv",
      "args": ["run", "fastmcp-template"],
      "cwd": "/path/to/fastmcp-template"
    }
  }
}
```

### Using with Zed

The template includes `.zed/settings.json` pre-configured for MCP context servers.

## Example Tools

The template includes several example tools demonstrating different patterns:

### Simple Tool (No Caching)

```python
@mcp.tool
def hello(name: str = "World") -> dict[str, Any]:
    """Say hello to someone."""
    return {"message": f"Hello, {name}!"}
```

### Cached Tool (Public Namespace)

```python
@mcp.tool
@cache.cached(namespace="public")
async def generate_items(count: int = 10, prefix: str = "item") -> list[dict]:
    """Generate items with automatic caching for large results."""
    return [{"id": i, "name": f"{prefix}_{i}"} for i in range(count)]
```

### Private Computation (EXECUTE Permission)

```python
@mcp.tool
def store_secret(name: str, value: float) -> dict[str, Any]:
    """Store a secret that agents can use but not read."""
    secret_policy = AccessPolicy(
        user_permissions=Permission.FULL,
        agent_permissions=Permission.EXECUTE,  # Can use, cannot see
    )
    ref = cache.set(key=f"secret_{name}", value=value, policy=secret_policy)
    return {"ref_id": ref.ref_id}

@mcp.tool
def compute_with_secret(secret_ref: str, multiplier: float = 1.0) -> dict[str, Any]:
    """Compute using a secret without revealing it."""
    secret = cache.resolve(secret_ref, actor=DefaultActor.system())
    return {"result": secret * multiplier}
```

## Project Structure

```
fastmcp-template/
├── app/                     # Application code (flat structure for containers)
│   ├── __init__.py          # Version export
│   ├── server.py            # Main server with example tools
│   └── tools/               # Additional tool modules
├── docker/
│   ├── Dockerfile.base      # Chainguard-based secure base image
│   ├── Dockerfile           # Production image (extends base)
│   └── Dockerfile.dev       # Development with hot reload
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   └── test_server.py       # Server tests
├── .github/
│   └── workflows/
│       ├── ci.yml           # CI pipeline
│       ├── docker.yml       # Docker build & publish to GHCR
│       └── release.yml      # Release automation
├── docker-compose.yml       # Local development & production
├── pyproject.toml           # Project config
├── flake.nix                # Nix dev shell
└── .rules                   # AI assistant guidelines
```

## Development

### Running Tests

```bash
uv run pytest
uv run pytest --cov  # With coverage
```

### Linting and Formatting

```bash
uv run ruff check . --fix
uv run ruff format .
```

### Type Checking

```bash
uv run mypy app/
```

### Docker Development

```bash
# Run development container with hot reload
docker compose --profile dev up

# Build base image (for publishing)
docker compose --profile build build base

# Build all images
docker compose build
```

### Using Nix (Optional)

```bash
nix develop  # Enter dev shell with all tools
```

## Customization

1. **Rename the project**: Update `pyproject.toml`, `app/`, and imports
2. **Add your tools**: Create new tools in `app/server.py` or add modules to `app/tools/`
3. **Configure caching**: Adjust `RefCache` settings in `app/server.py`
4. **Add Langfuse**: Install with `uv add langfuse` and configure environment variables
5. **Extend base image**: Use `FROM ghcr.io/l4b4r4b4b4/fastmcp-base:latest` in your Dockerfile

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | - |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | - |
| `LANGFUSE_HOST` | Langfuse host URL | `https://cloud.langfuse.com` |

### CLI Options

```bash
uv run fastmcp-template --help

Options:
  --transport {stdio,sse}  Transport mode (default: stdio)
  --port PORT              Port for SSE transport (default: 8000)
  --host HOST              Host for SSE transport (default: 127.0.0.1)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Related Projects

- [mcp-refcache](https://github.com/l4b4r4b4b4/mcp-refcache) - Reference-based caching for MCP servers
- [FastMCP](https://github.com/jlowin/fastmcp) - High-performance MCP server framework
- [Model Context Protocol](https://modelcontextprotocol.io/) - The underlying protocol specification