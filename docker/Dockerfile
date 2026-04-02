# FastMCP Server - Production Dockerfile
# ==============================================================================
# Self-contained multi-stage build for yt-api-mcp.
#
# Build: docker build -f docker/Dockerfile -t yt-mcp:latest .
# Run:   docker run -p 8000:8000 yt-mcp:latest
#
# The server uses streamable-http transport (recommended for remote/Docker).
# Configure via environment variables:
#   FASTMCP_PORT: Server port (default: 8000)
#   FASTMCP_HOST: Server host (default: 0.0.0.0)
#   CACHE_BACKEND: Cache backend - memory, sqlite, redis (default: redis for HTTP)
#   REDIS_URL: Redis connection URL (default: redis://localhost:6379)
# ==============================================================================

ARG PYTHON_VERSION=3.12

# ==============================================================================
# Stage 1: Builder
# Installs dependencies into a venv using uv for fast, reproducible builds.
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Build deps needed by some Python packages; git needed for VCS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency files first (layer cache: only reinstall when these change)
COPY pyproject.toml uv.lock ./

# Install dependencies into /opt/venv.
# --frozen: exact versions from lockfile
# --no-dev: skip dev dependencies for smaller image
# --no-install-project: don't install the project itself yet
# BuildKit secret mount keeps GitHub token out of layers and logs.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

RUN --mount=type=secret,id=github_token \
    GITHUB_TOKEN=$(cat /run/secrets/github_token 2>/dev/null || echo "") && \
    if [ -n "$GITHUB_TOKEN" ]; then \
      git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi && \
    uv venv /opt/venv && \
    uv sync --frozen --no-dev --no-install-project && \
    git config --global --unset-all url.https://${GITHUB_TOKEN}@github.com/.insteadOf 2>/dev/null || true

# ==============================================================================
# Stage 2: Production Runtime
# Minimal image with only the venv + application code.
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim AS production

ARG PYTHON_VERSION=3.12

# Non-root user for security
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Copy pre-built venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser app/ /app/app/

RUN chown -R appuser:appuser /app

USER appuser

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

LABEL org.opencontainers.image.source="https://github.com/l4b4r4b4b4/yt-api-mcp"
LABEL org.opencontainers.image.description="yt-api-mcp - YouTube MCP server with mcp-refcache"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.base.name="python:${PYTHON_VERSION}-slim"

CMD ["python", "-m", "app", "streamable-http"]

# ==============================================================================
# Stage 3: Development Runtime (optional, for docker-compose volume mounts)
# ==============================================================================
FROM production AS development

# Same as production but meant to be used with volume-mounted source
CMD ["python", "-m", "app", "streamable-http"]
