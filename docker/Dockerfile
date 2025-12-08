# FastMCP Template - Production Dockerfile
# Extends the base image with the template application
#
# Build: docker build -f docker/Dockerfile -t fastmcp-template:latest .
# Run:   docker run -p 8000:8000 fastmcp-template:latest

ARG BASE_IMAGE=ghcr.io/l4b4r4b4b4/fastmcp-base:latest

# ==============================================================================
# Production Runtime
# ==============================================================================
FROM ${BASE_IMAGE} AS production

# Copy application code
COPY app/ /app/app/

# Default: Run MCP server with SSE transport
CMD ["python", "-m", "app.server", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# Development Runtime (with source mounting support)
# ==============================================================================
FROM ${BASE_IMAGE} AS development

# Install dev dependencies (if base doesn't include them)
# This stage is meant for docker-compose volume mounts

# Copy application code
COPY app/ /app/app/

# Dev mode - can override with volume mount
CMD ["python", "-m", "app.server", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
