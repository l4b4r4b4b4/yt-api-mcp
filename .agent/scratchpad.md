# Side Quest: FastMCP Template Repository

## Status: Phase 1 Complete ✅

---

## Task Overview

### Goal (Phase 1 - Current Focus)
Create a **FastMCP template starter repo** based on finquant-mcp patterns, with:
- mcp-refcache integration
- Langfuse tracing (optional)
- Complete project scaffolding

### Goal (Phase 2 - Later)
Build a **Zed Management MCP Server** for tracking/managing chat sessions per project

---

## Phase 1 Progress

### ✅ Completed Files

#### Project Configuration
- [x] `pyproject.toml` - UV project with dependencies, ruff/pytest/mypy config
- [x] `.python-version` - Python 3.12
- [x] `flake.nix` - Nix dev shell with FHS environment, auto-venv, uv sync
- [x] `.gitignore` - Comprehensive Python gitignore + archive/, .venv, Nix result
- [x] `.pre-commit-config.yaml` - Ruff, mypy, bandit, safety hooks

#### GitHub Integration
- [x] `.github/workflows/ci.yml` - Python 3.12/3.13 matrix, lint, test, security scan
- [x] `.github/workflows/release.yml` - Build on version tags, GitHub release
- [x] `.github/copilot-instructions.md` - Copilot guidance for the project

#### IDE Configuration
- [x] `.zed/settings.json` - Pyright LSP, ruff format, MCP context servers

#### Source Code
- [x] `src/fastmcp_template/__init__.py` - Version export
- [x] `src/fastmcp_template/server.py` - **MAIN FILE** - Complete server with:
  - `hello` tool (no caching, simple example)
  - `generate_items` tool (cached in PUBLIC namespace - demonstrates shared caching)
  - `store_secret` tool (EXECUTE-only for agents)
  - `compute_with_secret` tool (private computation)
  - `get_cached_result` tool (pagination)
  - `health_check` tool
  - Admin tools registration
  - `template_guide` prompt
  - CLI with stdio/sse transport options
- [x] `src/fastmcp_template/tools/__init__.py` - Placeholder with usage example

#### Tests
- [x] `tests/__init__.py`
- [x] `tests/conftest.py` - RefCache fixture, sample_items fixture
- [x] `tests/test_server.py` - Tests for hello, health_check, MCP config

#### Documentation & Guidelines
- [x] `.rules` - Copied from finquant-mcp (needs project name updates)
- [x] `CONTRIBUTING.md` - Copied from finquant-mcp (needs project name updates)

### ✅ Completed Tasks

#### Documentation
- [x] `README.md` - Project overview, installation, usage, examples
- [x] `docs/README.md` - Extended documentation
- [x] `CHANGELOG.md` - Initial changelog entry
- [x] `LICENSE` - MIT license
- [x] `.agent/scratchpad.md` - Session scratchpad

#### File Updates
- [x] Update `.rules` - Replace finquant-mcp references with fastmcp-template
- [x] Update `CONTRIBUTING.md` - Replace mcp-refcache references with fastmcp-template
- [x] Update `pyproject.toml` - Remove deprecated ANN101/ANN102 ruff rules

#### Testing & Verification
- [x] Run `uv sync` to install dependencies
- [x] Run `uv run pytest` - 10 tests pass
- [x] Run `uv run ruff check . --fix && uv run ruff format .` - passes
- [x] Test server: `uv run fastmcp-template --help` - works

### 🔄 In Progress

#### Structure Changes (Current Session)
- [x] Moved `src/fastmcp_template/` → `app/` (flat structure for containerized servers)
- [x] Updated pyproject.toml, tests, CI workflows for `app/` structure
- [x] Created Docker setup:
  - `docker/Dockerfile.base` - Chainguard-based secure base image for all FastMCP servers
  - `docker/Dockerfile` - Production image extending base
  - `docker/Dockerfile.dev` - Development image with hot reload
  - `docker-compose.yml` - Local development and production
  - `.github/workflows/docker.yml` - Build & publish to GHCR

#### Final Verification
- [x] Run `nix develop` to test flake
- [x] Tests pass (10/10)
- [x] Linting passes
- [x] CLI works (`uv run fastmcp-template --help`)
- [ ] Verify Zed IDE settings work (LSP, MCP context servers)
- [ ] Build Docker images locally
- [ ] Push to GitHub repo `l4b4r4b4b4/fastmcp-template`
- [ ] Return to mcp-refcache, delete examples/fastmcp-template, add as submodule

---

## Key Design Decisions Made

1. **Public namespace for generate_items**: Uses `@cache.cached(namespace="public")` to demonstrate shared caching that all users can access.

2. **Copied patterns from mcp_server.py**: Server structure, tool patterns, Pydantic models all follow the calculator example.

3. **Minimal but complete**: Template has enough to be useful but isn't overwhelming - users can delete what they don't need.

4. **No separate cache.py**: Cache is created inline in server.py following the mcp_server.py pattern. No wrapper needed - use mcp-refcache directly.

5. **Python 3.12+ only**: Simplified matrix to 3.12 and 3.13 (dropped 3.10/3.11 support for cleaner code).

---

## File Locations

All files are in: `mcp-refcache/examples/fastmcp-template/`

```
fastmcp-template/
├── .agent/
│   └── scratchpad.md            ✅
├── .github/
│   ├── workflows/
│   │   ├── ci.yml               ✅
│   │   ├── docker.yml           ✅ (build & publish to GHCR)
│   │   └── release.yml          ✅
│   └── copilot-instructions.md  ✅
├── .zed/
│   └── settings.json            ✅
├── app/                         # Flat structure for containerized server
│   ├── __init__.py              ✅
│   ├── server.py                ✅ (main file)
│   └── tools/
│       └── __init__.py          ✅
├── docker/
│   ├── Dockerfile               ✅ (production, extends base)
│   ├── Dockerfile.base          ✅ (Chainguard-based, reusable)
│   └── Dockerfile.dev           ✅ (development with hot reload)
├── docs/
│   └── README.md                ✅
├── tests/
│   ├── __init__.py              ✅
│   ├── conftest.py              ✅
│   └── test_server.py           ✅
├── .gitignore                   ✅
├── .pre-commit-config.yaml      ✅
├── .python-version              ✅
├── .rules                       ✅
├── CHANGELOG.md                 ✅
├── CONTRIBUTING.md              ✅
├── LICENSE                      ✅
├── README.md                    ✅
├── docker-compose.yml           ✅
├── flake.nix                    ✅
└── pyproject.toml               ✅
```

---

## Session Log

### 2024-12-08: Research Complete
- Analyzed finquant-mcp, BundesMCP, calculator example
- Documented template specification
- Created implementation checklist

### 2024-12-09: Phase 1 Implementation Started
- Created GitHub repo `l4b4r4b4b4/fastmcp-template` (private)
- Scaffolded directory structure in `mcp-refcache/examples/fastmcp-template/`
- Created all config files (pyproject.toml, flake.nix, .pre-commit-config.yaml, etc.)
- Created GitHub workflows (ci.yml, release.yml)
- Copied and adapted server.py from mcp_server.py calculator example
- Created simplified tests from finquant-mcp patterns
- Copied .rules and CONTRIBUTING.md (need updates)
- **Key**: Used `@cache.cached(namespace="public")` for generate_items to demonstrate shared caching

### 2024-12-09: Phase 1 Completed
- Created README.md, CHANGELOG.md, LICENSE, docs/README.md
- Updated .rules and CONTRIBUTING.md with correct project references
- Fixed deprecated ruff rules in pyproject.toml
- Verified: uv sync, pytest (10 pass), ruff, CLI all work
- Switched to new Zed session in fastmcp-template directory with nix develop

### 2024-12-09: Docker & Structure Refactor
- Restructured from `src/fastmcp_template/` to `app/` (flat, containerized server pattern)
- Created Docker setup with Chainguard secure base image:
  - `docker/Dockerfile.base` - Reusable base for all FastMCP servers (GHCR: fastmcp-base)
  - `docker/Dockerfile` - Production image for this template
  - `docker/Dockerfile.dev` - Development with hot reload
  - `docker-compose.yml` - Easy local deployment
  - `.github/workflows/docker.yml` - CI/CD for Docker images
- Updated all config (pyproject.toml, tests, CI) for app/ structure
- All tests pass, linting passes, CLI works

---

## Future Feature Requests

### Zed MCP: Text Passage Copy/Insert Tool

**Problem**: Currently no MCP tool to copy a specific text passage from one file and insert it into another at a specific location.

**Available tools**:
- `read_file` - reads file content
- `edit_file` - creates/edits files  
- `copy_path` - copies entire files/directories

**Missing tool**: Something like `copy_text_passage` that could:
- Copy lines X-Y from file A
- Insert at line Z in file B
- Or insert before/after a specific pattern in file B

**Use case**: When migrating content between files, refactoring, or extracting sections - currently requires reading source, then manually editing destination.

---

## Next Steps

1. ~~Verify `nix develop` works correctly~~ ✅
2. Verify Zed LSP and MCP context servers work
3. Build Docker images locally to test
4. Push to GitHub
5. Add as submodule to mcp-refcache