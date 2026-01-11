{
  description = "youtube-mcp-server - YouTube search, transcripts, and semantic analysis with mcp-refcache";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        fhsEnv = pkgs.buildFHSEnv {
          name = "youtube-mcp-dev-env";

          targetPkgs = pkgs':
            with pkgs'; [
              # Python and uv
              python312
              uv

              # System libraries (required for some dependencies)
              zlib
              stdenv.cc.cc.lib

              # Shells
              zsh
              bash

              # Linting & Formatting
              ruff
              pre-commit

              # Development tools
              git
              git-lfs
              curl
              wget
              jq
              tree
              httpie
            ];

          profile = ''
            echo "🎥 YouTube MCP Server Development Environment"
            echo "=============================================="

            # Create and activate uv virtual environment if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "📦 Creating uv virtual environment..."
              uv venv --python python3.12 --prompt "youtube-mcp"
            fi

            # Activate the virtual environment
            source .venv/bin/activate

            # Set a recognizable name for IDEs
            export VIRTUAL_ENV_PROMPT="youtube-mcp"

            # Sync dependencies
            if [ -f "pyproject.toml" ]; then
              echo "🔄 Syncing dependencies..."
              uv sync --quiet
            else
              echo "⚠️  No pyproject.toml found. Run 'uv init' to create project."
            fi

            echo ""
            echo "✅ Python: $(python --version)"
            echo "✅ uv:     $(uv --version)"
            echo "✅ Virtual environment: activated (.venv)"
            echo "✅ PYTHONPATH: $PWD/src:$PWD"

            # Load environment variables from .envrc
            if [ -f .envrc ]; then
              echo "🔐 Loading environment from .envrc..."
              set -a  # Auto-export all variables
              source .envrc
              set +a
            fi

            # Load local overrides from .envrc.local (gitignored secrets)
            if [ -f .envrc.local ]; then
              echo "🔑 Loading secrets from .envrc.local..."
              set -a
              source .envrc.local
              set +a
            fi
          '';

          runScript = ''
            # Set shell for the environment
            SHELL=${pkgs.zsh}/bin/zsh

            # Set PYTHONPATH to project root for module imports
            export PYTHONPATH="$PWD/src:$PWD"
            export SSL_CERT_FILE="/etc/ssl/certs/ca-bundle.crt"

            echo ""
            echo "🎥 YouTube MCP Server Quick Reference:"
            echo ""
            echo "🔧 Development:"
            echo "  uv sync                    - Sync dependencies"
            echo "  uv run pytest              - Run tests"
            echo "  uv run ruff check .        - Lint code"
            echo "  uv run ruff format .       - Format code"
            echo "  uv lock --upgrade          - Update all dependencies"
            echo ""
            echo "📦 Package Management:"
            echo "  uv add <package>           - Add runtime dependency"
            echo "  uv add --dev <package>     - Add dev dependency"
            echo "  uv remove <package>        - Remove dependency"
            echo ""
            echo "🚀 Run Server:"
            echo "  uv run youtube-mcp         - Run MCP server (stdio)"
            echo "  uv run youtube-mcp --transport sse --port 8000"
            echo ""
            echo "🎯 Features:"
            echo "  • Search YouTube videos and channels"
            echo "  • Get video transcripts with caching"
            echo "  • Semantic search over video content"
            echo "  • Requires YouTube Data API v3 key"
            echo ""
            echo "🔗 mcp-refcache dependency:"
            echo "  Installed from: git+https://github.com/l4b4r4b4b4/mcp-refcache"
            echo ""
            echo "🔐 Environment Setup:"
            echo "  1. Copy: cp .envrc.local.example .envrc.local"
            echo "  2. Edit .envrc.local with your YOUTUBE_API_KEY"
            echo "  3. Restart shell (exit + nix develop)"
            echo ""
            if [ -n "''${YOUTUBE_API_KEY}" ]; then
              echo "✅ YOUTUBE_API_KEY is set"
            else
              echo "⚠️  YOUTUBE_API_KEY not set - create .envrc.local"
            fi
            echo ""
            echo "🚀 Ready to build!"
            echo ""

            # Start zsh shell
            exec ${pkgs.zsh}/bin/zsh
          '';
        };
      in {
        devShells.default = pkgs.mkShell {
          shellHook = ''
            exec ${fhsEnv}/bin/youtube-mcp-dev-env
          '';
        };

        packages.default = pkgs.python312;
      }
    );
}
