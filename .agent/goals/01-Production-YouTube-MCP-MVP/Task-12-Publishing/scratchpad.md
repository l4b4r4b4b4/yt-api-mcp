# Task 12: Publish to PyPI & GHCR (Version 0.0.0)

**Status:** 🟡 In Progress
**Goal:** [01-Production-YouTube-MCP-MVP](../scratchpad.md)
**Dependencies:** Tasks 1-11 (all complete)

---

## Objective

Publish the first experimental release (version 0.0.0) of the YouTube MCP server to:
1. **PyPI** - Python package installation via `pip install yt-mcp`
2. **GHCR** - Docker container registry at `ghcr.io/l4b4r4b4b4/yt-mcp`

This release tests both the implementation AND the publishing workflow, establishing the foundation for future releases.

---

## Pre-Publishing Validation

### ✅ Code Quality
- [x] 178/178 tests passing
- [x] 76% code coverage (exceeds 73% requirement)
- [x] Ruff linting clean
- [x] Type hints on all public APIs

### ✅ Docker Validation
- [x] Base image builds: 290MB
- [x] Production image builds: 229MB
- [x] All 16 tools work in container
- [x] Zed MCP client integration tested
- [x] Health checks configured

### ✅ Documentation
- [x] README.md: Complete tool docs, setup guides, examples
- [x] CHANGELOG.md: Detailed 0.0.0 release notes
- [x] TOOLS.md: Template with link to README
- [x] All examples validated against implementation

### ✅ Version Consistency
- [x] pyproject.toml: `version = "0.0.0"`
- [x] src/yt_mcp/__init__.py: `__version__ = "0.0.0"`
- [x] CHANGELOG.md: Version 0.0.0 documented

---

## Publishing Plan

### Phase 1: PyPI Publishing

**Prerequisites:**
- PyPI account with API token
- Token stored securely (not in repo)
- Package name `yt-mcp` available on PyPI

**Steps:**
1. **Build Python Package**
   ```bash
   cd yt-mcp
   uv build
   ```
   - Validates pyproject.toml configuration
   - Creates `dist/yt_mcp-0.0.0.tar.gz`
   - Creates `dist/yt_mcp-0.0.0-py3-none-any.whl`

2. **Test Package Build (Local Install)**
   ```bash
   # Create clean test environment
   uv venv .venv-test
   source .venv-test/bin/activate
   uv pip install dist/yt_mcp-0.0.0-py3-none-any.whl

   # Verify installation
   python -c "import yt_mcp; print(yt_mcp.__version__)"
   yt-mcp --help

   # Cleanup
   deactivate
   rm -rf .venv-test
   ```

3. **Publish to PyPI**
   ```bash
   # Using uv (requires TWINE_USERNAME and TWINE_PASSWORD env vars)
   # OR manually with twine:
   uv pip install twine
   twine upload dist/yt_mcp-0.0.0*
   ```
   - Enter PyPI API token when prompted
   - Verify upload success: https://pypi.org/project/yt-mcp/0.0.0/

4. **Validate PyPI Package**
   ```bash
   # Fresh environment
   uv venv .venv-pypi-test
   source .venv-pypi-test/bin/activate
   uv pip install yt-mcp==0.0.0

   # Test basic functionality
   python -c "from yt_mcp.server import main; print('Import successful')"

   deactivate
   rm -rf .venv-pypi-test
   ```

**Expected Outcomes:**
- Package available at: `https://pypi.org/project/yt-mcp/0.0.0/`
- Users can install with: `pip install yt-mcp==0.0.0`
- Package metadata displays correctly on PyPI

---

### Phase 2: GHCR Publishing

**Prerequisites:**
- GitHub account with package write permissions
- GHCR PAT (Personal Access Token) with `write:packages` scope
- Docker logged in to ghcr.io

**Steps:**
1. **Login to GHCR**
   ```bash
   echo $GITHUB_TOKEN | docker login ghcr.io -u l4b4r4b4b4 --password-stdin
   ```

2. **Tag Docker Images**
   ```bash
   cd yt-mcp

   # Base image (for development)
   docker tag yt-mcp:base ghcr.io/l4b4r4b4b4/yt-mcp:0.0.0-base
   docker tag yt-mcp:base ghcr.io/l4b4r4b4b4/yt-mcp:latest-base

   # Production image (main)
   docker tag yt-mcp:latest ghcr.io/l4b4r4b4b4/yt-mcp:0.0.0
   docker tag yt-mcp:latest ghcr.io/l4b4r4b4b4/yt-mcp:latest
   ```

3. **Push to GHCR**
   ```bash
   # Push base images
   docker push ghcr.io/l4b4r4b4b4/yt-mcp:0.0.0-base
   docker push ghcr.io/l4b4r4b4b4/yt-mcp:latest-base

   # Push production images
   docker push ghcr.io/l4b4r4b4b4/yt-mcp:0.0.0
   docker push ghcr.io/l4b4r4b4b4/yt-mcp:latest
   ```

4. **Make Package Public**
   - Navigate to: https://github.com/users/l4b4r4b4b4/packages/container/yt-mcp/settings
   - Change visibility to "Public"
   - Verify images are accessible without authentication

5. **Validate GHCR Images**
   ```bash
   # Remove local images
   docker rmi yt-mcp:latest
   docker rmi yt-mcp:base

   # Pull from GHCR
   docker pull ghcr.io/l4b4r4b4b4/yt-mcp:0.0.0

   # Test container
   docker run --rm \
     -e YOUTUBE_API_KEY=test_key \
     ghcr.io/l4b4r4b4b4/yt-mcp:0.0.0 \
     yt-mcp --help
   ```

**Expected Outcomes:**
- Images available at: `ghcr.io/l4b4r4b4b4/yt-mcp`
- Four tags published: `0.0.0`, `latest`, `0.0.0-base`, `latest-base`
- Public visibility (no auth required to pull)
- README displayed on GHCR package page

---

### Phase 3: Git Tagging & GitHub Release

**Steps:**
1. **Create Git Tag**
   ```bash
   cd yt-mcp
   git tag -a v0.0.0 -m "Release version 0.0.0 - First experimental release"
   git push origin v0.0.0
   ```

2. **Create GitHub Release**
   - Navigate to: https://github.com/l4b4r4b4b4/mcp-refcache/releases/new
   - Tag: `v0.0.0`
   - Title: `YouTube MCP Server v0.0.0 - First Experimental Release`
   - Description: Copy from CHANGELOG.md (0.0.0 section)
   - Mark as "pre-release" (this is 0.0.0)
   - Attach build artifacts (optional): `dist/yt_mcp-0.0.0*`

3. **Verify Release**
   - Check release page displays correctly
   - Verify CHANGELOG excerpt is readable
   - Test download links work

**Expected Outcomes:**
- Tag `v0.0.0` exists in repository
- GitHub release published and marked as pre-release
- Release notes visible on project page

---

## Success Criteria

### Publishing Success
- [ ] PyPI package published: `yt-mcp==0.0.0`
- [ ] PyPI page displays correctly with metadata
- [ ] GHCR images published with all 4 tags
- [ ] GHCR package is public
- [ ] Git tag `v0.0.0` pushed
- [ ] GitHub release created and visible

### Installation Success
- [ ] `pip install yt-mcp==0.0.0` works in clean environment
- [ ] `docker pull ghcr.io/l4b4r4b4b4/yt-mcp:0.0.0` works
- [ ] Installed package version reports `0.0.0`
- [ ] Docker container runs and responds to commands

### Documentation Success
- [ ] PyPI README renders correctly
- [ ] GHCR package page shows description
- [ ] GitHub release notes are complete
- [ ] All installation instructions are accurate

---

## Rollback Plan

If critical issues are discovered after publishing:

1. **DO NOT delete PyPI packages** (permanent record)
2. **DO NOT delete GHCR images** (users may depend on them)
3. **Instead:**
   - Document issues in GitHub release notes (edit)
   - Publish `0.0.1` with fixes immediately
   - Update README with known issues and upgrade path
   - Mark 0.0.0 as deprecated in CHANGELOG

**Rationale:** Version 0.0.0 signals "experimental" - users expect issues. Quick iteration to 0.0.1 is better than trying to hide mistakes.

---

## Known Limitations (Pre-Publishing)

These are expected for version 0.0.0 and documented in CHANGELOG:

1. **API Quota:** No built-in quota monitoring (user must track via Google Cloud Console)
2. **Rate Limiting:** No automatic retry with backoff (users handle HTTP 429)
3. **Live Chat:** Manual polling required (no automatic monitoring)
4. **Transcript Languages:** Basic selection (no auto-translation)
5. **Error Context:** Generic errors without detailed troubleshooting

These will be addressed in future releases (0.0.x, 0.1.0+).

---

## Post-Publishing Tasks

After successful publishing:

1. **Update Main Scratchpad**
   - Mark Task 12 as 🟠 Implemented
   - Note: Awaiting user validation in clean environment

2. **Create Task 13 Scratchpad**
   - Testing published versions
   - Clean environment validation
   - Real-world usage scenarios

3. **User Validation Needed**
   - Install from PyPI in fresh environment
   - Pull Docker image and test
   - Verify all setup instructions work
   - Test example workflows

4. **Mark Complete (🟢) Only After:**
   - User confirms PyPI package installs correctly
   - User confirms Docker image works
   - User validates setup instructions
   - Real-world testing shows no critical issues

---

## Risk Assessment

### Low Risk
- Build process (validated locally multiple times)
- Docker image quality (tested extensively)
- Documentation accuracy (all examples validated)

### Medium Risk
- PyPI publishing workflow (first time for this project)
- GHCR visibility settings (ensure public access)
- Package metadata rendering (PyPI README formatting)

### Mitigation Strategies
- Dry-run builds before publishing
- Test in clean environments after each step
- Verify public access before announcing
- Have 0.0.1 ready for quick iteration if needed

---

## Timeline

Estimated time: 1-2 hours

1. **Phase 1 (PyPI):** 30-45 minutes
   - Build: 5 minutes
   - Local validation: 10 minutes
   - Upload: 5 minutes
   - PyPI validation: 10 minutes

2. **Phase 2 (GHCR):** 20-30 minutes
   - Tagging: 5 minutes
   - Upload: 10 minutes (depends on network)
   - Validation: 10 minutes

3. **Phase 3 (Git/GitHub):** 10-15 minutes
   - Tagging: 2 minutes
   - Release notes: 8 minutes

4. **Buffer:** 15-30 minutes for unexpected issues

---

## Notes

- This is the first release - expect to learn and iterate
- Version 0.0.0 explicitly signals "experimental test release"
- PyPI is permanent - can't delete, only add newer versions
- GHCR images can be deleted but shouldn't be (users may depend)
- Success = published + basic validation, NOT production-ready
- User validation in Task 13 will inform 0.0.1 improvements

---

**Next Steps:**
1. Gather PyPI and GHCR credentials
2. Execute Phase 1 (PyPI publishing)
3. Execute Phase 2 (GHCR publishing)
4. Execute Phase 3 (Git tagging and GitHub release)
5. Update scratchpads and prepare for Task 13
