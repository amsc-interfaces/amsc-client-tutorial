# Authentication Guidance and Validation Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make credential handling and live-validation scope clear in the README and every tutorial notebook.

**Architecture:** Add static assertions first, then update the README with one authoritative authentication/validation matrix and add concise notebook-specific banners. Keep facility behavior unchanged and make only evidence-backed validation claims.

**Tech Stack:** Markdown, Jupyter Notebook JSON, Python 3.11, pytest, nbformat.

---

### Task 1: Add failing validation-status tests

**Files:**
- Modify: `tests/test_tutorial_content.py`

- [ ] Add tests requiring the README to state that AmSC staging and ALCF authentication/read-only paths were live-validated on 2026-09-22.
- [ ] Add tests requiring NERSC to be labeled built-in but not yet live-validated by the maintainers.
- [ ] Add tests requiring OLCF to be labeled as having no built-in `amsc-client 0.6.0` integration or tutorial.
- [ ] Add parameterized tests requiring each notebook to contain its designated authentication and validation banner.
- [ ] Run `python -m pytest tests/test_tutorial_content.py -q` and confirm the new tests fail for missing language.
- [ ] Commit with `test: require authentication validation guidance`.

### Task 2: Update README authentication guidance

**Files:**
- Modify: `README.md`

- [ ] Add an authentication quick-reference explaining `AMSC_TOKEN`, independent facility-native Globus authentication, browser/device login expectations, cache recovery, and the prohibition on storing tokens in notebooks.
- [ ] Add the dated validation matrix separating integration, live authentication/read-only validation, and mutation/job validation.
- [ ] State that public facility discovery does not prove authentication.
- [ ] State that NERSC has not yet been live-validated and that OLCF is not built into `amsc-client 0.6.0` or covered by a tutorial.
- [ ] Run focused README tests and confirm they pass.
- [ ] Commit with `docs: clarify authentication and validation status`.

### Task 3: Add per-notebook guidance

**Files:**
- Modify: `notebooks/catalog_explorer.ipynb`
- Modify: `notebooks/catalog_tutorial.ipynb`
- Modify: `notebooks/alcf_facility_tutorial.ipynb`
- Modify: `notebooks/filesystem_tutorial.ipynb`
- Modify: `notebooks/nersc_facility_tutorial.ipynb`

- [ ] Add concise authentication/validation banners matching the design specification.
- [ ] Preserve code, default-off gates, cleared outputs, and null execution counts.
- [ ] Run focused notebook tests and confirm they pass.
- [ ] Commit with `docs: add notebook authentication guidance`.

### Task 4: Verify and submit

**Files:**
- Verify all modified files.

- [ ] Install and test from the clean Python 3.11 environment containing the published wheel.
- [ ] Run `python -m pytest tests/ -q`.
- [ ] Compile every notebook code cell and Markdown Python fence.
- [ ] Verify outputs/execution counts are cleared and run `git diff --check`.
- [ ] Scan changed content for credential patterns.
- [ ] Perform independent read-only review and resolve Critical/Important findings.
- [ ] Push the branch, open a GitHub PR, and wait for CI without merging automatically.
