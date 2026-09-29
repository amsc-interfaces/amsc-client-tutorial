# ALCF RIG Connection Instructions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tested, secret-safe ALCF Tier-2 connection and recovery instructions near the top of the ALCF-via-RIG tutorial notebook.

**Architecture:** Add one Markdown-only setup section after the notebook introduction. Extend the existing static notebook contract so future edits cannot omit the active-project check, ALCF connection controls, correct Argonne identity, RIG verification gate, AmSC PAT distinction, recovery sequence, or credential hygiene.

**Tech Stack:** Jupyter Notebook JSON, Python, pytest.

---

### Task 1: Add the documentation contract

**Files:**
- Modify: `tests/test_rig_tutorial_content.py`

- [ ] **Step 1: Write a failing test**

Add `test_alcf_rig_notebook_documents_connection_setup_and_recovery`, loading `alcf_via_rig.ipynb` and asserting the approved vault URL, active-project guidance, connection controls, Argonne identity, connected and RIG-verification gates, AmSC PAT distinction, high-assurance recovery, and credential hygiene.

- [ ] **Step 2: Verify RED**

Run:

```bash
../fix-rig-live-failures/.venv/bin/python -m pytest -q tests/test_rig_tutorial_content.py::test_alcf_rig_notebook_documents_connection_setup_and_recovery
```

Expected: FAIL because the notebook does not yet contain the detailed setup section.

### Task 2: Add the notebook instructions

**Files:**
- Modify: `notebooks_via_rig/alcf_via_rig.ipynb`

- [ ] **Step 1: Insert one Markdown cell after the introduction**

Include all approved credential-domain, project, connection, verification, recovery, and safety guidance. Do not add executable credential handling.

- [ ] **Step 2: Verify GREEN**

Run the focused test from Task 1 and require PASS.

- [ ] **Step 3: Verify the repository**

Run:

```bash
../fix-rig-live-failures/.venv/bin/python -m pytest -q
```

Then parse and compile every notebook; require empty outputs, null execution counts, and no JWT-like token material.

- [ ] **Step 4: Commit**

```bash
git add tests/test_rig_tutorial_content.py notebooks_via_rig/alcf_via_rig.ipynb docs/superpowers/specs/2026-09-29-alcf-rig-connection-instructions-design.md docs/superpowers/plans/2026-09-29-alcf-rig-connection-instructions.md
git commit -m "docs: explain ALCF RIG connection setup"
```

### Task 3: Review and publish

- [ ] **Step 1: Request independent exact-commit review**

Require review of credential correctness, secret safety, notebook cleanliness, and test adequacy.

- [ ] **Step 2: Push and open a GitHub PR**

Push `docs/alcf-rig-connection-instructions`, create a PR against `main`, verify the PR head/files, wait for CI, merge, and verify post-merge CI.
