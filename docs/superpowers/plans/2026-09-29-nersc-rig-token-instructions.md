# NERSC RIG Token Instructions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tested, secret-safe NERSC IRI token setup and recovery instructions near the top of the NERSC-via-RIG tutorial notebook.

**Architecture:** Add one Markdown-only setup section after the notebook introduction. Extend the existing static notebook contract so future edits cannot omit the exact helper, required scope, direct-validation gate, project selection, vault rotation, RIG verification, credential-domain distinction, or inactive-token recovery.

**Tech Stack:** Jupyter Notebook JSON, Python, pytest.

---

### Task 1: Add the documentation contract

**Files:**
- Modify: `tests/test_rig_tutorial_content.py`

- [ ] **Step 1: Write a failing test**

Add `test_nersc_rig_notebook_documents_token_setup_and_recovery`, loading `nersc_via_rig.ipynb` and asserting the approved URLs, helper flags, NERSC scope, validation success phrase, project, Rotate, Verify via RIG, AmSC PAT distinction, and inactive/invalid-token recovery language.

- [ ] **Step 2: Verify RED**

Run:

```bash
../fix-rig-live-failures/.venv/bin/python -m pytest -q tests/test_rig_tutorial_content.py::test_nersc_rig_notebook_documents_token_setup_and_recovery
```

Expected: FAIL because the notebook does not yet contain the detailed setup section.

### Task 2: Add the notebook instructions

**Files:**
- Modify: `notebooks_via_rig/nersc_via_rig.ipynb`

- [ ] **Step 1: Insert one Markdown cell after the introduction**

Include the exact NERSC-only helper invocation and all approved credential-safety, validation, vault, project, and recovery guidance. Do not add executable token handling.

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
git add tests/test_rig_tutorial_content.py notebooks_via_rig/nersc_via_rig.ipynb docs/superpowers/specs/2026-09-29-nersc-rig-token-instructions-design.md docs/superpowers/plans/2026-09-29-nersc-rig-token-instructions.md
git commit -m "docs: explain NERSC RIG credential setup"
```

### Task 3: Review and publish

- [ ] **Step 1: Request independent exact-commit review**

Require review of credential correctness, secret safety, notebook cleanliness, and test adequacy.

- [ ] **Step 2: Push and open a GitHub PR**

Push `docs/nersc-rig-token-instructions`, create a PR against `main`, verify the PR head/files, and wait for CI.
