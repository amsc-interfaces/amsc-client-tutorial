# amsc-client 0.6 Tutorial Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update every tutorial notebook and guide to the published `amsc-client==0.6.0` API and mixed-authentication model, with automated static checks and opt-in read-only live verification.

**Architecture:** Treat tutorials as executable artifacts. A repository validator parses notebooks and Markdown, compiles code cells, checks links and banned legacy patterns, and imports the exact published package. A separate opt-in smoke script verifies protected central and facility reads without mutations.

**Tech Stack:** Python 3.11, pytest, nbformat, GitHub Actions, `amsc-client==0.6.0`, `amsc-auth`, direct ALCF IRI v1.

---

### Task 1: Establish exact dependency and validation harness

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `tests/test_tutorial_content.py`
- Create: `.github/workflows/validate.yml`

- [ ] Write tests asserting the exact 0.6.0 pin, required registries, notebook JSON validity, compilable Python cells, cleared outputs, valid README notebook links, and absence of known legacy patterns.
- [ ] Run `pytest -q` and confirm failures identify the current `<0.5` cap, stale links, outputs, and obsolete APIs.
- [ ] Pin `amsc-client==0.6.0`, add validation dependencies, and add GitHub Actions running the same clean-install and pytest commands.
- [ ] Re-run focused dependency/structural checks and commit the green subset.

### Task 2: Migrate central-service notebooks

**Files:**
- Modify: `notebooks/catalog_explorer.ipynb`
- Modify: `notebooks/catalog_tutorial.ipynb`

- [ ] Extend tests to require staging `AMSC_TOKEN` setup, reject ID-token/old Globus central-auth arguments, and require write gating in the CRUD notebook.
- [ ] Run focused tests and verify RED.
- [ ] Rewrite notebook setup and flow for an AmSC Keycard; make exploration population-independent and CRUD explicitly opt-in with deterministic cleanup.
- [ ] Clear outputs and execution counts, run focused tests, and commit.

### Task 3: Migrate facility notebooks

**Files:**
- Modify: `notebooks/alcf_facility_tutorial.ipynb`
- Modify: `notebooks/nersc_facility_tutorial.ipynb`
- Modify: `notebooks/filesystem_tutorial.ipynb`

- [ ] Extend tests to require explicit credential-domain language, built-in NERSC usage, parameterized user/account values, and only the supported 0.6.0 filesystem surface.
- [ ] Run focused tests and verify RED.
- [ ] Rewrite ALCF and NERSC notebooks around built-in native facility authenticators, with submissions opt-in.
- [ ] Rewrite filesystem examples around resolved compatibility tasks; remove `view`, `file`, live polling, and task cancellation assumptions.
- [ ] Clear outputs/execution counts, run focused tests, and commit.

### Task 4: Correct README and long-form guides

**Files:**
- Modify: `README.md`
- Modify: `docs/agentic-guide-to-polaris-with-iri.md`
- Modify: `docs/pytorch-distributed-training-on-polaris.md`

- [ ] Extend tests for the current notebook inventory, 0.6.0 installation, mixed-auth terminology, no personal executable defaults, and removal of obsolete capability claims.
- [ ] Run focused tests and verify RED.
- [ ] Rewrite README setup, tutorial ordering, supported-facility guidance, and troubleshooting.
- [ ] Update both guides to 0.6.0 auth setup; correct IRI operation claims and label dated HPC-stack observations.
- [ ] Run focused tests and commit.

### Task 5: Add read-only mixed-auth smoke test

**Files:**
- Create: `scripts/smoke_mixed_auth.py`
- Create: `tests/test_smoke_script.py`
- Modify: `README.md`

- [ ] Write tests that import the smoke module with no credentials, assert missing credentials fail safely, and verify requests are limited to read-only protected endpoints.
- [ ] Run tests and verify RED.
- [ ] Implement endpoint preflight, central `/account/me`, direct ALCF `/account/projects`, and one-client authenticator-isolation checks without logging tokens.
- [ ] Document invocation and expected environment variables.
- [ ] Run tests and commit.

### Task 6: Full clean-environment and live verification

**Files:**
- Modify only if verification exposes a documented defect.

- [ ] Create a clean Python 3.11 environment and install from all documented GitLab registries.
- [ ] Verify `amsc-client==0.6.0` and run the full pytest suite.
- [ ] Run notebook compilation and public API introspection checks.
- [ ] Run `scripts/smoke_mixed_auth.py` with the independently supplied staging and ALCF credentials; do not run writes or submissions.
- [ ] Run `git diff --check`, inspect the full diff, and scan for secrets.

### Task 7: Review and publish PR

**Files:**
- All modified files.

- [ ] Perform an adversarial review against the approved specification and fix Important/Critical findings.
- [ ] Re-run full verification after fixes.
- [ ] Push `docs/migrate-amsc-client-0.6` and open a GitHub PR with release/tag evidence and exact test results.
- [ ] Verify the remote branch SHA, PR head SHA, and GitHub Actions status before reporting completion.
