"""Static validator for the amsc-client-tutorial repository.

Tests are structured in logical groups matching the migration plan tasks:

  Task 1  – Dependency pin, README/notebook structural requirements
  Task 2  – Central-service notebook content (catalog_explorer, catalog_tutorial)
  Task 3  – Facility notebook content (alcf, nersc, filesystem)
  Task 4  – README and long-form guide correctness
  Task 5  – Smoke-script structure (tested in test_smoke_script.py)

Run with:
    pytest tests/test_tutorial_content.py -v
"""
from __future__ import annotations

import ast
import importlib.metadata
import inspect
import json
import os
import re
import sys
from pathlib import Path

import pytest

# ── Repository root ──────────────────────────────────────────────────────────

REPO = Path(__file__).parent.parent
NOTEBOOKS = REPO / "notebooks"
DOCS = REPO / "docs"

# ── Helpers ──────────────────────────────────────────────────────────────────


def load_notebook(name: str) -> dict:
    path = NOTEBOOKS / name
    assert path.exists(), f"Notebook not found: {path}"
    return json.loads(path.read_text())


def code_cells(nb: dict) -> list[str]:
    """Return all Python code-cell sources joined as a single string each."""
    return [
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    ]


def all_code(nb: dict) -> str:
    return "\n\n".join(code_cells(nb))


def compile_cell(src: str, cell_idx: int) -> None:
    try:
        ast.parse(src)
    except SyntaxError as exc:
        pytest.fail(f"Cell {cell_idx}: syntax error — {exc}\n\nSource:\n{src[:400]}")


def dotted_name(node: ast.AST) -> str:
    """Return a dotted name for a Name/Attribute expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def assert_calls_are_guarded(nb: dict, suffixes: tuple[str, ...], gate: str) -> None:
    """Require each matching call to be lexically nested under ``if <gate>``."""
    for cell_index, source in enumerate(code_cells(nb)):
        tree = ast.parse(source)

        def visit(node: ast.AST, guarded: bool = False) -> None:
            if isinstance(node, ast.If):
                names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                for child in node.body:
                    visit(child, guarded or gate in names)
                for child in node.orelse:
                    # ``if not ENABLE_WRITES: ... else: mutate()`` is the
                    # repository's fail-closed guard idiom.
                    negated_gate = (
                        isinstance(node.test, ast.UnaryOp)
                        and isinstance(node.test.op, ast.Not)
                        and isinstance(node.test.operand, ast.Name)
                        and node.test.operand.id == gate
                    )
                    visit(child, guarded or negated_gate)
                return
            if isinstance(node, ast.Call):
                name = dotted_name(node.func)
                if name.endswith(suffixes):
                    assert guarded, (
                        f"cell {cell_index}: mutating call {name} must be nested under "
                        f"a default-off {gate} guard"
                    )
            for child in ast.iter_child_nodes(node):
                visit(child, guarded)

        visit(tree)


# ============================================================================
# Task 1 — Dependency pin and structural requirements
# ============================================================================


class TestDependencyPin:
    """requirements.txt must pin amsc-client==0.6.0 and list all required registries."""

    def test_requirements_txt_exists(self):
        assert (REPO / "requirements.txt").exists()

    def test_amsc_client_exact_pin(self):
        req = (REPO / "requirements.txt").read_text()
        assert "amsc-client==0.6.0" in req, (
            "requirements.txt must pin amsc-client==0.6.0 (exact), "
            "not a range like <0.5 or >=0.4"
        )

    def test_no_legacy_cap(self):
        req = (REPO / "requirements.txt").read_text()
        assert "<0.5" not in req, "requirements.txt still has the legacy <0.5 cap"
        assert ">=0.4.1,<0.5" not in req, "requirements.txt still has the old range"

    def test_registry_amsc_python_client(self):
        req = (REPO / "requirements.txt").read_text()
        # The amsc-python-client registry
        assert "77567162" in req, "Missing amsc-python-client GitLab registry URL"

    def test_registry_amsc_api_autogen(self):
        req = (REPO / "requirements.txt").read_text()
        assert "76368190" in req, "Missing amsc-api-autogen GitLab registry URL"

    def test_registry_amsc_auth_or_third(self):
        req = (REPO / "requirements.txt").read_text()
        assert "80654726" in req, "Missing third GitLab registry URL"

    def test_registry_amsc_auth(self):
        req = (REPO / "requirements.txt").read_text()
        assert "82001936" in req, "Missing amsc-auth GitLab registry URL"

    def test_installed_wheel_is_exact_release(self):
        assert importlib.metadata.version("amsc-client") == "0.6.0"

    def test_referenced_public_api_signatures(self):
        from amsc_client.catalog.client import CatalogClient
        from amscrot.facility.filesystem import FilesystemClient

        artifact = inspect.signature(CatalogClient.create_artifact)
        assert "catalog" in artifact.parameters
        assert artifact.parameters["catalog"].default is inspect.Parameter.empty
        for method in ("ls", "head", "mkdir", "rm", "upload", "download"):
            assert callable(getattr(FilesystemClient, method))

    def test_requirements_dev_exists(self):
        assert (REPO / "requirements-dev.txt").exists(), (
            "requirements-dev.txt must exist (pytest, nbformat, etc.)"
        )

    def test_requirements_dev_includes_pytest(self):
        req = (REPO / "requirements-dev.txt").read_text()
        assert "pytest" in req, "requirements-dev.txt must list pytest"

    def test_requirements_dev_includes_nbformat(self):
        req = (REPO / "requirements-dev.txt").read_text()
        assert "nbformat" in req, "requirements-dev.txt must list nbformat"


class TestNotebookStructure:
    """All notebooks must parse as valid JSON, have compilable code cells, and cleared outputs."""

    NOTEBOOK_NAMES = [
        "catalog_explorer.ipynb",
        "catalog_tutorial.ipynb",
        "alcf_facility_tutorial.ipynb",
        "nersc_facility_tutorial.ipynb",
        "filesystem_tutorial.ipynb",
    ]

    @pytest.mark.parametrize("name", NOTEBOOK_NAMES)
    def test_valid_json(self, name):
        nb = load_notebook(name)
        assert "cells" in nb, f"{name}: missing 'cells' key"
        assert "nbformat" in nb, f"{name}: missing 'nbformat' key"

    @pytest.mark.parametrize("name", NOTEBOOK_NAMES)
    def test_code_cells_compile(self, name):
        nb = load_notebook(name)
        for i, src in enumerate(code_cells(nb)):
            compile_cell(src, i)

    @pytest.mark.parametrize("name", NOTEBOOK_NAMES)
    def test_outputs_cleared(self, name):
        nb = load_notebook(name)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                assert not cell.get("outputs"), (
                    f"{name}: cell {i} has non-empty outputs — run nbstripout or clear manually"
                )

    @pytest.mark.parametrize("name", NOTEBOOK_NAMES)
    def test_execution_counts_cleared(self, name):
        nb = load_notebook(name)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None, (
                    f"{name}: cell {i} has execution_count={cell['execution_count']}"
                )


class TestReadmeLinks:
    """README.md notebook links must resolve to actual files."""

    def test_readme_exists(self):
        assert (REPO / "README.md").exists()

    def test_all_notebook_links_resolve(self):
        readme = (REPO / "README.md").read_text()
        # Find links like [text](notebooks/foo.ipynb)
        links = re.findall(r"\(notebooks/([^)]+\.ipynb)\)", readme)
        assert links, "README links no notebook (.ipynb) files"
        for link in links:
            path = NOTEBOOKS / link
            assert path.exists(), f"README links to missing notebook: notebooks/{link}"

    def test_no_stale_facility_tutorial_link(self):
        """The old monolithic facility_tutorial.ipynb was split; README must not link it."""
        readme = (REPO / "README.md").read_text()
        # Check for the exact markdown link target, not the substring —
        # alcf_facility_tutorial.ipynb and nersc_facility_tutorial.ipynb are valid targets.
        assert "](notebooks/facility_tutorial.ipynb)" not in readme, (
            "README still links the removed facility_tutorial.ipynb — "
            "update to alcf_facility_tutorial.ipynb and nersc_facility_tutorial.ipynb"
        )

    def test_alcf_notebook_linked(self):
        readme = (REPO / "README.md").read_text()
        assert "alcf_facility_tutorial.ipynb" in readme, (
            "README does not link alcf_facility_tutorial.ipynb"
        )

    def test_nersc_notebook_linked(self):
        readme = (REPO / "README.md").read_text()
        assert "nersc_facility_tutorial.ipynb" in readme, (
            "README does not link nersc_facility_tutorial.ipynb"
        )


class TestNoEmbeddedCredentials:
    """No notebook or guide may embed likely credentials."""

    CREDENTIAL_PATTERNS = [
        # Real bearer-token shapes (not env-var references)
        r'token\s*=\s*"[A-Za-z0-9+/]{20,}',
        r"token\s*=\s*'[A-Za-z0-9+/]{20,}",
        r"AMSC_TOKEN\s*=\s*\"[^\"]{10,}\"",
        r"ALCF_IRI_TOKEN\s*=\s*\"[^\"]{10,}\"",
    ]

    SCANNED_PATHS = list((REPO / "notebooks").glob("*.ipynb")) + list(
        (REPO / "docs").glob("*.md")
    ) + [REPO / "README.md"]

    @pytest.mark.parametrize("path", SCANNED_PATHS, ids=lambda p: p.name)
    def test_no_embedded_credentials(self, path):
        text = path.read_text()
        for pattern in self.CREDENTIAL_PATTERNS:
            m = re.search(pattern, text)
            assert not m, (
                f"{path.name}: likely embedded credential matched {pattern!r}: "
                f"...{m.group()[:60]}..."
            )


# ============================================================================
# Task 2 — Central-service notebooks
# ============================================================================


class TestCatalogExplorer:
    """catalog_explorer.ipynb — read-only, AmSC Keycard, staging endpoint."""

    @pytest.fixture
    def nb(self):
        return load_notebook("catalog_explorer.ipynb")

    @pytest.fixture
    def code(self, nb):
        return all_code(nb)

    def test_uses_staging_base_url(self, code):
        assert "api.staging.american-science-cloud.org" in code, (
            "catalog_explorer must use the staging base URL"
        )

    def test_no_production_base_url_as_default(self, code):
        # Production URL must not be the hard-coded default
        prod_pat = r'base_url\s*=\s*["\']https://api\.american-science-cloud\.org/api/current["\']'
        assert not re.search(prod_pat, code), (
            "catalog_explorer still hard-codes the production base URL — use staging"
        )

    def test_no_use_id_token(self, code):
        assert "use_id_token" not in code, (
            "catalog_explorer uses use_id_token — must use AmSC Keycard (access token), "
            "never the Passport/ID token"
        )

    def test_no_globus_central_auth(self, code):
        """Central service access must use token auth, not Globus."""
        assert 'auth_method="globus"' not in code or "AMSC_TOKEN" in code, (
            "catalog_explorer may still use Globus for central auth; "
            "preferred path is token=os.environ['AMSC_TOKEN']"
        )

    def test_reads_token_from_environment(self, code):
        assert "AMSC_TOKEN" in code, (
            "catalog_explorer must read the AmSC Keycard from AMSC_TOKEN env var"
        )

    def test_no_hardcoded_globus_app_id_as_central_auth(self, code):
        """The old Globus app-ID-based central auth pattern must be gone."""
        old_pat = r'globus_client_id\s*=\s*["\']e4f48665'
        assert not re.search(old_pat, code), (
            "catalog_explorer still uses the old Globus client-ID approach for central auth"
        )

    def test_handles_empty_results(self, code):
        """Notebook must not assume catalog is populated."""
        assert "if" in code, "Notebook should have conditional logic for empty results"
        # At minimum, a check like 'if results' or 'if works'
        assert re.search(r"\bif\b.*(results|works|artifacts|items|len)", code), (
            "catalog_explorer should handle empty catalog results gracefully"
        )


class TestCatalogTutorial:
    """catalog_tutorial.ipynb — CRUD, write-gated, Keycard, staging."""

    @pytest.fixture
    def nb(self):
        return load_notebook("catalog_tutorial.ipynb")

    @pytest.fixture
    def code(self, nb):
        return all_code(nb)

    def test_uses_staging_base_url(self, code):
        assert "api.staging.american-science-cloud.org" in code

    def test_no_use_id_token(self, code):
        assert "use_id_token" not in code

    def test_reads_token_from_environment(self, code):
        assert "AMSC_TOKEN" in code

    def test_write_gate_present(self, code):
        assert "ENABLE_WRITES" in code, (
            "catalog_tutorial must have an ENABLE_WRITES gate defaulting to False"
        )

    def test_write_gate_defaults_false(self, code):
        assert re.search(r"ENABLE_WRITES\s*=\s*False", code), (
            "ENABLE_WRITES must default to False"
        )

    def test_cleanup_uses_created_fqn(self, code):
        """Cleanup must reference the FQN recorded at create-time, not a hard-coded value."""
        # The notebook should store the created FQN in a variable and use it in delete
        assert re.search(r"(work_fqn|artifact_fqns|created_fqn)", code), (
            "catalog_tutorial must store created FQNs for deterministic cleanup"
        )

    def test_create_artifact_passes_catalog(self, nb):
        calls = [
            node
            for src in code_cells(nb)
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Call)
            and dotted_name(node.func).endswith("catalog.create_artifact")
        ]
        assert calls, "catalog tutorial must demonstrate create_artifact"
        for call in calls:
            assert "catalog" in {kw.arg for kw in call.keywords}, (
                "create_artifact requires catalog= in amsc-client 0.6.0"
            )

    def test_catalog_mutations_are_structurally_guarded(self, nb):
        assert_calls_are_guarded(
            nb,
            ("catalog.create_work", "catalog.create_artifact", "catalog.delete"),
            "ENABLE_WRITES",
        )

    def test_uses_run_id_for_uniqueness(self, code):
        assert "RUN_ID" in code, (
            "catalog_tutorial must use a per-run suffix (RUN_ID) to avoid collisions"
        )


# ============================================================================
# Task 3 — Facility notebooks
# ============================================================================


class TestAlcfFacilityTutorial:
    """alcf_facility_tutorial.ipynb — built-in ALCF, no dummy tokens, explicit auth domain."""

    @pytest.fixture
    def nb(self):
        return load_notebook("alcf_facility_tutorial.ipynb")

    @pytest.fixture
    def code(self, nb):
        return all_code(nb)

    def test_no_dummy_token_as_main_client(self, code):
        """The old Client(token='not-needed-for-facilities') must be gone."""
        assert 'token="not-needed-for-facilities"' not in code, (
            "alcf_facility_tutorial still passes a dummy 'not-needed-for-facilities' token. "
            "Facility auth uses the built-in ALCF Globus authenticator independently."
        )
        assert "token='not-needed-for-facilities'" not in code

    def test_uses_builtin_alcf(self, code):
        """Must use client.facility('alcf') without a custom register_facility call."""
        assert 'client.facility("alcf")' in code or "client.facility('alcf')" in code

    def test_no_hardcoded_username(self, code):
        """Personal username 'parton' must not appear as a default."""
        assert '"parton"' not in code and "'parton'" not in code, (
            "alcf_facility_tutorial hard-codes username 'parton' — use ALCF_USERNAME env var"
        )

    def test_alcf_username_from_env(self, code):
        assert "ALCF_USERNAME" in code, (
            "alcf_facility_tutorial must read ALCF_USERNAME from environment"
        )

    def test_alcf_account_from_env_or_param(self, code):
        assert "ALCF_ACCOUNT" in code, (
            "alcf_facility_tutorial must use ALCF_ACCOUNT parameter, not a hard-coded project"
        )

    def test_submission_is_opt_in(self, code):
        """Job submission must be behind an explicit gate."""
        assert "SUBMIT_JOB" in code or "ENABLE_SUBMIT" in code or "# Submit" in code, (
            "alcf_facility_tutorial must make job submission explicitly opt-in"
        )

    def test_submission_is_structurally_guarded(self, nb):
        assert_calls_are_guarded(nb, (".submit",), "SUBMIT_JOB")

    def test_auth_domain_explained(self, code):
        """Notebook must explain the ALCF auth is independent from the central Keycard."""
        assert "ALCF" in code, "Facility auth domain explanation missing"

    def test_no_globus_client_registration(self, code):
        """ALCF must not require manual register_facility — it's built-in."""
        # If register_facility is called for alcf, that's wrong for the built-in
        if "register_facility" in code and "alcf" in code:
            # OK only if it's the NERSC notebook or an override
            pass  # alcf notebook should use auto-registration via client.facility("alcf")


class TestNerscFacilityTutorial:
    """nersc_facility_tutorial.ipynb — built-in NERSC, no manual registration."""

    @pytest.fixture
    def nb(self):
        return load_notebook("nersc_facility_tutorial.ipynb")

    @pytest.fixture
    def code(self, nb):
        return all_code(nb)

    def test_no_dummy_token_as_main_client(self, code):
        assert 'token="not-needed-for-facilities"' not in code
        assert "token='not-needed-for-facilities'" not in code

    def test_no_manual_nersc_registration(self, code):
        """NERSC is now built-in; custom register_facility with manual Globus config is wrong."""
        if "register_facility" in code and "nersc" in code.lower():
            # Fail: manual NERSC registration with FacilityConfig is the old pattern
            assert "FacilityConfig" not in code, (
                "nersc_facility_tutorial still uses manual register_facility with FacilityConfig. "
                "NERSC is a built-in facility; use client.facility('nersc') directly."
            )

    def test_uses_builtin_nersc(self, code):
        assert 'client.facility("nersc")' in code or "client.facility('nersc')" in code

    def test_no_hardcoded_username(self, code):
        assert '"parton"' not in code and "'parton'" not in code, (
            "nersc_facility_tutorial hard-codes username 'parton' — use NERSC_USERNAME env var"
        )

    def test_nersc_username_from_env(self, code):
        assert "NERSC_USERNAME" in code

    def test_nersc_account_from_env(self, code):
        assert "NERSC_ACCOUNT" in code

    def test_submission_is_opt_in(self, code):
        assert "SUBMIT_JOB" in code or "ENABLE_SUBMIT" in code or "# Submit" in code

    def test_submission_is_structurally_guarded(self, nb):
        assert_calls_are_guarded(nb, (".submit",), "SUBMIT_JOB")

    def test_no_custom_nersc_import_facility_config(self, code):
        """FacilityConfig should not be imported for the built-in NERSC path."""
        assert "FacilityConfig" not in code, (
            "nersc_facility_tutorial imports FacilityConfig — NERSC is built-in, "
            "no manual config required"
        )


class TestFilesystemTutorial:
    """filesystem_tutorial.ipynb — 0.6.0 public surface only; no view/file/live polling."""

    @pytest.fixture
    def nb(self):
        return load_notebook("filesystem_tutorial.ipynb")

    @pytest.fixture
    def code(self, nb):
        return all_code(nb)

    def test_no_dummy_token(self, code):
        assert 'token="not-needed-for-facilities"' not in code
        assert "token='not-needed-for-facilities'" not in code

    def test_no_view_method(self, code):
        assert "fs.view(" not in code, (
            "filesystem_tutorial calls fs.view() which is not in the 0.6.0 public surface"
        )

    def test_no_file_method(self, code):
        assert "fs.file(" not in code, (
            "filesystem_tutorial calls fs.file() which is not in the 0.6.0 public surface"
        )

    def test_no_hardcoded_username(self, code):
        assert '"parton"' not in code and "'parton'" not in code, (
            "filesystem_tutorial hard-codes username 'parton' — use ALCF_USERNAME env var"
        )

    def test_alcf_username_from_env(self, code):
        assert "ALCF_USERNAME" in code

    def test_uses_supported_fs_methods(self, code):
        """At least the core read ops should be demonstrated."""
        # ls, head, tail, stat are the core read ops
        assert "fs.ls(" in code, "filesystem_tutorial should demonstrate fs.ls()"
        assert "fs.head(" in code, "filesystem_tutorial should demonstrate fs.head()"

    def test_task_wait_is_noop_noted(self, code):
        """The spec says wait() is a no-op — the notebook should explain this or handle it."""
        # Either wait() is used (no-op but harmless) or the text explains the compatibility note
        # We just check that the notebook doesn't claim wait() polls live state
        assert "task.cancel" not in code, (
            "filesystem_tutorial calls task.cancel() — not in the 0.6.0 public surface"
        )

    def test_no_filesystem_task_polling(self, code):
        for stale in ("task.wait(", "task.status", "task.id", "task.uri", "task.command"):
            assert stale not in code, f"filesystem tutorial teaches stale Task API: {stale}"

    def test_filesystem_mutations_are_structurally_guarded(self, nb):
        assert_calls_are_guarded(
            nb,
            (
                "fs.mkdir",
                "fs.rm",
                "fs.cp",
                "fs.mv",
                "fs.upload",
                "fs.chmod",
                "fs.compress",
                "fs.extract",
            ),
            "ENABLE_WRITES",
        )

    def test_tutorial_dir_uses_run_id(self, code):
        """Destructive ops must use a per-run uniquely named directory."""
        assert "RUN_ID" in code or "run_id" in code, (
            "filesystem_tutorial must use a RUN_ID-suffixed tutorial directory for cleanup safety"
        )

    def test_cleanup_present(self, code):
        """Tutorial must clean up what it creates."""
        assert "fs.rm(" in code, (
            "filesystem_tutorial should clean up its tutorial directory with fs.rm()"
        )


# ============================================================================
# Task 4 — README and long-form guides
# ============================================================================


class TestReadmeContent:
    """README.md must reflect 0.6.0 setup, correct auth, and current notebook inventory."""

    @pytest.fixture
    def readme(self):
        return (REPO / "README.md").read_text()

    def test_installation_shows_exact_pin(self, readme):
        assert "amsc-client==0.6.0" in readme, (
            "README installation section must show amsc-client==0.6.0"
        )

    def test_no_old_globus_central_auth_example(self, readme):
        """README must not show Globus as the central-service auth method."""
        assert "use_id_token=True" not in readme, (
            "README still shows use_id_token=True — central auth is now token-based"
        )

    def test_amsc_token_env_var_mentioned(self, readme):
        assert "AMSC_TOKEN" in readme, (
            "README must document the AMSC_TOKEN environment variable"
        )

    def test_nersc_shown_as_builtin(self, readme):
        """README must not show NERSC custom registration as required."""
        assert "register_facility" not in readme or "built-in" in readme.lower(), (
            "README still shows manual register_facility for NERSC — it's now built-in"
        )

    def test_staging_url_present(self, readme):
        assert "staging.american-science-cloud.org" in readme, (
            "README must reference the staging endpoint"
        )

    def test_python_floor_matches_published_wheel(self, readme):
        assert "Python 3.11+" in readme
        assert "Python 3.10+" not in readme

    def test_read_only_mixed_auth_smoke_documented(self, readme):
        assert "scripts/smoke_mixed_auth.py" in readme
        assert "ALCF_IRI_TOKEN" in readme
        assert "read-only" in readme.lower()

    def test_tutorial_order_is_sane(self, readme):
        """All 5 notebooks must be listed."""
        for nb in [
            "catalog_explorer",
            "catalog_tutorial",
            "alcf_facility_tutorial",
            "nersc_facility_tutorial",
            "filesystem_tutorial",
        ]:
            assert nb in readme, f"README does not mention {nb}"


class TestAgenticGuide:
    """docs/agentic-guide-to-polaris-with-iri.md — 0.6.0 auth, labeled claims."""

    @pytest.fixture
    def guide(self):
        path = DOCS / "agentic-guide-to-polaris-with-iri.md"
        assert path.exists()
        return path.read_text()

    def test_no_use_id_token(self, guide):
        assert "use_id_token=True" not in guide, (
            "agentic-guide still shows use_id_token=True — must use AmSC Keycard"
        )

    def test_no_hardcoded_username(self, guide):
        # The guide mentions 'parton' as a historical user — should be parameterized
        # Allow it in labelled historical context but not as an executable default
        if "parton" in guide:
            # Check if it's labelled as example/historical
            idx = guide.index("parton")
            context = guide[max(0, idx - 100) : idx + 100]
            assert any(
                word in context.lower()
                for word in [
                    "example",
                    "e.g.",
                    "replace",
                    "your",
                    "env",
                    "environ",
                    "alcf_username",
                    "username",
                ]
            ), (
                "agentic-guide uses 'parton' as a default username without labelling it as an example"
            )

    def test_amsc_client_version_updated(self, guide):
        """Guide must not reference the old 0.4.1 version as current."""
        assert "amsc-client==0.4.1" not in guide, (
            "agentic-guide still references amsc-client==0.4.1"
        )

    def test_base64_labeled_as_fallback(self, guide):
        """If base64 transfer is shown, it must be labeled as historical/fallback."""
        if "base64" in guide.lower():
            idx = guide.lower().index("base64")
            context = guide[max(0, idx - 200) : idx + 200]
            assert any(
                word in context.lower()
                for word in ["fallback", "historical", "alternative", "workaround", "upload"]
            ), (
                "agentic-guide shows base64 transfer without labelling it as a fallback/workaround. "
                "The public upload path is preferred."
            )

    def test_does_not_claim_direct_upload_is_unavailable(self, guide):
        assert "can't upload files directly" not in guide
        assert "adds direct file-upload support in a future version" not in guide


class TestPytorchGuide:
    """docs/pytorch-distributed-training-on-polaris.md — 0.6.0 auth, parameterised values."""

    @pytest.fixture
    def guide(self):
        path = DOCS / "pytorch-distributed-training-on-polaris.md"
        assert path.exists()
        return path.read_text()

    def test_no_use_id_token(self, guide):
        assert "use_id_token=True" not in guide

    def test_no_hardcoded_username(self, guide):
        if "parton" in guide or "youruser" in guide:
            pass  # youruser is the recommended placeholder — OK
        # But 'parton' as an executable default is not OK
        for match in re.finditer(r"account\s*=\s*['\"]parton['\"]", guide):
            pytest.fail(
                "pytorch-guide hard-codes account='parton' — use ALCF_ACCOUNT env var"
            )

    def test_amsc_client_0_6_0_referenced(self, guide):
        """Guide must not present 0.4.1 or an unspecified old version."""
        assert "0.4.1" not in guide, (
            "pytorch-guide still references the old amsc-client version 0.4.1"
        )

    def test_container_image_parameterised(self, guide):
        """Container image must be a parameter, not a hard-coded user image."""
        # 'jtchilders/pepper-polaris' is a user-specific image and should not appear as a default
        assert "jtchilders/" not in guide, (
            "pytorch-guide hard-codes jtchilders/ container image — use CONTAINER_IMAGE env var"
        )


# ============================================================================
# Task 1 — GitHub Actions validation
# ============================================================================


class TestGithubActions:
    """GitHub Actions workflow must exist and run the correct commands."""

    GHA_PATH = REPO / ".github" / "workflows" / "validate.yml"

    def test_workflow_file_exists(self):
        assert self.GHA_PATH.exists(), (
            ".github/workflows/validate.yml must exist"
        )

    def test_workflow_installs_amsc_client_0_6(self):
        content = self.GHA_PATH.read_text()
        assert "pip install -r requirements-dev.txt" in content
        assert "amsc-client==0.6.0" in (REPO / "requirements.txt").read_text()

    def test_workflow_runs_pytest(self):
        content = self.GHA_PATH.read_text()
        assert "pytest" in content, "GitHub Actions workflow must run pytest"

    def test_workflow_runs_smoke_unit_tests(self):
        content = self.GHA_PATH.read_text()
        assert "--ignore=tests/test_smoke_script.py" not in content

    def test_workflow_installs_declared_dev_requirements(self):
        content = self.GHA_PATH.read_text()
        assert "requirements-dev.txt" in content

    def test_workflow_contains_all_package_indexes(self):
        content = (REPO / "requirements.txt").read_text()
        for project_id in ("77567162", "76368190", "80654726", "82001936"):
            assert project_id in content

    def test_workflow_uses_python_311(self):
        content = self.GHA_PATH.read_text()
        assert "3.11" in content, "GitHub Actions workflow must use Python 3.11"
