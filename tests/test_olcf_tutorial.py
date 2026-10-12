import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "olcf_facility_tutorial.ipynb"
HELPER = ROOT / "notebooks" / "olcf_s3m.py"


def _notebook():
    return json.loads(NOTEBOOK.read_text())


def _code_cells():
    return ["".join(cell.get("source", [])) for cell in _notebook()["cells"] if cell["cell_type"] == "code"]


def _load_helper():
    spec = importlib.util.spec_from_file_location("olcf_s3m_review", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_olcf_notebook_is_clean_and_compiles():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
            compile("".join(cell.get("source", [])), str(NOTEBOOK), "exec")


def test_all_job_operations_are_guarded_by_default_off_gate():
    cells = _code_cells()
    joined = "\n".join(cells)
    assert "SUBMIT_JOB = False" in joined
    assert joined.count("resource.submit(") == 2

    for source in cells:
        if any(marker in source for marker in ("resource.submit(", "job.status", "job.wait(")):
            tree = ast.parse(source)
            guarded_lines = {
                node.lineno
                for parent in ast.walk(tree)
                if isinstance(parent, ast.If)
                and isinstance(parent.test, ast.Name)
                and parent.test.id == "SUBMIT_JOB"
                for node in ast.walk(parent)
                if hasattr(node, "lineno")
            }
            sensitive_lines = {
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Attribute) and node.func.attr in {"submit", "wait"})
                    or (isinstance(node.func, ast.Name) and node.func.id == "status")
                )
            }
            assert sensitive_lines <= guarded_lines, source


def test_active_olcf_material_has_no_rig_or_staging_dependency():
    content = NOTEBOOK.read_text() + HELPER.read_text()
    for forbidden in (
        "notebooks_via_rig",
        "facility_via_rig",
        "rig.staging.american-science-cloud.org",
        "STAGING_RIG",
        "rig_whoami",
        "olcf_via_rig",
    ):
        assert forbidden not in content


def test_token_helpers_strip_schemes_and_reject_unknown_enclave(tmp_path):
    helper = _load_helper()
    assert helper._strip_scheme("Authorization: Bearer abc") == "abc"
    token_file = tmp_path / "token"
    token_file.write_text("Bearer xyz\n")
    assert helper.read_token_file(token_file, "open") == "xyz"
    with pytest.raises(ValueError):
        helper.token_env_vars("unknown")


def test_routing_rewrite_is_scoped_to_registered_s3m_host(monkeypatch):
    pytest.importorskip("amsc_iri")
    helper = _load_helper()

    helper.configure_s3m_routing(helper.ENCLAVES["open"])
    from amsc_iri.api_client import ApiClient
    from amsc_iri.configuration import Configuration

    routed = ApiClient(Configuration(host=helper.ENCLAVES["open"]))
    untouched = ApiClient(Configuration(host="https://api.alcf.anl.gov"))

    assert helper._route_for(routed) == (helper.API_PREFIX, "bearer")
    assert helper._route_for(untouched) is None
