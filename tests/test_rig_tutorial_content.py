"""Contract tests for the explicit experimental RIG tutorial notebooks."""
from __future__ import annotations

import ast
import importlib.metadata
import inspect
import json
import re
from pathlib import Path

import pytest

from amsc_client import Client
from amsc_client.core.exceptions import ApiError, AuthenticationError
from amsc_client.facility.client import FacilityClient

REPO = Path(__file__).parent.parent
RIG_DIR = REPO / "notebooks_via_rig"
RIG_NOTEBOOKS = {
    "alcf_via_rig.ipynb",
    "nersc_via_rig.ipynb",
    "multi_facility_via_rig.ipynb",
}
FACILITY_NOTEBOOKS = {
    "alcf_via_rig.ipynb": ("ALCF", "alcf"),
    "nersc_via_rig.ipynb": ("NERSC", "nersc"),
}
FORBIDDEN_IMPORTS = {"requests", "httpx", "urllib", "getpass", "jwt"}


def load(name: str) -> dict:
    path = RIG_DIR / name
    assert path.exists(), f"missing RIG notebook: {path}"
    return json.loads(path.read_text())


def cells(nb: dict, kind: str) -> list[str]:
    return ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == kind]


def code(nb: dict) -> str:
    return "\n\n".join(cells(nb, "code"))


def text(nb: dict) -> str:
    return "\n\n".join("".join(cell["source"]) for cell in nb["cells"])


def calls(nb: dict, suffix: str) -> list[ast.Call]:
    return [
        node
        for source in cells(nb, "code")
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and dotted_name(node.func).endswith(suffix)
    ]


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def guarded_calls(nb: dict, suffix: str, gate: str) -> list[ast.Call]:
    found: list[ast.Call] = []

    def visit(node: ast.AST, guarded: bool = False) -> None:
        if isinstance(node, ast.If):
            positive = isinstance(node.test, ast.Name) and node.test.id == gate
            negative = (
                isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id == gate
            )
            for child in node.body:
                visit(child, guarded or positive)
            for child in node.orelse:
                visit(child, guarded or negative)
            return
        if isinstance(node, ast.Call) and dotted_name(node.func).endswith(suffix):
            assert guarded, f"{dotted_name(node.func)} must be dominated by {gate}"
            found.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    for source in cells(nb, "code"):
        visit(ast.parse(source))
    return found


@pytest.mark.parametrize("name", sorted(RIG_NOTEBOOKS))
def test_notebook_exists_parses_compiles_and_is_clean(name):
    nb = load(name)
    assert nb["nbformat"] == 4
    for source in cells(nb, "code"):
        compile(source, name, "exec")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


@pytest.mark.parametrize("name", sorted(RIG_NOTEBOOKS))
def test_common_rig_auth_and_transport_contract(name):
    nb = load(name)
    source = code(nb)
    prose = text(nb).lower()
    trees = [ast.parse(src) for src in cells(nb, "code")]
    imported = {
        alias.name.split(".")[0]
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
    }
    assert not (imported & FORBIDDEN_IMPORTS)
    assert "AMSC_TOKEN" in source and ("os.environ" in source or "os.getenv" in source)
    assert "https://api.staging.american-science-cloud.org/api/current" in source
    assert "https://rig.staging.american-science-cloud.org" in source
    client_calls = [
        node
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and dotted_name(node.func) == "Client"
    ]
    assert len(client_calls) == 1
    assert {"token", "base_url"} <= {kw.arg for kw in client_calls[0].keywords}
    assert "keycard" in prose and "pat" in prose and "rig" in prose
    assert "credential vault" in prose and "allocation" in prose
    assert "statically validated" in prose and "not live-validated" in prose
    assert "~/.amsc_token.json" not in source
    assert not re.search(r"eyJ[A-Za-z0-9_-]{10,}", text(nb))


@pytest.mark.parametrize("name,config", sorted(FACILITY_NOTEBOOKS.items()))
def test_facility_notebooks_have_protected_read_and_guarded_submit(name, config):
    prefix, facility_name = config
    nb = load(name)
    source = code(nb)
    project_calls = calls(nb, ".projects")
    resource_calls = calls(nb, ".resources")
    routed_calls = calls(nb, ".facility_via_rig")
    submit_calls = guarded_calls(nb, ".submit", "SUBMIT_JOB")
    assert re.search(r"SUBMIT_JOB\s*=\s*False", source)
    assert len(routed_calls) == 1
    assert any(
        isinstance(arg, ast.Constant) and arg.value == facility_name
        for arg in routed_calls[0].args
    )
    assert project_calls and resource_calls
    assert source.index(".projects(") < source.index(".resources(")
    assert len(submit_calls) == 1
    keywords = {kw.arg for kw in submit_calls[0].keywords}
    assert {"stdout_path", "stderr_path", "queue", "account"} <= keywords
    if prefix == "ALCF":
        assert "filesystems" in keywords
    else:
        assert "constraint" in keywords
    for variable in (f"{prefix}_USERNAME", f"{prefix}_ACCOUNT", f"{prefix}_QUEUE"):
        assert variable in source
        assert re.search(rf'os\.(?:environ\[|getenv\(){re.escape(chr(34) + variable + chr(34))}', source)
    if prefix == "NERSC":
        assert "NERSC_CONSTRAINT" in source
    wait_calls = calls(nb, ".wait")
    assert len(wait_calls) == 1
    assert {"timeout", "poll_interval"} <= {kw.arg for kw in wait_calls[0].keywords}
    assert len(calls(nb, ".fs.head")) == 1
    assert "RUN_ID" in source


def test_multi_facility_notebook_is_read_only_and_probes_independently():
    nb = load("multi_facility_via_rig.ipynb")
    source = code(nb)
    assert len(calls(nb, ".rig_facilities")) == 1
    assert calls(nb, ".facility_via_rig")
    assert calls(nb, ".projects")
    assert "for meta in facilities" in source
    assert "try:" in source and "except AuthenticationError" in source and "except ApiError" in source
    assert "except Exception" in source and '"ERROR"' in source
    assert "AMSC_FACILITY" in source
    assert all(label in source for label in ("OK", "NO PROJECTS", "AUTH FAILED", "API FAILED"))
    forbidden = (".submit", ".mkdir", ".rm", ".upload", ".create_", ".delete")
    assert not any(calls(nb, suffix) for suffix in forbidden)


def test_released_client_api_and_repository_pin():
    assert importlib.metadata.version("amsc-client") == "0.7.1"
    requirements = (REPO / "requirements.txt").read_text()
    assert "amsc-client==0.7.1" in requirements
    assert callable(Client.rig_facilities)
    assert callable(Client.facility_via_rig)
    assert callable(FacilityClient.projects)
    assert callable(FacilityClient.info)
    assert callable(FacilityClient.resources)
    assert issubclass(AuthenticationError, Exception)
    assert issubclass(ApiError, Exception)
    assert "rig_url" in inspect.signature(Client.rig_facilities).parameters
    assert "rig_url" in inspect.signature(Client.facility_via_rig).parameters


def test_readme_links_and_explains_both_facility_routes():
    readme = (REPO / "README.md").read_text()
    for name in RIG_NOTEBOOKS:
        assert f"notebooks_via_rig/{name}" in readme
        assert (RIG_DIR / name).exists()
    lowered = readme.lower()
    assert "via rig" in lowered and "experimental" in lowered
    assert 'client.facility("alcf")' in readme
    assert 'client.facility_via_rig("alcf"' in readme
    assert "amsc-client==0.7.1" in readme
