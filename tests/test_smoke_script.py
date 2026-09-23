"""Tests for the opt-in, read-only mixed-auth smoke check."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts" / "smoke_mixed_auth.py"


def load_smoke():
    spec = importlib.util.spec_from_file_location("smoke_mixed_auth", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAuthenticator:
    def __init__(self, token: str):
        self.token = token

    def get_token(self) -> str:
        return self.token


class FakeClient:
    instances = []

    def __init__(self, *, token: str, base_url: str):
        self.central_token = token
        self.base_url = base_url
        self._authenticator = FakeAuthenticator(token)
        self._registered = {}
        self.account = SimpleNamespace(me=self._me)
        self.__class__.instances.append(self)

    def _me(self):
        return SimpleNamespace(username="test-user")

    def register_facility(self, name: str, *, authenticator):
        self._registered[name] = authenticator

    def facility(self, name: str):
        auth = self._registered[name]
        account_api = SimpleNamespace(get_projects=lambda: [SimpleNamespace(id="p1")])
        service = SimpleNamespace(_account_api=account_api)
        inner = SimpleNamespace(_service_client=service)
        return SimpleNamespace(_client=lambda: inner, _registry=SimpleNamespace(
            get_authenticator=lambda registered_name: self._registered[registered_name]
        ))


def test_module_is_import_safe(monkeypatch):
    monkeypatch.delenv("AMSC_TOKEN", raising=False)
    monkeypatch.delenv("ALCF_IRI_TOKEN", raising=False)
    module = load_smoke()
    assert callable(module.main)


def test_missing_credentials_fail_closed(monkeypatch, capsys):
    module = load_smoke()
    monkeypatch.delenv("AMSC_TOKEN", raising=False)
    monkeypatch.delenv("ALCF_IRI_TOKEN", raising=False)
    assert module.main() == 2
    captured = capsys.readouterr()
    assert "AMSC_TOKEN" in captured.err
    assert "ALCF_IRI_TOKEN" in captured.err


def test_script_does_not_compare_token_values():
    source = SCRIPT.read_text()
    assert "amsc_token == alcf_token" not in source
    assert "amsc_token != alcf_token" not in source


def prepare_success(module, monkeypatch):
    monkeypatch.setenv("AMSC_TOKEN", "central-secret-value")
    monkeypatch.setenv("ALCF_IRI_TOKEN", "facility-secret-value")
    monkeypatch.setattr(module, "Client", FakeClient)
    monkeypatch.setattr(module, "TokenAuthenticator", FakeAuthenticator)
    monkeypatch.setattr(module, "probe_openapi", lambda url: None)
    monkeypatch.setattr(module, "_installed_version", lambda: "0.6.1")


def test_success_uses_one_client_and_two_auth_domains(monkeypatch, capsys):
    module = load_smoke()
    FakeClient.instances.clear()
    prepare_success(module, monkeypatch)

    assert module.main() == 0
    assert len(FakeClient.instances) == 1
    client = FakeClient.instances[0]
    assert client.base_url == "https://api.staging.american-science-cloud.org/api/current"
    assert client.central_token == "central-secret-value"
    assert client._registered["alcf"].get_token() == "facility-secret-value"
    assert client._registered["alcf"] is not client._authenticator
    captured = capsys.readouterr()
    assert "central-secret-value" not in captured.out + captured.err
    assert "facility-secret-value" not in captured.out + captured.err
    assert "staging protected account" in captured.out
    assert "ALCF protected account" in captured.out


def test_reachability_failure_is_fatal(monkeypatch):
    module = load_smoke()
    prepare_success(module, monkeypatch)

    def fail(_url):
        raise RuntimeError("offline")

    monkeypatch.setattr(module, "probe_openapi", fail)
    assert module.main() == 1


def test_protected_central_failure_is_fatal(monkeypatch):
    module = load_smoke()
    prepare_success(module, monkeypatch)

    class BrokenClient(FakeClient):
        def _me(self):
            raise RuntimeError("unauthorized")

    monkeypatch.setattr(module, "Client", BrokenClient)
    assert module.main() == 1


def test_protected_alcf_failure_is_fatal(monkeypatch):
    module = load_smoke()
    prepare_success(module, monkeypatch)

    class BrokenAlcfClient(FakeClient):
        def facility(self, name: str):
            account_api = SimpleNamespace(
                get_projects=lambda: (_ for _ in ()).throw(RuntimeError("unauthorized"))
            )
            service = SimpleNamespace(_account_api=account_api)
            inner = SimpleNamespace(_service_client=service)
            return SimpleNamespace(_client=lambda: inner)

    monkeypatch.setattr(module, "Client", BrokenAlcfClient)
    assert module.main() == 1


def test_wrong_installed_version_is_fatal(monkeypatch):
    module = load_smoke()
    prepare_success(module, monkeypatch)
    monkeypatch.setattr(module, "_installed_version", lambda: "0.5.0")
    assert module.main() == 1


def test_script_contains_only_read_only_operations():
    source = SCRIPT.read_text()
    assert "account.me()" in source
    assert "_account_api.get_projects()" in source
    for forbidden in (
        ".submit(",
        "catalog.create",
        "catalog.delete",
        ".fs.mkdir(",
        ".fs.rm(",
        ".fs.cp(",
        ".fs.mv(",
        ".fs.upload(",
        ".fs.chmod(",
        ".fs.compress(",
        ".fs.extract(",
    ):
        assert forbidden not in source


def test_exact_openapi_urls():
    module = load_smoke()
    assert module.STAGING_OPENAPI_URL == (
        "https://api.staging.american-science-cloud.org/api/current/openapi.json"
    )
    assert module.ALCF_OPENAPI_URL == "https://api.alcf.anl.gov/openapi.json"
