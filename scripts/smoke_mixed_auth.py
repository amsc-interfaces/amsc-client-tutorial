"""Opt-in, read-only mixed-auth smoke test for ``amsc-client==0.6.0``.

The central AmSC staging API uses ``AMSC_TOKEN`` (an AmSC Keycard access
token). The direct ALCF IRI v1 API uses the independent ``ALCF_IRI_TOKEN``.
The script performs only public OpenAPI reads and protected account reads.
It never prints, compares via serialization, or otherwise exposes token values.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from importlib.metadata import PackageNotFoundError, version

from amsc_client import Client
from amsc_client.auth import TokenAuthenticator

STAGING_BASE_URL = "https://api.staging.american-science-cloud.org/api/current"
STAGING_OPENAPI_URL = f"{STAGING_BASE_URL}/openapi.json"
ALCF_OPENAPI_URL = "https://api.alcf.anl.gov/openapi.json"
EXPECTED_VERSION = "0.6.0"
_REQUIRED_VARS = ("AMSC_TOKEN", "ALCF_IRI_TOKEN")


def probe_openapi(url: str) -> None:
    """Require a reachable response that resembles an OpenAPI document."""
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "amsc-client-tutorial-smoke/0.6.0"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        body = response.read(4096)
    if b'"openapi"' not in body:
        raise RuntimeError("response is not an OpenAPI document")


def _installed_version() -> str:
    try:
        return version("amsc-client")
    except PackageNotFoundError:
        return "not-installed"


def main() -> int:
    missing = [name for name in _REQUIRED_VARS if not os.environ.get(name)]
    if missing:
        print(
            f"[FAIL] Missing required environment variable(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2

    amsc_token = os.environ["AMSC_TOKEN"]
    alcf_token = os.environ["ALCF_IRI_TOKEN"]
    if amsc_token == alcf_token:
        print("[FAIL] Central and facility credentials must be independent.", file=sys.stderr)
        return 2

    installed = _installed_version()
    if installed != EXPECTED_VERSION:
        print(
            f"[FAIL] Expected amsc-client {EXPECTED_VERSION}; installed version is {installed}.",
            file=sys.stderr,
        )
        return 1

    try:
        probe_openapi(STAGING_OPENAPI_URL)
        print("[OK] AmSC staging OpenAPI is reachable")
        probe_openapi(ALCF_OPENAPI_URL)
        print("[OK] ALCF IRI OpenAPI is reachable")

        client = Client(token=amsc_token, base_url=STAGING_BASE_URL)
        alcf_authenticator = TokenAuthenticator(token=alcf_token)
        if alcf_authenticator is client._authenticator:
            raise RuntimeError("central and facility authenticators are not independent")
        client.register_facility("alcf", authenticator=alcf_authenticator)

        client.account.me()
        print("[OK] staging protected account read authenticated")

        alcf = client.facility("alcf")
        projects = alcf._client()._service_client._account_api.get_projects()
        print(f"[OK] ALCF protected account read authenticated ({len(projects)} project(s))")
    except Exception as exc:
        # Do not render exception text: upstream HTTP errors can include request
        # details, and credentials must never be reflected in smoke-test output.
        print(f"[FAIL] Mixed-auth verification failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    print("[PASS] Independent staging and ALCF authentication verified read-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
