"""OLCF S3M helpers for amsc-client.

amsc-client 0.6.x/0.7.x delegates facility calls to the generated ``amsc_iri``
SDK, whose routes are hardcoded to ``/api/v1``. The ``api_prefix`` passed to
``Client.register_facility`` is stored but never used.

OLCF's S3M deployments serve the IRI API under ``/api/v2``, with v2 resource
type URNs. Its IRI routes take ``Authorization: Bearer <token>`` (the SDK's
default); ``account/whoami`` also accepts the raw token. This module bridges
the gap until the client supports v2 natively:

* ``register_olcf`` registers an OLCF enclave with amsc-client and routes its
  requests to ``/api/v2``.
* ``load_token`` reads per-enclave tokens from environment variables (or a
  git-ignored ``.env`` file); ``read_token_file`` and ``resolve_token`` add
  token files and a hidden prompt as alternatives.
* ``whoami`` calls ``/api/v2/account/whoami``, which the generated SDK lacks.
* ``verify_api_version`` and ``recent_requests`` show which API version the
  client actually calls.
* ``olcf_via_rig`` and ``rig_whoami`` do the same through the AmSC Resource
  Integration Gateway (RIG), whose OLCF mounts (``olcf-open``,
  ``olcf-moderate``) are IRI v2 while amsc-client 0.7.1's RIG helper only
  accepts v1.

The routing patch only applies to requests whose host is a registered S3M base
URL, so other facilities (ALCF, NERSC) are unaffected.
"""

from __future__ import annotations

from collections import deque
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import requests

logger = logging.getLogger(__name__)

ENCLAVES = {
    "open": "https://amsc-open.s3m.olcf.ornl.gov",
    "moderate": "https://amsc-moderate.s3m.olcf.ornl.gov",
}
DISPLAY_NAMES = {
    "open": "Oak Ridge Leadership Computing Facility (Open)",
    "moderate": "Oak Ridge Leadership Computing Facility (Moderate)",
}
API_PREFIX = "/api/v2"
STAGING_RIG = "https://rig.staging.american-science-cloud.org"
GENERATED_PREFIX = "/api/v1"

# S3M's IRI routes reject the raw token with 401 "Not authenticated" and
# accept "Bearer <token>". "raw" sends the token as the whole header value.
AUTH_SCHEMES = {"raw", "bearer"}

# base URL -> (api_prefix, auth_scheme)
_S3M_ROUTES: dict[str, tuple[str, str]] = {}

DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

TOKEN_SOURCES = {"env", "file", "prompt"}

# "METHOD URL" (decoded; never headers) for recent S3M requests.
_RECENT_REQUESTS: deque[str] = deque(maxlen=50)


class MissingTokenError(RuntimeError):
    """Raised when no S3M token is configured for an enclave."""


class RigAccessError(RuntimeError):
    """A RIG request was refused; ``status`` and ``detail`` say why (never the token)."""

    def __init__(self, status: int, detail: str, url: str) -> None:
        super().__init__(f"HTTP {status} from {url}: {detail}")
        self.status = status
        self.detail = detail
        self.url = url


def token_env_vars(enclave: str) -> list[str]:
    """Environment variables checked (in order) for an enclave's token."""
    names = [f"OLCF_S3M_TOKEN_{enclave.upper()}"]
    if enclave == "open":
        names.append("OLCF_S3M_TOKEN")
    return names


def load_env_file(path: str | Path = DEFAULT_ENV_FILE) -> None:
    """Load ``KEY=VALUE`` lines into ``os.environ`` without overriding.

    Blank lines, ``#`` comments, an optional ``export`` prefix and matching
    surrounding quotes are handled; nothing else is interpreted.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_token(enclave: str, *, env_file: str | Path | None = DEFAULT_ENV_FILE) -> str:
    """Return the S3M token for ``enclave`` from the environment or ``.env``."""
    _check_enclave(enclave)
    if env_file is not None:
        load_env_file(env_file)
    for name in token_env_vars(enclave):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    raise MissingTokenError(
        f"No S3M token for OLCF {enclave}: set {' or '.join(token_env_vars(enclave))} "
        "(or add it to a .env file at the repository root)."
    )


def read_token_file(path: str | Path, enclave: str) -> str:
    """Read an S3M token from a file.

    The file may hold either the bare token on one line, or ``KEY=VALUE``
    lines (``OLCF_S3M_TOKEN_<ENCLAVE>=...``, as in a ``.env`` file). A leading
    ``Bearer `` or ``Authorization:`` is stripped.
    """
    _check_enclave(enclave)
    token_path = Path(path).expanduser()
    if not token_path.is_file():
        raise MissingTokenError(f"Token file not found: {token_path}")
    lines = [
        line.strip()
        for line in token_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assignments = {}
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            assignments[key.removeprefix("export ").strip()] = value.strip().strip("\"'")
    for name in token_env_vars(enclave):
        if assignments.get(name):
            return _strip_scheme(assignments[name])
    if len(lines) == 1 and "=" not in lines[0]:
        return _strip_scheme(lines[0])
    raise MissingTokenError(
        f"{token_path} has no bare token and no {' or '.join(token_env_vars(enclave))} entry"
    )


def resolve_token(
    enclave: str,
    source: str = "env",
    *,
    token_file: str | Path | None = None,
) -> str | None:
    """Return the token for ``enclave`` from ``source``; ``None`` if skipped.

    ``env``: environment variables or the repository ``.env`` (:func:`load_token`).
    ``file``: :func:`read_token_file` on ``token_file``.
    ``prompt``: a hidden input box; press Enter to skip the enclave. Nothing
    is echoed or stored in the notebook.
    """
    _check_enclave(enclave)
    if source not in TOKEN_SOURCES:
        raise ValueError(f"source must be one of {sorted(TOKEN_SOURCES)}")
    if source == "env":
        return load_token(enclave)
    if source == "file":
        if token_file is None:
            raise MissingTokenError(f"No token file configured for {enclave}")
        return read_token_file(token_file, enclave)
    from getpass import getpass

    token = getpass(f"S3M token for OLCF {enclave} (Enter to skip): ").strip()
    return _strip_scheme(token) or None


def _strip_scheme(token: str) -> str:
    token = token.strip()
    if token.lower().startswith("authorization:"):
        token = token.split(":", 1)[1].strip()
    if token.lower().startswith("bearer "):
        token = token[len("bearer "):].strip()
    return token


def configure_s3m_routing(
    base_url: str,
    *,
    api_prefix: str = API_PREFIX,
    auth_scheme: str = "bearer",
) -> None:
    """Route generated-SDK requests for ``base_url`` to ``api_prefix``."""
    if auth_scheme not in AUTH_SCHEMES:
        raise ValueError(f"auth_scheme must be one of {sorted(AUTH_SCHEMES)}")
    _install_patch()
    _S3M_ROUTES[base_url.rstrip("/")] = (api_prefix.rstrip("/"), auth_scheme)


def register_olcf(
    client: Any,
    enclave: str = "open",
    *,
    token: str | None = None,
    auth_scheme: str = "bearer",
) -> Any:
    """Register an OLCF S3M enclave with ``client`` and return its facility.

    Args:
        client: An ``amsc_client.Client``.
        enclave: ``"open"`` or ``"moderate"``.
        token: S3M token; defaults to :func:`load_token`.
        auth_scheme: ``"bearer"`` (default, required by S3M's IRI routes) or ``"raw"``.
    """
    _check_enclave(enclave)
    base_url = ENCLAVES[enclave]
    configure_s3m_routing(base_url, auth_scheme=auth_scheme)
    name = f"olcf-{enclave}"
    client.register_facility(
        name,
        base_url=base_url,
        display_name=DISPLAY_NAMES[enclave],
        auth_method="token",
        token=token or load_token(enclave),
        api_prefix=API_PREFIX,
    )
    facility = client.facility(name)
    _fresh_discovery(facility._client())
    return facility


def _fresh_discovery(amscrot_client: Any) -> None:
    """Run resource discovery now, ignoring any cached result.

    amscrot 1.2 (amsc-client 0.6.x) caches discovery on disk for an hour, and
    entries written by a v1 client or a failed run carry different resource
    ids. amscrot 1.1 (amsc-client 0.7.x) only caches in memory and its
    ``_get_discovery`` takes no ``refresh`` argument.
    """
    import inspect

    if "refresh" in inspect.signature(amscrot_client._get_discovery).parameters:
        amscrot_client._get_discovery(refresh=True)
    else:
        amscrot_client._discovery = None
        amscrot_client._get_discovery()


def authorization_header(token: str, auth_scheme: str = "bearer") -> str:
    return token if auth_scheme == "raw" else f"Bearer {token}"


def whoami(
    enclave: str = "open",
    token: str | None = None,
    *,
    auth_scheme: str = "bearer",
    timeout: float = 30,
) -> dict[str, Any]:
    """Return the identity behind an S3M token (raises on 4xx/5xx)."""
    _check_enclave(enclave)
    response = requests.get(
        f"{ENCLAVES[enclave]}{API_PREFIX}/account/whoami",
        headers={
            "Accept": "application/json",
            "Authorization": authorization_header(token or load_token(enclave), auth_scheme),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def recent_requests(limit: int = 10, *, base_url: str | None = None) -> list[str]:
    """The last S3M requests sent through amsc-client, as ``METHOD URL``.

    Pass ``base_url`` (e.g. ``facility.base_url``) to show one enclave only.
    """
    requests_sent = list(_RECENT_REQUESTS)
    if base_url:
        prefix = base_url.rstrip("/") + "/"
        requests_sent = [r for r in requests_sent if r.split(" ", 1)[1].startswith(prefix)]
    return requests_sent[-limit:]


def verify_api_version(
    facility: Any, *, token: str | None = None, timeout: float = 30
) -> dict[str, str]:
    """Show and check the API version a registered OLCF facility uses.

    Builds (without sending) the URL amsc-client would call for the resource
    list, and fetches the server's v2 OpenAPI document (through RIG, pass the
    AmSC ``token``). Raises ``RuntimeError`` if the client would not call
    ``/api/v2``; an unreachable OpenAPI document is reported, not fatal.
    """
    base_url = facility.base_url.rstrip("/")
    api_client = facility._client()._service_client._api_client
    _, sample_url, _, _, _ = api_client.param_serialize(
        method="GET", resource_path=f"{GENERATED_PREFIX}/status/resources"
    )
    client_prefix = sample_url[len(base_url):].split("/status/", 1)[0]
    if _RECENT_REQUESTS and _RECENT_REQUESTS[-1] == f"GET {unquote(sample_url)}":
        _RECENT_REQUESTS.pop()  # built, never sent: keep the request log honest

    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        spec = requests.get(f"{base_url}{API_PREFIX}/openapi.json", headers=headers, timeout=timeout)
        spec.raise_for_status()
        info = spec.json().get("info", {})
        server_api = f"{info.get('title', '?')} {info.get('version', '?')}"
    except (requests.RequestException, ValueError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        server_api = f"OpenAPI document unavailable ({f'HTTP {status}' if status else type(exc).__name__})"
    result = {
        "facility": facility.name,
        "generated_prefix": GENERATED_PREFIX,
        "client_prefix": client_prefix,
        "sample_url": sample_url,
        "server_api": server_api,
    }
    print(f"API check for {facility.name}")
    print(f"  amsc-client generated routes : {GENERATED_PREFIX}  (rewritten by olcf_s3m)")
    print(f"  client will call             : {sample_url}")
    print(f"  server reports               : {result['server_api']}")
    if client_prefix != API_PREFIX:
        raise RuntimeError(
            f"{facility.name} would call {client_prefix}, not {API_PREFIX}; "
            "register it with olcf_s3m.register_olcf()"
        )
    print(f"  ✅ requests go to {API_PREFIX}")
    return result


def rig_mount(enclave: str = "open", *, rig_url: str = STAGING_RIG) -> str:
    """RIG mount for an OLCF enclave, e.g. ``<rig>/rig/external/olcf-open``."""
    from amsc_client.rig import rig_facility_mount

    _check_enclave(enclave)
    return rig_facility_mount(rig_url, f"olcf-{enclave}")


def olcf_via_rig(client: Any, enclave: str = "open", *, rig_url: str = STAGING_RIG) -> Any:
    """Return an amsc-client facility for an OLCF enclave routed through RIG.

    RIG authenticates the caller with the client's AmSC Keycard/PAT and
    attaches the S3M token saved in the Credential Vault. RIG advertises the
    OLCF mounts as IRI v2, but ``Client.facility_via_rig`` in amsc-client 0.7.1
    only accepts v1. So the facility is requested with an explicit
    ``/api/v1`` prefix, which satisfies that check, and this module rewrites
    the generated v1 routes to v2 for the RIG mount only. The Authorization
    header stays ``Bearer <AmSC token>``, as RIG requires.
    """
    mount = rig_mount(enclave, rig_url=rig_url)
    configure_s3m_routing(mount, auth_scheme="bearer")
    facility = client.facility_via_rig(
        f"olcf-{enclave}", rig_url=rig_url, api_prefix=GENERATED_PREFIX
    )
    _fresh_discovery(facility._client())
    return facility


def rig_whoami(
    amsc_token: str,
    enclave: str = "open",
    *,
    rig_url: str = STAGING_RIG,
    timeout: float = 30,
) -> dict[str, Any]:
    """Return the S3M identity behind the vaulted token, through RIG.

    This is RIG's ``probe_path`` for OLCF (``/account/whoami``). A 200 proves
    the whole chain: AmSC token -> RIG -> Credential Vault S3M token -> S3M.
    Raises :class:`RigAccessError` with RIG's or S3M's reason otherwise.
    """
    url = f"{rig_mount(enclave, rig_url=rig_url)}{API_PREFIX}/account/whoami"
    response = requests.get(
        url,
        headers={"Accept": "application/json", "Authorization": f"Bearer {amsc_token}"},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RigAccessError(response.status_code, _rig_error_detail(response), url)
    return response.json()


def _rig_error_detail(response: Any) -> str:
    """RIG answers ``{error, detail, hint}``; S3M ``{code, message}``."""
    try:
        body = response.json()
    except ValueError:
        return (response.text or response.reason or "")[:200]
    if not isinstance(body, dict):
        return str(body)[:200]
    parts = [str(body[key]) for key in ("error", "title", "detail", "message", "hint") if body.get(key)]
    if "code" in body and "message" in body:
        parts.append(f"(S3M code {body['code']})")
    return " — ".join(parts)[:300] or response.reason or ""


def _check_enclave(enclave: str) -> None:
    if enclave not in ENCLAVES:
        raise ValueError(f"Unknown OLCF enclave {enclave!r}; expected one of {sorted(ENCLAVES)}")


# IRI v2 identifies resource types by URN; the generated v1 models use short
# enum values ("compute"). Translate in both directions.
RESOURCE_TYPE_URN_PREFIX = "urn:doe-iri:resource:"


def to_v2_resource_type(value: Any) -> Any:
    text = str(getattr(value, "value", value))
    if text.startswith("urn:"):
        return text
    return RESOURCE_TYPE_URN_PREFIX + text


def to_v1_resource_type(value: Any) -> Any:
    """Map a v2 URN onto the closest value the v1 ``ResourceType`` enum accepts."""
    from amsc_iri.models.resource_type import ResourceType

    if not isinstance(value, str) or not value.startswith("urn:"):
        return value
    allowed = {member.value for member in ResourceType}
    parts = value.split(":")[2:]  # drop "urn:doe-iri"
    for part in reversed(parts):
        if part in allowed:
            return part
    return ResourceType.UNKNOWN.value


def normalize_v2_payload(data: Any) -> Any:
    """Rewrite v2 response fields that the generated v1 models would reject."""
    if isinstance(data, list):
        return [normalize_v2_payload(item) for item in data]
    if isinstance(data, dict):
        return {
            key: to_v1_resource_type(value) if key == "resource_type" else normalize_v2_payload(value)
            for key, value in data.items()
        }
    return data


def _route_for(api_client: Any) -> tuple[str, str] | None:
    return _S3M_ROUTES.get((api_client.configuration.host or "").rstrip("/"))


def _install_patch() -> None:
    """Wrap ``amsc_iri.ApiClient`` request/response handling once, process-wide.

    Patching the class (not an instance) matters: amscrot rebuilds its service
    client, and with it the ApiClient, whenever it refreshes a token.
    """
    from amsc_iri.api_client import ApiClient

    if getattr(ApiClient.param_serialize, "_s3m_patched", False):
        return
    original_serialize = ApiClient.param_serialize
    original_deserialize = ApiClient.deserialize

    def param_serialize(self: Any, method: str, resource_path: str, *args: Any, **kwargs: Any):
        route = _route_for(self)
        if route is None:
            return original_serialize(self, method, resource_path, *args, **kwargs)

        api_prefix, auth_scheme = route
        if resource_path.startswith(GENERATED_PREFIX + "/"):
            resource_path = api_prefix + resource_path[len(GENERATED_PREFIX):]
        if kwargs.get("query_params"):
            kwargs["query_params"] = [
                (key, to_v2_resource_type(value) if key == "resource_type" else value)
                for key, value in kwargs["query_params"]
            ]
        method, url, headers, body, post_params = original_serialize(
            self, method, resource_path, *args, **kwargs
        )
        authorization = headers.get("Authorization", "")
        if auth_scheme == "raw" and authorization.startswith("Bearer "):
            headers["Authorization"] = authorization[len("Bearer "):]
        logger.debug("S3M %s %s", method, url)
        _RECENT_REQUESTS.append(f"{method} {unquote(url)}")
        return method, url, headers, body, post_params

    def deserialize(self: Any, response_text: str, response_type: str, content_type: str | None):
        if _route_for(self) is not None and response_text:
            try:
                response_text = json.dumps(normalize_v2_payload(json.loads(response_text)))
            except ValueError:
                pass
        return original_deserialize(self, response_text, response_type, content_type)

    param_serialize._s3m_patched = True  # type: ignore[attr-defined]
    ApiClient.param_serialize = param_serialize  # type: ignore[method-assign]
    ApiClient.deserialize = deserialize  # type: ignore[method-assign]
    _skip_unavailable_account_discovery()


def _skip_unavailable_account_discovery() -> None:
    """Return no capabilities/projects for S3M instead of calling account routes.

    amscrot's resource discovery queries the account API after resources and
    sites. S3M only implements ``account/whoami``, so those calls would fail and
    log an error on every discovery.
    """
    from amsc_iri.api.account_api import AccountApi

    for name in ("get_capabilities", "get_projects"):
        original = getattr(AccountApi, name)

        def skip(self: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            if _route_for(self.api_client) is not None:
                return []
            return _original(self, *args, **kwargs)

        setattr(AccountApi, name, skip)
