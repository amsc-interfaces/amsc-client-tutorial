# RIG-routed facility helper and tutorial notebooks

**Date:** 2026-09-28  
**Status:** Approved design; implementation pending  
**Repositories:** `amsc-python-client`, `amsc-client-tutorial`

## Purpose

Provide a temporary, explicit high-level Python path for calling IRI facility APIs through the AmSC Resource Integration Gateway (RIG), then teach that path in three tutorial notebooks under `notebooks_via_rig/`:

- `alcf_via_rig.ipynb`
- `nersc_via_rig.ipynb`
- `multi_facility_via_rig.ipynb`

This removes most raw `requests`, URL construction, request-model conversion, and task polling from the RIG examples while keeping the security boundary visible.

## Current architecture and invariant

The released client's existing facility path is direct:

```text
client.facility("alcf")
  -> facility-native authenticator
  -> direct ALCF IRI v1 endpoint
```

The new path is different:

```text
client.facility_via_rig("alcf", rig_url=...)
  -> primary AmSC authenticator (Keycard/PAT)
  -> RIG /rig/external/alcf
  -> RIG performs facility credential selection or exchange
  -> facility IRI API
```

These paths must remain explicit and independent. The client must never silently fall back from one to the other, reuse a primary AmSC credential for a direct facility call, or register RIG routing into the existing direct-facility registry.

## Considered approaches

### 1. Explicit methods on `Client` — selected

Add `rig_facilities(...)` and `facility_via_rig(...)`. This makes the routing and credential-domain choice visible at the call site while preserving the existing `FacilityClient` experience.

### 2. Generic `client.via_rig()` proxy

A proxy could expose discovery and facility operations through a new object. It is conceptually neat but adds a larger temporary public object graph and more documentation burden.

### 3. Change `client.facility()` to prefer RIG

Rejected. This would silently change authentication, policy enforcement, refresh behavior, and network routing for existing users.

## Python-client API

### `Client.rig_facilities`

Proposed signature:

```python
def rig_facilities(
    self,
    *,
    rig_url: str = "https://rig.staging.american-science-cloud.org",
    timeout: float = 30.0,
) -> list[RigFacility]:
    ...
```

Behavior:

- Obtain the bearer from the client's primary authenticator.
- Call authenticated `GET <rig_url>/ready`.
- Parse only the fields the helper supports into a typed immutable model:
  - `name`
  - `display_name`
  - `tier`
  - `api_version`
  - `metadata_path`
  - `probe_path`
- Preserve neither the bearer nor arbitrary response bodies in the returned object or its representation.
- Map transport, authentication, malformed-payload, and unsupported-facility metadata failures into `amsc_client` exceptions with bounded, non-secret messages.
- On 401/403, invalidate or refresh only the primary AmSC authenticator and retry once when supported. It must not touch the direct facility `TokenRegistry`.
- Require HTTPS except for loopback test/development URLs. Normalize trailing slashes.

`RigFacility` is deliberately metadata, not a second facility client abstraction.

### `Client.facility_via_rig`

Proposed signature:

```python
def facility_via_rig(
    self,
    name: str,
    *,
    rig_url: str = "https://rig.staging.american-science-cloud.org",
    api_prefix: str | None = None,
) -> FacilityClient:
    ...
```

Behavior:

1. Discover the named facility through `rig_facilities()` unless `api_prefix` is explicitly supplied for controlled testing or forward compatibility.
2. Derive the IRI prefix from RIG metadata:
   - Prefer the prefix implied by `metadata_path` when it ends in `/facility`.
   - Otherwise use `/api/<api_version>`.
3. Construct the RIG facility base as `<rig_url>/rig/external/<url-encoded-name>`.
4. Configure the generated IRI stack so its embedded versioned operation paths resolve exactly once. The implementation must prove this with request-URL tests; it must not assume the currently ineffective direct-facility `FacilityConfig.api_prefix` will solve routing.
5. Supply an adapter around the *same primary authenticator* to the RIG-specific credential registry. This adapter preserves refresh/invalidation behavior without copying token strings into config or object representations.
6. Return the existing high-level `FacilityClient`.
7. Cache by normalized `(rig_url, facility name, resolved API prefix)`, separately from `_facility_clients` used by direct access.

The method is experimental and temporary. Its docstring and release notes will say so. Removal or replacement requires a normal deprecation path if it ships in a release.

## Authentication and refresh isolation

- `Client(...)` owns the AmSC Keycard/PAT authenticator.
- `facility_via_rig(...)` borrows that authenticator through a narrow adapter because RIG expects the AmSC bearer.
- `facility(...)` continues to use an independently registered facility-native authenticator.
- A RIG 401/403 may invalidate and reacquire the primary credential once.
- A direct facility 401/403 may invalidate and reacquire only that facility credential.
- Neither path may invalidate the other credential domain.
- Tests must use distinct sentinel values and verify object identity and invalidation calls, not merely successful responses.

## Error handling

The helper fails closed on:

- missing or unusable primary AmSC credential;
- non-HTTPS non-loopback RIG URL;
- facility absent from authenticated RIG discovery;
- malformed `/ready` responses;
- missing API-version metadata when no explicit prefix is supplied;
- ambiguous or duplicate facility records;
- response status outside the supported authentication retry policy.

Exceptions must not include authorization headers, token values, arbitrary upstream request dumps, or full unbounded bodies. A facility probe failure in the multi-facility notebook is reported per facility and does not abort probes for unrelated facilities.

## Notebook design

All three notebooks:

- live under `notebooks_via_rig/`;
- use `AMSC_TOKEN` from the environment and never persist, decode, print, compare, or embed it;
- identify the credential domain as AmSC Keycard/PAT → RIG;
- explain that Credential Vault setup and facility allocations may still be required;
- use the high-level client helper rather than raw `requests` for RIG calls;
- have cleared outputs and null execution counts in git;
- distinguish static validation from dated live validation;
- show non-secret, bounded error summaries;
- pin to the first released client version that contains the helper. The tutorial dependency pin is updated only after that immutable release is published and clean-install verified.

### ALCF notebook

Flow:

1. Validate required environment-backed inputs.
2. Construct one `Client` with the staging central API and AmSC token.
3. Discover ALCF through RIG and create `facility_via_rig("alcf")`.
4. Perform protected facility/account discovery through a public high-level helper where available; do not use public resource discovery alone as authentication proof.
5. List resources and select Polaris/Home by discovered names.
6. Keep `SUBMIT_JOB = False` by default.
7. If enabled, require `ALCF_USERNAME`, `ALCF_ACCOUNT`, queue, existing home output directory, explicit stdout/stderr paths, and ALCF filesystem attributes.
8. Submit, poll to a bounded terminal result, then read a bounded stdout excerpt.

The notebook describes ALCF's RIG tier and Credential Vault setup without claiming that an AmSC PAT alone grants facility access.

### NERSC notebook

Flow mirrors ALCF but uses NERSC metadata and parameters:

- `NERSC_USERNAME`
- `NERSC_ACCOUNT`
- `NERSC_QUEUE`
- `NERSC_CONSTRAINT`
- existing NERSC home output path
- explicit stdout/stderr paths

Submission remains default-off. The notebook must not claim live validation unless a protected read actually succeeds through staging RIG on a dated run. It explains the Tier-3/vaulted credential requirement without teaching users to paste a bearer into notebook cells.

### Multi-facility notebook

This notebook is read-only:

1. Call `rig_facilities()` at runtime.
2. Display name, tier, API version, and display name.
3. Probe each facility independently through `facility_via_rig()` and a protected account/project read where the public façade supports it.
4. Report `OK`, `NO PROJECTS`, or a bounded failure category without exposing upstream bodies.
5. Allow selection of a reachable facility for metadata and resource inspection.
6. Perform no job submission or filesystem mutation.

The notebook does not hard-code a supposedly permanent facility inventory. Unsupported version metadata is shown as unsupported rather than guessed.

## Public protected-read gap

The current high-level `FacilityClient` has no stable public account/projects method. The design requires adding the narrow public method needed by all three notebooks (for example `projects()`), delegating through AmSCROT and translating exceptions consistently. We will inspect the installed dependency's exact surface before fixing the method name and return type in the implementation plan. We will not teach a private `_client()._service_client...` chain in a new tutorial.

## Testing strategy

### Client tests — test first

Tests will first fail for the desired API and then drive the implementation. They must cover:

- authenticated `/ready` request URL and headers;
- typed parsing and deterministic ordering;
- malformed, duplicate, and missing facility metadata;
- HTTPS enforcement with loopback allowance;
- exact RIG operation URLs with no doubled or omitted API prefix;
- URL encoding of facility names;
- distinct direct and RIG client caches;
- primary authenticator identity and token use;
- primary-only 401/403 refresh/invalidation and one retry;
- no direct-facility registry mutation;
- token-safe `repr` and exceptions;
- protected projects method delegation and exception translation.

Mocked tests establish client behavior, not live RIG compatibility.

### Tutorial tests

Extend the repository validator to include all three notebooks and verify:

- valid notebook JSON and compilable Python cells;
- cleared outputs and execution counts;
- environment-backed token input;
- absence of likely embedded credentials and raw `requests` RIG calls;
- expected helper calls;
- default-off, structurally dominant submission gates;
- required ALCF/NERSC output fields and environment-backed parameters;
- no mutation calls in the multi-facility notebook;
- README links and authentication explanation;
- exact published client pin after release.

### Verification levels

Report separately:

1. unit/static validation;
2. clean installation of the published wheel;
3. protected-read live validation through staging RIG;
4. live scheduler or filesystem mutation.

No scheduler submission is part of normal automated verification. It requires explicit authorization and appropriate user credentials/allocation.

## Repository and release sequence

1. Implement and test the Python-client helper on an isolated branch/worktree based on current remote `main`.
2. Independently review the exact client commit and resolve all Critical/Important findings.
3. Publish and clean-install an immutable client release containing the helper.
4. Implement the notebooks in an isolated tutorial worktree and pin that exact release.
5. Run the full clean tutorial suite and independent exact-HEAD review.
6. Publish branches/MRs only after explicit approval.

The user's dirty `notebooks/nersc_facility_tutorial.ipynb` on the tutorial main worktree remains untouched.

## Out of scope

- Replacing RIG or moving token exchange into `amsc-api`.
- Changing existing direct `client.facility()` semantics.
- Persisting PATs in a new client-managed token file.
- Supporting arbitrary IRI v2 behavior without compatible generated SDK evidence.
- Reimplementing all IRI REST models in the tutorial.
- Live submissions during ordinary tests.
- Claiming ALCF, NERSC, or cross-facility success without dated protected endpoint evidence.
