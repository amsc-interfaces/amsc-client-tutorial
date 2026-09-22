# amsc-client 0.6 Tutorial Migration Design

## Status

Approved in conversation by Taylor Childers on 2026-09-22. The release version was resolved after approval and is fixed at `amsc-client==0.6.0`.

## Goal

Migrate the complete `amsc-client-tutorial` repository to the public API and authentication model shipped in `amsc-client 0.6.0`, while making future documentation drift detectable with automated checks.

## Release prerequisite

The tutorial must target an installable release, not a source checkout or prerelease. The client release chain is:

- Git tag: `v0.6.0`
- Commit: `cc608a4ba93e1fba2e2e1132a8bbfef08cdac256`
- GitLab pipeline: `2872466872`
- Published package: `amsc-client 0.6.0`, registry package ID `70251794`

The tutorial dependency will use an exact `amsc-client==0.6.0` pin so examples and validation resolve the reviewed API.

## Scope

Update all user-facing material:

- `README.md`
- `notebooks/catalog_explorer.ipynb`
- `notebooks/catalog_tutorial.ipynb`
- `notebooks/alcf_facility_tutorial.ipynb`
- `notebooks/nersc_facility_tutorial.ipynb`
- `notebooks/filesystem_tutorial.ipynb`
- `docs/agentic-guide-to-polaris-with-iri.md`
- `docs/pytorch-distributed-training-on-polaris.md`
- dependency metadata

Add repository-local validation and GitHub Actions CI. Do not migrate the facility examples to IRI v2, route them through RIG, or introduce live write tests.

## Authentication architecture

The tutorials will teach two independent credential domains:

```text
Primary Client authenticator / AmSC Keycard
  -> staging AmSC services (Catalog, Account, Workflow, MLflow)

Named facility authenticator in TokenRegistry
  -> direct facility-native IRI v1 endpoint
```

Rules:

- The AmSC API base is `https://api.staging.american-science-cloud.org/api/current` for executable central-service examples.
- Central examples consume `AMSC_TOKEN`, which must be an AmSC Keycard/access token, never a Passport/ID token.
- Facility examples use the built-in ALCF or NERSC native Globus configuration or explicitly register a facility authenticator. They never imply that a dummy primary token authenticates the facility.
- A mixed-auth example demonstrates one `Client` with an AmSC Keycard plus an independent ALCF authenticator.
- Tokens are read from environment variables or interactive authenticators and are never printed, embedded in notebooks, or committed.
- IRI remains on v1.

## Notebook design

### Catalog Explorer

A read-only introduction using `AMSC_TOKEN`. It validates caller identity, searches staging catalog data, and handles empty results without assuming a particular catalog population.

### Catalog Tutorial

Uses the same Keycard setup. All mutating calls are behind an explicit `ENABLE_WRITES = False` gate. The notebook requires users to supply an authorized catalog name and records created FQNs for deterministic cleanup. A failed create must not leave later cells referencing an unbound result.

### ALCF facility tutorial

Explains that the ALCF built-in uses its own Globus authenticator. Public discovery is not presented as authentication proof. Authenticated operations are clearly separated, and job submission remains opt-in with user/project/output configuration supplied by the reader.

### NERSC facility tutorial

Uses the built-in `nersc` facility instead of obsolete custom registration. NERSC-specific account, path, queue, and resource names are parameters rather than universal constants.

### Filesystem tutorial

Uses only the public filesystem surface in 0.6.0: `ls`, `stat`, `head`, `tail`, `checksum`, `download`, `mkdir`, `rm`, `cp`, `mv`, `symlink`, `upload`, `chmod`, `compress`, and `extract`. It removes `view()` and `file()`. It describes AmSCROT's compatibility `Task` as already resolved: `wait()` is a no-op and live task polling/cancellation fields are not demonstrated. Destructive operations use a uniquely named tutorial directory and explicit cleanup.

All notebook outputs and execution counts remain cleared in git.

## Long-form guide design

Both guides retain their scientific/HPC value but adopt the 0.6.0 setup and terminology. Personal values become environment-backed examples such as `ALCF_USERNAME`, `ALCF_ACCOUNT`, and `CONTAINER_IMAGE`, with no executable user-specific defaults.

The agentic Polaris guide will correct stale capability claims. It will distinguish:

- operations exposed by `amsc-client 0.6.0`;
- operations known to work at ALCF today;
- generated/spec operations that remain unimplemented;
- historical workarounds that are no longer the preferred path.

Base64-via-job script transfer is not presented as the default when the public upload path is available. If retained as a historical/fallback technique, it is labeled accordingly and not represented as an API requirement.

Machine-stack details that may drift—module versions, library paths, queue limits, container runtime versions—are labeled as observations from a dated tested configuration rather than timeless API facts.

## Repository validation

Create a focused Python validator and pytest suite with these checks:

1. Every notebook parses as valid notebook JSON.
2. Every Python code cell compiles.
3. Outputs and execution counts are cleared.
4. README notebook links resolve.
5. Dependency metadata pins `amsc-client==0.6.0` and includes every required package registry.
6. Legacy patterns are absent, including the `<0.5` cap, central `use_id_token=True`, dummy-token wording, custom built-in NERSC registration, and removed filesystem methods.
7. Imports and referenced public APIs exist in a clean environment containing the published 0.6.0 wheel.
8. Likely credentials are not embedded in tracked tutorial content.

GitHub Actions will install the exact released wheel and run the validation suite on pushes and pull requests. Live tests remain opt-in and are not run in untrusted PR CI.

## Live read-only smoke test

Provide an explicit script or pytest marker requiring two independently supplied credentials:

- `AMSC_TOKEN` for staging;
- `ALCF_IRI_TOKEN` for direct ALCF IRI v1.

The smoke test will:

1. verify staging and ALCF endpoint/version reachability;
2. call a protected staging account endpoint;
3. call a protected ALCF account endpoint;
4. construct one client with separate primary and ALCF authenticators;
5. verify object and token-domain separation without logging token values;
6. report central and facility results separately.

It will not submit jobs, mutate catalog records, or change filesystems. Public facility/resource discovery and empty job lists are not accepted as authentication proof.

## Error handling and safety

- Missing environment variables produce actionable skip/error messages.
- Write examples default off and require explicit configuration.
- Cleanup operates only on artifacts created during the current tutorial run.
- Authentication failures distinguish missing/invalid Keycards from facility-native failures and ALCF high-assurance expiration.
- Validation never prints bearer tokens.
- The repository contains no user-specific executable defaults.

## Acceptance criteria

The migration is complete when:

- all scoped notebooks and guides use the 0.6.0 public interface and mixed-auth model;
- a clean environment installs `amsc-client==0.6.0` from the documented registries;
- static validation passes for every notebook and guide;
- all documented imports and public method references resolve against the published wheel;
- read-only staging and ALCF smoke tests pass with independent credentials;
- no write, filesystem mutation, or job submission is performed during verification;
- GitHub Actions passes on the PR;
- the PR records the exact release tag, commit, and test evidence.
