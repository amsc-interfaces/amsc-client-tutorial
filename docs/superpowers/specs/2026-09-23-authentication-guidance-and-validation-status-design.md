# Authentication Guidance and Validation Status Design

## Status

Approved by Taylor Childers on 2026-09-23.

## Goal

Make authentication requirements and validation scope unmistakable as users progress through the tutorial notebooks, without implying that untested facilities or workflows have been live-validated.

## Authentication model

The documentation will distinguish two independent credential domains:

1. **AmSC central services** use `AMSC_TOKEN`, an AmSC Keycard OAuth2 access token, against the staging AmSC API. An OIDC Passport/ID token must never be used as the API bearer.
2. **Direct facility APIs** use facility-native credentials. ALCF and NERSC are built-in `amsc-client 0.6.0` facilities whose Globus authenticators are resolved independently from the central Keycard. Facility-only notebooks do not require `AMSC_TOKEN`.

The README will explain what users should expect when authentication occurs, which environment variables are required, how to recover from stale facility credentials, and that tokens must never be pasted into or saved by notebooks.

## Notebook guidance

Each notebook will begin with a concise authentication and validation banner:

- `catalog_explorer.ipynb`: AmSC Keycard; read-only; live authentication and protected account access validated against staging.
- `catalog_tutorial.ipynb`: AmSC Keycard; catalog mutations remain default-off; authentication/read-only paths validated, but write execution requires the user's own authorized staging catalog.
- `alcf_facility_tutorial.ipynb`: independent ALCF Globus authentication; protected ALCF IRI v1 authentication validated; submission remains default-off and requires an account, allocation, and allowlist access.
- `filesystem_tutorial.ipynb`: independent ALCF Globus authentication; mutations remain default-off and operate only in run-unique paths.
- `nersc_facility_tutorial.ipynb`: independent NERSC Globus authentication; API examples are statically validated against the published client but have not yet been live-validated by the maintainers.

No notebook will print, embed, compare, or serialize credential values.

## Facility status

The README will use a dated validation matrix that separates:

- client integration status;
- live authentication/read-only validation;
- live mutation or job-workflow validation.

As of 2026-09-23:

- AmSC staging authentication/read-only paths: live-validated.
- ALCF direct IRI v1 authentication/read-only paths: live-validated.
- NERSC: built into `amsc-client 0.6.0`, but not yet live-validated by the tutorial maintainers.
- OLCF: no built-in `amsc-client 0.6.0` facility configuration and no tutorial; not currently supported by this repository. OLCF must not be described as merely awaiting validation.

Public facility discovery is not authentication proof. A protected account read is required to claim live authentication validation.

## Validation

The static suite will require:

- the README validation matrix and credential-domain guidance;
- explicit NERSC not-yet-live-validated language;
- explicit OLCF unsupported/no-built-in-integration language;
- per-notebook authentication and validation banners;
- no embedded credential values;
- default-off write and submission gates;
- cleared notebook outputs and execution counts.

The existing clean-install tests, notebook compilation, API-signature checks, and CI remain mandatory.

## Out of scope

- Adding an OLCF facility implementation or notebook.
- Claiming NERSC authentication or job submission works without a live test.
- Running live submissions or filesystem mutations.
- Changing client authentication behavior.
