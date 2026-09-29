# NERSC RIG token instructions design

## Goal

Add safe, reproducible NERSC credential setup and recovery instructions near the top of `notebooks_via_rig/nersc_via_rig.ipynb`.

## Design

The notebook will use Markdown only; it will not acquire, read, print, or upload credentials. The new section will:

1. Link the NERSC token helper and staging RIG Credential Vault.
2. Give the exact NERSC-only helper command using `--facilities nersc --force-login --validate-iri --print-token`.
3. Require successful direct NERSC validation before vaulting the token.
4. Identify the correct value as the `NERSC IRI API access token`, scoped to `https://auth.globus.org/scopes/ed3e577d-f7f3-4639-b96e-ff5a8445d699/iri_api`.
5. Tell users to select the same active project represented by their PAT, save or rotate the NERSC credential, and use **Verify via RIG**.
6. Keep `AMSC_TOKEN` explicitly defined as the AmSC PAT rather than the NERSC token.
7. Explain that `Inactive token` or `Invalid token` requires obtaining, directly validating, and rotating a fresh NERSC IRI token.

## Verification

A repository contract test will require the links, command flags, scope, credential distinction, project guidance, vault actions, and recovery wording. Existing checks will continue to require valid notebook JSON, compilable code cells, no outputs, null execution counts, and no JWT-like token material.
