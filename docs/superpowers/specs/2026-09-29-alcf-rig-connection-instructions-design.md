# ALCF RIG Connection Instructions Design

## Goal

Add safe, reproducible ALCF Tier-2 connection and recovery instructions near the top of `notebooks_via_rig/alcf_via_rig.ipynb`.

## Design

The notebook will use Markdown only; it will not acquire, read, print, or upload credentials or authorization codes. The new section will:

1. Link the staging RIG Credential Vault.
2. Keep `AMSC_TOKEN` explicitly defined as the AmSC PAT rather than an ALCF facility credential.
3. Tell users to confirm that the Credential Vault's active project matches the project encoded in the PAT.
4. On the ALCF card, use **Get Auth URL**, authenticate through Globus with the **Argonne LCF (`alcf.anl.gov`) identity**, enter the returned value in **Authorization Code**, and click **Submit Code**.
5. Require **Connected** status and **Test with RIG** before running protected notebook cells.
6. Explain recovery for disconnected credentials, ordinary 401 failures, and ALCF high-assurance timeout failures by repeating a fresh interactive connection.
7. Prohibit putting the authorization code or resulting ALCF credential in notebook cells, chat, shell history, screenshots, or committed files.

## Verification

A repository contract test will require the vault URL, project and credential-domain distinction, exact connection controls, identity guidance, verification gate, recovery wording, and credential hygiene. Existing checks will continue to require valid notebook JSON, compilable code cells, no outputs, null execution counts, and no JWT-like token material.
