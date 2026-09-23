# AmSC Client Tutorials

Tutorial notebooks for the [AmSC Python Client](https://gitlab.com/amsc2/infrastructure-and-services/amsc-interfaces/amsc-python-client) — a unified SDK for the American Science Cloud APIs, targeting `amsc-client==0.6.1`.

## Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/amsc-interfaces/amsc-client-tutorial.git
cd amsc-client-tutorial
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `amsc-client==0.6.1` and Jupyter from the four public AmSC GitLab package registries.

### 4. Set up authentication

Central-service tutorials (catalog) require an **AmSC Keycard** — an OAuth2 access token for the AmSC staging API:

```bash
export AMSC_TOKEN='<your-amsc-keycard>'
```

The AmSC staging API endpoint is `https://api.staging.american-science-cloud.org/api/current`.

Facility tutorials (ALCF, NERSC, filesystem) use independent facility-native Globus authenticators. `client.facility("alcf")` and `client.facility("nersc")` do not reuse `AMSC_TOKEN`. When a protected facility call needs a credential and no usable cached credential exists, the client prints an authorization URL. Open it, log in with the appropriate facility identity, then paste the returned authorization code into the prompt.

**Credential safety:** Never paste a token into a notebook cell, save it in notebook output, commit it, or print it. Export credentials in the shell before starting Jupyter. If a token is exposed, revoke or rotate it and clear the notebook output. The AmSC Passport/ID token is not an API bearer; central API calls require the AmSC Keycard access token.

### 5. Launch Jupyter

```bash
jupyter notebook notebooks/
```

## Tutorials

| Notebook | Description | Auth Required |
|----------|-------------|---------------|
| [**Catalog Explorer**](notebooks/catalog_explorer.ipynb) | Browse the AmSC data catalog — search, filter, and inspect scientific works and artifacts on staging | `AMSC_TOKEN` (AmSC Keycard) |
| [**Catalog Tutorial**](notebooks/catalog_tutorial.ipynb) | Full CRUD operations — create, update, search, and delete catalog entities | `AMSC_TOKEN` + write access |
| [**ALCF Facility Tutorial**](notebooks/alcf_facility_tutorial.ipynb) | Connect to ALCF, explore Polaris and other resources, and optionally submit a job | ALCF Globus (facility-native) |
| [**NERSC Facility Tutorial**](notebooks/nersc_facility_tutorial.ipynb) | Connect to NERSC, explore Perlmutter resources, and optionally submit a job | NERSC Globus (facility-native) |
| [**Filesystem Tutorial**](notebooks/filesystem_tutorial.ipynb) | Filesystem operations on ALCF resources — ls, head, tail, stat, cp, mv, mkdir, rm, and more | ALCF Globus (facility-native) |

### Recommended order

1. **Catalog Explorer** — read-only, requires only an AmSC Keycard
2. **ALCF Facility Tutorial** — requires an ALCF account, allocation, and IRI API access
3. **NERSC Facility Tutorial** — requires a NERSC account, allocation, and IRI API access
4. **Filesystem Tutorial** — requires an ALCF account and IRI API access
5. **Catalog Tutorial** — requires write access to a staging catalog

## Authentication architecture

The client uses **two independent credential domains**:

```text
AMSC_TOKEN (AmSC Keycard / access token)
  → staging AmSC services: Catalog, Account, Workflow, MLflow
  → base URL: https://api.staging.american-science-cloud.org/api/current

Facility-native Globus authenticator (ALCF or NERSC)
  → direct IRI v1 facility API (alcf or nersc built-in)
  → no AMSC_TOKEN required for facility-only access
```

**Important:** `AMSC_TOKEN` must be an AmSC access token (Keycard), not an OIDC ID token (Passport). Never print or embed tokens in notebooks.

### Authentication quick reference

- **Catalog notebooks:** export `AMSC_TOKEN` before starting Jupyter. These calls target the AmSC staging API.
- **ALCF and filesystem notebooks:** no `AMSC_TOKEN` is needed for facility-only access. The built-in ALCF authenticator starts an ALCF Globus login when a protected IRI v1 call first needs a token.
- **NERSC notebook:** no `AMSC_TOKEN` is needed for facility-only access. The built-in NERSC authenticator uses a separate NERSC Globus login.
- **Stale facility login:** restart the kernel and reauthenticate. If necessary, remove `~/.amsc/credentials.json` to force a fresh login. Never copy credentials from that file into a notebook.
- **Write safety:** leave `ENABLE_WRITES=False` and `SUBMIT_JOB=False` until the required account, allocation, API access, destination paths, and project values have been confirmed.

### Validation status

Validation claims are scoped and dated; they do not imply that every example or write path has run at every facility.

- **AmSC staging — Live-validated 2026-09-22:** the AmSC Keycard authenticated a protected central account read. Read-only catalog examples are also covered by the static suite; catalog writes still require the user's authorized staging catalog.
- **ALCF direct IRI v1 — Live-validated 2026-09-22:** an independent facility-native Globus token authenticated a protected account-project read through the same `Client`. Public facility discovery does not prove authentication. Job submission and filesystem mutations remain explicit opt-in operations.
- **NERSC — Not yet live-validated by the tutorial maintainers:** NERSC is built into `amsc-client 0.6.1`, and the notebook is statically checked against the published API, but its login, protected reads, filesystem access, and submission flow still require live validation.
- **OLCF — Not currently covered:** No built-in `amsc-client 0.6.1` facility configuration exists, and this repository has no OLCF tutorial. Do not adapt the NERSC or ALCF examples by changing only the facility name.

### Verify both authentication domains

After obtaining both credentials, run the opt-in, read-only mixed-auth smoke test:

```bash
export AMSC_TOKEN='<your-amsc-keycard>'
export ALCF_IRI_TOKEN='<your-alcf-iri-token>'
python scripts/smoke_mixed_auth.py
```

It uses one `Client`, but keeps the central Keycard and facility-native ALCF
credential independent. It proves each credential against a protected account
read and performs no job submissions, catalog writes, or filesystem mutations.

## Client-integrated facilities

Both ALCF and NERSC are **built-in** facilities — no manual registration required:

```python
from amsc_client import Client

client = Client()  # No token needed for facility-only access

alcf  = client.facility("alcf")
nersc = client.facility("nersc")
```

OLCF is not listed because `amsc-client 0.6.1` does not provide a built-in OLCF facility configuration and this repository has no OLCF tutorial.

### ALCF (Argonne Leadership Computing Facility)

```python
alcf    = client.facility("alcf")
polaris = alcf.resource("Polaris")

# Set SUBMIT_JOB = True to actually submit; False (default) is safe to explore
SUBMIT_JOB = False

if SUBMIT_JOB:
    job = polaris.submit(
        executable="/bin/echo",
        arguments=["Hello from Polaris!"],
        nodes=1,
        queue="debug",
        account=os.environ["ALCF_ACCOUNT"],
        duration=300,
        filesystems="home",     # ALCF-specific: PBS filesystem mounts
    )
```

| Detail | Value |
|--------|-------|
| API endpoint | `https://api.alcf.anl.gov/api/v1/` |
| Compute resources | Polaris, Aurora, Sophia |
| Storage resources | Home, Eagle |
| Scheduler | PBS |
| Account signup | [accounts.alcf.anl.gov](https://accounts.alcf.anl.gov/) |
| Custom attributes | `filesystems` — comma-separated list of filesystem mounts (e.g., `"home"`, `"home,eagle"`) |

### NERSC (National Energy Research Scientific Computing Center)

NERSC is a built-in facility in `amsc-client 0.6.1` — use `client.facility("nersc")` directly:

```python
nersc      = client.facility("nersc")
perlmutter = nersc.resource("compute")

# Set SUBMIT_JOB = True to actually submit; False (default) is safe to explore
SUBMIT_JOB = False

if SUBMIT_JOB:
    job = perlmutter.submit(
        executable="/bin/echo",
        arguments=["Hello from Perlmutter!"],
        nodes=1,
        queue="debug",
        account=os.environ["NERSC_ACCOUNT"],
        duration=300,
        constraint="gpu",       # NERSC-specific: Slurm constraint
    )
```

| Detail | Value |
|--------|-------|
| API endpoint | `https://api.iri.nersc.gov/api/v1/` |
| Compute resources | Perlmutter (`compute`) |
| Storage resources | `scratch`, `homes`, `common`, `cfs` |
| Scheduler | Slurm |
| Account signup | [iris.nersc.gov](https://iris.nersc.gov/) |
| Custom attributes | `constraint` — Slurm constraint (e.g., `"gpu"`, `"cpu"`) |

## Prerequisites

### For all tutorials
- Python 3.11+
- A [Globus](https://www.globus.org/) account

### For catalog tutorials
- An AmSC Keycard (access token) in `AMSC_TOKEN`
- For write operations: staging catalog write access

### For ALCF tutorials
- An [ALCF account](https://accounts.alcf.anl.gov/)
- An active ALCF project allocation
- **IRI API allowlist access** — having an ALCF account is not sufficient. Email [ALCF support](https://help.alcf.anl.gov) with your ALCF username and use case to request access. Without it, job submission returns HTTP 401.

### For NERSC tutorials
- A [NERSC account](https://iris.nersc.gov/)
- An active NERSC project allocation
- **IRI API allowlist access** — Email [NERSC support](https://help.nersc.gov) with your NERSC username and use case. Without it, job submission returns HTTP 401.

## Troubleshooting

### 401 errors on job submission (not on the IRI API allowlist)

If you receive an `HTTP 401` error when submitting a job (distinct from a login failure), your account may not be on the facility's IRI API access list. Contact the relevant support team:

- **ALCF:** [help.alcf.anl.gov](https://help.alcf.anl.gov)
- **NERSC:** [help.nersc.gov](https://help.nersc.gov)

### ALCF high-assurance token expiry

ALCF uses a high-assurance Globus auth policy — the login session carries a time-limited assurance level that can expire independently of the Globus token itself. If you see repeated `AuthenticationError` after a previously successful login:

1. **Re-authenticate interactively** using the official ALCF login flow at [alcf.anl.gov](https://www.alcf.anl.gov) with your ALCF (not personal Globus) identity. Clearing browser cookies or deleting the local credential cache alone is not sufficient — the high-assurance session is issued by ALCF's Keycloak and must be renewed through the official ALCF identity portal.
2. After renewing your ALCF session, restart the notebook kernel and re-run from the beginning.

### Package not found

`amsc-client 0.6.1` is distributed across four public GitLab package registries. If `pip install -r requirements.txt` fails, try:

```bash
pip install amsc-client==0.6.1 \
  --extra-index-url https://gitlab.com/api/v4/projects/77567162/packages/pypi/simple \
  --extra-index-url https://gitlab.com/api/v4/projects/76368190/packages/pypi/simple \
  --extra-index-url https://gitlab.com/api/v4/projects/80654726/packages/pypi/simple \
  --extra-index-url https://gitlab.com/api/v4/projects/82001936/packages/pypi/simple
```

## Links

- [AmSC Python Client](https://gitlab.com/amsc2/infrastructure-and-services/amsc-interfaces/amsc-python-client) — source code and API docs
- [AmSC Portal](https://my.american-science-cloud.org) — web interface
- [Globus](https://www.globus.org/) — authentication and data transfer
- [ALCF](https://www.alcf.anl.gov/) — Argonne Leadership Computing Facility
- [NERSC](https://www.nersc.gov/) — National Energy Research Scientific Computing Center
- [DOE IRI](https://www.exascaleproject.org/research-group/iri/) — Integrated Research Infrastructure
