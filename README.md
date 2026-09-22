# AmSC Client Tutorials

Tutorial notebooks for the [AmSC Python Client](https://gitlab.com/amsc2/infrastructure-and-services/amsc-interfaces/amsc-python-client) — a unified SDK for the American Science Cloud APIs, targeting `amsc-client==0.6.0`.

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

This installs `amsc-client==0.6.0` and Jupyter from the four public AmSC GitLab package registries.

### 4. Set up authentication

Central-service tutorials (catalog) require an **AmSC Keycard** — an OAuth2 access token for the AmSC staging API:

```bash
export AMSC_TOKEN='<your-amsc-keycard>'
```

The AmSC staging API endpoint is `https://api.staging.american-science-cloud.org/api/current`.

Facility tutorials (ALCF, NERSC, filesystem) use independent Globus-based authenticators. The Globus login is triggered automatically on the first facility call — a browser window will open.

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

## Supported Facilities

Both ALCF and NERSC are **built-in** facilities — no manual registration required:

```python
from amsc_client import Client

client = Client()  # No token needed for facility-only access

alcf  = client.facility("alcf")
nersc = client.facility("nersc")
```

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

NERSC is a built-in facility in `amsc-client 0.6.0` — use `client.facility("nersc")` directly:

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

`amsc-client 0.6.0` is distributed across four public GitLab package registries. If `pip install -r requirements.txt` fails, try:

```bash
pip install amsc-client==0.6.0 \
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
