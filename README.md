# AmSC Client Tutorials

Tutorial notebooks for the [AmSC Python Client](https://gitlab.com/amsc2/infrastructure-and-services/amsc-interfaces/amsc-python-client) — a unified SDK for the American Science Cloud APIs.

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

This installs the AmSC client, Globus SDK, and Jupyter from the public AmSC package registries.

### 4. Launch Jupyter

```bash
jupyter notebook notebooks/
```

## Tutorials

| Notebook | Description | Auth Required |
|----------|-------------|---------------|
| [**Catalog Explorer**](notebooks/catalog_explorer.ipynb) | Browse the AmSC data catalog — search, filter, and inspect scientific works and artifacts on staging | Globus (AmSC) |
| [**Catalog Tutorial**](notebooks/catalog_tutorial.ipynb) | Full CRUD operations — create, update, search, and delete catalog entities | Globus (AmSC) + write access |
| [**Facility Tutorial**](notebooks/facility_tutorial.ipynb) | Connect to a DOE facility, list resources, submit a job, monitor status, and read output via the filesystem API | Globus (facility) |
| [**Filesystem Tutorial**](notebooks/filesystem_tutorial.ipynb) | Filesystem operations on facility resources — ls, head, tail, stat, cp, mv, mkdir, and more | Globus (facility) |

### Recommended order

1. **Catalog Explorer** — read-only, works for everyone with a Globus account
2. **Facility Tutorial** — requires an account and allocation at a DOE facility
3. **Filesystem Tutorial** — requires an account at a DOE facility
4. **Catalog Tutorial** — requires write access to a data catalog

## Supported Facilities

The AmSC Python Client uses the [DOE IRI Facility API](https://www.exascaleproject.org/research-group/iri/) standard, which provides a uniform interface across DOE computing facilities. The tutorial notebooks demonstrate ALCF (Polaris) but the same API works at any IRI-compliant facility.

### ALCF (Argonne Leadership Computing Facility)

ALCF is built into the client — no extra configuration needed:

```python
alcf = client.facility("alcf")
polaris = alcf.resource("Polaris")

job = polaris.submit(
    executable="/bin/echo",
    arguments=["Hello from Polaris!"],
    nodes=1,
    queue="debug",
    account="myproject",
    duration=300,
    filesystems="home",         # ALCF-specific: PBS filesystem mounts
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

NERSC also provides an IRI Facility API. To use it, register NERSC as a custom facility:

```python
from amsc_client.facility.config import FacilityConfig

client.register_facility(
    name="nersc",
    config=FacilityConfig(
        name="nersc",
        display_name="National Energy Research Scientific Computing Center",
        base_url="https://api.iri.nersc.gov",
        auth_method="globus",
        globus_client_id="YOUR_GLOBUS_CLIENT_ID",
        globus_scope="YOUR_NERSC_SCOPE",
    ),
)

nersc = client.facility("nersc")
perlmutter = nersc.resource("compute")  # NERSC resource name for Perlmutter

job = perlmutter.submit(
    executable="/bin/echo",
    arguments=["Hello from Perlmutter!"],
    nodes=1,
    queue="regular",
    account="myproject",
    duration=3600,
    constraint="gpu",           # NERSC-specific: Slurm constraint
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

### Key Differences Between Facilities

The IRI API is the same across facilities. The differences are in **scheduler-specific custom attributes** — these are passed as keyword arguments to `submit()`:

| Facility | Scheduler | Common Custom Attributes |
|----------|-----------|--------------------------|
| ALCF | PBS | `filesystems="home"` |
| NERSC | Slurm | `constraint="gpu"` |

Standard IRI parameters (`nodes`, `queue`, `account`, `duration`, `executable`, `arguments`, etc.) work identically across all facilities.

## Prerequisites

### For all tutorials
- Python 3.10+
- A [Globus](https://www.globus.org/) account

### For ALCF tutorials
- An [ALCF account](https://accounts.alcf.anl.gov/)
- An active ALCF project allocation (e.g., `datascience`)
- Access to Polaris (or another ALCF compute resource)
- **IRI API allowlist access** — having an ALCF account is not sufficient on its own. Email [ALCF support](https://help.alcf.anl.gov) with your ALCF username and a brief description of your use case to request access. Without it, job submission returns HTTP 401.

### For NERSC
- A [NERSC account](https://iris.nersc.gov/)
- An active NERSC project allocation
- Access to Perlmutter
- **IRI API allowlist access** — having a NERSC account is not sufficient on its own. Email [NERSC support](https://help.nersc.gov) with your NERSC username and a brief description of your use case to request access. Without it, job submission returns HTTP 401.

## Troubleshooting

### 401 errors on job submission (not on the IRI API allowlist)

If you receive an `HTTP 401` error when submitting a job (distinct from an authentication failure after login), your account may not be on the facility's IRI API access list. Contact the relevant support team with your username and use case:

- **ALCF:** [help.alcf.anl.gov](https://help.alcf.anl.gov)
- **NERSC:** [help.nersc.gov](https://help.nersc.gov)

### Persistent 401 errors after re-authentication (ALCF)

ALCF tokens embed a Keycloak identity token inside the Globus access token. This embedded token can expire even while the Globus token itself remains valid. If you see repeated `AuthenticationError: Authentication failed (401)` errors even after re-authenticating:

1. **Clear your browser cookies** for `globus.org` and `globusid.org` domains
2. **Delete cached credentials**: `rm ~/.amsc/credentials.json`
3. **Restart your notebook kernel** and re-run from the beginning

This forces a full fresh login through the facility's identity provider.

### `globus-sdk` not found

If you get an import error for `globus_sdk`, make sure you installed from the requirements file:

```bash
pip install -r requirements.txt
```

### Package not found

If `amsc-client` can't be found, the GitLab package registry URLs may not be resolving. Try installing manually:

```bash
pip install \
  --extra-index-url https://gitlab.com/api/v4/projects/77567162/packages/pypi/simple \
  --extra-index-url https://gitlab.com/api/v4/projects/76368190/packages/pypi/simple \
  --extra-index-url https://gitlab.com/api/v4/projects/80654726/packages/pypi/simple \
  amsc-client
```

## Links

- [AmSC Python Client](https://gitlab.com/amsc2/infrastructure-and-services/amsc-interfaces/amsc-python-client) — source code and API docs
- [AmSC Portal](https://my.american-science-cloud.org) — web interface
- [Globus](https://www.globus.org/) — authentication and data transfer
- [ALCF](https://www.alcf.anl.gov/) — Argonne Leadership Computing Facility
- [NERSC](https://www.nersc.gov/) — National Energy Research Scientific Computing Center
- [DOE IRI](https://www.exascaleproject.org/research-group/iri/) — Integrated Research Infrastructure
