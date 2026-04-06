# AmSC Client Tutorials

Tutorial notebooks for the [AmSC Python Client](https://gitlab.com/amsc2/infrastructure-and-services/amsc-interfaces/amsc-python-client) — a unified SDK for the American Science Cloud APIs.

## Getting Started

### 1. Clone this repository

```bash
git clone https://gitlab.com/amsc2/amsc-client-tutorial.git
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
| [**Facility Tutorial**](notebooks/facility_tutorial.ipynb) | Connect to ALCF, list resources, submit a job to Polaris, monitor status, and read output via the filesystem API | Globus (ALCF) |
| [**Filesystem Tutorial**](notebooks/filesystem_tutorial.ipynb) | Filesystem operations on ALCF resources — ls, head, tail, stat, cp, mv, mkdir, and more | Globus (ALCF) |

### Recommended order

1. **Catalog Explorer** — read-only, works for everyone with a Globus account
2. **Facility Tutorial** — requires an ALCF account and allocation
3. **Filesystem Tutorial** — requires an ALCF account
4. **Catalog Tutorial** — requires write access to a data catalog

## Prerequisites

### For all tutorials
- Python 3.10+
- A [Globus](https://www.globus.org/) account

### For ALCF tutorials (facility + filesystem)
- An [ALCF account](https://accounts.alcf.anl.gov/)
- An active ALCF project allocation (e.g., `datascience`)
- Access to Polaris (or another ALCF compute resource)

## Troubleshooting

### Persistent 401 errors after re-authentication (ALCF)

ALCF tokens embed a Keycloak identity token inside the Globus access token. This embedded token can expire even while the Globus token itself remains valid. If you see repeated `AuthenticationError: Authentication failed (401)` errors even after re-authenticating:

1. **Clear your browser cookies** for `globus.org` and `globusid.org` domains
2. **Delete cached credentials**: `rm ~/.amsc/credentials.json`
3. **Restart your notebook kernel** and re-run from the beginning

This forces a full fresh login through ALCF's Keycloak identity provider.

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
