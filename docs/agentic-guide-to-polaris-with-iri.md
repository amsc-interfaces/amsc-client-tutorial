# Agentic Guide: Running Containerized MPI on Polaris via IRI API

**Target audience:** AI agents using the AmSC Python Client (amsc-client) to build, submit, and run containerized MPI applications on ALCF Polaris.

**What this documents:** Every hard-won lesson from building Pepper (a HEP event generator) in a multi-stage Docker container, pushing it to a registry, and running a successful 2-node, 8-rank MPI job on Polaris — all orchestrated through the IRI API without SSH access.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [IRI API Reference — What Works and What Doesn't](#3-iri-api-reference)
4. [Container Build](#4-container-build)
5. [Polaris Job Submission via IRI](#5-polaris-job-submission-via-iri)
6. [Job Script Template](#6-job-script-template)
7. [The MPI/Container Stack](#7-the-mpicontainer-stack)
8. [Known Failure Modes](#8-known-failure-modes)
9. [Verification Steps](#9-verification-steps)
10. [Output Retrieval](#10-output-retrieval)
11. [Working Code](#11-working-code)

---

## 1. Overview

### What we're doing

Running a containerized MPI+GPU application on ALCF Polaris entirely through the IRI API (no direct SSH, no manual intervention). The workflow:

```
[Local machine]                        [ALCF Polaris]
     │                                       │
     ├─ Build Docker image                   │
     ├─ Push to Docker Hub                   │
     │                                       │
     └─ amsc-client ──────── IRI API ───────>│
          │                                  ├─ PBS job: pull SIF
          │                                  ├─ PBS job: write run script (base64)
          │                                  └─ PBS job: mpiexec + apptainer exec
          │                                       │
          └─ Read stdout/stderr ◄── IRI API ──────┘
```

### Why this matters

The IRI API has significant constraints that require specific workarounds. This guide documents the exact patterns that work, saving you from the ~20 iteration cycles that were needed to discover them.

### Key facts

- **ALCF user:** `parton`, **project:** `datascience`
- **Container registry:** Docker Hub (`jtchilders/pepper-polaris`)
- **amsc-client version:** 0.4.1
- **Polaris nodes:** NVIDIA A100 GPUs (Ampere80), Cray Slingshot network, PBS scheduler
- **Apptainer version on Polaris:** 1.4.1

---

## 2. Prerequisites

### 2.0 IRI API allowlist access

**Having an ALCF account and allocation is not sufficient to use the IRI API.** Your account must be added to the ALCF IRI API access list separately. Email [ALCF support](https://help.alcf.anl.gov) with your ALCF username and a description of your use case to request access. Without this, job submission returns HTTP 401.

### 2.1 amsc-client installation

amsc-client is distributed via a private GitLab package registry. Three extra index URLs are required:

```bash
pip install amsc-client==0.4.1 \
  --extra-index-url https://gitlab.com/api/v4/projects/... \
  --extra-index-url https://... \
  --extra-index-url https://...
```

Check the AmSC client tutorial repo for the current install command:
`github.com/amsc-interfaces/amsc-client-tutorial`

### 2.2 Globus authentication

Auth uses cached Globus credentials. The client reads from `~/.amsc/credentials.json` automatically.

```python
from amsc_client import Client

GLOBUS_APP_ID    = 'e4f48665-38b5-4833-a89e-849c71f5b3e3'
RESOURCE_SERVER  = '8b84fc2d-49e9-49ea-b54d-b3a29a70cf31'

client = Client(
    base_url='https://api.american-science-cloud.org/api/current',
    auth_method="globus",
    globus_client_id=GLOBUS_APP_ID,
    requested_scopes=(
        f'openid profile email '
        f'https://auth.globus.org/scopes/{GLOBUS_APP_ID}/amsc_test'
    ),
    resource_server=RESOURCE_SERVER,
    use_id_token=True,
)
```

On first run, Globus will prompt for browser-based authentication. After that, credentials are cached.

### 2.3 Resource handles

```python
alcf    = client.facility("alcf")
polaris = alcf.resource("Polaris")   # compute resource — submit jobs here
home    = alcf.resource("Home")      # storage resource — read files here
```

**Critical:** `polaris.fs.*` operations return HTTP 400. Filesystem ops only work on storage resources (`home`, Eagle). Never call `polaris.fs.ls()`.

### 2.4 Docker Hub credentials (macOS)

If you see `error -61` from the macOS keychain when pushing images:

```bash
# Remove keychain-based credential store from Docker config
# Edit ~/.docker/config.json and remove: "credsStore": "osxkeychain"
# Then: docker login -u <username> --password-stdin
```

---

## 3. IRI API Reference

### 3.1 Filesystem operations

| Operation | Status | Notes |
|-----------|--------|-------|
| `ls` | ✅ Working | On Home/Eagle only |
| `head` | ✅ Working | Read first N bytes |
| `view` | ✅ Working | Read full file content |
| `chmod` | ✅ Working | |
| `chown` | ✅ Working | |
| `tail` | ❌ 501 Not Implemented | |
| `stat` | ❌ 501 Not Implemented | |
| `checksum` | ❌ 501 Not Implemented | |
| `file` | ❌ 501 Not Implemented | |
| `download` | ❌ 501 Not Implemented | |
| `mkdir` | ❌ 501 Not Implemented | Create dirs via job instead |
| `cp` | ❌ 501 Not Implemented | |
| `mv` | ❌ 501 Not Implemented | |
| `rm` | ❌ 501 Not Implemented | |
| `symlink` | ❌ 501 Not Implemented | |
| `compress` | ❌ 501 Not Implemented | |
| `extract` | ❌ 501 Not Implemented | |

**Filesystem path constraints:**
- Paths must start with `/home/<user>` or `/eagle` / `/lus/eagle`
- Polaris compute paths are not accessible via the filesystem API

### 3.2 Job submission

```python
job = polaris.submit(
    executable="/bin/bash",
    arguments=["-l", "/home/parton/script.sh"],
    directory="/home/parton/workdir",
    name="my-job",
    queue="debug",          # debug | debug-scaling | prod
    account="datascience",
    duration=1800,          # seconds
    nodes=2,
    filesystems="home",
)
job.wait(timeout=1800, poll_interval=15)
print(f"state={job.state}, exit={job.exit_code}")
```

**Queue limits:**
- `debug`: 1–2 nodes, 1 hour max
- `debug-scaling`: 1–10 nodes, 1 hour max
- `prod`: 10–496 nodes, 24 hours max

### 3.3 Job output routing

Output always lands at: `{directory}/{name}.stdout` and `{directory}/{name}.stderr`

The `directory` parameter in `submit()` controls where output files are written.

### 3.4 Critical API limitations

**`pre_launch` parameter: NOT IMPLEMENTED**
```python
# This returns 501:
polaris.submit(..., pre_launch="module load apptainer")  # ❌ BROKEN
```
Use a shell script submitted as the job instead.

**Complex bash arguments break GraphQL parser:**
```python
# HTTP 400 — colons, dollar signs, quotes break the parser:
polaris.submit(executable="/bin/bash",
    arguments=["-c", "export HTTP_PROXY=http://proxy:3128 && mpiexec ..."])  # ❌ BROKEN
```

**The workaround for both:** Two-step pattern — write a script via base64, then execute it.

### 3.5 The base64 script transfer pattern

Because you can't upload files directly and complex bash args break GraphQL, use this two-step pattern for every non-trivial job:

```python
import base64
import time

SCRIPT_CONTENT = r"""#!/bin/bash -l
# ... your full script here ...
"""

# Step 1: base64-encode the script and submit a job to decode it
b64 = base64.b64encode(SCRIPT_CONTENT.encode()).decode()
decode_cmd = (
    f"echo {b64} | base64 -d > {WORK_DIR}/run.sh "
    f"&& chmod +x {WORK_DIR}/run.sh && echo OK"
)

write_job = polaris.submit(
    executable="/bin/bash",
    arguments=["-c", decode_cmd],
    directory=f"/home/{ALCF_USER}",
    name="write-script",
    queue="debug",
    account=PROJECT,
    duration=300,
    nodes=1,
    filesystems="home",
)
write_job.wait(timeout=600, poll_interval=10)
time.sleep(3)  # Let filesystem settle

# Step 2: run the script
run_job = polaris.submit(
    executable="/bin/bash",
    arguments=["-l", f"{WORK_DIR}/run.sh"],
    directory=WORK_DIR,
    name="actual-job",
    queue="debug",
    account=PROJECT,
    duration=1800,
    nodes=2,
    filesystems="home",
)
run_job.wait(timeout=1800, poll_interval=15)
```

**Notes:**
- `echo {b64} | base64 -d` is safe because base64 output contains no shell-special chars
- `time.sleep(3)` after write job helps avoid race conditions with filesystem sync
- The write job only needs 1 node; the run job can use however many you need
- Always use `-l` (login shell flag) when executing the script so `module` commands work

---

## 4. Container Build

### 4.1 Non-negotiable requirements

| Requirement | Why |
|-------------|-----|
| Base image: Ubuntu 24.04+ | glibc ≥ 2.38 required by Cray libfabric 2.2.0rc1 |
| Install to `/usr/local` | Apptainer `-B /opt` bind-mount overwrites container's `/opt` at runtime |
| MPICH 4.1.2 from source | Must match Cray MPICH 9.0.1 (which is ANL MPICH 4.1.2) |
| `--with-device=ch4:ofi` | Required for Slingshot network; `ch4:ucx` (apt default) won't work cross-node |
| Do NOT install `libfabric-dev` | Causes linker errors for psm2/rdmacm/ibverbs/efa providers; MPICH uses embedded OFI |
| `-fno-lto` build flags | LTO + CUDA fatbinData symbol collision (GCC LTO + CUDA fatbinaries conflict) |

### 4.2 How to verify MPICH version on Polaris

Submit this as a probe job to discover what version you need to match:

```bash
#!/bin/bash -l
module use /soft/modulefiles
module load spack-pe-base cray-mpich
/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/bin/mpichversion
```

At time of writing: Cray MPICH 9.0.1 = ANL MPICH base **4.1.2**

### 4.3 Complete working Dockerfile

See [Section 11](#11-working-code) for the full Dockerfile. Key structure:

```
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Build dependencies (no apt mpich — build from source)
RUN apt-get install build-essential cmake wget git libjson-c-dev gfortran ...

# Build MPICH 4.1.2 from source with ch4:ofi
RUN wget mpich-4.1.2.tar.gz && ./configure \
    --prefix=/usr/local \
    --with-device=ch4:ofi \
    --disable-wrapper-rpath \
    --enable-shared \
    && make && make install

# Build your application to /usr/local
# Use -fno-lto for GPU apps

# Add mpi_hello diagnostic binary
RUN mpicc -o /usr/local/bin/mpi_hello mpi_hello.c
```

### 4.4 Container pull on Polaris

Polaris nodes need internet access via proxy. The SIF pull is done inside a job script:

```bash
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

export APPTAINER_TMPDIR=/local/scratch/apptainer-tmpdir
export APPTAINER_CACHEDIR=/local/scratch/apptainer-cachedir
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

apptainer pull /home/parton/pepper-iri-test/pepper-polaris-v4.sif \
    docker://docker.io/jtchilders/pepper-polaris:latest
```

Pull time: ~2 minutes for a 4GB SIF (3.9GB compressed from 12.6GB Docker image).

**Docker tag caching gotcha:** After rebuilding and pushing `docker.io/user/image:latest`, the old SIF may be cached on Polaris. Always use a versioned SIF filename (`pepper-polaris-v4.sif`, `pepper-polaris-v5.sif`) to force a fresh pull.

---

## 5. Polaris Job Submission via IRI

### 5.1 Full two-step submission pattern

```python
import base64
import time
from amsc_client import Client

GLOBUS_APP_ID   = 'e4f48665-38b5-4833-a89e-849c71f5b3e3'
RESOURCE_SERVER = '8b84fc2d-49e9-49ea-b54d-b3a29a70cf31'
ALCF_USER = 'parton'
PROJECT   = 'datascience'
WORK_DIR  = f"/home/{ALCF_USER}/my-iri-job"

client = Client(
    base_url='https://api.american-science-cloud.org/api/current',
    auth_method="globus",
    globus_client_id=GLOBUS_APP_ID,
    requested_scopes=(
        f'openid profile email '
        f'https://auth.globus.org/scopes/{GLOBUS_APP_ID}/amsc_test'
    ),
    resource_server=RESOURCE_SERVER,
    use_id_token=True,
)

alcf    = client.facility("alcf")
polaris = alcf.resource("Polaris")
home    = alcf.resource("Home")

SCRIPT = r"""#!/bin/bash -l
# ... full job script (see Section 6) ...
"""

# Step 1: Write the script
b64 = base64.b64encode(SCRIPT.encode()).decode()
write_job = polaris.submit(
    executable="/bin/bash",
    arguments=["-c",
        f"mkdir -p {WORK_DIR} && "
        f"echo {b64} | base64 -d > {WORK_DIR}/run.sh && "
        f"chmod +x {WORK_DIR}/run.sh && echo OK"
    ],
    directory=f"/home/{ALCF_USER}",
    name="write-script",
    queue="debug",
    account=PROJECT,
    duration=300,
    nodes=1,
    filesystems="home",
)
print(f"Write job: {write_job.id}")
write_job.wait(timeout=600, poll_interval=10)
assert write_job.exit_code == 0, f"Script write failed: {write_job.state}"
time.sleep(3)

# Step 2: Execute the script
run_job = polaris.submit(
    executable="/bin/bash",
    arguments=["-l", f"{WORK_DIR}/run.sh"],
    directory=WORK_DIR,
    name="mpi-run",
    queue="debug",
    account=PROJECT,
    duration=1800,
    nodes=2,
    filesystems="home",
)
print(f"Run job: {run_job.id}")
run_job.wait(timeout=1800, poll_interval=15)
print(f"Done: state={run_job.state}, exit={run_job.exit_code}")
```

### 5.2 Job listing and monitoring

```python
# List recent jobs
jobs = polaris.jobs(limit=20)
for j in jobs:
    print(f"{j.id}: {j.name} state={j.state}")

# Poll a specific job
job = polaris.job(job_id)
print(job.state, job.exit_code)
```

### 5.3 Resource names

| Name | Type | Use |
|------|------|-----|
| `"Polaris"` | Compute | Submit jobs |
| `"Home"` | Filesystem | Read/write `/home/<user>` |
| `"Eagle"` | Filesystem | Read/write `/eagle` or `/lus/eagle` |

---

## 6. Job Script Template

This is the exact working script structure for a 2-node MPI job with GPU containers on Polaris. Every detail matters.

```bash
#!/bin/bash -l
# ^^^ -l (login shell) is MANDATORY for 'module' commands to work
set -e

# ============================================================
# 1. Load modules
# ============================================================
module use /soft/modulefiles
module load spack-pe-base apptainer cray-mpich-abi
# Note: Lmod will auto-replace 'cray-mpich' with 'cray-mpich-abi' — that's correct

# ============================================================
# 2. Apptainer scratch dirs (local NVMe, not home filesystem)
# ============================================================
export APPTAINER_TMPDIR=/local/scratch/apptainer-tmpdir
export APPTAINER_CACHEDIR=/local/scratch/apptainer-cachedir
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

# ============================================================
# 3. Proxy (required for container pulls from login/compute nodes)
# ============================================================
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# ============================================================
# 4. MPI/container environment
# ============================================================
# LD_LIBRARY_PATH visible inside the container:
# - $CRAY_LD_LIBRARY_PATH: Cray's MPI and fabric libraries
# - $LD_LIBRARY_PATH: additional host libraries
# - /opt/cray/pe/pals/1.2.12/lib: PALS launcher library
# - /host/usr/lib64: Cray's libcxi.so.1 (not under /opt!)
export APPTAINERENV_LD_LIBRARY_PATH="\
$CRAY_LD_LIBRARY_PATH:\
$LD_LIBRARY_PATH:\
/opt/cray/pe/pals/1.2.12/lib:\
/host/usr/lib64"

# CRITICAL: Disable CMA (Cross-Memory Attach) — broken in Apptainer containers
# MPI collectives (MPI_Reduce, MPI_Allreduce, etc.) will crash without this
# FI_CXI_DISABLE_CMA=1 and MPICH_SHM_DISABLE_CMA=1 do NOT work — use this:
export APPTAINERENV_MPICH_SMP_SINGLE_COPY_MODE=NONE

# ============================================================
# 5. Job parameters
# ============================================================
SIF=$HOME/my-iri-job/my-app-v4.sif
RUNDIR=$HOME/my-iri-job/run
mkdir -p $RUNDIR

NODES=$(cat $PBS_NODEFILE | wc -l)
RANKS_PER_NODE=4
RANKS=$((NODES * RANKS_PER_NODE))

echo "=== Job Info ==="
echo "Nodes: $NODES"
echo "Ranks: $RANKS"
cat $PBS_NODEFILE

# ============================================================
# 6. Pull the container SIF (if not already present)
# ============================================================
if [ ! -f "$SIF" ]; then
    apptainer pull "$SIF" docker://docker.io/myuser/my-app:v4
fi

# ============================================================
# 7. Run the MPI job
# ============================================================
mpiexec \
    -n $RANKS \
    -ppn $RANKS_PER_NODE \
    --hostfile $PBS_NODEFILE \
    apptainer exec \
        --fakeroot \
        --nv \
        --writable-tmpfs \
        -B /opt \
        -B /var/run/palsd/ \
        -B /usr/lib64:/host/usr/lib64 \
        -B $HOME:$HOME \
        --pwd $RUNDIR \
        "$SIF" \
        my-app --my-args

echo "=== Done ==="
```

### Critical flags explained

| Flag | Why |
|------|-----|
| `#!/bin/bash -l` | Login shell — required for `module` command |
| `--fakeroot` | Required for `apptainer exec` in this configuration |
| `--nv` | Expose NVIDIA GPU and CUDA to container |
| `--writable-tmpfs` | Application cache writes fail without this (read-only filesystem error) |
| `-B /opt` | Bind-mounts host Cray libs (libfabric, MPICH, etc.) |
| `-B /var/run/palsd/` | PALS daemon socket — required for MPI launcher |
| `-B /usr/lib64:/host/usr/lib64` | Cray's `libcxi.so.1` lives here, NOT under `/opt` |
| `--pwd $RUNDIR` | Working directory inside container; use a clean writable dir to avoid stale config files |
| `APPTAINERENV_MPICH_SMP_SINGLE_COPY_MODE=NONE` | Disables CMA — required for MPI collectives |

---

## 7. The MPI/Container Stack

### 7.1 Module load sequence

```bash
module use /soft/modulefiles
module load spack-pe-base apptainer cray-mpich-abi
```

Order matters. `spack-pe-base` must come first. `cray-mpich-abi` provides the host-side MPI ABI that overrides the container's MPICH at runtime.

Lmod will print: `The following have been reloaded with a version change: cray-mpich -> cray-mpich-abi`. This is expected and correct for container workflows.

### 7.2 MPICH ABI compatibility chain

```
Container MPICH 4.1.2 (ch4:ofi)
        ↓ ABI override at runtime
Cray MPICH 9.0.1 (ANL MPICH base 4.1.2)
        ↓
HPE Cray Slingshot (libfabric + libcxi)
        ↓
Physical HSN (High-Speed Network)
```

Why version matching matters: MPICH uses its own ABI internally. If container MPICH is 3.4.3 and host is 4.1.2, the ABI mismatch causes crashes in MPI collectives (but basic `MPI_Send`/`MPI_Recv` may still work — `mpi_hello` passes, Pepper's `MPI_Reduce` crashes).

### 7.3 LD_LIBRARY_PATH assembly

The `APPTAINERENV_LD_LIBRARY_PATH` value is evaluated on the **host** before being passed into the container. The variable expansion happens at the `export` statement, so `$CRAY_LD_LIBRARY_PATH` and `$LD_LIBRARY_PATH` are expanded from the host environment at that point.

```bash
# After module load, $CRAY_LD_LIBRARY_PATH contains paths like:
# /opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib-abi-mpich
# /opt/cray/pe/pmi/6.1.14/lib
# /opt/cray/libfabric/2.2.0rc1/lib64
# etc.
export APPTAINERENV_LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/opt/cray/pe/pals/1.2.12/lib:/host/usr/lib64"
```

### 7.4 Bind mounts and what they provide

| Bind mount | What's inside | Why needed |
|------------|---------------|------------|
| `-B /opt` | Cray PE, libfabric, MPICH, CUDA, etc. | Host MPI and fabric runtime |
| `-B /var/run/palsd/` | PALS daemon Unix socket | MPI job launcher communication |
| `-B /usr/lib64:/host/usr/lib64` | `libcxi.so.1`, `libcxi.so` | Cray Slingshot fabric library |
| `-B $HOME:$HOME` | User's home directory | Output files, SIF file, run directory |

Note: `/usr/lib64` is mounted to `/host/usr/lib64` (not `/usr/lib64`) to avoid overwriting the container's own `/usr/lib64`.

### 7.5 CMA (Cross-Memory Attach) issue

Apptainer containers on Polaris lack the Linux capabilities needed for CMA. MPI collectives (`MPI_Reduce`, `MPI_Allreduce`, `MPI_Bcast`, etc.) use CMA for intra-node shared memory transfers and will crash:

```
CMA does not have sufficient permission
```

**The fix:**
```bash
export APPTAINERENV_MPICH_SMP_SINGLE_COPY_MODE=NONE
```

This disables CMA entirely and falls back to a different shared memory mechanism. **Do not use:**
- `FI_CXI_DISABLE_CMA=1` — does not fix this
- `MPICH_SHM_DISABLE_CMA=1` — does not fix this

`mpi_hello` will pass even without this fix (it only calls `MPI_Comm_rank`/`MPI_Comm_size`). The CMA crash only manifests when your application uses collectives.

---

## 8. Known Failure Modes

| Error | Root Cause | Fix |
|-------|------------|-----|
| `module: command not found` | Script not running as login shell | Use `#!/bin/bash -l` |
| `executable not found in $PATH` | Container `/opt` was bind-overwritten | Install app to `/usr/local`, not `/opt` |
| `libcuda.so.1: cannot open` | GPU library not exposed | Add `--nv` to `apptainer exec` |
| `libcxi.so.1: cannot open` | Cray fabric library not in container | Add `-B /usr/lib64:/host/usr/lib64` and add `/host/usr/lib64` to `APPTAINERENV_LD_LIBRARY_PATH` |
| `glibc 2.35 < 2.38` (crash or SIGABRT) | Ubuntu 22.04 glibc too old for Cray libfabric | Use Ubuntu 24.04+ (glibc 2.39) |
| `read-only file system` (app cache) | Container filesystem is read-only | Add `--writable-tmpfs` |
| `Unknown settings section [phase_space.chili]` | Stale config file in working dir | Use `--pwd $RUNDIR` pointing to a clean directory |
| `/dev/null: is a directory` (runcard) | Positional arg mistakenly treated as runcard path | Remove positional args; use CLI flags only |
| `CMA does not have sufficient permission` | Apptainer lacks Linux capabilities for CMA | Set `APPTAINERENV_MPICH_SMP_SINGLE_COPY_MODE=NONE` |
| MPI collective crash (MPI_Reduce) | MPICH version mismatch (e.g., 3.4.3 in container vs 4.1.2 on host) | Build container MPICH matching host: 4.1.2 |
| `libpsm_infinipath` / `librdmacm` linker errors during MPICH build | `libfabric-dev` apt package installed — pulls in unwanted providers | Remove `libfabric-dev`; MPICH bundles OFI internally |
| `ch4:ucx` MPICH fails cross-node | Ubuntu apt MPICH uses UCX, not OFI | Build MPICH from source with `--with-device=ch4:ofi` |
| HTTP 401 on job submit (not on allowlist) | Account not added to IRI API access list | Email ALCF support with username and use case |
| HTTP 400 on job submit | Complex bash args with special chars break GraphQL | Use base64 encoding workaround |
| HTTP 501 on `pre_launch` | Not implemented in IRI | Embed module loads in the job script |
| HTTP 400 on `polaris.fs.ls()` | Filesystem API only works on storage resources | Use `home.fs.ls()` or `eagle.fs.ls()` |
| Old SIF used despite new Docker push | `apptainer pull :latest` hits Polaris cache | Use versioned SIF filenames; delete old SIF before pull |
| `docker login` keychain error -61 (macOS) | macOS keychain credential store issue | Remove `"credsStore": "osxkeychain"` from `~/.docker/config.json` |

---

## 9. Verification Steps

Before running your real application, verify the full stack with `mpi_hello` (a minimal binary that reports hostname and MPI rank). Build it into your container:

```c
// mpi_hello.c
#include <stdio.h>
#include <mpi.h>
#include <limits.h>
#include <unistd.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char host[HOST_NAME_MAX];
    gethostname(host, HOST_NAME_MAX);
    printf("MPI rank %d of %d on %s\n", rank, size, host);
    MPI_Finalize();
    return 0;
}
```

```bash
mpicc -o /usr/local/bin/mpi_hello mpi_hello.c
```

### Verification job script (add to your run.sh before the real app)

```bash
echo "=== mpi_hello verification ==="
mpiexec -n $RANKS -ppn $RANKS_PER_NODE --hostfile $PBS_NODEFILE \
    apptainer exec --fakeroot --nv --writable-tmpfs \
    -B /opt -B /var/run/palsd/ \
    -B /usr/lib64:/host/usr/lib64 \
    "$SIF" \
    mpi_hello
```

### Expected output (2 nodes, 4 ranks/node = 8 total)

```
MPI rank 0 of 8 on x3210c0s25b0n0
MPI rank 1 of 8 on x3210c0s25b0n0
MPI rank 2 of 8 on x3210c0s25b0n0
MPI rank 3 of 8 on x3210c0s25b0n0
MPI rank 4 of 8 on x3210c0s37b1n0
MPI rank 5 of 8 on x3210c0s37b1n0
MPI rank 6 of 8 on x3210c0s37b1n0
MPI rank 7 of 8 on x3210c0s37b1n0
```

Two distinct hostnames confirms cross-node MPI is working. If all ranks show the same hostname, your multi-node MPI is broken (probably MPICH version mismatch or wrong device).

**Important:** `mpi_hello` passing does NOT guarantee collectives work. You need your actual app (or an `MPI_Reduce` test) to confirm `APPTAINERENV_MPICH_SMP_SINGLE_COPY_MODE=NONE` is correctly set.

---

## 10. Output Retrieval

### 10.1 Read stdout/stderr via IRI

After a job completes, read its output through the Home filesystem resource:

```python
import time

def read_output(home, work_dir, job_name, max_chars=10000):
    time.sleep(5)  # Let filesystem sync
    results = {}
    for label, fname in [("stdout", f"{job_name}.stdout"),
                          ("stderr", f"{job_name}.stderr")]:
        try:
            task = home.fs.view(f"{work_dir}/{fname}")
            task.wait(timeout=60)
            r = task.result
            # result may be a dict or raw string depending on client version
            if isinstance(r, dict):
                content = r.get('output', r).get('content', '')
            else:
                content = str(r)
            results[label] = content
            print(f"\n-- {label.upper()} ({len(content)} chars) --")
            if len(content) > max_chars:
                half = max_chars // 2
                print(content[:half])
                print(f"\n... ({len(content) - max_chars} chars omitted) ...\n")
                print(content[-half:])
            else:
                print(content)
        except Exception as e:
            print(f"{label}: {type(e).__name__}: {e}")
    return results
```

### 10.2 What to check in stderr

For MPI container jobs, stderr is more informative than stdout:
- Module load messages (normal)
- Apptainer startup messages (normal)
- Kokkos/CUDA initialization (look for GPU detection)
- MPI rank assignments
- Application output
- Any errors

A clean 2-node run with MPICH 4.1.2 + CMA disabled produces ~8KB of stderr (mostly Kokkos init). A failing run with stack traces produces ~25KB+.

### 10.3 Tail workaround

Since `home.fs.tail()` returns 501, read the full file with `view()` and take the last N chars:

```python
task = home.fs.view(f"{work_dir}/job.stderr")
task.wait(timeout=60)
content = task.result.get('output', {}).get('content', '')
print(content[-3000:])  # Last 3000 chars
```

---

## 11. Working Code

### 11.1 Final Dockerfile

```dockerfile
# Pepper on Polaris — Multi-node MPI + GPU container
#
# Target: ALCF Polaris (NVIDIA A100, Cray MPICH ABI via Hybrid mode)
# Runtime: Apptainer with -B /opt for Cray libs + --nv for NVIDIA
#
# Key requirements for Polaris Hybrid MPI:
#   1. MPICH 4.1.2 built from source with --with-device=ch4:ofi
#      (must match Cray MPICH 9.0.1 = ANL MPICH 4.1.2)
#   2. Install to /usr/local (not /opt) — -B /opt overwrites container's /opt
#   3. Ubuntu 24.04 for glibc >= 2.38 (Cray libfabric 2.2.0rc1 needs it)
#   4. NO libfabric-dev (causes linker errors; MPICH uses embedded OFI)
#   5. -fno-lto (GCC LTO + CUDA fatbinData conflict)
#
# Build: docker build -t pepper-polaris .
# Push:  docker push jtchilders/pepper-polaris:latest

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies (no apt mpich — we build from source)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    pkg-config \
    python3 \
    python3-dev \
    libjson-c-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Build MPICH 4.1.2 from source with ch4:ofi device
# Must match Cray MPICH 9.0.1 which is based on ANL MPICH 4.1.2
WORKDIR /build/mpich
RUN wget -q https://www.mpich.org/static/downloads/4.1.2/mpich-4.1.2.tar.gz \
    && tar xzf mpich-4.1.2.tar.gz \
    && cd mpich-4.1.2 \
    && ./configure \
        --prefix=/usr/local \
        --with-device=ch4:ofi \
        --disable-wrapper-rpath \
        --enable-shared \
        FFLAGS='-O3 -fallow-argument-mismatch' \
        FCFLAGS='-O3 -fallow-argument-mismatch' \
        CFLAGS='-O3' \
        CXXFLAGS='-O3' \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd / && rm -rf /build/mpich

# Build Pepper from source
WORKDIR /build
RUN git clone --depth 1 --branch 1.11.0 https://gitlab.com/spice-mc/pepper.git

WORKDIR /build/pepper

# Configure: CUDA A100 (sm_80), MPI enabled
# -fno-lto prevents fatbinData symbol collision (GCC LTO + CUDA fatbinaries)
RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DCMAKE_CXX_FLAGS="-fno-lto" \
    -DCMAKE_C_FLAGS="-fno-lto" \
    -DCMAKE_EXE_LINKER_FLAGS="-fno-lto" \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_AMPERE80=ON \
    -DMPI_DISABLED=FALSE \
    -DHepMC3_DISABLED=TRUE \
    -DHDF5_DISABLED=TRUE \
    -DLHAPDF_DISABLED=TRUE \
    -Dpybind11_DISABLED=TRUE \
    2>&1 | tee /tmp/cmake-configure.log

RUN cmake --build build --parallel $(nproc) 2>&1 | tee /tmp/cmake-build.log
RUN cmake --build build --target install

# Add a simple MPI diagnostic binary for cross-node verification
RUN printf '#include <stdio.h>\n#include <mpi.h>\n#include <limits.h>\n#include <unistd.h>\nint main(int argc, char** argv) {\n    MPI_Init(&argc, &argv);\n    int rank, size;\n    MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n    MPI_Comm_size(MPI_COMM_WORLD, &size);\n    char host[HOST_NAME_MAX];\n    gethostname(host, HOST_NAME_MAX);\n    printf("MPI rank %%d of %%d on %%s\\n", rank, size, host);\n    MPI_Finalize();\n    return 0;\n}\n' > /tmp/mpi_hello.c \
    && mpicc -o /usr/local/bin/mpi_hello /tmp/mpi_hello.c \
    && rm /tmp/mpi_hello.c

# Clean up build dir to reduce image size
RUN rm -rf /build

WORKDIR /work

CMD ["pepper", "--help"]
```

### 11.2 Final job submission script (submit_mpi_test_v7.py)

```python
#!/usr/bin/env python3
"""
Final working script: 2-node Pepper MPI job on Polaris via IRI API.
Uses base64 pattern to transfer script (workaround for GraphQL special-char limitations).
"""
import base64
import time
from amsc_client import Client

GLOBUS_APP_ID   = 'e4f48665-38b5-4833-a89e-849c71f5b3e3'
RESOURCE_SERVER = '8b84fc2d-49e9-49ea-b54d-b3a29a70cf31'
ALCF_USER = 'parton'
PROJECT   = 'datascience'
WORK_DIR  = f"/home/{ALCF_USER}/pepper-iri-test"

client = Client(
    base_url='https://api.american-science-cloud.org/api/current',
    auth_method="globus",
    globus_client_id=GLOBUS_APP_ID,
    requested_scopes=(
        f'openid profile email '
        f'https://auth.globus.org/scopes/{GLOBUS_APP_ID}/amsc_test'
    ),
    resource_server=RESOURCE_SERVER,
    use_id_token=True,
)

alcf    = client.facility("alcf")
polaris = alcf.resource("Polaris")
home    = alcf.resource("Home")

SCRIPT_CONTENT = r"""#!/bin/bash -l
set -e

module use /soft/modulefiles
module load spack-pe-base apptainer cray-mpich-abi

export APPTAINER_TMPDIR=/local/scratch/apptainer-tmpdir
export APPTAINER_CACHEDIR=/local/scratch/apptainer-cachedir
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

export ADDITIONAL_PATH=/opt/cray/pe/pals/1.2.12/lib
export APPTAINERENV_LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:$ADDITIONAL_PATH:/host/usr/lib64"

# KEY FIX: disable CMA — Cray MPICH says to use this exact env var
export APPTAINERENV_MPICH_SMP_SINGLE_COPY_MODE=NONE

SIF=$HOME/pepper-iri-test/pepper-polaris-v4.sif
RUNDIR=$HOME/pepper-iri-test/run-v7
mkdir -p $RUNDIR

NODES=$(cat $PBS_NODEFILE | wc -l)
RANKS=$((NODES * 4))

echo "=== MPI Test v7 (MPICH 4.1.2 + CMA disabled) ==="
echo "Nodes: $NODES"
cat $PBS_NODEFILE
echo ""

# Test 1: mpi_hello (verifies cross-node MPI)
echo "=== Test 1: mpi_hello ==="
mpiexec -n $RANKS -ppn 4 --hostfile $PBS_NODEFILE \
    apptainer exec --fakeroot --nv --writable-tmpfs \
    -B /opt -B /var/run/palsd/ \
    -B /usr/lib64:/host/usr/lib64 \
    "$SIF" \
    mpi_hello
echo ""

# Test 2: Pepper (real MPI collective workload)
echo "=== Test 2: Pepper ==="
mpiexec -n $RANKS -ppn 4 --hostfile $PBS_NODEFILE \
    apptainer exec --fakeroot --nv --writable-tmpfs \
    -B /opt -B /var/run/palsd/ \
    -B /usr/lib64:/host/usr/lib64 \
    -B /home/parton:/home/parton \
    --pwd $RUNDIR \
    "$SIF" \
    pepper \
    --process "d db -> e+ e-" \
    --collision-energy 91.2 \
    --n-events 10000 \
    --output-disabled

echo ""
echo "=== Done ==="
"""

# Step 1: Write the script to Polaris via base64 decode job
b64_script = base64.b64encode(SCRIPT_CONTENT.encode()).decode()
decode_cmd = (
    f"echo {b64_script} | base64 -d > {WORK_DIR}/run_mpi_test_v7.sh "
    f"&& chmod +x {WORK_DIR}/run_mpi_test_v7.sh && echo OK"
)

print("Writing script...")
write_job = polaris.submit(
    executable="/bin/bash",
    arguments=["-c", decode_cmd],
    directory=f"/home/{ALCF_USER}",
    name="mpi7-write",
    queue="debug",
    account=PROJECT,
    duration=300,
    nodes=1,
    filesystems="home",
)
print(f"  Write job: {write_job.id}")
write_job.wait(timeout=600, poll_interval=10)
print(f"  Write done: state={write_job.state}, exit={write_job.exit_code}")
time.sleep(3)

# Step 2: Submit 2-node MPI job
print("\nSubmitting 2-node MPI job...")
time.sleep(3)
run_job = polaris.submit(
    executable="/bin/bash",
    arguments=["-l", f"{WORK_DIR}/run_mpi_test_v7.sh"],
    directory=WORK_DIR,
    name="mpi-test7",
    queue="debug",
    account=PROJECT,
    duration=1800,
    nodes=2,
    filesystems="home",
)
print(f"  Run job: {run_job.id}")
print("  Waiting (up to 30 min)...")
run_job.wait(timeout=1800, poll_interval=15)
print(f"  Done: state={run_job.state}, exit={run_job.exit_code}")

# Read output
time.sleep(5)
for label, fname in [("STDOUT", "mpi-test7.stdout"), ("STDERR", "mpi-test7.stderr")]:
    try:
        task = home.fs.view(f"{WORK_DIR}/{fname}")
        task.wait(timeout=60)
        r = task.result
        content = r.get('output', r).get('content', '') if isinstance(r, dict) else str(r)
        if content.strip():
            print(f"\n-- {label} ({len(content)} chars) --")
            if len(content) > 6000:
                print(content[:2000])
                print(f"\n... ({len(content) - 4000} chars omitted) ...\n")
                print(content[-2000:])
            else:
                print(content)
    except Exception as e:
        print(f"\n{label}: {type(e).__name__}: {str(e)[:200]}")

print("\nDone!")
```

### 11.3 Proven results

This exact configuration produced:

```
=== Test 1: mpi_hello ===
MPI rank 0 of 8 on x3210c0s25b0n0
MPI rank 1 of 8 on x3210c0s25b0n0
MPI rank 2 of 8 on x3210c0s25b0n0
MPI rank 3 of 8 on x3210c0s25b0n0
MPI rank 4 of 8 on x3210c0s37b1n0
MPI rank 5 of 8 on x3210c0s37b1n0
MPI rank 6 of 8 on x3210c0s37b1n0
MPI rank 7 of 8 on x3210c0s37b1n0

=== Test 2: Pepper ===
[Kokkos] CUDA: detected 4 A100 GPUs on each node
...
σ(d d̄ → e⁺e⁻) = 2.109 ± 0.019 pb at √s = 91.2 GeV
10,000 events, ~13.5M events/hour
Exit code: 0
```

---

## Appendix A: amsc-client API Patterns

```python
# List all facilities
for f in client.facilities():
    print(f.name)

# List all resources under a facility
for r in alcf.resources():
    print(r.name, r.type)

# List recent jobs
jobs = polaris.jobs(limit=20)

# Wait for job with timeout
job.wait(timeout=1800, poll_interval=15)

# Access job fields after completion
job.state      # e.g., "completed", "failed", "active", "queued"
job.exit_code  # integer exit code
job.id         # PBS job ID

# Read a file
task = home.fs.view("/home/parton/some/file.txt")
task.wait(timeout=60)
content = task.result.get('output', {}).get('content', '')
```

## Appendix B: Probing the Polaris Software Stack

Use this script pattern (via base64 job) to discover what's available before writing your run script:

```bash
#!/bin/bash -l
module use /soft/modulefiles
module load spack-pe-base cray-mpich-abi

echo "=== MPICH version ==="
/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/bin/mpichversion

echo "=== Apptainer version ==="
module load apptainer
apptainer --version

echo "=== libcxi location ==="
find /usr/lib64 /opt -name "libcxi.so*" 2>/dev/null

echo "=== PALS lib location ==="
find /opt/cray/pe/pals -name "*.so" 2>/dev/null | head -5

echo "=== PBS nodefile ==="
cat $PBS_NODEFILE
echo "Nodes: $(cat $PBS_NODEFILE | wc -l)"
```

## Appendix C: Iteration History (for context)

The path to the working configuration required these iterations. Each represents a distinct failure mode worth knowing about:

| Iteration | Script | Key fix |
|-----------|--------|---------|
| v1 | submit_pepper_job.py | Added `#!/bin/bash -l` (login shell) |
| v2 | fix_and_resubmit.py | Moved install from `/opt` to `/usr/local` |
| v3 | fix_v3.py | Added `--nv` flag for CUDA |
| v4 | fix_v4.py | Ubuntu 24.04 for glibc 2.38; added `-B /opt` |
| v5 | fix_v5.py | Added `--writable-tmpfs` for app cache |
| v6 | submit_final_v4.py | Added `--pwd $RUNDIR` for clean working dir |
| v7 | (Dockerfile rebuild) | Removed positional runcard arg |
| v8 | submit_mpi_test.py | Added MPICH 3.4.3 ch4:ofi (replaced apt MPICH) |
| v9 | submit_debug_libcxi.py | Discovered libcxi in `/usr/lib64`, not `/opt` |
| v10 | submit_mpi_test_v2.py | Added `-B /usr/lib64:/host/usr/lib64` |
| v11 | submit_mpich_version.py | Probed Cray MPICH → found it's MPICH 4.1.2 base |
| v12 | (Dockerfile rebuild v2) | Rebuilt with MPICH 4.1.2 instead of 3.4.3 |
| v13 | submit_mpi_test_v6.py | Added `APPTAINERENV_MPICH_SMP_SINGLE_COPY_MODE=NONE` |
| **v14** | **submit_mpi_test_v7.py** | **✅ FULL SUCCESS: 2 nodes, 8 ranks, 8 GPUs** |

---

*This guide was written based on direct hands-on experience with the ALCF IRI API and Polaris system. All commands and configurations were tested and produced the results documented above. Software versions and API behavior may change; verify critical details against current ALCF documentation.*
