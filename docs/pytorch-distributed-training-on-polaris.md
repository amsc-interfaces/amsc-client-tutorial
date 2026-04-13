# PyTorch Distributed Training on Polaris via IRI API

**An Agent's Guide to Multi-Node GPU Training with Containers**

---

## Overview

This guide documents how to run **distributed PyTorch training** (DDP with `torchrun`) on **ALCF Polaris** using the **IRI API** and **Apptainer containers**. It covers:

- Building a PyTorch container compatible with Polaris
- Configuring NCCL for Slingshot-11 interconnect
- Launching multi-node training via `mpiexec` + `torchrun`
- Critical environment variables and bind mounts
- Submitting and monitoring jobs via IRI

**Target audience:** AI agents automating scientific workflows, researchers deploying distributed training.

**Tested configuration:**
- **System:** ALCF Polaris (A100 40GB GPUs, Slingshot-11 HPE interconnect)
- **PyTorch:** 2.5.1+cu124 (pip install)
- **Container runtime:** Apptainer 1.3.5
- **Launcher:** Cray MPICH ABI + `mpiexec` wrapping `torchrun`
- **NCCL plugin:** AWS OFI NCCL v1.9.1-aws (host bind-mount)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Container Strategy](#container-strategy)
4. [Building the Container](#building-the-container)
5. [Training Script](#training-script)
6. [Job Script Anatomy](#job-script-anatomy)
7. [Critical Environment Variables](#critical-environment-variables)
8. [Submitting via IRI](#submitting-via-iri)
9. [Debugging Common Failures](#debugging-common-failures)
10. [Performance Notes](#performance-notes)
11. [Complete Working Example](#complete-working-example)

---

## Quick Start

**For the impatient:**

```bash
# 1. Build container (on a machine with Docker)
docker build -t youruser/pytorch-polaris:v1 .
docker push youruser/pytorch-polaris:v1

# 2. Submit via IRI (Python + amsc-client)
polaris.submit(
    executable="/bin/bash",
    arguments=["-l", "/path/to/run_ddp.sh"],
    directory="/home/youruser/work",
    name="ddp-job",
    queue="debug",
    account="yourproject",
    duration=1200,  # 20 minutes (in seconds)
    nodes=2,
    filesystems="home",
)
```

**Expected outcome:** Multi-GPU distributed training across 2 nodes (8 GPUs) with NCCL communication.

**Time to first result:** ~15 minutes (container pull + queue + training)

---

## System Architecture

### Polaris Hardware
- **Nodes:** 560 HPE Apollo Gen10+ nodes
- **GPUs:** 4× NVIDIA A100 40GB per node (total 8 GPUs for 2-node jobs)
- **Interconnect:** HPE Slingshot-11 (not InfiniBand — requires AWS OFI NCCL plugin)
- **Node OS:** SUSE Linux Enterprise Server 15 SP3

### Software Stack
- **Scheduler:** PBS Pro
- **Container runtime:** Apptainer 1.3.5 (formerly Singularity)
- **MPI:** Cray MPICH 9.0.1 → `cray-mpich-abi/9.0.1` (ABI-compatible module)
- **Launcher pattern:** `mpiexec` (host) → `torchrun` (container)

### Why This Matters
- **No MPICH in container:** Host `mpiexec` handles process launching; container just needs PyTorch + `torchrun`
- **NCCL plugin required:** Slingshot-11 needs AWS OFI NCCL plugin (bind-mounted from host)
- **Library paths:** Must expose Cray libs (`libpmi`, `libpals`) and hwloc to container

---

## Container Strategy

### Design Principles

1. **Minimal container scope:** Container = PyTorch + training code. No MPI stack needed.
2. **Host-side launching:** `mpiexec` on host wraps `torchrun` in container (avoids MPI version conflicts).
3. **Bind-mount critical libs:** NCCL plugin, Cray PMI/PALS, hwloc from host.

### Container vs. Host Responsibilities

| Component | Location | Reason |
|-----------|----------|--------|
| PyTorch + torchrun | Container | Training framework |
| Training script | Container (via bind-mount or baked-in) | Workload |
| `mpiexec` | Host | Process launcher |
| Cray MPICH ABI | Host | Slingshot interconnect driver |
| AWS OFI NCCL plugin | Host (bind-mount to container) | NCCL transport for Slingshot-11 |
| hwloc | Host (bind-mount to container) | Topology detection |

### Why Skip MPICH in Container?

Mixing container MPICH with host Cray MPICH causes ABI mismatches. Letting the host handle launching is simpler and more robust.

---

## Building the Container

### Dockerfile

```dockerfile
# File: Dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Install Python + pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        ca-certificates \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch (CUDA 12.4 wheel works with CUDA 12.6 driver on Polaris)
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
```

### Build and Push

```bash
docker build -t youruser/pytorch-polaris:v1 .
docker push youruser/pytorch-polaris:v1
```

**Image size:** ~7.2 GB (CUDA base + PyTorch)

**CUDA compatibility note:** PyTorch 2.5.1 built for CUDA 12.4 works on Polaris (CUDA 12.6 driver). CUDA maintains forward compatibility within minor versions.

### Alternative: Extend ALCF Base Containers

ALCF provides `pytorch/2.4.1` containers. You can layer on top:

```dockerfile
FROM docker.io/alcf/pytorch:2.4.1
# Add your dependencies
RUN pip install transformers datasets
```

Check available containers: `https://www.alcf.anl.gov/support/user-guides/polaris/data-science-workflows/containers/container-list/`

---

## Training Script

### Minimal DDP Example

Save as `train_ddp.py`:

```python
#!/usr/bin/env python3
"""
Minimal PyTorch DDP training example for multi-node GPU clusters.
Launches via: torchrun --nnodes=N --nproc_per_node=4 train_ddp.py
"""
import os
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time

# ─── Config ───────────────────────────────────────────────────────────────────
VOCAB_SIZE = 10000
SEQ_LEN = 512
BATCH_SIZE = 8      # per GPU
STEPS = 20
D_MODEL = 512
N_HEAD = 8
N_LAYERS = 2
DIM_FF = 2048

# ─── Setup ────────────────────────────────────────────────────────────────────
def setup():
    """Initialize distributed training (torchrun sets env vars automatically)."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

# ─── Synthetic Dataset ────────────────────────────────────────────────────────
class RandomTokenDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size, seq_len, size=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randint(0, self.vocab_size, (self.seq_len,))

# ─── Model ────────────────────────────────────────────────────────────────────
class ToyTransformer(torch.nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, dim_ff):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_ff, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, n_layers)
        self.fc = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.fc(x)

# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    setup()
    
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    
    # Print config (rank 0 only)
    if rank == 0:
        print("=== Toy DDP Transformer ===")
        print(f"World size : {world_size}")
        print(f"Device     : {torch.cuda.get_device_name(device)}")
        print(f"PyTorch    : {torch.__version__}")
        print(f"CUDA       : {torch.version.cuda}")
        print(f"NCCL       : {torch.cuda.nccl.version()}")
        print(f"Model      : {N_LAYERS}L d={D_MODEL} h={N_HEAD} ffn={DIM_FF}")
        print(f"Seq len    : {SEQ_LEN}  Batch/GPU: {BATCH_SIZE}\n")
    
    # Model + DDP
    model = ToyTransformer(VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYERS, DIM_FF).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Data
    dataset = RandomTokenDataset(VOCAB_SIZE, SEQ_LEN, size=STEPS * BATCH_SIZE * world_size)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
    
    # Optimizer + loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    start_time = time.time()
    total_tokens = 0
    
    for step, batch in enumerate(loader, 1):
        if step > STEPS:
            break
        
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward
        logits = model(batch)  # [B, L, V]
        loss = criterion(logits.view(-1, VOCAB_SIZE), batch.view(-1))
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics (rank 0)
        if rank == 0 and step % 5 == 0:
            elapsed = time.time() - start_time
            tokens_processed = step * BATCH_SIZE * SEQ_LEN * world_size
            tokens_per_sec = tokens_processed / elapsed
            print(f"step {step:3d}/{STEPS}  loss={loss.item():.4f}  "
                  f"tokens/s={tokens_per_sec:,.0f}  elapsed={elapsed:.1f}s")
        
        total_tokens += BATCH_SIZE * SEQ_LEN * world_size
    
    # Final stats
    if rank == 0:
        total_time = time.time() - start_time
        print(f"\n=== Done ===")
        print(f"Total steps    : {STEPS}")
        print(f"Total tokens   : {total_tokens:,}")
        print(f"Tokens/sec     : {total_tokens/total_time:,.0f}")
        print(f"Tokens/sec/GPU : {total_tokens/total_time/world_size:,.0f}")
        print(f"Elapsed        : {total_time:.1f}s")
    
    cleanup()

if __name__ == "__main__":
    train()
```

**Key points:**
- Uses `torch.distributed.init_process_group(backend="nccl")` — torchrun sets env vars automatically
- `DistributedSampler` ensures each GPU sees different data
- DDP wraps model for gradient synchronization
- Metrics printed only on rank 0

---

## Job Script Anatomy

### Complete PBS Job Script

Save as `run_pytorch_ddp.sh`:

```bash
#!/bin/bash -l
#PBS -N pt-ddp
#PBS -l select=2:system=polaris
#PBS -l walltime=00:20:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home

# ─── Header explanation ───────────────────────────────────────────────────────
# -l: Login shell (required for module command)
# select=2: 2 nodes (8 GPUs total, 4 per node)
# walltime: Max 1 hour for debug queue
# filesystems=home: Mount /home (add :eagle:grand for project filesystems)

# ─── Job info ─────────────────────────────────────────────────────────────────
NNODES=$(wc -l < ${PBS_NODEFILE})
NPROC_PER_NODE=4  # 4 A100 GPUs per Polaris node
HEAD_NODE=$(head -n 1 ${PBS_NODEFILE})
WORK_DIR="${PBS_O_WORKDIR:-/home/$(whoami)/pytorch-test}"
SIF="${WORK_DIR}/pytorch-polaris-v1.sif"
OFI_NCCL_LIB="/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib"

echo "=== Job info ==="
echo "Nodes      : ${NNODES}"
echo "Head node  : ${HEAD_NODE}"
echo "GPUs/node  : ${NPROC_PER_NODE}"
echo "Total GPUs : $((NNODES * NPROC_PER_NODE))"
echo "SIF        : ${SIF}"
echo ""

# ─── Modules ──────────────────────────────────────────────────────────────────
module use /soft/modulefiles
module load spack-pe-base apptainer cray-mpich-abi

# ─── LD_LIBRARY_PATH inside container ─────────────────────────────────────────
# Include OFI NCCL plugin, Cray MPICH/PMI libs, and /soft for hwloc
export APPTAINERENV_LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}:/opt/cray/pe/pals/1.2.12/lib:/host/usr/lib64:/ofi-nccl:/soft/libraries/hwloc/lib"

# ─── Tell NCCL where to find the AWS OFI plugin ───────────────────────────────
# The plugin .so must be explicitly preloaded; LD_LIBRARY_PATH alone is not enough
export APPTAINERENV_LD_PRELOAD="/ofi-nccl/libnccl-net.so"

# ─── NCCL tuning for Slingshot-11 ─────────────────────────────────────────────
export APPTAINERENV_NCCL_NET="AWS Libfabric"
export APPTAINERENV_NCCL_NET_GDR_LEVEL=PHB
export APPTAINERENV_NCCL_CROSS_NIC=1
export APPTAINERENV_NCCL_COLLNET_ENABLE=1
export APPTAINERENV_FI_CXI_DISABLE_HOST_REGISTER=1
export APPTAINERENV_FI_MR_CACHE_MONITOR=userfaultfd
export APPTAINERENV_FI_CXI_DEFAULT_CQ_SIZE=131072

# ─── Apptainer cache ──────────────────────────────────────────────────────────
export APPTAINER_TMPDIR=/local/scratch/apptainer-tmpdir
export APPTAINER_CACHEDIR=/local/scratch/apptainer-cachedir
mkdir -p ${APPTAINER_TMPDIR} ${APPTAINER_CACHEDIR}

# ─── Pull container (first run only) ──────────────────────────────────────────
if [ ! -f "${SIF}" ]; then
    echo "Pulling container..."
    apptainer pull ${SIF} docker://youruser/pytorch-polaris:v1
fi

# ─── Launch: mpiexec spawns one torchrun per node ─────────────────────────────
echo "=== Launching DDP training ==="
mpiexec \
    -n ${NNODES} \
    -ppn 1 \
    --hostfile ${PBS_NODEFILE} \
    --depth=${NPROC_PER_NODE} \
    --cpu-bind depth \
    apptainer exec \
        -B /opt \
        -B /soft \
        -B /var/run/palsd/ \
        -B /usr/lib64:/host/usr/lib64 \
        -B ${OFI_NCCL_LIB}:/ofi-nccl \
        --nv \
        --writable-tmpfs \
        --fakeroot \
        ${SIF} \
    torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${HEAD_NODE}:29500 \
        /workspace/train_ddp.py

echo "=== Job complete ==="
```

### Critical Bind Mounts Explained

| Bind Mount | Purpose | Why Required |
|------------|---------|--------------|
| `-B /opt` | Cray PE libraries | Slingshot drivers |
| `-B /soft` | ALCF software stack | Modules, hwloc |
| `-B /var/run/palsd/` | PALS daemon socket | Process management |
| `-B /usr/lib64:/host/usr/lib64` | System libs | Cray MPICH dependencies |
| `-B ${OFI_NCCL_LIB}:/ofi-nccl` | AWS OFI NCCL plugin | NCCL transport for Slingshot |

**Omitting any of these will cause failures.** See [Debugging](#debugging-common-failures) for symptoms.

---

## Critical Environment Variables

### NCCL Configuration

These **must** be set for Slingshot-11:

```bash
export APPTAINERENV_NCCL_NET="AWS Libfabric"
export APPTAINERENV_NCCL_NET_GDR_LEVEL=PHB
export APPTAINERENV_LD_PRELOAD="/ofi-nccl/libnccl-net.so"
```

**Why `LD_PRELOAD`?** NCCL won't auto-discover the plugin via `LD_LIBRARY_PATH`. Explicit preload is mandatory.

### Library Paths

```bash
export APPTAINERENV_LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}:/opt/cray/pe/pals/1.2.12/lib:/host/usr/lib64:/ofi-nccl:/soft/libraries/hwloc/lib"
```

**Key components:**
- `${CRAY_LD_LIBRARY_PATH}`: Set by `cray-mpich-abi` module (includes PMI/PALS)
- `/ofi-nccl`: Bind-mounted NCCL plugin
- `/soft/libraries/hwloc/lib`: hwloc (required by Cray MPICH)

### Optional: NCCL Debugging

```bash
export APPTAINERENV_NCCL_DEBUG=INFO        # Verbose NCCL logs
export APPTAINERENV_NCCL_DEBUG_SUBSYS=ALL # All subsystems
```

---

## Submitting via IRI

### Python Script with amsc-client

```python
import amsc_client
import base64

# ─── Authenticate ─────────────────────────────────────────────────────────────
CLIENT_ID = "your-iri-client-id"
client = amsc_client.Client(
    base_url='https://api.american-science-cloud.org/api/current',
    auth_method="globus",
    globus_client_id=CLIENT_ID,
)
alcf = client.facility("alcf")
polaris = alcf.resource("Polaris")

# ─── Write script to Polaris (base64 workaround) ──────────────────────────────
WORK_DIR = "/home/youruser/pytorch-test"
with open('run_pytorch_ddp.sh') as f:
    script = f.read()
script_b64 = base64.b64encode(script.encode()).decode()

# Submit script-writing job first
cmd = f"printf '%s' '{script_b64}' | base64 -d > {WORK_DIR}/run_pytorch_ddp.sh && chmod +x {WORK_DIR}/run_pytorch_ddp.sh"
setup_job = polaris.submit(
    executable="/bin/bash",
    arguments=["-c", cmd],
    directory=WORK_DIR,
    name="setup",
    queue="debug",
    account="yourproject",
    duration=300,  # 5 minutes (minimum for debug queue)
    nodes=1,
    filesystems="home",
)
print(f"Setup job: {setup_job.id}")

# ─── Wait for setup to complete ───────────────────────────────────────────────
import time
for _ in range(60):
    time.sleep(10)
    try:
        raw = alcf._get_job_raw(polaris.id, str(setup_job.id))
        if raw.state in ("complete", "failed", "cancelled"):
            break
    except:
        break  # Job cleared from queue

# Extra buffer for queue to clear (debug queue = 1 job max)
time.sleep(30)

# ─── Submit DDP job ───────────────────────────────────────────────────────────
ddp_job = polaris.submit(
    executable="/bin/bash",
    arguments=["-l", f"{WORK_DIR}/run_pytorch_ddp.sh"],
    directory=WORK_DIR,
    name="pt-ddp",
    queue="debug",
    account="yourproject",
    duration=1200,  # 20 minutes
    nodes=2,
    filesystems="home",
)
print(f"DDP job: {ddp_job.id}")

# ─── Fetch output ─────────────────────────────────────────────────────────────
# (Wait ~10-15 min for job to complete)
time.sleep(600)  # Adjust based on queue wait time

home = alcf.resource("Home")
stdout_task = home.fs.view(f"{WORK_DIR}/pt-ddp.stdout")
stdout_task.wait(timeout=60, poll_interval=5)
print(stdout_task.result['output']['content'].decode('utf-8'))
```

### Queue Limits

**Debug queue:**
- Max 1 job queued per user at a time
- Max 60 minutes walltime
- Minimum 5 minutes (`duration=300` seconds)

**Prod queue:**
- More capacity but longer wait times
- Use for production runs

---

## Debugging Common Failures

### 1. `NCCL WARN Error: network AWS Libfabric not found`

**Cause:** NCCL plugin not loaded.

**Fix:** Ensure `APPTAINERENV_LD_PRELOAD="/ofi-nccl/libnccl-net.so"` is set **before** `apptainer exec`.

**Verification:**
```bash
# Inside container (add to script for debugging):
echo $LD_PRELOAD  # Should show /ofi-nccl/libnccl-net.so
ldd /ofi-nccl/libnccl-net.so  # Should resolve all dependencies
```

---

### 2. `error while loading shared libraries: libhwloc.so.0`

**Cause:** hwloc not accessible to container.

**Fix:** Add `-B /soft` and include `/soft/libraries/hwloc/lib` in `LD_LIBRARY_PATH`.

**Full fix:**
```bash
export APPTAINERENV_LD_LIBRARY_PATH="...:/ soft/libraries/hwloc/lib"
apptainer exec -B /soft ...
```

---

### 3. `Job violates queue and/or server resource limits`

**Causes:**
- Another job already queued (debug queue = 1 max)
- Walltime < 5 min (debug queue minimum)
- Walltime > 60 min (debug queue maximum)

**Fix:** Wait for previous job to clear, or use `queue="prod"`.

---

### 4. Hangs at `init_process_group`

**Cause:** NCCL can't communicate across nodes (network issue).

**Debug:**
```bash
export APPTAINERENV_NCCL_DEBUG=INFO
export APPTAINERENV_NCCL_DEBUG_SUBSYS=NET
# Re-run and check stderr for NCCL initialization logs
```

**Common issues:**
- Missing `-B /opt` (Slingshot drivers)
- Wrong `NCCL_NET_GDR_LEVEL` (should be `PHB`)

---

### 5. Rank Mismatch / Duplicate Ranks

**Symptom:** `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.`

**Cause:** `mpiexec` and `torchrun` rank assignments conflict.

**Fix:** Use `-ppn 1` with `mpiexec` (one `torchrun` process per node, which spawns `--nproc_per_node` workers).

**Wrong:**
```bash
mpiexec -n 8 torchrun --nproc_per_node=4 train.py  # Creates 8×4=32 ranks!
```

**Right:**
```bash
mpiexec -n 2 -ppn 1 torchrun --nproc_per_node=4 train.py  # 2×4=8 ranks
```

---

## Performance Notes

### Observed Throughput (2-layer Transformer, 512 seq len)

| Nodes | GPUs | Tokens/sec | Tokens/sec/GPU | Speedup |
|-------|------|------------|----------------|---------|
| 1 | 4 | ~190K | ~47.5K | 1.0× |
| 2 | 8 | ~383K | ~47.9K | 2.0× |

**Near-linear scaling** for this toy workload. Real models may see communication overhead.

### NCCL Performance Tips

1. **Use `NCCL_CROSS_NIC=1`** — enables multi-NIC striping
2. **Tune `FI_CXI_DEFAULT_CQ_SIZE`** — larger = fewer interrupts (131072 is good)
3. **Enable `NCCL_COLLNET_ENABLE=1`** — offload collectives to network
4. **Profile with `nsys` or `torch.profiler`** — identify communication bottlenecks

---

## Complete Working Example

### Directory Structure

```
pytorch-test/
├── Dockerfile
├── train_ddp.py
└── run_pytorch_ddp.sh
```

### Step-by-Step Execution

```bash
# 1. Build container (local machine)
cd pytorch-test
docker build -t youruser/pytorch-polaris:v1 .
docker push youruser/pytorch-polaris:v1

# 2. Submit via IRI (Python script or interactive session)
python submit_job.py

# 3. Monitor (check every few minutes)
python check_output.py

# 4. Expected output
# ==> pt-ddp.stdout <==
# === Toy DDP Transformer ===
# World size : 8
# Device     : NVIDIA A100-SXM4-40GB
# PyTorch    : 2.5.1+cu124
# CUDA       : 12.4
# NCCL       : (2, 21, 5)
# Model      : 2L d=512 h=8 ffn=2048
# Seq len    : 512  Batch/GPU: 8
#
# step   5/20  loss=8.4874  tokens/s=139,284  elapsed=1.2s
# step  10/20  loss=8.4685  tokens/s=237,290  elapsed=1.4s
# step  15/20  loss=8.4883  tokens/s=344,331  elapsed=1.4s
# step  20/20  loss=8.4912  tokens/s=407,831  elapsed=1.6s
#
# === Done ===
# Total steps    : 20
# Total tokens   : 655,360
# Tokens/sec     : 382,967
# Tokens/sec/GPU : 47,871
# Elapsed        : 1.7s
```

---

## Lessons Learned (Agent's Perspective)

### What Worked

1. **No-MPICH container approach** — simpler than trying to match host/container MPI versions
2. **`mpiexec` wrapping `torchrun`** — clean separation of concerns (host launches, container trains)
3. **Explicit `LD_PRELOAD` for NCCL plugin** — `LD_LIBRARY_PATH` alone was insufficient
4. **Base64 file transfer** — only reliable way to write files via IRI (no direct `scp`/`rsync`)

### Pitfalls to Avoid

1. **Assuming `LD_LIBRARY_PATH` is enough** — NCCL plugin requires `LD_PRELOAD`
2. **Forgetting `-B /soft`** — hwloc is required but not obvious
3. **Submitting jobs too quickly** — debug queue = 1 job max, need 30-60s buffer between submits
4. **Using `duration` in minutes** — API expects **seconds** (misleading error messages)

### Automation Strategy

For agents orchestrating workflows:

```python
def submit_ddp_training(script_path, nodes=2, walltime_min=20):
    """
    Submit PyTorch DDP job with automatic retries and queue handling.
    
    Returns job ID or raises exception after max retries.
    """
    # 1. Write script (with retry on queue limit)
    setup_job = submit_with_retry(write_script_job, max_attempts=3, delay=60)
    
    # 2. Wait for queue to clear
    wait_for_job_clear(setup_job.id, timeout=300)
    time.sleep(30)  # Extra buffer
    
    # 3. Submit DDP job
    ddp_job = submit_with_retry(
        ddp_job_spec,
        max_attempts=3,
        delay=60,
        duration=walltime_min * 60,  # Convert to seconds
        nodes=nodes,
    )
    
    # 4. Poll and return when complete
    return wait_for_completion(ddp_job.id, poll_interval=30, timeout=3600)
```

---

## References

- **ALCF Polaris User Guide:** https://docs.alcf.anl.gov/polaris/
- **ALCF Container Guide:** https://www.alcf.anl.gov/support/user-guides/polaris/data-science-workflows/containers/
- **AWS OFI NCCL Plugin:** https://github.com/aws/aws-ofi-nccl
- **PyTorch DDP Tutorial:** https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- **IRI API Docs:** https://api.american-science-cloud.org/docs
- **Companion MPI Guide:** `agentic-guide-to-polaris-with-iri.md` (in this repo)

---

## Acknowledgments

This guide synthesized from live debugging sessions (2026-04-11 to 2026-04-13) where an AI agent (Wesley Crusher / OpenClaw) iteratively diagnosed and fixed:

1. NCCL plugin loading failures
2. hwloc library path issues
3. Queue limit handling quirks
4. PBS walltime minimum/maximum constraints

**Human collaborator:** Taylor Childers (ALCF, `parton` account)  
**Test system:** Polaris debug queue, `datascience` project  
**Total iterations to working solution:** 6 job submissions

---

**Last updated:** 2026-04-13  
**Guide version:** 1.0  
**Tested PyTorch version:** 2.5.1+cu124  
**Tested Polaris software stack:** 2025-09-25 conda env
