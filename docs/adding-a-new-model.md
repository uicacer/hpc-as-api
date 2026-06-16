# How to Add a New LLM to the hpc-as-api Infrastructure

This guide walks through deploying a new language model on Lakeshore HPC and
connecting it to the hpc-as-api gateway so users can access it via an
OpenAI-compatible API.

We use **Gemma 4 31B** (`google/gemma-4-31B-it`) as the worked example. Wherever
you see a Gemma-specific value, the surrounding text explains the general rule.

**No prior HPC experience is assumed.** Every concept is explained the first
time it appears.

---

## Background: How This All Fits Together

### What is Lakeshore?

Lakeshore is a shared computing cluster at UIC — a collection of many computers
("nodes") connected by a fast internal network. Some nodes have GPUs (the ones
we care about for running LLMs). You don't interact with GPU nodes directly.
Instead you describe what you want to run in a script, submit it to a scheduler,
and the scheduler finds a free node and runs it for you.

There are two types of nodes you'll deal with:

- **Login node** (`lakeshore.acer.uic.edu`) — the machine you SSH into. It is a
  shared gateway: you use it to write scripts, submit jobs, and check results.
  **Do not run heavy computation here** — it is shared by all users and has no GPUs.

- **Compute nodes** (e.g. `ghi2-002`, `ga-002`) — machines with GPUs where your
  actual jobs run. You never SSH into these directly; the scheduler manages them.

### GPU nodes on Lakeshore

| Node | GPUs | VRAM | Partition | Account | Driver / CUDA |
|------|------|------|-----------|---------|---------------|
| `ghi2-002` | 4× H100 NVL | 95.8 GB each | `batch_gpu2` | (default) | 550 → CUDA 12.4 |
| `ga-002` | 2× A100 SXM4 | 80 GB each | `batch_gpuapi` | `ts_acer` | 580 → CUDA 13.0; **NV4 NVLink** |

`ga-002` has 4-bond NVLink between its two A100s (~600 GB/s bidirectional) —
confirmed with `nvidia-smi topo -m`. This makes tensor-parallel inference fast.

### What is SLURM?

SLURM is the job scheduler on Lakeshore. You write a shell script with special
`#SBATCH` lines at the top (resource requests) and submit it with `sbatch`.
SLURM queues it and runs it on a compute node when resources are free, saving
all output to a log file.

### What is a container?

A container (here, an **Apptainer** sandbox) bundles a complete software
environment: Python, PyTorch, CUDA libraries, and vLLM — isolated from whatever
is installed on the host. Containers are stored as unpacked directories
("sandboxes") on disk, typically 20–30 GB each.

### What does "module load" mean?

`module load apptainer` makes the already-installed Apptainer binary available
in your `$PATH`. Without it the binary exists on disk but your shell can't find
it.

> **Critical:** SLURM batch scripts run as non-login shells — the module system
> is **not initialized by default**. You must add `source /etc/profile` before
> `module load apptainer` in every SLURM script that uses it. Without it you
> get `module: command not found`.

```bash
# Required at the top of every SLURM script that uses apptainer:
source /etc/profile
module load apptainer
```

### How a request flows end-to-end

```
Student / client
    │  HTTPS POST /v1/chat/completions  (Authorization: Bearer <api-key>)
    ▼
relay.stream.acer.uic.edu:8001          ← public — Caddy handles TLS
    │  reverse proxy to localhost:8002
    ▼
stream-proxy  (hpc_as_api, port 8002)   ← reads HPC_MODELS from /home/ubuntu/proxy-env
    │  submits task via Globus Compute (outbound AMQP — no inbound ports on cluster)
    ▼
Globus Compute endpoint on Lakeshore   (UUID: 8d978809-eec4-413d-bbd4-b099e488100a)
    │  spawns workers via SLURM — pinned to ga-002
    ▼
Globus worker on ga-002
    │  HTTP to 127.0.0.1:8001 (loopback — same machine)
    ▼
vLLM server on ga-002 (port 8001, bound to 127.0.0.1)
    │  streaming tokens → WebSocket relay
    ▼
relay.stream.acer.uic.edu              ← SSE stream back to client
```

**Why workers run on ga-002 and vLLM binds to 127.0.0.1:**
vLLM must not listen on `0.0.0.0` — that would expose port 8001 to every user
on the cluster with no authentication. Binding to `127.0.0.1` (loopback) means
only processes on ga-002 itself can reach vLLM. The Globus Compute workers are
pinned to ga-002 via SLURM so they connect via loopback — no network exposure.

**The key insight:** adding a new model = download weights + start vLLM +
update one JSON entry in `proxy-env` + restart service. **No code changes.**

---

## Storage Layout on Lakeshore

Everything lives under the shared project space — never in `$HOME` (too small).

```
/projects/acer_hpc_admin/nassar/
├── huggingface/                    ← all model weights (HF_HOME points here)
│   └── hub/
│       ├── models--Qwen--Qwen2.5-VL-72B-Instruct-AWQ/
│       └── models--google--gemma-4-31B-it/
├── containers/                     ← Apptainer sandboxes (pinned versions)
│   ├── vllm-cu124/                 ← CUDA 12.4 build, AWQ Marlin (ghi2-002)
│   ├── vllm-0.22.1/                ← official image; Gemma 4 support (ghi2-002)
│   └── vllm-0.23.0/                ← latest official image (ga-002 — current)
├── logs/                           ← SLURM job output files
└── .hf_token                       ← HuggingFace token (chmod 600, never committed)
```

SLURM scripts live in `~/STREAM/scripts/`.

### Storage requirements

| Item | Size |
|------|------|
| Model weights (BF16, ~31B params) | ~62 GB |
| vLLM container (sandbox) | ~20–30 GB |
| Logs | ~1 GB |
| **Total** | **~95 GB** |

```bash
df -h /projects/acer_hpc_admin/nassar
```

---

## The Containers — Which One to Use?

| Container | vLLM | CUDA req | Use for | Node |
|-----------|------|----------|---------|------|
| `vllm-cu124` | 0.8.5.post1 | 12.4 | AWQ quantized models (Qwen) | ghi2-002 |
| `vllm-0.22.1` | 0.22.1 | 12.8+ | BF16 models, Gemma 4 | ghi2-002 (check driver first) |
| `vllm-0.23.0` | 0.23.0 | 12.8+ | BF16 models, latest tool-use fixes | ga-002 ✓ |

**Always pin to an exact version — never use `latest`.**
`ga-002` has driver 580 (CUDA 13.0) so both 0.22.1 and 0.23.0 work.
`ghi2-002` has driver 550 (CUDA 12.4) — only `vllm-cu124` runs AWQ at full speed.

### Pulling a new container version

```bash
#!/bin/bash
#SBATCH --job-name=pull-vllm-VERSION
#SBATCH --partition=batch_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/acer_hpc_admin/nassar/logs/pull-vllm-VERSION-%j.out

PROJECT_DIR="/projects/acer_hpc_admin/nassar"
VERSION="0.23.0"   # ← change this

source /etc/profile
module load apptainer

apptainer build \
    --sandbox \
    "${PROJECT_DIR}/containers/vllm-${VERSION}" \
    "docker://vllm/vllm-openai:v${VERSION}"

echo "Done: $(date)"
```

---

## Before You Start

- SSH access: `ssh nassar@lakeshore.acer.uic.edu`
- HuggingFace token stored securely at `/projects/acer_hpc_admin/nassar/.hf_token`:

```bash
echo "hf_your_token_here" > /projects/acer_hpc_admin/nassar/.hf_token
chmod 600 /projects/acer_hpc_admin/nassar/.hf_token
```

The token is read by all SLURM scripts via `$(cat "${PROJECT_DIR}/.hf_token")`.
It is never hardcoded in any script and never committed to git.

---

## Part 1 — Accept the Model License on HuggingFace

Some models require accepting terms before downloading (e.g. Llama). Gemma 4
uses Apache 2.0 and is currently ungated — skip if the model page shows files
without a login prompt.

1. Go to the model page on HuggingFace
2. Log in → click **"Expand to review and access the repository"** → Accept

---

## Part 2 — Download Model Weights

Use a CPU-only SLURM job — `apptainer` isn't available on the login node, and
downloads are too large to run there anyway.

```bash
#!/bin/bash
#SBATCH --job-name=download-model
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/acer_hpc_admin/nassar/logs/download-%j.out

# ── Edit these ────────────────────────────────────────────────────────────────
MODEL_ID="google/gemma-4-31B-it"
PROJECT_DIR="/projects/acer_hpc_admin/nassar"
CONTAINER="${PROJECT_DIR}/containers/vllm-0.23.0"
# ─────────────────────────────────────────────────────────────────────────────

source /etc/profile
module load apptainer

export HF_HOME="${PROJECT_DIR}/huggingface"
export HF_TOKEN=$(cat "${PROJECT_DIR}/.hf_token")

apptainer exec \
    --env HF_HOME="${HF_HOME}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    "${CONTAINER}" \
    huggingface-cli download "${MODEL_ID}"

echo "Download complete: $(date)"
```

```bash
sbatch ~/STREAM/scripts/download-MODEL.sh
tail -f /projects/acer_hpc_admin/nassar/logs/download-JOBID.out
# Wait for "Download complete"

# Verify
ls /projects/acer_hpc_admin/nassar/huggingface/hub/ | grep MODEL
```

---

## Part 3 — Start vLLM on a GPU Node

### Choosing the right node and container

| Requirement | Node | Container | GPUs |
|-------------|------|-----------|------|
| 128K context, BF16, tool-use | `ga-002` | `vllm-0.23.0` | 2× A100 80GB (TP2) |
| 32K context, BF16 | `ghi2-002` | `vllm-0.22.1` | 1× H100 NVL |
| AWQ quantized | `ghi2-002` | `vllm-cu124` | 1× H100 NVL |

### Memory budget for ga-002

```
Total VRAM (2× A100 SXM4):     160.0 GiB
vLLM budget (0.92):             147.2 GiB
Gemma 4 31B weights (BF16 TP2): ~62   GiB  (31 GiB per GPU)
KV cache available:              ~85   GiB  → 128K context, 6.47 concurrent max sessions
PyTorch/NCCL overhead:          ~13   GiB  (outside 0.92 budget)
```

### The SLURM serving script

The production script is `~/STREAM/scripts/vllm-gemma4-ga002.sh`.
Use it as the template for any new model on ga-002:

```bash
#!/bin/bash
#SBATCH --job-name=gemma4-ga002
#SBATCH --partition=batch_gpuapi
#SBATCH --account=ts_acer
#SBATCH --nodelist=ga-002
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.log

MODEL="google/gemma-4-31B-it"
PORT=8001
PROJECT_DIR="/projects/acer_hpc_admin/nassar"
CONTAINER="${PROJECT_DIR}/containers/vllm-0.23.0"

echo "=========================================="
echo "vLLM: ${MODEL}"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

source /etc/profile
module load apptainer

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME="${PROJECT_DIR}/huggingface"
export HF_TOKEN=$(cat "${PROJECT_DIR}/.hf_token")
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

CHAT_TEMPLATE="/vllm-workspace/examples/tool_chat_template_gemma4.jinja"

apptainer exec --nv --no-home --env PYTHONNOUSERSITE=1 "${CONTAINER}" \
    vllm serve "${MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 128 \
    --dtype auto \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --chat-template "${CHAT_TEMPLATE}" \
    --served-model-name gemma4-31b

echo "Service stopped: $(date)"
```

Key flags:

| Flag | Why |
|------|-----|
| `--host 127.0.0.1` | **Security**: binds to loopback only — no other cluster node can reach this port. Globus workers run on ga-002 and connect via loopback. Never use `0.0.0.0`. |
| `--served-model-name gemma4-31b` | The alias clients use in `"model": "gemma4-31b"`. Must match `hf_name` in `proxy-env`. |
| `--tensor-parallel-size 2` | Splits weights across 2 A100s via NV4 NVLink (~600 GB/s) |
| `--max-model-len 131072` | 128K context — **hard requirement, never reduce** |
| `--gpu-memory-utilization 0.92` | Leaves 8% for PyTorch/NCCL overhead |
| `--enable-auto-tool-choice` | OpenAI-compatible parallel tool calling |
| `--tool-call-parser gemma4` | Parses Gemma 4's native tool-call format |
| `--reasoning-parser gemma4` | Surfaces `<think>...</think>` as `reasoning_content` |
| `--chat-template` | Required for tool calling — Gemma 4's default template silently ignores tools |

OOM fallback order (never reduce `--max-model-len`):
1. `--gpu-memory-utilization 0.88`
2. `--max-num-seqs 64`
3. `--enforce-eager`

### Submit and monitor

```bash
cd ~/STREAM/scripts
sbatch vllm-gemma4-ga002.sh
squeue -u nassar                          # confirm RUNNING on ga-002
tail -f logs/gemma4-ga002-JOBID.log
# Wait for: "Application startup complete."
# First run: ~15 min (torch.compile from scratch)
# Subsequent restarts: ~5 min (compile cache hit)
```

### Verify vLLM is up

Since vLLM is bound to `127.0.0.1`, you **cannot** curl it from the login node.
Verify via an interactive shell on ga-002:

```bash
srun --jobid=JOBID --overlap --pty bash
curl http://127.0.0.1:8001/v1/models | python3 -m json.tool
# Expected: {"data": [{"id": "gemma4-31b", ...}]}
exit
```

Or check the log — `Application startup complete.` is definitive.

---

## Part 4 — Configure the Globus Compute Endpoint

The Globus Compute endpoint controls where worker jobs run. Workers must be
**pinned to ga-002** so they can reach vLLM via `127.0.0.1`.

The config lives at `~/.globus_compute/lakeshore-research/config.yaml`:

```yaml
display_name: UIC Lakeshore Research Computing Endpoint

heartbeat_period: 30
idle_heartbeats_soft: 0
idle_heartbeats_hard: 0

engine:
  type: GlobusComputeEngine

  # Workers run on ga-002 alongside vLLM — connect to 127.0.0.1:8001 (loopback).
  # No GPU needed — workers only make HTTP calls to vLLM.
  max_workers_per_node: 32

  provider:
    type: SlurmProvider
    account: ts_acer
    partition: batch_gpuapi
    walltime: "04:00:00"

    nodes_per_block: 1
    init_blocks: 8            # 256 workers pre-warmed at startup
    min_blocks: 8             # keep alive between class bursts
    max_blocks: 10            # 320 total slots ≥ 300 students

    cores_per_node: 4
    mem_per_node: 8
    exclusive: false

    scheduler_options: "--nodelist=ga-002"   # pin workers to same node as vLLM

    worker_init: |
      export PATH=/cm/shared/apps/slurm/23.11.11/bin:/software/EasyBuild/AMD_EPYC_7763_64-Core_Processor/software/Python/3.12.3-GCCcore-13.3.0/bin:$PATH
      export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
      export PYTHONPATH=/home/nassar/.local/lib/python3.12/site-packages:$PYTHONPATH
      export RELAY_ENCRYPTION_KEY=...   # from proxy-env
      export RELAY_SECRET=...           # from proxy-env

    parallelism: 1.0
    cmd_timeout: 60
```

> **Note:** `SlurmProvider` does not have a `nodelist` parameter — use
> `scheduler_options: "--nodelist=ga-002"` to pass it as a raw SLURM flag.

After editing the config, restart the endpoint:

```bash
globus-compute-endpoint stop lakeshore-research
globus-compute-endpoint start lakeshore-research
globus-compute-endpoint list   # confirm Running
```

---

## Part 5 — Register the Model in the Gateway

### Architecture of the relay VM

The relay VM (`relay.stream.acer.uic.edu`, SSH alias: `stream-relay`) runs:

| Service | Port | Description |
|---------|------|-------------|
| Caddy | 8001 (public) | TLS termination → proxies to localhost:8002 |
| `stream-proxy` | 8002 (localhost) | hpc-as-api gateway |
| `streamrelay` | internal | WebSocket relay for token streaming |

Configuration: `/home/ubuntu/proxy-env` — read at service startup.

### Edit the model registry

```bash
ssh stream-relay
sudo nano /home/ubuntu/proxy-env
```

The `HPC_MODELS` value must be a single line of JSON (no newlines):

```bash
HPC_MODELS={"qwen25-vl-72b": {"hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "url": "http://ghi2-002:8000", "context_reserve_output": 4096}, "gemma4-31b": {"hf_name": "gemma4-31b", "url": "http://127.0.0.1:8001", "context_reserve_output": 8192}}
```

The three fields per model:

| Field | Meaning |
|-------|---------|
| `hf_name` | **Must match `--served-model-name`** in the SLURM script — not the HuggingFace model ID. This is what the gateway sends in the `"model"` field of every request to vLLM. |
| `url` | Where vLLM is reachable **from the Globus worker**. Since workers run on ga-002 and vLLM is bound to `127.0.0.1`, use `http://127.0.0.1:PORT`. For ghi2-002 models where workers run elsewhere, use the node hostname. |
| `context_reserve_output` | Max output tokens per call. 4096 for standard chat, 8192 for reasoning models. |

### Restart the service

```bash
sudo systemctl restart stream-proxy
sudo systemctl status stream-proxy   # must show "active (running)"
```

### Verify from outside

```bash
curl https://relay.stream.acer.uic.edu:8001/v1/models \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
```

---

## Part 6 — End-to-End Test

```bash
# Non-streaming
curl -X POST https://relay.stream.acer.uic.edu:8001/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma4-31b",
       "messages": [{"role": "user", "content": "What is the capital of France?"}],
       "max_tokens": 80, "stream": false}' | python3 -m json.tool

# Streaming
curl -N -X POST https://relay.stream.acer.uic.edu:8001/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma4-31b",
       "messages": [{"role": "user", "content": "Count to five."}],
       "max_tokens": 50, "stream": true}'
# Expected: SSE lines ending with data: [DONE]

# Tool calling
curl -X POST https://relay.stream.acer.uic.edu:8001/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-31b",
    "messages": [{"role": "user", "content": "What is 42 * 17?"}],
    "tools": [{"type": "function", "function": {"name": "calculator",
               "parameters": {"type": "object",
                              "properties": {"expression": {"type": "string"}}}}}],
    "tool_choice": "auto", "max_tokens": 200
  }' | python3 -m json.tool
```

---

## Part 7 — Monitoring

### Check throughput and latency (vLLM Prometheus metrics)

From the login node, run an interactive shell on ga-002 to reach the loopback:

```bash
srun --partition=batch_gpuapi --account=ts_acer --nodelist=ga-002 \
     --cpus-per-task=1 --mem=1G --time=00:05:00 --pty bash -c "
curl -s http://127.0.0.1:8001/metrics \
  | grep -E '^vllm:(time_to_first_token|time_per_output_token|e2e_request_latency|request_success)' \
  | grep -v '_created\|_bucket'
"
```

Key metrics:

| Metric | Meaning |
|--------|---------|
| `time_to_first_token_seconds_sum / count` | Mean TTFT (target: ≤ 2s per MLPerf SLO) |
| `time_per_output_token_seconds_sum / count` | Mean TPOT (target: ≤ 200ms per MLPerf SLO) |
| `e2e_request_latency_seconds_sum / count` | Mean end-to-end latency |
| `request_success_total{finished_reason="stop"}` | Successful completions |

### Observed performance (Gemma 4 31B, 2× A100 SXM4, BF16, TP2)

Measured 2026-06-15 via the full relay stack (Globus Compute + WebSocket relay).

| Metric | Value |
|--------|-------|
| Decode throughput (single user) | **25–38 tok/s** |
| TTFT via relay (warm, single user) | **0.5–1.2s** |
| Thinking mode TTFT (first `reasoning_content` token) | **~0.5s** |
| TTFT (first ever request, Triton JIT) | ~3–5s (one-time spike) |
| Max concurrent 128K sessions (VRAM) | 6.47× |
| `torch.compile` — first run | ~68s |
| `torch.compile` — cached restarts | ~22s |

**Relay overhead floor:** Globus Compute dispatch adds ~0.4s to every TTFT regardless of vLLM warm state. Raw vLLM TTFT on the node is ~70ms; the relay round-trip accounts for the rest.

The Triton JIT spike only happens once — kernels are JIT-compiled on first use and cached for all subsequent requests.

### Concurrent capacity

| Concurrent users | Observed TTFT | Experience |
|-----------------|---------------|------------|
| 1–2 | 0.5–1.2s | Interactive — comfortable |
| 3–4 | 1–4s | Acceptable for coursework |
| 8 | 5–10s | Noticeable queuing |
| 16+ | 10s+ | Slow — batching saturated |

Bottleneck is Globus relay overhead + vLLM KV queue, not GPU throughput.

### Capacity for a 300-student class

At 25–38 tok/s single-user and ~10% duty cycle (students read/think between requests):

- **Async homework / self-paced lab**: comfortable — natural stagger means ~30
  truly concurrent requests at any time, which vLLM handles via continuous batching
- **Synchronized "everyone run this now"**: queuing — last users may wait 2–10 min
  during a 100+ request spike. Advise students to avoid simultaneous submission.

---

## Part 8 — Common Problems and Fixes

### `module: command not found` in SLURM job

Add `source /etc/profile` before `module load apptainer`. Required on ga-002
because SLURM batch scripts run as non-login shells without module initialization.

### `apptainer: command not found` on the login node

Expected — only available inside SLURM jobs on compute nodes.

### `QOSMaxCpuPerUserLimit` — job stays pending

Hit a per-user CPU quota. Reduce `--cpus-per-task` (16 → 8) and resubmit.

### CUDA out of memory at startup

Reduce in order (never reduce `--max-model-len` for ga-002):
1. `--gpu-memory-utilization 0.88`
2. `--max-num-seqs 64`
3. `--enforce-eager`

### vLLM is running but `/v1/models` returns connection refused from login node

Expected — vLLM is bound to `127.0.0.1`. You can only reach it from ga-002 itself.
Use `srun --overlap` to get a shell on the running job's node, or just watch the
log for `Application startup complete.`.

### Gateway returns `{"detail": "Model not found"}`

1. `hf_name` in `proxy-env` must exactly match `--served-model-name` in the SLURM script
2. Restart `stream-proxy` after every `proxy-env` change
3. Check: `sudo systemctl status stream-proxy`

### `torch.compile` takes 60+ seconds on first startup

Expected — vLLM compiles Triton/Inductor kernels on first run. Cached at
`/home/nassar/.cache/vllm/torch_compile_cache/` — subsequent restarts take ~22s.

### `hostname -I` IP in log is unreachable from other nodes

`hostname -I` can print a secondary interface IP. Always use the node hostname
(`ga-002`, `ghi2-002`) — not the IP. Exception: when workers are co-located on
the same node, use `127.0.0.1`.

---

## Quick Reference — Adding Any New Model

```bash
# ── LAKESHORE (login node) ────────────────────────────────────────────────────

# 1. Accept license on HuggingFace if gated

# 2. Download weights
sbatch ~/STREAM/scripts/download-MODEL.sh
tail -f /projects/acer_hpc_admin/nassar/logs/download-JOBID.out
# Wait for "Download complete"

# 3. Submit vLLM job (use vllm-gemma4-ga002.sh as template)
#    Key: --host 127.0.0.1, --served-model-name YOUR-ALIAS
sbatch ~/STREAM/scripts/vllm-YOURMODEL.sh
squeue -u nassar
tail -f ~/STREAM/scripts/logs/YOURMODEL-JOBID.log
# Wait for "Application startup complete."

# 4. Restart Globus endpoint (only if switching nodes or changing config)
globus-compute-endpoint stop lakeshore-research
globus-compute-endpoint start lakeshore-research

# ── RELAY VM ─────────────────────────────────────────────────────────────────

ssh stream-relay

# 5. Add model entry to HPC_MODELS (one JSON line)
#    hf_name = --served-model-name value (not HF model ID)
#    url = http://127.0.0.1:PORT  (workers run on same node)
sudo nano /home/ubuntu/proxy-env

# 6. Restart gateway
sudo systemctl restart stream-proxy
sudo systemctl status stream-proxy   # must be "active (running)"

# ── VERIFY ────────────────────────────────────────────────────────────────────

curl https://relay.stream.acer.uic.edu:8001/v1/models \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
```

---

## Appendix — Current Deployment

```
ga-002:8001 (127.0.0.1)  ← Gemma 4 31B BF16  (2× A100 80GB, TP2, NV4 NVLink, 128K)
ghi2-002:8000             ← Qwen 2.5-VL 72B AWQ (1× H100 NVL, TP1, 64K)
```

`proxy-env` on relay VM:

```bash
HPC_MODELS={"qwen25-vl-72b": {"hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "url": "http://ghi2-002:8000", "context_reserve_output": 4096}, "gemma4-31b": {"hf_name": "gemma4-31b", "url": "http://127.0.0.1:8001", "context_reserve_output": 8192}}
```

Globus endpoint (`~/.globus_compute/lakeshore-research/config.yaml`):
- Workers pinned to `ga-002` via `scheduler_options: "--nodelist=ga-002"`
- 8 blocks × 32 workers = 256 pre-warmed slots (scales to 320)
- Workers connect to vLLM at `http://127.0.0.1:8001`

---

## Appendix — Collaborator Onboarding

This section is for collaborators who want to run their own model on Lakeshore and expose it through the shared hpc-as-api gateway.

### What collaborators can and cannot do

| | Anas / Steve | Collaborator |
|---|---|---|
| GPU partition | `batch_gpuapi` (`ts_acer` account) | `batch_gpu2` (default account) |
| Available nodes | `ga-002` (2× A100 SXM4 80GB) | `ghi2-002` (1× H100 NVL 95.8GB) — same cluster, same container |
| Tensor parallelism | TP2 (both A100s) | TP1 only (1 GPU) |
| Max context | 128K (62GB weights + 85GB KV) | ~64K for a 13B model; ~32K for a 34B |
| Register model in gateway | Yes | No — contact Anas |

Both nodes are on the same Lakeshore cluster. Collaborators use the same vLLM container, same `huggingface-cli` workflow, and the same gateway URL — only the SLURM partition and GPU count differ.

### GPU memory budget for ghi2-002 (1× H100 NVL, 95.8GB)

```
Total VRAM:                   95.8 GiB
vLLM budget (0.90):           86.2 GiB
Example — Qwen2.5-7B BF16:    ~14 GiB weights → ~72 GiB KV cache → 64K context easy
Example — Llama-3-34B BF16:   ~68 GiB weights → ~18 GiB KV cache → 32K context
Example — 70B AWQ (4-bit):    ~35 GiB weights → ~51 GiB KV cache → 64K context
```

AWQ or GPTQ quantization is recommended for models larger than 30B on a single H100.

### Step-by-step for collaborators

**Prerequisites:**
- Lakeshore account (request via [UIC ACER help desk](https://acer.uic.edu))
- HuggingFace account + token

**1. Set up storage**

```bash
mkdir -p /projects/acer_hpc_admin/$USER/huggingface
mkdir -p /projects/acer_hpc_admin/$USER/logs
echo "hf_your_token_here" > /projects/acer_hpc_admin/$USER/.hf_token
chmod 600 /projects/acer_hpc_admin/$USER/.hf_token
```

You can reuse Anas's vLLM container (world-readable):
```bash
CONTAINER="/projects/acer_hpc_admin/nassar/containers/vllm-0.23.0"
```

**2. Download weights** — same as Part 2 of this guide; just change `PROJECT_DIR` to your space.

**3. Start vLLM on ghi2-002 (1 GPU)**

Key differences from the ga-002 script:

```bash
#SBATCH --partition=batch_gpu2      # no --account needed
#SBATCH --nodelist=ghi2-002
#SBATCH --gres=gpu:1                # 1 GPU only

export CUDA_VISIBLE_DEVICES=0
    --tensor-parallel-size 1 \
    --port 8002 \                   # 8001 is taken by Anas's model on this node
    --max-model-len 32768           # adjust based on model size — see budget above
```

See `quickstart-add-model.md` for the full script template.

**4. Contact Anas**

Once `Application startup complete.` appears in your log, send Anas:
- Model alias (`--served-model-name` value)
- Port (e.g. `8002`)
- Node (`ghi2-002`)

Anas will add it to `proxy-env` on the relay VM. The `url` entry uses the node hostname (not loopback) because the Globus workers are pinned to `ga-002`, not `ghi2-002`:

```bash
# Anas adds to HPC_MODELS:
"your-alias": {"hf_name": "your-alias", "url": "http://ghi2-002:8002", "context_reserve_output": 4096}
```

No Globus endpoint changes needed — the existing workers on `ga-002` reach `ghi2-002` over the cluster's internal network.

**5. Keep your SLURM job alive**

The model is only accessible while your SLURM job is running. If it ends (walltime, OOM, or scancel), the gateway returns `503` for that model until you resubmit. Monitor with:

```bash
squeue -u $USER
tail -f /projects/acer_hpc_admin/$USER/logs/MODELNAME-JOBID.log
```
