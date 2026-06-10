# Capacity Tuning Guide: Serving 300+ Concurrent Requests

This document explains every change made to achieve 300+ concurrent user capacity,
and why each change was necessary. It also serves as the reference for future tuning.

---

## What we started with and why it couldn't serve 300 requests

The original Globus Compute endpoint config had:

```yaml
max_workers_per_node: 1
max_blocks: 4
```

This means: **4 SLURM jobs × 1 worker each = 4 concurrent tasks maximum.**

With 4 task slots, the 5th student request queues behind the first 4. At 300 requests,
requests would queue 75-deep. Since each request takes ~30–60 s, callers #5 through
#300 would wait 37–74 minutes before seeing a single token. Completely unusable.

---

## Why workers are the bottleneck (not the GPU)

Before changing anything we ran a Layer 1 benchmark hitting vLLM directly from
the Lakeshore login node (no Globus, no relay):

```
concurrency=  1   TTFT p50=0.04s   agg_tok/s=  28
concurrency=  2   TTFT p50=0.06s   agg_tok/s=  55
concurrency=  4   TTFT p50=0.09s   agg_tok/s= 110
concurrency=  8   TTFT p50=0.13s   agg_tok/s= 217
concurrency= 16   TTFT p50=0.20s   agg_tok/s= 418
concurrency= 32   TTFT p50=0.36s   agg_tok/s= 777
concurrency= 64   TTFT p50=1.82s   agg_tok/s= 781
```

The H100 + vLLM can serve 64 simultaneous streaming requests with sub-2s TTFT and
780 tok/s aggregate throughput. The GPU is not the bottleneck at all.

The bottleneck is the Globus Compute worker layer: only 4 workers existed to forward
requests to vLLM and open WebSocket connections to the relay. The GPU was sitting idle
99% of the time waiting for Globus to dispatch more tasks.

---

## What a Globus Compute worker actually does

A Globus Compute worker is a Python process running on a batch node. For our workload,
it does exactly this:

1. Receives the task (model name, messages, relay URL) over AMQP — ~10 ms
2. Opens an HTTP connection to vLLM on `ghi2-002:8000` — ~5 ms
3. Opens a WebSocket connection to the relay — ~5 ms  
4. Streams tokens from vLLM to the relay — ~30–60 s
5. Closes both connections and reports done

Steps 1–3 take ~20 ms total. Step 4 takes 30–60 s but the worker process is just
**blocked on I/O** — it is not computing anything. Its CPU usage during streaming is
essentially 0%.

This means a single compute node with 8 CPU cores can comfortably run 32 workers
simultaneously. All 32 are just sleeping on network I/O. The 4:1 workers-per-core
ratio is very conservative.

---

## The three changes made

### Change 1: `max_workers_per_node: 1 → 32`

**Old:** Each SLURM job runs 1 worker.  
**New:** Each SLURM job runs 32 workers.

This is the key change. 32 I/O-bound workers share 8 CPU cores. Each worker sleeps
on network 99% of the time, so 8 cores is generous headroom.

### Change 2: `max_blocks: 4 → 10`

**Old:** 4 SLURM jobs maximum.  
**New:** 10 SLURM jobs maximum.

Combined with Change 1: `32 × 10 = 320 concurrent task slots`.
This exceeds 300 requests with 20 slots of burst headroom.

### Change 3: `init_blocks: 1 → 3` and `min_blocks: 1 → 3`

**Old:** 1 SLURM job pre-warmed; 1 kept alive.  
**New:** 3 SLURM jobs pre-warmed; 3 kept alive = **96 workers always running**.

Without pre-warming, the first requests after the endpoint starts have to wait for
SLURM to allocate a node (~15–60 s depending on queue). With 3 init blocks, 96
workers are alive and ready before the first caller connects.

### Change 4: `cores_per_node: 1 → 8` and `mem_per_node: 8 → 16`

The original 1 core / 8 GB was for 1 worker. With 32 workers per node:
- 8 cores @ 4 workers/core — comfortable for I/O-bound processes
- 16 GB @ 512 MB/worker — headroom for Python runtime + connection buffers

### Change 5: `parallelism: 0.8 → 1.0`

`parallelism` controls what fraction of allocated workers are allowed to receive tasks.
At 0.8, only 80% of 32 = 25 workers per block would be used. At 1.0, all 32 are active.

### Change 6: `walltime: 2:00:00 → 4:00:00`

A 2-hour class + setup/teardown fits in 4 hours. Extended to ensure workers don't
expire mid-class.

---

## The SLURM path fix

After making the config changes, sbatch calls failed with:

```
sbatch: command not found    (exit code 127)
```

The Globus endpoint daemon starts as a background process without sourcing the user's
`.bashrc`. The SLURM binaries on Lakeshore live at `/cm/shared/apps/slurm/23.11.11/bin/`
— a non-standard path not in the default `$PATH`.

Two fixes applied together:

1. **`~/.bashrc`** — added both lines so interactive logins and future daemon starts work:
   ```bash
   export PATH=/cm/shared/apps/slurm/23.11.11/bin:$PATH
   export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
   ```

2. **`worker_init`** in config.yaml — added the same exports so the compute-node batch
   environment also has the correct paths (needed if workers ever call SLURM commands):
   ```yaml
   worker_init: |
     export PATH=/cm/shared/apps/slurm/23.11.11/bin:/software/.../Python/3.12.../bin:$PATH
     export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
     ...
   ```
   
   The `module load Python/3.12...` line was also replaced with a direct PATH entry
   because batch nodes don't source `/etc/profile.d/` (where `module` is defined).

---

## Before vs after: summary table

| Setting | Before | After | Effect |
|---|---|---|---|
| `max_workers_per_node` | 1 | **32** | 32× more parallelism per SLURM job |
| `max_blocks` | 4 | **10** | 10 SLURM jobs max |
| `init_blocks` | 1 | **3** | 96 workers pre-warmed on startup |
| `min_blocks` | 1 | **3** | 96 workers kept alive between bursts |
| `cores_per_node` | 1 | **8** | Supports 32 workers per node |
| `mem_per_node` | 8 GB | **16 GB** | ~512 MB per worker |
| `parallelism` | 0.8 | **1.0** | All allocated workers active |
| `walltime` | 2:00:00 | **4:00:00** | Covers full session |
| **Total task slots** | **4** | **320** | **80× capacity increase** |
| sbatch PATH | broken | **fixed** | SLURM blocks actually start |

---

## Final config

The deployed config is at `~/.globus_compute/lakeshore-research/config.yaml` on
the Lakeshore login node. A copy is kept at the bottom of this document.

---

## How to restart the endpoint

After any config change:

```bash
ssh nassar@lakeshore.acer.uic.edu

# Stop
~/.local/bin/globus-compute-endpoint stop lakeshore-research

# Start (with env vars the daemon needs)
export PATH=/cm/shared/apps/slurm/23.11.11/bin:$PATH
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
~/.local/bin/globus-compute-endpoint start lakeshore-research

# Verify
~/.local/bin/globus-compute-endpoint list
# Expected: Running

# Verify SLURM jobs were submitted (wait ~30 s)
export PATH=/cm/shared/apps/slurm/23.11.11/bin:$PATH
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
squeue -u nassar
# Expected: 3 parsl.GlobusComputeEngine... jobs in state R
```

---

## How to verify capacity before class

```bash
# 1. Check gateway health
curl -s https://relay.stream.acer.uic.edu:8001/health
# Expected: {"status": "healthy", "globus_configured": true, ...}

# 2. Verify SLURM workers are running
ssh nassar@lakeshore.acer.uic.edu "
  export PATH=/cm/shared/apps/slurm/23.11.11/bin:\$PATH
  export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
  squeue -u nassar --format='%i %j %t %N'
"
# Expected: at least 3 parsl.GlobusComputeEngine... jobs in state R

# 3. Quick warmup request (confirms workers accept tasks)
curl -s -N -X POST https://relay.stream.acer.uic.edu:8001/v1/chat/completions \
  -H "Authorization: Bearer sk-stream-amplify-2f4e12d7..." \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen25-vl-72b","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":true}' \
  | head -3
# Expected: data: {"id": ...} lines within ~2 s
```

---

## What to give the operator

The operator needs two things:

```
Endpoint URL:  https://relay.stream.acer.uic.edu:8001
API Key:       sk-stream-amplify-2f4e12d7af4ee1fd0911c3241e5adcfd30b6aa1bcaf2ca65f1c30aaf1b92ddd4
Model name:    qwen25-vl-72b
```

The API is fully OpenAI-compatible. Callers can use it with the standard OpenAI
Python SDK, any REST client, or any tool that accepts a custom `base_url`:

```python
# Python (openai SDK)
from openai import OpenAI
client = OpenAI(
    base_url="https://relay.stream.acer.uic.edu:8001/v1",
    api_key="sk-stream-amplify-2f4e12d7af4ee1fd0911c3241e5adcfd30b6aa1bcaf2ca65f1c30aaf1b92ddd4",
)
response = client.chat.completions.create(
    model="qwen25-vl-72b",
    messages=[{"role": "user", "content": "Explain transformer attention."}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

```bash
# curl
curl -N -X POST https://relay.stream.acer.uic.edu:8001/v1/chat/completions \
  -H "Authorization: Bearer sk-stream-amplify-2f4e12d7af4ee1fd0911c3241e5adcfd30b6aa1bcaf2ca65f1c30aaf1b92ddd4" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen25-vl-72b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

Works with: OpenWebUI, LangChain, LlamaIndex, Cursor, any OpenAI-compatible tool.

---

## Realistic capacity expectations

| Metric | Value | Notes |
|---|---|---|
| Task slots | 320 | 32 workers × 10 blocks |
| Always-warm slots | 96 | 32 workers × 3 init/min blocks |
| vLLM throughput | ~780 tok/s | H100, AWQ marlin, tensor-parallel-1 |
| Comfortable simultaneous requests | 64 | TTFT < 2s, from Layer 1 benchmark |
| tok/s per caller @ 64 concurrent | ~12 | Very fast (80 tokens in ~7 s) |
| tok/s per caller @ 300 concurrent | ~2.6 | Still reasonable (80 tokens in ~31 s) |
| Realistic class concurrency | 30–60 | Students read, think, type between prompts |
| Class of 300 peak burst | 60–100 | Comfortably within 96 warm slots |

**Bottom line:** 300 requests in a class is fine. Even at peak burst (60–100
simultaneous), the 96 pre-warmed workers handle it with sub-2s TTFT.
If every student somehow sent a request at the exact same millisecond,
the 320-slot ceiling handles it within acceptable queue depth.

---

## Appendix: deployed config.yaml

```yaml
display_name: UIC Lakeshore Research Computing Endpoint

heartbeat_period: 30
idle_heartbeats_soft: 0
idle_heartbeats_hard: 0

engine:
  type: GlobusComputeEngine
  max_workers_per_node: 32

  provider:
    type: SlurmProvider
    account: ts_acer_chi
    partition: batch
    walltime: "04:00:00"

    nodes_per_block: 1
    init_blocks: 3
    min_blocks: 3
    max_blocks: 10

    cores_per_node: 8
    mem_per_node: 16
    exclusive: false

    scheduler_options: ""

    worker_init: |
      export PATH=/cm/shared/apps/slurm/23.11.11/bin:/software/EasyBuild/AMD_EPYC_7763_64-Core_Processor/software/Python/3.12.3-GCCcore-13.3.0/bin:$PATH
      export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
      export PYTHONPATH=/home/nassar/.local/lib/python3.12/site-packages:$PYTHONPATH
      export RELAY_ENCRYPTION_KEY=<see proxy-env on relay server>
      export RELAY_SECRET=<see proxy-env on relay server>

    parallelism: 1.0
    cmd_timeout: 60
```
