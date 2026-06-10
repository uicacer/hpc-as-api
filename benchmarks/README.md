# hpc-as-api Benchmark Suite

**Goal:** Can 300 students use the gateway concurrently for a class?

This folder contains two complementary tests — a **load test** (validate the target) and a **stress test** (find the breaking point) — run at two layers of the stack.

---

## Terminology

| Term | Definition |
|---|---|
| **Load test** | Run at a target concurrency level; verify latency stays within SLA. Answers: "does it work for 300 users?" |
| **Stress test** | Ramp concurrency until the system fails; find the breaking point. Answers: "where does it break?" |
| **TTFT** | Time-to-first-token — how long the student waits before seeing any response. Most important UX metric. |
| **p50 / p95** | Median and 95th percentile. p95 = 19 out of 20 students see this or better. |
| **Layer 1** | vLLM direct — hits `ghi2-002:8000` with no Globus or relay in the path. Measures raw GPU ceiling. |
| **Layer 2** | End-to-end — hits the public gateway through Globus Compute + WebSocket relay. Measures the full stack. |

---

## Architecture recap

```
[Student laptop]
      │ POST /v1/chat/completions
      ▼
[relay.stream.acer.uic.edu:8001]  ← hpc-as-api (FastAPI)
      │ Globus Compute AMQP
      ▼
[Lakeshore login node]  →  SLURM job  →  ghi2-002 (H100 NVL)
      │                                        │
      │                                   vLLM HTTP :8000
      │                                        │ tokens
      ▼                                        ▼
[relay.stream.acer.uic.edu:8765]  ← WebSocket relay (streamrelay)
      │ SSE stream
      ▼
[Student laptop]
```

**Known bottlenecks (before running):**

| Layer | Limit | Config |
|---|---|---|
| vLLM continuous batching | 256 simultaneous sequences | `--max-num-seqs 256` |
| Single H100 NVL throughput | ~25 tok/s total | `--tensor-parallel-size 1` |
| Globus Compute workers | 4 max | `max_blocks: 4` in endpoint config |
| Globus AMQP dispatch overhead | ~1–2 s cold, <1 s warm | persistent Executor |
| Relay server | Not a bottleneck | memory-copy forwarder |

> **Expected finding:** The GPU can handle 300 students in a batched sense (25 tok/s ÷ 300 = 0.08 tok/s/user, which means a student's 80-token response takes ~30–60 s). The harder limit is Globus Compute `max_blocks=4` — only 4 SLURM workers can be dispatching at once.

---

## Step-by-step instructions

### Prerequisites

```bash
# On your laptop — Python 3.9+ stdlib only (no pip install needed)
python3 --version

# Confirm you can SSH to Lakeshore
ssh nassar@lakeshore.acer.uic.edu "hostname"

# Confirm the gateway is healthy
curl -s https://relay.stream.acer.uic.edu:8001/health
# Expected: {"status": "healthy", ...}
```

You need an API key for the gateway. Get it from the proxy EnvironmentFile:

```bash
ssh stream-relay "grep PROXY_API_KEY /home/ubuntu/proxy-env"
# Pick any key from the output (PROXY_API_KEY_xxx=sk-...)
```

---

### Layer 1: vLLM direct (run from Lakeshore)

This measures the raw GPU + vLLM ceiling with zero gateway overhead.
You must run it **from the Lakeshore login node** because `ghi2-002:8000` is only reachable on the internal LAN.

```bash
# 1. SSH to Lakeshore
ssh nassar@lakeshore.acer.uic.edu

# 2. Copy the script (from your laptop, in a separate terminal)
scp benchmarks/layer1_vllm_direct.py nassar@lakeshore.acer.uic.edu:~/

# 3. Run it (Python 3.9 is available by default; no deps needed)
python3 layer1_vllm_direct.py --url http://ghi2-002:8000 --output results/layer1.json

# Expected output (approximate — real numbers will differ):
#   concurrency=  1  TTFT p50=0.62s p95=0.70s  total p50=3.1s  agg_tok/s=24.8  errors=0
#   concurrency=  2  TTFT p50=0.65s p95=0.80s  total p50=5.8s  agg_tok/s=24.5  errors=0
#   concurrency=  4  TTFT p50=1.10s p95=1.40s  total p50=9.2s  agg_tok/s=23.9  errors=0
#   concurrency=  8  TTFT p50=2.20s p95=2.80s  total p50=17s   agg_tok/s=22.1  errors=0
#   concurrency= 16  TTFT p50=4.50s p95=6.10s  total p50=32s   agg_tok/s=19.8  errors=0
#   concurrency= 32  TTFT p50=9.10s p95=12.5s  total p50=60s   agg_tok/s=17.2  errors=2

# 4. Copy results back to your laptop
scp nassar@lakeshore.acer.uic.edu:~/results/layer1.json benchmarks/results/
```

---

### Layer 2a: Load test (run from your laptop)

Sweeps concurrency 1→2→4→8→16→32 against the full stack and extrapolates to 300 users.

```bash
cd benchmarks

python3 layer2_gateway.py \
    --url https://relay.stream.acer.uic.edu:8001 \
    --api-key YOUR_API_KEY_HERE \
    --mode load \
    --output results/layer2_load.json

# Expected output:
#   c=   1  TTFT p50=1.9s p95=2.1s  total p50=5.2s p95=5.8s  agg=25 tok/s  err=0/1
#   c=   2  TTFT p50=2.1s p95=2.4s  total p50=9.1s p95=9.8s  agg=24 tok/s  err=0/2
#   c=   4  TTFT p50=2.8s p95=3.5s  total p50=16s p95=18s    agg=23 tok/s  err=0/4
#   c=   8  TTFT p50=5.1s p95=6.8s  total p50=30s p95=34s    agg=21 tok/s  err=0/8
#   c=  16  TTFT p50=9.8s p95=13s   total p50=55s p95=62s    agg=18 tok/s  err=2/16
#   c=  32  TTFT p50=18s  p95=25s   total p50=90s p95=110s   agg=15 tok/s  err=8/32
```

---

### Layer 2b: Stress test (run from your laptop)

Doubles concurrency (8→16→32→64→128→256→512) until ≥50% of requests fail.

```bash
python3 layer2_gateway.py \
    --url https://relay.stream.acer.uic.edu:8001 \
    --api-key YOUR_API_KEY_HERE \
    --mode stress \
    --output results/layer2_stress.json
```

---

### Analyze and compare

```bash
python3 analyze.py \
    --layer1 results/layer1.json \
    --layer2-load results/layer2_load.json \
    --layer2-stress results/layer2_stress.json \
    --output results/report.txt

# Outputs:
#   results/report.txt  — full text report with projections and recommendations
#   results/report.csv  — data table for plotting
```

---

## Results

> Results are populated after running the tests. See [results/report.txt](results/report.txt).

### Layer 1: vLLM direct

<!-- RESULTS_L1_TABLE -->
*Run `layer1_vllm_direct.py` from the Lakeshore login node to populate this.*

### Layer 2: End-to-end gateway

<!-- RESULTS_L2_TABLE -->
*Run `layer2_gateway.py` to populate this.*

### 300-user projection

<!-- RESULTS_PROJECTION -->
*Run `analyze.py` to populate this.*

---

## Interpretation guide

### What "300 concurrent students" actually means

A class of 300 students is **not** 300 simultaneous requests. Students:
- Read the response (~30–90 s)
- Think and type the next prompt (~30–120 s)
- Take notes, look at code, etc.

A realistic **concurrency factor** is 10–20%: at any given second, 30–60 students are actively waiting for a response. Plan for **burst peaks of 60–100** simultaneous requests, not 300.

### Reading the numbers

| Metric | Acceptable for class use | Notes |
|---|---|---|
| TTFT p95 | ≤ 10 s | Students start seeing tokens within 10 s |
| Total p95 | ≤ 60 s | 80-token response completes within 1 minute |
| Error rate | < 5% | Occasional retries are fine |
| tok/s per user | ≥ 0.5 | At 25 tok/s total, comfortable for ≤ 50 concurrent |

### If numbers are worse than expected

| Symptom | Root cause | Fix |
|---|---|---|
| TTFT > 5 s even at low concurrency | Globus cold start (first request) | Increase `init_blocks: 1` (already set) |
| TTFT grows fast with concurrency | `max_blocks: 4` is the ceiling | Increase `max_blocks` to 16–32 |
| Total latency >> expected | vLLM queue depth high | Enable tensor-parallel across all 4 H100s |
| 503 errors | Globus Compute workers not running | Restart endpoint; check SLURM queue |
| Breaking point < 32 | Relay or proxy overwhelmed | Check relay server load; unlikely |

### How to increase capacity for the class

**Option A (recommended): Increase `max_blocks`**
```yaml
# ~/.globus_compute/lakeshore-research/config.yaml
engine:
  provider:
    max_blocks: 32     # was 4
    init_blocks: 4     # pre-warm 4 workers before class starts
```
This lets 32 SLURM workers dispatch simultaneously, drastically reducing queue depth.

**Option B: Use all 4 H100s**
```bash
# On ghi2-002, restart vLLM with tensor parallelism:
vllm serve Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
    --tensor-parallel-size 4 \
    --max-num-seqs 256 \
    ...
```
4× GPUs → ~4× throughput (100 tok/s). At 0.5 tok/s/student, comfortable for 200 concurrent.

**Option C: Pre-warm before class**
```bash
# 10 minutes before class, send a warmup request so Globus workers are running:
curl -s -X POST https://relay.stream.acer.uic.edu:8001/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen25-vl-72b","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":true}' \
  | head -5
```
