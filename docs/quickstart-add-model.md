# Quick Guide: Add a New Model to HPC-as-API

**Prerequisites:** SSH access to Lakeshore, HuggingFace account.

**Storage layout (quick reminder):**
- Model weights → `/projects/acer_hpc_admin/nassar/huggingface/` (persists across jobs; shared project storage, not home dir)
- vLLM container (Apptainer SIF) → `/projects/acer_hpc_admin/nassar/containers/vllm-0.23.0` — pinned version with all CUDA deps; Docker is not available on Lakeshore (no root), so we use Apptainer

---

## Step 1 — Accept the model license

Go to the model page on HuggingFace, log in, and accept the license if prompted.
Some models (e.g. Llama) are gated; others (e.g. Gemma 4) are open.

---

## Step 2 — Store your HuggingFace token

On the Lakeshore login node (`ssh YOUR_NETID@lakeshore.acer.uic.edu`):

```bash
echo "hf_your_token_here" > /projects/acer_hpc_admin/nassar/.hf_token  # save token to a private file
chmod 600 /projects/acer_hpc_admin/nassar/.hf_token                    # restrict to your user only
```

Get your token at: huggingface.co → Settings → Access Tokens (read scope is enough).

---

## Step 3 — Download the model weights

Create `~/STREAM/scripts/download-MODELNAME.sh` and submit it:

```bash
#!/bin/bash
#SBATCH --job-name=download-model      # name shown in squeue
#SBATCH --partition=batch              # CPU-only nodes — no GPU needed for download
#SBATCH --cpus-per-task=4              # parallel download threads
#SBATCH --mem=16G                      # memory for the download process
#SBATCH --time=02:00:00                # max runtime — 2h is enough for ~60GB
#SBATCH --output=/projects/acer_hpc_admin/nassar/logs/download-%j.out  # log file (%j = job ID)

MODEL_ID="google/gemma-4-31B-it"           # ← change this to your model's HuggingFace ID
PROJECT_DIR="/projects/acer_hpc_admin/nassar"
CONTAINER="${PROJECT_DIR}/containers/vllm-0.23.0"  # container that has huggingface-cli

source /etc/profile       # initialize the module system (required in SLURM batch jobs)
module load apptainer     # make the apptainer command available

export HF_HOME="${PROJECT_DIR}/huggingface"          # where weights are saved on disk
export HF_TOKEN=$(cat "${PROJECT_DIR}/.hf_token")    # read token from the file saved in Step 2

apptainer exec \
    --env HF_HOME="${HF_HOME}" \         # pass HF_HOME into the container
    --env HF_TOKEN="${HF_TOKEN}" \       # pass token into the container
    "${CONTAINER}" \                     # run inside the vLLM container
    huggingface-cli download "${MODEL_ID}"  # download all model files to HF_HOME

echo "Download complete: $(date)"
```

```bash
sbatch ~/STREAM/scripts/download-MODELNAME.sh              # submit the job
tail -f /projects/acer_hpc_admin/nassar/logs/download-JOBID.out  # watch progress
# Wait for: "Download complete"
```

---

## Step 4 — Start vLLM on ga-002

Create `~/STREAM/scripts/vllm-MODELNAME.sh`. Change `MODEL`, `PORT`,
`--served-model-name`, and `--tool-call-parser` / `--chat-template` if the
model family differs from Gemma 4.

```bash
#!/bin/bash
#SBATCH --job-name=MODELNAME           # name shown in squeue
#SBATCH --partition=batch_gpuapi       # GPU partition for ga-002
#SBATCH --account=ts_acer              # required billing account for batch_gpuapi
#SBATCH --nodelist=ga-002              # pin to this specific GPU node
#SBATCH --gres=gpu:2                   # request both A100 GPUs (needed for 128K context)
#SBATCH --cpus-per-task=8              # CPU cores for the vLLM process
#SBATCH --mem=80G                      # system RAM for model loading and workers
#SBATCH --time=48:00:00                # keep running for 48 hours
#SBATCH --output=logs/%x-%j.log        # log file (%x = job name, %j = job ID)

MODEL="google/gemma-4-31B-it"              # ← HuggingFace model ID (used to load weights)
PORT=8001                                   # ← change if running a second model on the same node
PROJECT_DIR="/projects/acer_hpc_admin/nassar"
CONTAINER="${PROJECT_DIR}/containers/vllm-0.23.0"  # pinned container version

echo "=========================================="
echo "vLLM: ${MODEL} | Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

source /etc/profile       # initialize the module system (required in SLURM batch jobs)
module load apptainer     # make the apptainer command available

export CUDA_VISIBLE_DEVICES=0,1    # use both GPUs on ga-002
export HF_HOME="${PROJECT_DIR}/huggingface"          # where weights are stored
export HF_TOKEN=$(cat "${PROJECT_DIR}/.hf_token")    # HuggingFace token for gated models
export NCCL_P2P_DISABLE=0          # enable direct GPU-to-GPU communication via NVLink
export NCCL_SHM_DISABLE=0          # enable shared memory transport between GPUs

apptainer exec --nv \               # --nv: pass GPUs into the container
    --no-home \                     # don't mount home dir (keeps container isolated)
    --env PYTHONNOUSERSITE=1 \      # ignore user's local Python packages inside container
    "${CONTAINER}" \
    vllm serve "${MODEL}" \
    --host 127.0.0.1 \              # loopback only — no other cluster node can reach this port
    --port "${PORT}" \
    --tensor-parallel-size 2 \      # split model across 2 GPUs via NVLink
    --max-model-len 131072 \        # 128K context window (131072 = 128 × 1024) — do not reduce
    --gpu-memory-utilization 0.92 \ # reserve 92% of VRAM for weights + KV cache
    --max-num-seqs 128 \            # max concurrent requests
    --dtype auto \                  # auto-select best dtype (BF16 on A100)
    --enable-prefix-caching \       # cache repeated system prompts — free speedup
    --enable-chunked-prefill \      # interleave prefill and decode for better throughput
    --enable-auto-tool-choice \     # enable OpenAI-compatible tool/function calling
    --tool-call-parser gemma4 \     # parse Gemma 4's tool-call format → OpenAI format
    --reasoning-parser gemma4 \     # expose <think>...</think> as reasoning_content field
    --chat-template /vllm-workspace/examples/tool_chat_template_gemma4.jinja \  # required for tool calling to work
    --served-model-name gemma4-31b  # ← the alias clients use in "model": "gemma4-31b"

echo "Service stopped: $(date)"
```

```bash
sbatch ~/STREAM/scripts/vllm-MODELNAME.sh         # submit the job
squeue -u nassar                                   # confirm STATE=RUNNING on ga-002
tail -f ~/STREAM/scripts/logs/MODELNAME-JOBID.log  # watch startup log
# Wait for: "Application startup complete."        # first run ~15min, restarts ~5min
```

---

## Step 5 — Contact Anas ✉️

Once you see `Application startup complete.` in the log, send Anas:

1. The **model alias** you used in `--served-model-name` (e.g. `gemma4-31b`)
2. The **port** (e.g. `8001`)

**Anas will do the rest** — register the model in the gateway and restart the
service. This takes about 2 minutes on his side.

---

## Step 6 — (Anas only) Register in the gateway

```bash
ssh stream-relay                       # SSH to the relay VM
sudo nano /home/ubuntu/proxy-env       # edit the gateway config
```

Add the new model to `HPC_MODELS` — must be a **single JSON line, no newlines**:

```bash
HPC_MODELS={..., "YOUR-ALIAS": {"hf_name": "YOUR-ALIAS", "url": "http://127.0.0.1:PORT", "context_reserve_output": 8192}}
#                               ^ must match --served-model-name   ^ loopback: Globus workers run on ga-002
```

```bash
sudo systemctl restart stream-proxy         # apply the new config
sudo systemctl status stream-proxy          # must show "active (running)"
```

---

## Step 7 — Verify end-to-end

```bash
# List models — your alias should appear
curl https://relay.stream.acer.uic.edu:8001/v1/models \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# Quick chat test
curl -X POST https://relay.stream.acer.uic.edu:8001/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "YOUR-ALIAS",
       "messages": [{"role": "user", "content": "Say hello."}],
       "max_tokens": 50, "stream": false}' | python3 -m json.tool
```

---

## Reference — Current models

| Alias | Model | Node | Context | Status |
|-------|-------|------|---------|--------|
| `gemma4-31b` | google/gemma-4-31B-it | ga-002 (2× A100 80GB) | 128K | **Online** |
| `qwen25-vl-72b` | Qwen/Qwen2.5-VL-72B-Instruct-AWQ | ghi2-002 (H100 NVL) | 64K | Offline (SLURM job not running) |

**Performance expectations (gemma4-31b via relay):**
- TTFT warm, single user: 0.5–1.2s
- Decode throughput: 25–38 tok/s
- Concurrent interactive (TTFT ≤1s): 1–2 users
- Concurrent acceptable (TTFT ≤4s): 3–4 users

For full details, troubleshooting, and memory budgets see
[adding-a-new-model.md](adding-a-new-model.md).
