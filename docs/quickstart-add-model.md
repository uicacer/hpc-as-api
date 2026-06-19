# Quick Guide: Add a New Model to HPC-as-API

**Prerequisites:** SSH access to Lakeshore, HuggingFace account.

**Storage layout:**
- Model weights → `/projects/acer_hpc_admin/<username>/huggingface/`
- vLLM container → `/projects/acer_hpc_admin/<username>/containers/vllm-0.23.0`
- SLURM scripts → `~/STREAM/scripts/`
- Logs → `~/STREAM/scripts/logs/` and `/projects/acer_hpc_admin/<username>/logs/`

---

## Step 0 — Pull the vLLM container

This only needs to be done once per account. The container is an Apptainer sandbox built from the official vLLM Docker image.

```bash
#!/bin/bash
#SBATCH --job-name=pull-vllm-0.23.0
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/acer_hpc_admin/<username>/logs/pull-vllm-0.23.0-%j.log

source /etc/profile
module load apptainer

CONTAINER_DIR="/projects/acer_hpc_admin/<username>/containers"
TARGET="${CONTAINER_DIR}/vllm-0.23.0"

if [ -d "${TARGET}" ]; then
    echo "Container already exists at ${TARGET} — exiting."
    exit 0
fi

echo "============================================"
echo "Pulling vLLM v0.23.0 from Docker Hub"
echo "Target: ${TARGET}"
echo "Started: $(date)"
echo "============================================"

apptainer build --sandbox "${TARGET}" docker://vllm/vllm-openai:v0.23.0

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "PULL SUCCESSFUL — $(date)"
    apptainer exec "${TARGET}" python3 -c "import vllm; print('vLLM version:', vllm.__version__)"
else
    echo "PULL FAILED (exit code: ${EXIT_CODE}) — $(date)"
    rm -rf "${TARGET}"
fi
```

**Why v0.23.0?** It's the first release with correct Gemma 4 support: MTP speculative decoding, tool parser fixes, and Model Runner V2 Gemma 4 support. Use this version or newer.

```bash
mkdir -p /projects/acer_hpc_admin/<username>/logs
mkdir -p /projects/acer_hpc_admin/<username>/containers
sbatch ~/STREAM/scripts/pull-vllm-0.23.0.sh
tail -f /projects/acer_hpc_admin/<username>/logs/pull-vllm-0.23.0-JOBID.log
# Wait for: "PULL SUCCESSFUL" and the version line
# Takes ~1–2 hours
```

---

## Step 1 — Accept the model license

Go to the model page on HuggingFace, log in, and accept the license if prompted.
Some models (e.g. Llama) are gated; others (e.g. Gemma 4) are open.

---

## Step 2 — Store your HuggingFace token

On the Lakeshore login node (`ssh <username>@lakeshore.acer.uic.edu`):

```bash
echo "hf_your_token_here" > /projects/acer_hpc_admin/<username>/.hf_token
chmod 600 /projects/acer_hpc_admin/<username>/.hf_token
```

Get your token at: huggingface.co → Settings → Access Tokens (read scope is enough).

---

## Step 3 — Download the model weights

This is the exact script used to download Gemma 4. Copy it, change the model ID, local-dir path, and job name for your model:

```bash
#!/bin/bash
# #SBATCH lines are instructions to SLURM, not shell comments.
# SLURM reads them before the job starts; bash ignores them during execution.
#SBATCH --job-name=download-gemma4       # name shown in squeue; change to match your model
#SBATCH --partition=batch                # CPU-only nodes; no GPU needed for downloading
#SBATCH --ntasks=1                       # one process (huggingface-cli is single-process)
#SBATCH --cpus-per-task=4               # huggingface-cli downloads in parallel threads; more = faster
#SBATCH --mem=16G                        # download uses ~1-2 GB; 16G is a safe buffer
#SBATCH --time=02:00:00                  # 2 hours is enough for ~66 GB at cluster network speeds
#SBATCH --output=/projects/acer_hpc_admin/<username>/logs/download-gemma4-%j.out  # %j = job ID

# Make the apptainer command available on this compute node.
module load apptainer

# Tell HuggingFace where to store files.
# Never use $HOME — home directory quota is only 100 GiB; the model alone is ~66 GiB.
export HF_HOME=/projects/acer_hpc_admin/<username>/huggingface

# Run huggingface-cli inside the vLLM container (huggingface_hub is installed inside it).
# --env HF_HOME passes the variable into the isolated container environment.
# --local-dir saves files to a path matching HuggingFace's cache layout (required by vLLM).
apptainer exec \
    --env HF_HOME=${HF_HOME} \
    /projects/acer_hpc_admin/<username>/containers/vllm-0.23.0 \
    huggingface-cli download \
        google/gemma-4-31B-it \
        --local-dir ${HF_HOME}/hub/models--google--gemma-4-31B-it

echo "Download complete: $(date)"
```

```bash
sbatch ~/STREAM/scripts/download-gemma4.sh
tail -f /projects/acer_hpc_admin/<username>/logs/download-gemma4-JOBID.out
# Wait for: "Download complete"
```

---

## Step 4 — Start vLLM on ga-002

This is the exact production script currently running Gemma 4. Copy it and adjust `MODEL`, `PORT`, `--served-model-name`, and the chat template for your model:

```bash
#!/bin/bash
#SBATCH --job-name=gemma4-ga002          # name shown in squeue
#SBATCH --partition=batch_gpuapi         # GPU partition for ga-002; requires --account=ts_acer
#SBATCH --account=ts_acer                # billing account — only Anas and Steve have access
#SBATCH --nodelist=ga-002                # pin to ga-002; Globus workers also run here, reach vLLM via 127.0.0.1
#SBATCH --gres=gpu:2                     # both A100s; 1 GPU gives only ~18 GiB KV cache (~16K context)
#SBATCH --cpus-per-task=4                # CPU cores for vLLM HTTP server, tokenizer, scheduling loop
#SBATCH --mem=80G                        # system RAM for weight loading and worker buffers
#SBATCH --time=48:00:00                  # resubmit before expiry to avoid downtime
#SBATCH --output=logs/%x-%j.log          # %x = job name, %j = job ID; logs go to ~/STREAM/scripts/logs/

# =============================================================================
# Memory budget (tensor-parallel-size 2, gpu-memory-utilization 0.90):
#   Total VRAM (2x A100):     160.0 GiB
#   vLLM budget (0.90):       144.0 GiB
#   Weights (BF16, TP2):       ~62   GiB
#   KV cache available:        ~82   GiB  -> 128K context
#   Reserved (PyTorch/NCCL):   ~16   GiB
#
# IF OOM AT STARTUP (step down in order):
#   1. --gpu-memory-utilization 0.88  (do NOT reduce --max-model-len)
#   2. --max-num-seqs 64
#   3. --enforce-eager
# =============================================================================

MODEL="google/gemma-4-31B-it"
PORT=8001
PROJECT_DIR="/projects/acer_hpc_admin/<username>"
CONTAINER="${PROJECT_DIR}/containers/vllm-0.23.0"

echo "=========================================="
echo "vLLM: ${MODEL}"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Initialize the module system. Required in SLURM batch jobs — they run as
# non-login shells and don't load /etc/profile automatically.
source /etc/profile
module load apptainer

export HF_HOME="${PROJECT_DIR}/huggingface"
export HF_TOKEN=$(cat "${PROJECT_DIR}/.hf_token")

NODE_IP=$(hostname -I | awk '{print $1}')
echo "Service: http://${NODE_IP}:${PORT}"

# Gemma 4's default chat template does not activate tool-calling.
# vLLM ships a dedicated jinja template that wires up function-call tokens.
# Without this, --enable-auto-tool-choice has no effect.
# The path is inside the container (vllm-workspace = vLLM's source tree in the SIF).
CHAT_TEMPLATE="/vllm-workspace/examples/tool_chat_template_gemma4.jinja"

# --nv: expose GPUs inside the container (required for CUDA)
# --no-home: don't mount home dir — keeps container isolated
# --env PYTHONNOUSERSITE=1: ignore user's local Python packages inside container
# --host 127.0.0.1: loopback only — vLLM not exposed to other cluster users;
#   Globus workers are pinned to ga-002 and connect via 127.0.0.1
# --tensor-parallel-size 2: split weights across 2 GPUs via NVLink
# --max-model-len 131072: 128K context window — do not reduce, hard requirement
# --gpu-memory-utilization 0.90: reserve 90% of VRAM for weights + KV cache
# --max-num-seqs 128: max concurrent requests in flight
# --dtype auto: auto-select best dtype — BF16 on A100
# --enable-prefix-caching: cache repeated system prompts — free speedup
# --enable-chunked-prefill: interleave prefill and decode — better concurrency
# --enable-auto-tool-choice: OpenAI-compatible parallel tool/function calling
# --tool-call-parser gemma4: parse Gemma 4's tool-call format → OpenAI format
# --reasoning-parser gemma4: expose <think>...</think> as reasoning_content field
# --chat-template: required for tool calling — see note above
# --limit-mm-per-prompt: disable image/audio — text-only deployment
# --async-scheduling: decouple scheduling from GPU loop — better concurrency
# --served-model-name: alias clients use in "model": "gemma4-31b";
#   must match hf_name in proxy-env on the relay VM
apptainer exec \
    --nv \
    --no-home \
    --env PYTHONNOUSERSITE=1 \
    "${CONTAINER}" \
    vllm serve "${MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 128 \
    --dtype auto \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --chat-template "${CHAT_TEMPLATE}" \
    --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
    --async-scheduling \
    --served-model-name gemma4-31b

echo "Service stopped: $(date)"
```

```bash
cd ~/STREAM/scripts
sbatch vllm-gemma4-ga002.sh
squeue -u <username>                     # confirm STATE=RUNNING on ga-002
tail -f logs/gemma4-ga002-JOBID.log
# Wait for: "Application startup complete."
# First run: ~15 min (torch.compile from scratch)
# Subsequent restarts: ~5 min (compile cache hit)
```

---

## Step 5 — Contact Anas ✉️

Once you see `Application startup complete.` in the log, send Anas:

1. The **model alias** used in `--served-model-name` (e.g. `gemma4-31b`)
2. The **port** (e.g. `8001`)

**Anas will do the rest** — register the model in the gateway and restart the service. This takes about 2 minutes on his side.

---

## Step 6 — (Anas only) Register the model on the proxy/relay VM

```bash
ssh stream-relay
sudo nano /home/ubuntu/proxy-env
```

Add the new model to `HPC_MODELS` — must be a **single JSON line, no newlines**:

```bash
HPC_MODELS={..., "YOUR-ALIAS": {"hf_name": "YOUR-ALIAS", "url": "http://127.0.0.1:PORT", "context_reserve_output": 8192}}
```

`hf_name` must exactly match `--served-model-name`. `url` uses `127.0.0.1` because Globus workers are pinned to ga-002 and connect via loopback.

```bash
sudo systemctl restart stream-proxy
sudo systemctl status stream-proxy   # must show "active (running)"
```

---

## Step 7 — Verify end-to-end

```bash
export API_KEY="your-key-here"

# List models
curl https://relay.stream.acer.uic.edu:8001/v1/models \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# Quick chat test
curl -X POST https://relay.stream.acer.uic.edu:8001/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma4-31b",
       "messages": [{"role": "user", "content": "Say hello."}],
       "max_tokens": 50, "stream": false}' | python3 -m json.tool
```
