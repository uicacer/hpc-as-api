# Quick Guide: Add a New Model to HPC-as-API

**Prerequisites:** SSH access to Lakeshore, HuggingFace account.

---

## Who can do what

| | Anas / Steve | Collaborator |
|---|---|---|
| GPU partition | `batch_gpuapi` / `ts_acer` — up to 2× A100 80GB (ga-002) | `batch_gpu2` — 1× H100 NVL 95.8GB (ghi2-002) |
| Tensor parallelism | TP2 (2 GPUs) | TP1 (1 GPU only) |
| Max BF16 model size | ~62B params (Gemma 4 31B fills both A100s) | ~80B params in theory; ~34B comfortably |
| Register model in gateway | Yes (owns proxy-env on relay VM) | No — contact Anas with alias + port |

> Collaborators use `ghi2-002` (H100 NVL) — the same cluster, same vLLM container, same gateway. Only the SLURM partition and number of GPUs differ.

**Collaborator onboarding path:**
1. Get a Lakeshore account (contact UIC ACER)
2. Follow Steps 1–4 below using `batch_gpu2` and `ghi2-002` (1 GPU)
3. Send Anas the model alias and port → he registers it in the gateway (Step 6)
4. Anas sends back an API key — use the gateway exactly like any other model

**Storage layout (quick reminder):**
- Model weights → your own project space (e.g. `/projects/acer_hpc_admin/<YOUR_NETID>/huggingface/`)
- vLLM container → you can use Anas's container at `/projects/acer_hpc_admin/nassar/containers/vllm-0.23.0` (world-readable) or pull your own
- Anas's containers and weights are at `/projects/acer_hpc_admin/nassar/` — readable but not writable

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

## Step 4 — Start vLLM

Choose the right template based on your GPU allocation:

### If you are Anas/Steve (2× A100, ga-002, batch_gpuapi)

See the full script in `adding-a-new-model.md` Part 3. Use `--gres=gpu:2`, `--tensor-parallel-size 2`, `--partition=batch_gpuapi`, `--account=ts_acer`, `--nodelist=ga-002`.

### If you are a collaborator (1× H100 NVL, ghi2-002, batch_gpu2)

Create `~/scripts/vllm-MODELNAME.sh` using the template below.
Change `MODEL`, `PORT`, `--served-model-name`, and `--tool-call-parser` / `--chat-template` for your model family.

```bash
#!/bin/bash
#SBATCH --job-name=MODELNAME           # name shown in squeue
#SBATCH --partition=batch_gpu2         # collaborator GPU partition — ghi2-002
#SBATCH --nodelist=ghi2-002            # pin to the H100 node
#SBATCH --gres=gpu:1                   # 1× H100 NVL 95.8GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/logs/%x-%j.log

MODEL="your-org/your-model"            # ← HuggingFace model ID
PORT=8002                              # ← use 8002+ (8001 is taken by Anas's model)
PROJECT_DIR="/projects/acer_hpc_admin/nassar"      # reuse Anas's container and weights cache
YOUR_PROJECT="/projects/acer_hpc_admin/YOUR_NETID" # ← your own project space for weights
CONTAINER="${PROJECT_DIR}/containers/vllm-0.23.0"

echo "vLLM: ${MODEL} | Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Started: $(date)"

source /etc/profile
module load apptainer

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${YOUR_PROJECT}/huggingface"
export HF_TOKEN=$(cat "${YOUR_PROJECT}/.hf_token")

apptainer exec --nv --no-home --env PYTHONNOUSERSITE=1 "${CONTAINER}" \
    vllm serve "${MODEL}" \
    --host 127.0.0.1 \             # loopback only — security requirement
    --port "${PORT}" \
    --tensor-parallel-size 1 \     # 1 GPU only — collaborator allocation
    --max-model-len 32768 \        # 32K context fits in 1× H100 for most models
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --dtype auto \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --served-model-name YOUR-ALIAS # ← the alias clients use in "model": "YOUR-ALIAS"

echo "Service stopped: $(date)"
```

**Key differences from Anas's script:**
- `batch_gpu2` + `ghi2-002` instead of `batch_gpuapi` + `ga-002` — no `--account=ts_acer` needed
- `--gres=gpu:1` and `--tensor-parallel-size 1` — 1 GPU only
- Port `8002+` — port 8001 is already used by Gemma 4 on that node
- `--max-model-len 32768` — conservative default; H100 NVL has 95.8GB so you can go higher for smaller models

```bash
sbatch ~/scripts/vllm-MODELNAME.sh
squeue -u YOUR_NETID                               # confirm STATE=RUNNING on ghi2-002
tail -f ~/logs/MODELNAME-JOBID.log
# Wait for: "Application startup complete."        # first run ~15min, restarts ~5min
```

---

## Step 5 — Contact Anas ✉️

Once you see `Application startup complete.` in the log, send Anas:

1. The **model alias** you used in `--served-model-name` (e.g. `my-model-7b`)
2. The **port** (e.g. `8002`)
3. The **node** it's running on (`ghi2-002` for collaborators)

**Anas will do the rest** — register the model in the gateway and issue you an API key. This takes about 2 minutes on his side.

> **Note:** Your SLURM job must stay running for the model to be accessible through the gateway. If the job ends (walltime, OOM, manual cancel), the model disappears from the gateway until you resubmit. Keep an eye on `squeue -u YOUR_NETID`.

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
