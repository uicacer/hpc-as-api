# Deployment Guide — hpc-as-api

End-to-end instructions for a sysadmin deploying `hpc-as-api` from scratch on a
relay VM, connecting it to a Globus Compute endpoint on an HPC cluster.

---

## What you need

| Component | Notes |
|-----------|-------|
| A small public VM | AWS t3.small, DigitalOcean Droplet, or equivalent; Ubuntu 22.04+ |
| A domain name | e.g. `hpc-api.institution.edu` — points to the VM's public IP |
| Python 3.11+ on the VM | `python3 --version` |
| A Globus Compute endpoint | Already running on your HPC cluster |
| Outbound HTTPS from VM | For Globus Auth introspection (port 443) |

---

## Architecture

```
Client (curl / OpenAI SDK / notebook)
    │  HTTPS :8001 (TLS via Caddy)
    ▼
relay VM  (your-domain.edu)
    │  localhost:8002
    ▼
hpc-as-api  (stream-proxy systemd service)
    │  Globus Compute AMQP — outbound only, no HPC firewall holes
    ▼
HPC Cluster (Globus Compute endpoint)
    │  spawns SLURM workers
    ▼
vLLM on GPU node (127.0.0.1:8001, loopback only)
    │  streaming tokens via WebSocket relay
    ▼
streamrelay (same VM)  →  SSE stream back to client
```

No inbound firewall rules needed on the HPC side — all traffic flows outbound
from the cluster to Globus Compute's managed AMQP and to the relay.

---

## Step 1 — Install on the relay VM

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-pip python3.12-venv caddy

# Install hpc-as-api with Globus support
python3.12 -m pip install "hpc-as-api[globus]"

# Install streamrelay (WebSocket relay for token streaming)
python3.12 -m pip install streamrelay
```

Verify:
```bash
python3.12 -c "import hpc_as_api; print(hpc_as_api.__version__)"
```

---

## Step 2 — Generate secrets

Every value below should be freshly generated — never reuse across deployments.

```bash
# Relay shared secret (auth between proxy and relay)
python3 -c "import secrets; print(secrets.token_hex(32))"

# End-to-end encryption key (set on both relay VM and HPC endpoint worker_init)
python3 -c "import secrets; print(secrets.token_hex(32))"

# API key for a calling service
python3 -c "import secrets; print('sk-' + secrets.token_hex(32))"
```

All keys are 64-character hex strings. Hex avoids `+`, `/`, `=` characters that
cause parsing problems in `.env` files, shell exports, and YAML `worker_init`.

---

## Step 3 — Configure environment

Create `/home/ubuntu/proxy-env` (chmod 600 — **this file contains secrets**):

```bash
sudo -u ubuntu bash -c "cat > /home/ubuntu/proxy-env << 'EOF'
# ── Globus Compute ────────────────────────────────────────────────────────────
GLOBUS_COMPUTE_ENDPOINT_ID=your-endpoint-uuid-here

# ── Model registry ────────────────────────────────────────────────────────────
# One JSON object per model. hf_name must match --served-model-name in your
# SLURM script. url is where vLLM is reachable FROM the Globus worker.
HPC_MODELS={"gemma4-31b": {"hf_name": "gemma4-31b", "url": "http://127.0.0.1:8001", "context_reserve_output": 8192}}

# ── Relay ─────────────────────────────────────────────────────────────────────
RELAY_URL=wss://your-domain.edu
RELAY_SECRET=<output of: python3 -c "import secrets; print(secrets.token_hex(32))">
RELAY_ENCRYPTION_KEY=<output of: python3 -c "import secrets; print(secrets.token_hex(32))">

# ── API keys ──────────────────────────────────────────────────────────────────
# Any PROXY_API_KEY_<NAME>=<value> pair becomes an accepted Bearer token.
# The <NAME> suffix (lowercased) is used in logs and rate-limit overrides.
PROXY_API_KEY_AMPLIFY=sk-your-amplify-key-here
PROXY_API_KEY_DEMO=sk-your-demo-key-here

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Global default: max requests per window (sliding, per-caller).
# Set high so shared classroom keys aren't throttled under normal load.
PROXY_RATE_LIMIT_REQUESTS=10000
PROXY_RATE_LIMIT_WINDOW=60

# Per-key overrides (PROXY_RATE_LIMIT_REQUESTS_<NAME>, same suffix as API key).
# Use these to throttle specific keys more tightly without affecting others.
# Example: limit the demo key to 20 req/min to prevent accidental abuse.
PROXY_RATE_LIMIT_REQUESTS_DEMO=20
# PROXY_RATE_LIMIT_REQUESTS_AMPLIFY=500   # uncomment to cap the Amplify key

# ── Proxy bind ────────────────────────────────────────────────────────────────
HPC_PROXY_HOST=127.0.0.1
HPC_PROXY_PORT=8002
EOF
chmod 600 /home/ubuntu/proxy-env"
```

> **Security**: `RELAY_ENCRYPTION_KEY` must also be set in your Globus Compute
> endpoint's `worker_init` block (not as a task argument) so it never travels
> over Globus Compute's AMQP channel.

---

## Step 4 — systemd service for hpc-as-api

Create `/etc/systemd/system/stream-proxy.service`:

```ini
[Unit]
Description=HPC-as-API Gateway (stream-proxy)
After=network.target

[Service]
User=ubuntu
EnvironmentFile=/home/ubuntu/proxy-env
ExecStart=/usr/bin/python3.12 -m uvicorn hpc_as_api.app:app \
    --host 127.0.0.1 --port 8002 --log-level info
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stream-proxy
sudo systemctl status stream-proxy   # must show "active (running)"
```

Logs:
```bash
sudo journalctl -u stream-proxy -f
```

---

## Step 5 — systemd service for streamrelay

```bash
python3.12 -m pip install streamrelay
```

Create `/etc/systemd/system/streamrelay.service`:

```ini
[Unit]
Description=WebSocket relay for token streaming
After=network.target

[Service]
User=ubuntu
Environment=RELAY_SECRET=<same value as in proxy-env>
Environment=RELAY_HOST=0.0.0.0
Environment=RELAY_PORT=8765
ExecStart=/home/ubuntu/.local/bin/streamrelay
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now streamrelay
```

---

## Step 6 — Caddy (TLS + reverse proxy)

Edit `/etc/caddy/Caddyfile`:

```caddyfile
# WebSocket relay — used for streaming tokens
your-domain.edu {
    reverse_proxy localhost:8765
}

# API gateway — OpenAI-compatible endpoint
your-domain.edu:8001 {
    reverse_proxy localhost:8002
}
```

```bash
sudo systemctl restart caddy
sudo systemctl status caddy
```

Open inbound TCP **443** and **8001** in your cloud provider's security group.

---

## Step 7 — Verify

```bash
# Set your API key in the shell first
export API_KEY="$(grep PROXY_API_KEY_AMPLIFY /home/ubuntu/proxy-env | cut -d= -f2)"

# Health check
curl https://your-domain.edu:8001/health
# Expected: {"status":"healthy","service":"HPC Gateway","models":["gemma4-31b"],...}

# List models
curl https://your-domain.edu:8001/v1/models \
  -H "Authorization: Bearer $API_KEY"

# Non-streaming chat
curl -X POST https://your-domain.edu:8001/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-31b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50,"stream":false}'
```

---

## Configuration reference

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GLOBUS_COMPUTE_ENDPOINT_ID` | — | UUID of the Globus Compute endpoint on the HPC cluster |
| `HPC_MODELS` | `{}` | JSON dict: model alias → `{"hf_name", "url", "context_reserve_output"}` |
| `RELAY_URL` | — | WebSocket relay URL, e.g. `wss://your-domain.edu` |
| `RELAY_SECRET` | — | Shared secret for relay auth |
| `RELAY_ENCRYPTION_KEY` | — | AES-256 hex key for E2E encryption |
| `PROXY_API_KEY_<NAME>` | — | API key for service `<NAME>` (any suffix, any number of keys) |
| `PROXY_RATE_LIMIT_REQUESTS` | `10000` | Global max requests per window (per-caller sliding window) |
| `PROXY_RATE_LIMIT_WINDOW` | `60` | Window size in seconds |
| `PROXY_RATE_LIMIT_REQUESTS_<NAME>` | — | Per-key override — `<NAME>` matches the suffix in `PROXY_API_KEY_<NAME>` (lowercased) |
| `PROXY_ALLOWED_DOMAINS` | — | Comma-separated email domains allowed for Globus token auth |
| `HPC_PROXY_HOST` | `0.0.0.0` | Host to bind to (use `127.0.0.1` behind a reverse proxy) |
| `HPC_PROXY_PORT` | `8001` | Port to listen on |

### Scaling to per-student keys (future work)

The current `PROXY_API_KEY_<NAME>` pattern works well for a handful of service accounts but doesn't scale to 300 students without manual proxy-env edits. The planned approach is a `PROXY_KEYS_FILE` pointing to a JSON file of `{"student_name": "sk-..."}` pairs that the proxy loads and merges with env-var keys at startup:

```json
{"alice_smith": "sk-class-a1b2c3...", "bob_jones": "sk-class-d4e5f6..."}
```

A bulk generation script would produce all keys at once; distribution happens via Canvas. Adding/revoking a student = edit the JSON file + hit `/reload-keys` (no restart needed). No OAuth, no login, no extra infrastructure — students use their key exactly like any API key. This is not yet implemented.

### Rate limiting examples

**Shared classroom key, tight demo key:**
```bash
PROXY_API_KEY_CLASS=sk-class-key
PROXY_API_KEY_DEMO=sk-demo-key
PROXY_RATE_LIMIT_REQUESTS=10000    # class key: 10k req/min (essentially unlimited)
PROXY_RATE_LIMIT_REQUESTS_DEMO=20  # demo key: 20 req/min
```

**Cap a specific integration:**
```bash
PROXY_RATE_LIMIT_REQUESTS_AMPLIFY=500   # Amplify: 500 req/min
```

---

## Adding a new model

See [adding-a-new-model.md](adding-a-new-model.md) for the full guide.
The one-line summary:

1. Start vLLM on the HPC node with `--served-model-name your-alias`
2. Add `"your-alias": {"hf_name": "your-alias", "url": "...", "context_reserve_output": 4096}` to `HPC_MODELS` in `proxy-env`
3. `sudo systemctl restart stream-proxy`

No code changes, no reinstall, no restart of Globus endpoint.

---

## Upgrading hpc-as-api

```bash
python3.12 -m pip install --upgrade "hpc-as-api[globus]"
sudo systemctl restart stream-proxy
```

Or pin to a specific version:
```bash
python3.12 -m pip install "hpc-as-api[globus]==0.4.0"
```

**If a local `hpc_as_api/` directory exists in the home directory**, pip's installed package will be shadowed by it — `python3.12 -c "import hpc_as_api; print(hpc_as_api.__version__)"` will show the old version even after upgrading. Fix by syncing the source tree into the local directory:

```bash
# From your dev machine:
rsync -av --delete /path/to/hpc-as-api/hpc_as_api/ stream-relay:/home/ubuntu/hpc_as_api/
ssh stream-relay "sudo systemctl restart stream-proxy"
```

After the rsync, verify the version matches what you expect:
```bash
ssh stream-relay "python3.12 -c 'import hpc_as_api; print(hpc_as_api.__version__)'"
```

---

## Threat model

| Attack vector | Defense |
|---|---|
| Eavesdropping on caller → proxy | TLS (Caddy, auto Let's Encrypt) |
| Relay operator reading token payloads | AES-256-GCM end-to-end encryption |
| Unauthorized relay connections | Shared secret (post-handshake, not in URL) |
| Unauthorized proxy access | Bearer token (Globus) or API key auth |
| API key exposure in logs | Keys validated in-memory; only the SHA-256 hash is logged |
| Runaway scripts / abuse | Per-caller sliding-window rate limiter (configurable per key) |
| vLLM exposed to other cluster users | `--host 127.0.0.1` — loopback only; only Globus workers on the same node can reach it |
| Globus credentials leaving proxy VM | Stored only in `~/.globus_compute/storage.db`; never transmitted |
