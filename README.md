# hpc-as-api

[![PyPI](https://img.shields.io/pypi/v/hpc-as-api)](https://pypi.org/project/hpc-as-api/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/uicacer/hpc-as-api/blob/main/LICENSE)

**HTTP gateway for any HPC function — real-time streaming from any HPC workload.**

`hpc-as-api` turns any Python function running on an HPC cluster into a streaming HTTP endpoint. Register your function, define its input schema with Pydantic, and get a production-ready REST API with authentication, rate limiting, and live SSE streaming — no open ports, no VPN, no firewall changes on the HPC side.

```python
from hpc_as_api.core import HPCApp
from pydantic import BaseModel

class SimRequest(BaseModel):
    steps: int = 1000
    grid_size: int = 100

def hpc_simulation(steps, grid_size, relay_url, channel_id, relay_secret=""):
    from streamrelay import RelayProducer
    with RelayProducer(relay_url, channel_id, relay_secret=relay_secret) as relay:
        for i in range(steps):
            result = run_timestep(i, grid_size)
            relay.send_token(f"step={i} energy={result:.4f}\n")

app = HPCApp(endpoint_id="...", relay_url="wss://relay.example.com") \
    .mount("/simulate", hpc_simulation, SimRequest) \
    .create_app()
```

Any output produced incrementally on the HPC side arrives in real time: simulation checkpoints, solver residuals, genome alignment progress, molecular dynamics snapshots, LLM tokens — anything.

## Why

HPC clusters run workloads impossible on commodity hardware — 72B+ parameter models, climate simulations, molecular dynamics at scale. But they expose no standard API. Each cluster has its own SLURM scripts, SSH tunnels, authentication systems, and job submission conventions.

`hpc-as-api` provides a uniform HTTP interface over any HPC function using [Globus Compute](https://www.globus.org/compute) for authentication and job dispatch and [streamrelay](https://github.com/uicacer/streamrelay) for real-time output streaming. Callers send a POST request; the framework handles everything else.

## Architecture

```
Your Application / HTTP Client
        │  POST /your-endpoint  (any input schema)
        ▼
  hpc-as-api (FastAPI)
        │  Globus Compute (AMQP — no HPC firewall holes)
        ▼
  HPC Cluster (SLURM / PBS / …)
        │  your function runs; output flows via streamrelay
        ▼
  GPU / CPU Compute Node
        │  tokens / results / checkpoints via WebSocket relay
        ▼
  hpc-as-api → SSE stream → Your Application
```

Key design points:
- **No open ports on HPC**: Globus Compute is outbound-only from the cluster
- **Real-time streaming**: Any incremental output arrives as SSE via [streamrelay](https://github.com/uicacer/streamrelay)
- **E2E encryption**: Optional AES-256-GCM encryption — relay sees only ciphertext
- **Domain-agnostic**: Register any Python function; not limited to LLMs

## Installation

```bash
# Base package (no Globus SDK)
pip install hpc-as-api

# With Globus Compute support
pip install "hpc-as-api[globus]"
```

## Quickstart: Domain-agnostic gateway

Register any HPC function and stream its output:

```python
from hpc_as_api.core import HPCApp
from pydantic import BaseModel

class RunRequest(BaseModel):
    steps: int = 1000
    param: float = 0.5

def my_hpc_function(steps, param, relay_url, channel_id, relay_secret=""):
    from streamrelay import RelayProducer
    with RelayProducer(relay_url, channel_id, relay_secret=relay_secret) as relay:
        for i in range(steps):
            relay.send_token(f"step={i} value={compute(i, param)}\n")

gateway = HPCApp(
    endpoint_id="your-globus-endpoint-uuid",
    relay_url="wss://relay.example.com",
    relay_secret="your-relay-secret",
)
gateway.mount("/run", my_hpc_function, RunRequest)
app = gateway.create_app()
```

Run with:
```bash
uvicorn mymodule:app --host 0.0.0.0 --port 8001
```

Clients stream the output in real time:
```bash
export API_KEY="sk-your-key"
curl -N -X POST http://localhost:8001/run \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"steps": 500, "param": 0.7}'
```

## Built-in preset: OpenAI-compatible LLM gateway

For vLLM-served language models, the OpenAI preset provides a drop-in
`/v1/chat/completions` endpoint compatible with any OpenAI client:

```python
from hpc_as_api.presets.openai import create_openai_app

app = create_openai_app(
    endpoint_id="8d978809-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    models={
        "gemma4-31b": {
            "hf_name": "gemma4-31b",
            "url": "http://127.0.0.1:8001",
            "context_reserve_output": 8192,
        }
    },
    relay_url="wss://relay.example.com",
    relay_secret="your-relay-secret",
)
```

Or run as a service from environment variables:

```bash
export GLOBUS_COMPUTE_ENDPOINT_ID="your-endpoint-uuid"
export HPC_MODELS='{"gemma4-31b": {"hf_name": "gemma4-31b", "url": "http://127.0.0.1:8001", "context_reserve_output": 8192}}'
export RELAY_URL="wss://relay.example.com"
export RELAY_SECRET="your-relay-secret"
export PROXY_API_KEY_MYSERVICE="sk-your-key"

uvicorn hpc_as_api.app:app --host 127.0.0.1 --port 8002
```

Any OpenAI client works without modification:
```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key=os.environ["PROXY_API_KEY_MYSERVICE"],
)
response = client.chat.completions.create(
    model="gemma4-31b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Multiple independent gateways

`create_openai_app()` returns a fresh, independent instance each time — safe to
run multiple gateways with different configurations in the same process:

```python
from hpc_as_api.presets.openai import create_openai_app

llm_a = create_openai_app(endpoint_id="endpoint-a", models={...}, relay_url="wss://relay.example.com")
llm_b = create_openai_app(endpoint_id="endpoint-b", models={...}, relay_url="wss://relay.example.com")
```

## Programmatic auth configuration

```python
from hpc_as_api import AuthConfig
from hpc_as_api.core import HPCApp

gateway = HPCApp(
    endpoint_id="...",
    relay_url="wss://relay.example.com",
    auth=AuthConfig(
        globus_client_id="your-client-id",
        globus_client_secret="your-client-secret",
        allowed_domains=["university.edu"],
        api_keys={"my-service": "sk-xxxx"},
        rate_limit_requests=10000,
        rate_limit_window=60,
    ),
)
```

## Configuration reference

### HPCApp / create_openai_app()

| Argument | Env var fallback | Description |
|---|---|---|
| `endpoint_id` | `GLOBUS_COMPUTE_ENDPOINT_ID` | Globus endpoint UUID for the HPC cluster |
| `relay_url` | `RELAY_URL` | WebSocket relay URL for streaming |
| `relay_secret` | `RELAY_SECRET` | Shared secret for relay auth |
| `relay_encryption_key` | `RELAY_ENCRYPTION_KEY` | AES-256 hex key for E2E encryption |
| `auth` | — | `AuthConfig` or `Authenticator` instance |

### OpenAI preset environment variables

| Variable | Default | Description |
|---|---|---|
| `HPC_MODELS` | `{}` | JSON dict: model alias → `{"hf_name", "url", "context_reserve_output"}` |
| `PROXY_API_KEY_<NAME>` | — | API key for service `<NAME>` — any number of keys, any suffix |
| `PROXY_RATE_LIMIT_REQUESTS` | `10000` | Global max requests per window (per-caller sliding window) |
| `PROXY_RATE_LIMIT_WINDOW` | `60` | Window size in seconds |
| `PROXY_RATE_LIMIT_REQUESTS_<NAME>` | — | Per-key override; `<NAME>` must match the suffix in `PROXY_API_KEY_<NAME>` (lowercased) |
| `USE_GLOBUS_COMPUTE` | `true` | `false` to route directly to a vLLM URL without Globus |

### HPC_MODELS schema

```json
{
  "my-model-alias": {
    "hf_name": "my-model-alias",
    "url": "http://127.0.0.1:8001",
    "context_reserve_output": 8192
  }
}
```

`hf_name` must exactly match `--served-model-name` in the vLLM SLURM script.
`url` is where vLLM is reachable from the Globus Compute worker (usually `http://127.0.0.1:PORT` when workers are co-located).

## Authentication

Two auth modes coexist automatically, configured via `AuthConfig` or environment variables:

**Mode A — Globus token** (for institutional users)
The caller presents a Globus access token validated via introspection. The job runs under the caller's Globus identity. Set `GLOBUS_CLIENT_ID`, `GLOBUS_CLIENT_SECRET`, and optionally `PROXY_ALLOWED_DOMAINS`.

**Mode B — API key** (for service-to-service callers)
The caller presents a static key. Set one or more `PROXY_API_KEY_<NAME>=<value>` env vars. The `<NAME>` suffix (lowercased) identifies the caller in logs and rate-limit overrides.

```bash
# Example: two keys, different rate limits
PROXY_API_KEY_CLASS=sk-class-key
PROXY_API_KEY_DEMO=sk-demo-key
PROXY_RATE_LIMIT_REQUESTS=10000      # class key: 10k req/min
PROXY_RATE_LIMIT_REQUESTS_DEMO=20    # demo key: 20 req/min
```

> **Scaling to per-student keys (future work):** For classroom deployments with hundreds of students, the planned approach is a `PROXY_KEYS_FILE` pointing at a JSON file of `{"student_name": "sk-..."}` pairs loaded and merged with env-var keys at startup. A bulk generation script produces all keys at once; students receive theirs via Canvas. No OAuth, no login, no extra infrastructure. Not yet implemented.

## Development

```bash
git clone https://github.com/uicacer/hpc-as-api
cd hpc-as-api
uv sync --extra dev

# Install pre-commit hooks (ruff, mypy, gitleaks, hygiene checks)
pre-commit install

uv run pytest
```

## Deployment

See [docs/deployment.md](docs/deployment.md) for the full sysadmin guide (systemd, Caddy TLS, Globus endpoint, secrets management).

See [docs/tutorial.ipynb](docs/tutorial.ipynb) for a zero-to-hero walkthrough from relay primitives through production deployment.

## Related

- [streamrelay](https://github.com/uicacer/streamrelay) — WebSocket relay for real-time output streaming from Globus Compute
- [STREAM](https://github.com/uicacer/STREAM) — Full tiered LLM routing system that uses hpc-as-api

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

If you use hpc-as-api in research, please cite:

```bibtex
@software{nassar2025hpcgateway,
  author = {Nassar, Anas},
  title  = {hpc-as-api: HTTP gateway for any HPC function via Globus Compute and WebSocket relay},
  year   = {2025},
  url    = {https://github.com/uicacer/hpc-as-api}
}
```
