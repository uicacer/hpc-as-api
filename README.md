# hpc-as-api

[![PyPI](https://img.shields.io/pypi/v/hpc-as-api)](https://pypi.org/project/hpc-as-api/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/uicacer/hpc-as-api/blob/main/LICENSE)
[![Tests](https://github.com/uicacer/hpc-as-api/actions/workflows/tests.yml/badge.svg)](https://github.com/uicacer/hpc-as-api/actions)

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

![Relay architecture: the HPC compute node and gateway consumer both connect outbound to the WebSocket relay, traversing firewalls without VPN or inbound ports.](Relay_Architecture.png)

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
curl -X POST http://localhost:8001/run \
  -H "Authorization: Bearer <token>" \
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
        "qwen25-vl-72b": {
            "hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
            "url": "http://ghi2-002:8000",
            "context_reserve_output": 4096,
        }
    },
    relay_url="wss://relay.example.com",
    relay_secret="your-relay-secret",
)
```

Or run as a service from environment variables:

```bash
export GLOBUS_COMPUTE_ENDPOINT_ID="your-endpoint-uuid"
export HPC_MODELS='{"qwen25-vl-72b": {"hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "url": "http://ghi2-002:8000", "context_reserve_output": 4096}}'
export RELAY_URL="wss://relay.example.com"
export RELAY_SECRET="your-relay-secret"

uvicorn hpc_as_api.app:app --host 0.0.0.0 --port 8001
```

Any OpenAI client works without modification:
```python
import openai
client = openai.OpenAI(base_url="http://localhost:8001/v1", api_key="sk-xxxx")
response = client.chat.completions.create(model="qwen25-vl-72b", messages=[...], stream=True)
```

## Multiple independent gateways

`make_app()` returns a fresh, independent instance each time — safe to use
multiple gateways with different configurations in the same process:

```python
from hpc_as_api.app import make_app

sim_app = make_app(endpoint_id="endpoint-a", relay_url="wss://relay.example.com", models={...})
llm_app = make_app(endpoint_id="endpoint-b", relay_url="wss://relay.example.com", models={...})
```

## Embed in an existing FastAPI app

```python
from fastapi import FastAPI
from hpc_as_api.app import router

app = FastAPI()
app.include_router(router, prefix="/hpc")
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
        rate_limit_requests=20,
        rate_limit_window=60,
    ),
)
```

## Configuration reference

### HPCApp / make_app()

| Argument | Env var fallback | Description |
|---|---|---|
| `endpoint_id` | `GLOBUS_COMPUTE_ENDPOINT_ID` | Globus endpoint UUID for the HPC cluster |
| `relay_url` | `RELAY_URL` | WebSocket relay URL for streaming |
| `relay_secret` | `RELAY_SECRET` | Shared secret for relay auth |
| `relay_encryption_key` | `RELAY_ENCRYPTION_KEY` | AES-256 hex key for E2E encryption |
| `auth` | — | `AuthConfig` or `Authenticator` instance |

### OpenAI preset additional settings

| Variable | Default | Description |
|---|---|---|
| `HPC_MODELS` | `{}` | JSON dict: model name → HPC config |
| `USE_GLOBUS_COMPUTE` | `true` | `false` to route directly via vLLM URL |
| `LAKESHORE_VLLM_ENDPOINT` | `http://localhost:8000` | Direct vLLM URL (non-Globus mode) |
| `PROXY_RATE_LIMIT_REQUESTS` | `10000` | Global max requests per window (per-caller) |
| `PROXY_RATE_LIMIT_WINDOW` | `60` | Window size in seconds |
| `PROXY_RATE_LIMIT_REQUESTS_<NAME>` | — | Per-key override; `<NAME>` matches `PROXY_API_KEY_<NAME>` suffix |

### HPC_MODELS schema (LLM preset)

```json
{
  "my-model-name": {
    "hf_name": "org/ModelName",
    "url": "http://compute-node:8000",
    "context_reserve_output": 4096
  }
}
```

## Authentication

Two auth modes, configurable via `AuthConfig` or environment variables:

- **Globus token**: Bearer token from Globus Auth, validated via introspection; email domain filtering supported
- **API key**: Static key from `HPC_API_KEYS` env var (comma-separated `name:key` pairs)

Both modes coexist on the same endpoint.

## Development

```bash
git clone https://github.com/uicacer/hpc-as-api
cd hpc-as-api
uv sync --extra dev
uv run pytest
```

## Related

- [streamrelay](https://github.com/uicacer/streamrelay) — WebSocket relay for real-time output streaming from Globus Compute
- [STREAM](https://github.com/uicacer/stream) — Full tiered LLM routing system that uses hpc-as-api

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
