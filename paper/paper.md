---
title: 'hpc-as-api: A Domain-Agnostic HTTP Gateway for HPC Functions via Globus Compute and WebSocket Relay'
tags:
  - Python
  - HPC
  - API gateway
  - Globus Compute
  - LLM
  - streaming
  - SLURM
  - scientific computing
  - domain-agnostic
authors:
  - name: Anas Nassar
    orcid: 0009-0008-4225-5745
    corresponding: true
    affiliation: 1
affiliations:
  - name: Advanced Cyberinfrastructure for Education and Research (ACER), University of Illinois Chicago, USA
    index: 1
date: 2026-05-22
bibliography: paper.bib
---

# Summary

HPC clusters run workloads impossible on commodity hardware — 72-billion-parameter AI models,
large-scale climate simulations, molecular dynamics, genome alignment, and more. But they
expose no standard API surface. Each cluster has its own SLURM scripts, SSH tunnels,
authentication systems, and job submission conventions. Developers who want to call an
HPC-hosted function must navigate these heterogeneous interfaces directly, requiring HPC
expertise that application developers do not typically have.

`hpc-as-api` closes this gap by turning any Python function running on an HPC cluster into a
streaming HTTP endpoint. The developer registers a function and its Pydantic input schema;
the framework provides authentication, rate limiting, real-time SSE streaming, and optional
end-to-end encryption — no open ports, no VPN, no firewall changes on the HPC side.

The framework is **domain-agnostic**: any workload that produces incremental output can be
exposed this way — simulation checkpoints, solver residuals, genome alignment progress,
molecular dynamics snapshots, LLM tokens, or any other incrementally produced result.
An OpenAI-compatible LLM preset is included as a built-in application of the framework,
not its primary purpose.

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
            relay.send_token(f"step={i} energy={run_timestep(i, grid_size):.4f}\n")

app = HPCApp(endpoint_id="...", relay_url="wss://relay.example.com") \
    .mount("/simulate", hpc_simulation, SimRequest) \
    .create_app()
```

The HPC cluster requires no public IP, no inbound ports, and no changes to its scheduler
or firewall configuration. Both the HPC producer and the gateway consumer connect
*outbound* to a lightweight WebSocket relay.

# Statement of Need

## The accessibility gap in HPC resources

Institutional HPC clusters provide GPU and CPU resources unavailable on personal hardware
at no marginal cost to researchers. Yet these resources are structurally inaccessible to
the majority of application developers:

- **HPC expertise barrier**: Accessing a cluster requires knowledge of SLURM, SSH key
  management, module systems, batch scripts, and cluster-specific conventions. Application
  developers do not typically have this expertise.
- **No standard interface**: Each HPC center exposes resources differently. Code written
  for one cluster does not transfer to another.
- **No streaming**: Standard HPC job dispatch returns a single result when a job completes.
  Applications that need incremental output — a chat interface seeing the first token, a
  monitoring dashboard watching solver convergence, a real-time data pipeline — are
  incompatible with the batch execution model.

## What hpc-as-api provides

`hpc-as-api` closes this gap by providing:

1. **A domain-agnostic HTTP gateway**: Any Python function that produces incremental output
   can be registered as a streaming endpoint using the `HPCApp` framework. The function
   author writes normal Python; the framework handles HTTP, authentication, streaming, and
   job dispatch.

2. **A standard API surface**: HTTP POST with a bearer token. No HPC knowledge is required
   by callers. The gateway handles Globus authentication, job dispatch, relay connection,
   and SSE streaming internally.

3. **Dual-mode authentication**: Globus Token Auth for direct institutional users (per-user
   attribution via email domain mapping); pre-issued API keys for external service callers
   (e.g., an AWS backend authenticating its own users via Cognito). Both modes coexist on
   the same endpoint.

4. **Real-time streaming**: Integration with `streamrelay` [@nassar2026streamrelay]
   provides sub-second time-to-first-output from HPC via the dual-channel WebSocket relay
   architecture. A batch fallback handles relay-unavailable scenarios.

5. **End-to-end encryption**: Optional AES-256-GCM encryption between the HPC node and
   the gateway consumer, so the relay operator never sees plaintext payloads even if the
   relay VM is compromised.

6. **An OpenAI-compatible LLM preset**: The built-in `create_openai_app()` preset exposes
   vLLM-served models as a standard `/v1/chat/completions` endpoint, compatible with any
   OpenAI client library. This is one built-in application of the domain-agnostic
   framework, not a constraint on its use.

# Design and Implementation

## Architecture

`hpc-as-api` is built around three separation-of-concerns principles:

**Control plane stays unchanged.** Job authentication, dispatch, and scheduling continue
to use Globus Compute [@globuscompute2024]. The gateway adds no new dependencies to the
HPC cluster — it reuses the Globus Compute endpoint already required for job submission.

**Data plane via streamrelay.** Output streaming uses the dual-channel WebSocket relay
architecture from `streamrelay` [@nassar2026streamrelay]. The HPC compute node connects
outbound to the relay as producer; the gateway connects as consumer. Neither side opens
an inbound port.

![Relay architecture: the HPC compute node and gateway consumer both connect outbound to the relay, traversing firewalls without VPN or inbound ports.](../Relay_Architecture.png)

**Configuration via constructor arguments or environment.** All settings (endpoint ID,
relay URL, secrets, auth config) can be supplied as Python arguments to `HPCApp` or
`make_app()`, falling back to environment variables. This makes the framework
equally usable in embedded applications, test suites, and production deployments.

## Key components

**`HPCApp`** (`hpc_as_api/core.py`): The domain-agnostic gateway builder. Developers call
`HPCApp(...).mount(path, fn, schema)` to register one or more HPC functions, then
`create_app()` to get a FastAPI application. Each mounted function becomes a `POST`
endpoint that submits the job via Globus Compute and streams the output as SSE. The
`mount()` method accepts a custom `output_handler` to transform relay messages into
SSE tokens, making the output format fully customizable. Multiple routes can be
mounted on a single `HPCApp` instance, and multiple `HPCApp` instances can be combined
into a single FastAPI application.

**`make_app()` factory** (`hpc_as_api/app.py`): A factory function that returns a fresh,
independent FastAPI instance each time, capturing its configuration in closures. Each call
produces an isolated gateway with its own model registry, auth configuration, and connection
state — two gateways with different endpoints or models can run in the same process without
interfering. The built-in OpenAI-compatible preset (`create_openai_app()`) is implemented
as a thin wrapper over this factory.

**`GlobusComputeClient`** (`hpc_as_api/compute.py`): Manages the persistent Globus Compute
Executor (AMQP connection reuse saves 1–2 s per request), handles authentication checks
and credential reload, manages payload size (stripping images from older conversation
history to stay under Globus's 10 MB limit), and submits both batch and streaming jobs.
Remote functions are defined as source strings and compiled via `exec()` to produce clean
bytecode — a workaround for PyInstaller-bundled environments where standard serialization
fails with missing internal modules.

**`authenticate` / `AuthConfig`** (`hpc_as_api/auth.py`): FastAPI dependency that
validates every request. Accepts either a Globus access token (introspected against Globus
Auth's public endpoint, email domain checked) or a pre-issued API key (constant-time
comparison). Per-caller rate limiting uses a sliding window (default 20 requests/60 s).
`AuthConfig` allows all authentication settings to be supplied as Python arguments,
enabling programmatic configuration without environment variables.

**`decrypt_message`** (`hpc_as_api/crypto.py`): AES-256-GCM decryption for end-to-end
encrypted relay payloads. The encryption key is set as an environment variable on the
Globus Compute endpoint (`worker_init`) and never transmitted as a task argument, so it
does not travel over Globus's AMQP channel.

## Deployment

The recommended deployment places the gateway on the same VM as the `streamrelay` relay
server (lower latency, single TLS certificate, one security perimeter). Caddy handles
TLS automatically. The full deployment requires:

- One small public VM (e.g., AWS t3.micro) running `streamrelay` and `hpc-as-api`
- One Globus Compute endpoint on the HPC cluster (outbound AMQP only)
- One pre-issued API key per external calling service

Complete deployment instructions and a threat model covering all five attack surfaces
(Globus AMQP, relay channel, proxy endpoint, API key storage, TLS termination) are
provided in `docs/deployment.md`.

# Performance

Deployed in the STREAM system [@nassar2026stream] at the University of Illinois Chicago
against a Qwen 2.5 72B AWQ model on an NVIDIA H100 NVL GPU on the Lakeshore HPC cluster
(50-run medians):

| Metric | Value |
|---|---|
| Median time-to-first-token (relay streaming, steady state) | **0.60 s** ± 0.20 s |
| Median end-to-end latency (batch, warm cache) | **11.17 s** ± 2.0 s |
| Throughput | ~25 tok/s (vLLM with `--enforce-eager` on CUDA 12.4) |

The 0.60 s TTFT includes Globus Compute authentication and job dispatch on a dedicated
single-user endpoint. The relay itself adds no measurable per-message overhead — it is a
memory-copy forwarder with no parsing on message content. These numbers are for the LLM
preset; the framework overhead is identical for any other HPC function type.

# Acknowledgements

`hpc-as-api` was developed as part of the STREAM project at the Advanced
Cyberinfrastructure for Education and Research (ACER) group at the University of Illinois
Chicago. We thank Marius Horga (Assistant Director of Advanced Platforms for Research,
ACER) for support of this work, and the UIC ACER team for providing and maintaining the
Lakeshore HPC cluster used in development and evaluation.

# AI Usage Disclosure

Claude Code (Anthropic, claude-sonnet-4-6) was used to assist with: code generation and
refactoring (`GlobusComputeClient`, FastAPI routes, authentication module, encryption
module, test scaffolding), documentation drafting (README, deployment guide, CONTRIBUTING),
and paper text editing. All architectural decisions — the domain-agnostic `HPCApp`
framework design, `make_app()` factory pattern, configuration-injectable client design,
dual-mode authentication architecture, end-to-end encryption key separation (endpoint env
var vs. task argument), embeddable FastAPI router pattern, and multimodal payload
management strategy — are the author's original work. All AI-assisted outputs were
reviewed, validated, and revised by the author. The author takes full responsibility for
the accuracy, correctness, and integrity of all submitted materials.

# References
