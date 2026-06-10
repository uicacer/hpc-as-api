"""
hpc-as-api — Domain-agnostic HTTP gateway for any HPC function via Globus Compute + WebSocket relay.

Turn any Python function running on an HPC cluster into a streaming HTTP endpoint.
Register your function, define its input schema with Pydantic, and get a production-ready
REST API with authentication, rate limiting, and live SSE streaming — no open ports,
no VPN, no firewall changes required.

Any workload that produces incremental output works: simulation checkpoints, solver
residuals, genome alignment progress, LLM tokens, molecular dynamics snapshots, etc.

Two usage styles:

1. Domain-agnostic (primary interface):
   Stream any HPC function output through a WebSocket relay::

       from hpc_as_api.core import HPCApp
       from pydantic import BaseModel

       class Request(BaseModel):
           steps: int = 100

       def my_fn(steps, relay_url, channel_id, relay_secret=""):
           from streamrelay import RelayProducer
           with RelayProducer(relay_url, channel_id, relay_secret=relay_secret) as r:
               for i in range(steps):
                   r.send_token(f"step {i}\\n")

       app = HPCApp(endpoint_id="...", relay_url="wss://...").mount("/run", my_fn, Request).create_app()

2. OpenAI-compatible LLM preset (built-in application of the framework):
   Drop-in OpenAI-compatible gateway for vLLM on HPC::

       from hpc_as_api.presets.openai import create_openai_app
       app = create_openai_app(endpoint_id="...", models={...}, relay_url="wss://...")

3. Low-level Globus Compute client::

       from hpc_as_api.compute import GlobusComputeClient
       client = GlobusComputeClient(endpoint_id="...", models={...})
       result = await client.submit_inference(messages=[...], model="qwen25-vl-72b")
"""

from hpc_as_api.auth import AuthConfig, Authenticator
from hpc_as_api.utils import (
    count_images,
    extract_text_content,
    has_images,
    strip_old_images,
)

try:
    from hpc_as_api.core import HPCApp

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

# GlobusComputeClient depends on globus_compute_sdk (optional extra [globus]).
try:
    from hpc_as_api.compute import GlobusComputeClient

    _GLOBUS_AVAILABLE = True
except ImportError:
    _GLOBUS_AVAILABLE = False

__version__ = "0.3.4"
__all__ = [
    "AuthConfig",
    "Authenticator",
    "HPCApp",
    "GlobusComputeClient",
    "extract_text_content",
    "has_images",
    "count_images",
    "strip_old_images",
]
