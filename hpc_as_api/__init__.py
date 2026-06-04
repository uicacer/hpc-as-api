"""
hpc-as-api — HTTP gateway for HPC functions via Globus Compute + WebSocket relay.

Two usage styles:

1. Domain-agnostic (new in v0.2.0):
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

2. OpenAI-compatible LLM preset:
   Drop-in OpenAI-compatible gateway for vLLM on HPC::

       from hpc_as_api.presets.openai import create_openai_app
       app = create_openai_app(endpoint_id="...", models={...}, relay_url="wss://...")

3. Low-level Globus Compute client::

       from hpc_as_api.compute import GlobusComputeClient
       client = GlobusComputeClient(endpoint_id="...", models={...})
       result = await client.submit_inference(messages=[...], model="qwen25-vl-72b")
"""

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

__version__ = "0.2.0"
__all__ = [
    "HPCApp",
    "GlobusComputeClient",
    "extract_text_content",
    "has_images",
    "count_images",
    "strip_old_images",
]
