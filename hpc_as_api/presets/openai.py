"""
hpc_as_api.presets.openai — OpenAI-compatible LLM gateway preset.

Creates a FastAPI app with ``POST /v1/chat/completions`` and ``GET /v1/models``
that routes LLM inference to a vLLM server on an HPC cluster via Globus Compute.

The response is OpenAI wire-compatible: streaming clients (LangChain, OpenAI SDK,
LiteLLM) can point at this gateway and work unchanged.

Usage::

    from hpc_as_api.presets.openai import create_openai_app

    app = create_openai_app(
        endpoint_id="8d978809-eec4-413d-bbd4-b099e488100a",
        models={
            "qwen25-vl-72b": {
                "hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
                "url": "http://ghi2-002:8000",
                "context_reserve_output": 4096,
            }
        },
        relay_url="wss://relay.stream.acer.uic.edu",
    )
    # uvicorn mymodule:app --host 0.0.0.0 --port 8001
"""

import json
import logging
import os

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def create_openai_app(
    endpoint_id: str | None = None,
    models: dict | None = None,
    relay_url: str | None = None,
    relay_secret: str = "",
    relay_encryption_key: str = "",
    host: str = "0.0.0.0",  # nosec B104
    port: int = 8001,
    log_level: str = "INFO",
) -> FastAPI:
    """
    Create an OpenAI-compatible FastAPI gateway for LLM inference on HPC.

    Returns a :class:`fastapi.FastAPI` instance with these endpoints:

    - ``GET /health``               — service health, no auth
    - ``GET /v1/models``            — list available models (auth required)
    - ``POST /v1/chat/completions`` — OpenAI-compatible chat (auth required)
    - ``POST /reload-auth``         — reload Globus credentials from disk

    Args:
        endpoint_id: Globus Compute endpoint UUID.
            Falls back to ``GLOBUS_COMPUTE_ENDPOINT_ID`` env var.
        models: Dict mapping model names to their config.
            Each entry: ``{"hf_name": str, "url": str, "context_reserve_output": int}``.
            Falls back to ``HPC_MODELS`` env var (JSON string).
        relay_url: WebSocket URL of the relay server.
            Falls back to ``RELAY_URL`` env var.
        relay_secret: Shared relay auth secret.
            Falls back to ``RELAY_SECRET`` env var.
        relay_encryption_key: AES-256 hex key for E2E encryption.
            Falls back to ``RELAY_ENCRYPTION_KEY`` env var.
        host: Bind address for standalone ``main()`` mode. Default ``0.0.0.0``.
        port: Port for standalone ``main()`` mode. Default ``8001``.
        log_level: Uvicorn log level. Default ``INFO``.

    Returns:
        A :class:`fastapi.FastAPI` app ready for uvicorn.

    Example — embed in an existing FastAPI app::

        from hpc_as_api.presets.openai import create_openai_app
        from fastapi import FastAPI

        main_app = FastAPI()
        hpc_app = create_openai_app(endpoint_id="...", models={...}, relay_url="...")
        main_app.mount("/hpc", hpc_app)
    """
    # Resolve configuration: explicit args > env vars
    _endpoint_id = endpoint_id or os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID")
    _relay_url = relay_url or os.getenv("RELAY_URL", "")
    _relay_secret = relay_secret or os.getenv("RELAY_SECRET", "")
    _relay_encryption_key = relay_encryption_key or os.getenv("RELAY_ENCRYPTION_KEY", "")

    # Inject into env vars so hpc_as_api.app picks them up at import time
    if models is not None:
        os.environ["HPC_MODELS"] = json.dumps(models)
    if _endpoint_id:
        os.environ["GLOBUS_COMPUTE_ENDPOINT_ID"] = _endpoint_id
    if _relay_url:
        os.environ["RELAY_URL"] = _relay_url
    if _relay_secret:
        os.environ["RELAY_SECRET"] = _relay_secret
    if _relay_encryption_key:
        os.environ["RELAY_ENCRYPTION_KEY"] = _relay_encryption_key

    from hpc_as_api.app import app as _app

    return _app


def main():
    """CLI entry point for the OpenAI-compatible LLM gateway."""
    import uvicorn

    host = os.getenv("HPC_PROXY_HOST", "0.0.0.0")  # nosec B104
    port = int(os.getenv("HPC_PROXY_PORT", "8001"))
    log_level = os.getenv("LOG_LEVEL", "INFO").lower()

    app = create_openai_app(host=host, port=port, log_level=log_level)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
