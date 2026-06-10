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

Multiple independent instances in the same process::

    app_prod = create_openai_app(endpoint_id="uuid-1", models={...}, relay_url="wss://...")
    app_dev  = create_openai_app(endpoint_id="uuid-2", models={...}, relay_url="wss://...")
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from hpc_as_api.auth import AuthConfig, Authenticator

logger = logging.getLogger(__name__)


def create_openai_app(
    endpoint_id: str | None = None,
    models: dict | None = None,
    relay_url: str | None = None,
    relay_secret: str = "",
    relay_encryption_key: str = "",
    host: str = "0.0.0.0",  # nosec B104  # noqa: S104
    port: int = 8001,
    log_level: str = "INFO",
    auth: AuthConfig | Authenticator | None = None,
) -> FastAPI:
    """
    Create an OpenAI-compatible FastAPI gateway for LLM inference on HPC.

    Returns a :class:`fastapi.FastAPI` instance with these endpoints:

    - ``GET /health``               — service health, no auth
    - ``GET /v1/models``            — list available models (auth required)
    - ``POST /v1/chat/completions`` — OpenAI-compatible chat (auth required)
    - ``POST /reload-auth``         — reload Globus credentials from disk

    Each call returns a **new, independent** FastAPI app — calling this function
    twice with different arguments gives two isolated gateways that can run in the
    same process without interfering with each other.

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
        auth: Authentication configuration. Pass an :class:`~hpc_as_api.auth.AuthConfig`
            to configure all auth settings in Python (Globus credentials, allowed
            domains, API keys, rate limits). Falls back to env vars when not supplied.

    Returns:
        A :class:`fastapi.FastAPI` app ready for uvicorn.

    Example — restrict to one institution, all config in Python::

        from hpc_as_api.presets.openai import create_openai_app
        from hpc_as_api.auth import AuthConfig

        app = create_openai_app(
            endpoint_id="8d978809-...",
            models={"llama3": {"hf_name": "meta-llama/Llama-3-70b", "url": "http://gpu01:8000"}},
            relay_url="wss://relay.example.com",
            auth=AuthConfig(
                globus_client_id="your-client-id",
                globus_client_secret="your-client-secret",
                allowed_domains=["ornl.gov", "anl.gov"],
                api_keys={"myservice": "sk-my-service-key"},
                rate_limit_requests=20,
                rate_limit_window=60,
            ),
        )

    Example — embed in an existing FastAPI app::

        from fastapi import FastAPI
        main_app = FastAPI()
        hpc_app = create_openai_app(endpoint_id="...", models={...}, relay_url="...")
        main_app.mount("/hpc", hpc_app)
    """
    from hpc_as_api.app import make_app

    return make_app(
        endpoint_id=endpoint_id,
        models=models,
        relay_url=relay_url or os.getenv("RELAY_URL", ""),
        relay_secret=relay_secret or os.getenv("RELAY_SECRET", ""),
        relay_encryption_key=relay_encryption_key or os.getenv("RELAY_ENCRYPTION_KEY", ""),
        auth=auth,
        title="HPC Gateway (OpenAI-compatible)",
    )


def main():
    """CLI entry point for the OpenAI-compatible LLM gateway."""
    import uvicorn

    host = os.getenv("HPC_PROXY_HOST", "0.0.0.0")  # nosec B104  # noqa: S104
    port = int(os.getenv("HPC_PROXY_PORT", "8001"))
    log_level = os.getenv("LOG_LEVEL", "INFO").lower()

    app = create_openai_app(host=host, port=port, log_level=log_level)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
