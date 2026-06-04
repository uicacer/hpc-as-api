"""
hpc_as_api.core — Domain-agnostic HPCApp framework.

HPCApp exposes any HPC function as a streaming HTTP endpoint.
The function runs on an HPC cluster via Globus Compute; its output
streams to clients in real time via a WebSocket relay.

Both ends connect *outbound* to the relay — no inbound ports needed,
no VPN, no firewall changes. Works with any incrementally produced
output: LLM tokens, simulation checkpoints, solver metrics, etc.

Quickstart::

    from hpc_as_api.core import HPCApp
    from pydantic import BaseModel

    class SimRequest(BaseModel):
        steps: int = 1000
        grid_size: int = 100

    def hpc_sim(steps, grid_size, relay_url, channel_id, relay_secret=""):
        from streamrelay import RelayProducer
        with RelayProducer(relay_url, channel_id, relay_secret=relay_secret) as relay:
            for i in range(steps):
                relay.send_token(f"step={i}\\n")

    gateway = HPCApp(endpoint_id="...", relay_url="wss://relay.example.com")
    gateway.mount("/simulate", hpc_sim, SimRequest)
    app = gateway.create_app()
    # uvicorn mymodule:app --port 8001
"""

import json
import logging
import os
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _default_output_handler(msg: dict) -> str | None:
    """
    Convert a relay message to an SSE data string.

    Returns a string to yield it as ``data: <string>\\n\\n``,
    or ``None`` to skip it (used for "done" which signals end of stream).
    """
    msg_type = msg.get("type")
    if msg_type == "token":
        return msg.get("content", "")
    elif msg_type == "error":
        return f"[ERROR] {msg.get('message', 'unknown error')}"
    return None


class _Route:
    """Internal holder for a registered route."""

    def __init__(
        self,
        path: str,
        remote_fn: Callable,
        input_schema: type[BaseModel],
        output_handler: Callable[[dict], str | None],
        auth: bool,
        description: str,
    ):
        self.path = path
        self.remote_fn = remote_fn
        self.input_schema = input_schema
        self.output_handler = output_handler
        self.auth = auth
        self.description = description


class HPCApp:
    """
    Domain-agnostic HTTP gateway for HPC functions via Globus Compute + WebSocket relay.

    Any Python function that produces incremental output can be exposed as a
    streaming HTTP endpoint. The function runs on the HPC cluster; its output
    arrives in real time through the WebSocket relay.

    Configuration is read from constructor arguments (explicit) or environment
    variables (implicit), in that order:

    +-----------------------------+---------------------------------+
    | Constructor arg             | Env var                         |
    +=============================+=================================+
    | ``endpoint_id``             | ``GLOBUS_COMPUTE_ENDPOINT_ID``  |
    | ``relay_url``               | ``RELAY_URL``                   |
    | ``relay_secret``            | ``RELAY_SECRET``                |
    | ``relay_encryption_key``    | ``RELAY_ENCRYPTION_KEY``        |
    +-----------------------------+---------------------------------+

    Usage::

        from hpc_as_api.core import HPCApp
        from pydantic import BaseModel

        class RunRequest(BaseModel):
            steps: int = 1000

        def my_hpc_function(steps, relay_url, channel_id, relay_secret=""):
            from streamrelay import RelayProducer
            with RelayProducer(relay_url, channel_id, relay_secret=relay_secret) as relay:
                for i in range(steps):
                    relay.send_token(f"{i}\\n")

        gateway = HPCApp(endpoint_id="...", relay_url="wss://relay.example.com")
        gateway.mount("/run", my_hpc_function, RunRequest)
        app = gateway.create_app()
    """

    def __init__(
        self,
        endpoint_id: str | None = None,
        relay_url: str | None = None,
        relay_secret: str = "",
        relay_encryption_key: str = "",
        title: str = "HPC Gateway",
        description: str = "HTTP gateway for HPC functions via Globus Compute",
    ):
        self.endpoint_id = endpoint_id or os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID")
        self.relay_url = relay_url or os.getenv("RELAY_URL", "")
        self.relay_secret = relay_secret or os.getenv("RELAY_SECRET", "")
        self.relay_encryption_key = relay_encryption_key or os.getenv("RELAY_ENCRYPTION_KEY", "")
        self.title = title
        self.description = description
        self._routes: list[_Route] = []

    def mount(
        self,
        path: str,
        remote_fn: Callable,
        input_schema: type[BaseModel],
        output_handler: Callable[[dict], str | None] = _default_output_handler,
        auth: bool = True,
        description: str = "",
    ) -> "HPCApp":
        """Register a remote HPC function as a streaming HTTP endpoint.

        Args:
            path: URL path, e.g. ``"/run"`` or ``"/simulate"``.
            remote_fn: Python callable executed on the HPC cluster.
                Receives all ``input_schema`` fields as kwargs plus:
                ``relay_url``, ``channel_id``, ``relay_secret``.
                Use :class:`~streamrelay.producer.RelayProducer` inside it
                to stream output back through the relay.
            input_schema: Pydantic model for the HTTP request body.
                Each field is forwarded as a keyword argument to ``remote_fn``.
            output_handler: Converts each relay message to an SSE data string,
                or ``None`` to skip it. The default yields token content and
                skips "done" (the framework sends ``[DONE]`` automatically).
            auth: Require API key or Globus token auth. Default ``True``.
            description: Description shown in the OpenAPI (Swagger) docs.

        Returns:
            ``self`` — allows chaining: ``gateway.mount(...).mount(...)``.
        """
        self._routes.append(
            _Route(
                path=path,
                remote_fn=remote_fn,
                input_schema=input_schema,
                output_handler=output_handler,
                auth=auth,
                description=description,
            )
        )
        return self

    def create_app(self) -> FastAPI:
        """Build and return the FastAPI application.

        Creates ``GET /health`` plus one ``POST`` endpoint for every
        registered route.

        Returns:
            A :class:`fastapi.FastAPI` instance ready for uvicorn.
        """
        endpoint_id = self.endpoint_id
        relay_url = self.relay_url
        relay_secret = self.relay_secret
        relay_encryption_key = self.relay_encryption_key
        routes_snapshot = list(self._routes)

        _client: list[Any] = [None]

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                from hpc_as_api.compute import GlobusComputeClient

                if endpoint_id:
                    client = GlobusComputeClient(
                        endpoint_id=endpoint_id,
                        relay_secret=relay_secret,
                    )
                    _client[0] = client
                    logger.info(
                        f"HPCApp ready: endpoint={endpoint_id}, "
                        f"routes={[r.path for r in routes_snapshot]}"
                    )
                else:
                    logger.warning("HPCApp: no endpoint_id — Globus Compute unavailable")
            except ImportError:
                logger.warning(
                    "HPCApp: globus-compute-sdk not installed. "
                    "Install with: pip install hpc-as-api[globus]"
                )

            yield

            if _client[0]:
                _client[0].shutdown()

        fastapi_app = FastAPI(
            title=self.title,
            description=self.description,
            lifespan=lifespan,
        )
        router = APIRouter()

        @router.get("/health", summary="Service health check")
        async def health():
            """Health check — no authentication required."""
            client = _client[0]
            return {
                "status": "healthy",
                "endpoint_configured": bool(endpoint_id),
                "globus_ready": bool(client and client.is_available()),
                "relay_configured": bool(relay_url),
                "routes": [r.path for r in routes_snapshot],
            }

        for route in routes_snapshot:
            _add_route(
                router=router,
                route=route,
                client_ref=_client,
                relay_url=relay_url,
                relay_secret=relay_secret,
                relay_encryption_key=relay_encryption_key,
            )

        fastapi_app.include_router(router)
        return fastapi_app


def _add_route(
    router: APIRouter,
    route: _Route,
    client_ref: list,
    relay_url: str,
    relay_secret: str,
    relay_encryption_key: str,
) -> None:
    """Register one POST endpoint on the router for the given route."""
    from hpc_as_api.auth import authenticate

    path = route.path
    schema_cls = route.input_schema
    remote_fn = route.remote_fn
    output_handler = route.output_handler
    route_auth = route.auth
    route_description = route.description

    dependencies = [Depends(authenticate)] if route_auth else []

    async def _endpoint(body: schema_cls):  # type: ignore[valid-type]
        client = client_ref[0]
        if not client or not client.is_available():
            raise HTTPException(
                status_code=503, detail="Globus Compute not configured or unavailable"
            )
        if not relay_url:
            raise HTTPException(status_code=503, detail="RELAY_URL not configured")

        channel_id = str(uuid.uuid4())
        kwargs = body.model_dump()

        try:
            gce = client._get_executor()
            gce.submit(
                remote_fn,
                **kwargs,
                relay_url=relay_url,
                channel_id=channel_id,
                relay_secret=relay_secret,
            )
        except Exception as exc:
            logger.error(f"Failed to submit job: {exc}", exc_info=True)
            raise HTTPException(
                status_code=503, detail=f"Failed to submit HPC job: {exc}"
            ) from exc

        async def sse_gen():
            from websockets.asyncio.client import connect as ws_connect

            try:
                from hpc_as_api.crypto import decrypt_message as _decrypt
            except ImportError:
                _decrypt = None

            try:
                consume_url = f"{relay_url}/consume/{channel_id}"
                async with ws_connect(consume_url) as ws:
                    if relay_secret:
                        await ws.send(json.dumps({"type": "auth", "secret": relay_secret}))
                    async for raw in ws:
                        if relay_encryption_key and _decrypt:
                            raw = _decrypt(relay_encryption_key, raw)
                        msg = json.loads(raw)
                        result = output_handler(msg)
                        if msg.get("type") == "done":
                            if result is not None:
                                yield f"data: {result}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                        elif result is not None:
                            yield f"data: {result}\n\n"
            except Exception as exc:
                logger.error(f"Relay stream error on channel {channel_id[:8]}: {exc}")
                yield f"data: [ERROR] {exc}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    _endpoint.__name__ = f"_hpcapp_route_{path.strip('/').replace('/', '_')}"

    router.add_api_route(
        path=path,
        endpoint=_endpoint,
        methods=["POST"],
        dependencies=dependencies,
        description=route_description or f"Stream results of HPC function at {path}",
        summary=path.strip("/").replace("/", " ").title(),
    )
