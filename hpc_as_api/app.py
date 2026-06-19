"""
HPC Gateway — FastAPI application for routing LLM requests to HPC clusters.

DUAL-USE DESIGN:
----------------
1. STANDALONE SERVICE:
   Run as its own process with uvicorn. The caller sends OpenAI-compatible
   /v1/chat/completions requests; the gateway dispatches them to the HPC cluster
   via Globus Compute and streams tokens back through the WebSocket relay.
   → Start with: uvicorn hpc_as_api.app:app --host 0.0.0.0 --port 8001

2. EMBEDDED ROUTER:
   Import `router` from the standalone `app` object and mount it:
       from hpc_as_api.app import router
       main_app.include_router(router, prefix="/hpc")

3. PROGRAMMATIC FACTORY:
   Create multiple independent app instances with different configurations:
       from hpc_as_api.app import make_app
       app = make_app(endpoint_id="...", models={...}, relay_url="wss://...")

CONFIGURATION:
--------------
All settings come from environment variables when using the standalone `app`.
When using make_app(), pass them as arguments (env vars are the fallback).

  GLOBUS_COMPUTE_ENDPOINT_ID         UUID of the HPC cluster's Globus endpoint
  HPC_MODELS                         JSON dict mapping model names to their config
  RELAY_URL                          WebSocket URL of the relay server
  RELAY_SECRET                       Shared secret for relay authentication
  RELAY_ENCRYPTION_KEY               AES-256 key (hex) for E2E relay encryption
  HPC_PROXY_HOST                     Host to bind to (default: 0.0.0.0)
  HPC_PROXY_PORT                     Port to listen on (default: 8001)
  USE_GLOBUS_COMPUTE                 "true"/"false" (default: true)
  VLLM_SERVER_URL                    Fallback vLLM URL when not using Globus
  LOG_LEVEL                          Logging level (default: INFO)
  PROXY_RATE_LIMIT_REQUESTS          Global max requests per caller per window (default: 10000)
  PROXY_RATE_LIMIT_WINDOW            Window size in seconds (default: 60)
  PROXY_RATE_LIMIT_REQUESTS_<NAME>   Per-key override; <NAME> matches PROXY_API_KEY_<NAME> suffix

AUTHENTICATION:
---------------
Two modes coexist on the same instance:
  - Globus token: Bearer token from Globus Auth (introspected, per-user attribution)
  - API key: PROXY_API_KEY_<NAME> env vars (for service-to-service callers)
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from websockets.asyncio.client import connect as ws_connect

# Shared persistent HTTP client reuses TCP connections across requests.
# Per-request AsyncClient creation was the relay throughput bottleneck at >3.7 req/s,
# because each request paid a full TCP handshake through the SSH tunnel.
_SHARED_LIMITS = httpx.Limits(
    max_connections=200,
    max_keepalive_connections=100,
    keepalive_expiry=60.0,
)
_shared_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _shared_http_client
    if _shared_http_client is None or _shared_http_client.is_closed:
        _shared_http_client = httpx.AsyncClient(timeout=120.0, limits=_SHARED_LIMITS)
    return _shared_http_client


from hpc_as_api.auth import AuthConfig, Authenticator, CallerIdentity, validate_messages  # noqa: E402
from hpc_as_api.crypto import decrypt_message  # noqa: E402

if TYPE_CHECKING:
    from hpc_as_api.compute import GlobusComputeClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standalone defaults (env vars, resolved once at import time)
# ---------------------------------------------------------------------------
PROXY_HOST = os.getenv("HPC_PROXY_HOST", "0.0.0.0")  # nosec B104  # noqa: S104
PROXY_PORT = int(os.getenv("HPC_PROXY_PORT", "8001"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_app(
    endpoint_id: str | None = None,
    models: dict | None = None,
    relay_url: str | None = None,
    relay_secret: str | None = None,
    relay_encryption_key: str | None = None,
    use_globus_compute: bool | None = None,
    direct_vllm_url: str | None = None,
    auth: "AuthConfig | Authenticator | None" = None,
    title: str = "HPC Gateway",
) -> FastAPI:
    """
    Create an independent OpenAI-compatible HPC gateway FastAPI app.

    Every argument falls back to its corresponding environment variable so
    env-var-based deployments work with zero code changes.  Passing arguments
    explicitly lets you create multiple independent app instances in the same
    process — something the old module-global pattern could not do.

    Args:
        endpoint_id:          Globus Compute endpoint UUID.
        models:               ``{"name": {"hf_name": ..., "url": ..., "context_reserve_output": ...}}``
        relay_url:            WebSocket relay URL for token streaming.
        relay_secret:         Shared relay auth secret.
        relay_encryption_key: Hex AES-256 key for E2E encryption.
        use_globus_compute:   ``True`` (default) → Globus path; ``False`` → direct SSH/vLLM.
        direct_vllm_url:      Fallback vLLM URL when ``use_globus_compute=False``.
        auth:                 ``AuthConfig`` or ``Authenticator``.  Falls back to env vars.
        title:                FastAPI app title.

    Returns:
        A :class:`fastapi.FastAPI` instance ready for uvicorn.
    """
    # Resolve all config values: explicit arg > env var > built-in default
    _endpoint_id = endpoint_id or os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID")

    if models is not None:
        _models: dict = models
    else:
        raw = os.getenv("HPC_MODELS", "{}")
        try:
            _models = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("HPC_MODELS env var is not valid JSON — no models registered")
            _models = {}

    _relay_url = relay_url if relay_url is not None else os.getenv("RELAY_URL", "")
    _relay_secret = relay_secret if relay_secret is not None else os.getenv("RELAY_SECRET", "")
    _relay_enc_key = relay_encryption_key if relay_encryption_key is not None else os.getenv("RELAY_ENCRYPTION_KEY", "")

    if use_globus_compute is not None:
        _use_globus = use_globus_compute
    else:
        _use_globus = os.getenv("USE_GLOBUS_COMPUTE", "true").lower() == "true"

    _direct_url = direct_vllm_url or os.getenv("LAKESHORE_VLLM_ENDPOINT", "http://localhost:8000")

    # Always track the tunnel URL for health reporting and direct routing.
    # When USE_GLOBUS_COMPUTE=false the tunnel is the sole path, so we must
    # still detect it. When true, the tunnel is preferred over Globus when alive.
    _tunnel_url = os.getenv("LAKESHORE_VLLM_ENDPOINT", "")
    _tunnel_check_timeout = float(os.getenv("TUNNEL_CHECK_TIMEOUT", "1.0"))

    # Build the Authenticator from AuthConfig, Authenticator, or env vars.
    if isinstance(auth, Authenticator):
        _authenticator = auth
    elif isinstance(auth, AuthConfig):
        _authenticator = Authenticator(auth)
    else:
        _authenticator = Authenticator(AuthConfig())

    # Mutable single-element list so the lifespan closure can write into it.
    _client: list[GlobusComputeClient | None] = [None]

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        if _use_globus and _endpoint_id:
            try:
                from hpc_as_api.compute import GlobusComputeClient

                client = GlobusComputeClient(
                    endpoint_id=_endpoint_id,
                    models=_models,
                    relay_secret=_relay_secret,
                )
                _client[0] = client
                logger.info(
                    f"HPC Gateway ready: endpoint={_endpoint_id}, models={list(_models.keys())}, relay={_relay_url}"
                )
            except ImportError:
                logger.warning("globus-compute-sdk not installed. Install with: pip install hpc-as-api[globus]")
        elif _use_globus:
            logger.warning("USE_GLOBUS_COMPUTE=true but GLOBUS_COMPUTE_ENDPOINT_ID is not set")
        else:
            logger.info(f"HPC Gateway ready (direct mode): vllm={_direct_url}")

        yield

        if _client[0]:
            logger.info("Shutting down Globus Compute client...")
            _client[0].shutdown()

    fastapi_app = FastAPI(title=title, lifespan=lifespan)
    _router = APIRouter()

    # -----------------------------------------------------------------------
    # /health
    # -----------------------------------------------------------------------

    @_router.get("/health")
    async def health_check():
        tunnel_up = await _tunnel_alive(_tunnel_url, _tunnel_check_timeout) if _tunnel_url else False
        if tunnel_up:
            mode = "ssh_tunnel"
        elif _use_globus:
            mode = "globus_compute"
        else:
            mode = "direct"
        return {
            "status": "healthy",
            "service": "HPC Gateway",
            "mode": mode,
            "globus_configured": bool(_client[0] and _client[0].is_available()),
            "tunnel_up": tunnel_up,
            "models": list(_models.keys()),
            "relay_configured": bool(_relay_url),
        }

    # -----------------------------------------------------------------------
    # GET /v1/models
    # -----------------------------------------------------------------------

    @_router.get("/v1/models")
    async def list_models(caller: CallerIdentity = Depends(_authenticator)):
        from time import time as now

        return {
            "object": "list",
            "data": [
                {
                    "id": info.get("hf_name", name),
                    "object": "model",
                    "created": int(now()),
                    "owned_by": "hpc-as-api",
                    "gateway_name": name,
                }
                for name, info in _models.items()
            ],
        }

    # -----------------------------------------------------------------------
    # POST /reload-auth
    # -----------------------------------------------------------------------

    @_router.post("/reload-auth")
    async def reload_authentication():
        if not _use_globus or not _client[0]:
            return {"success": False, "message": "Globus Compute not configured"}
        try:
            success, message = _client[0].reload_credentials()
            return {"success": success, "message": message}
        except Exception as e:
            logger.error(f"Failed to reload credentials: {e}")
            return {"success": False, "message": f"Failed to reload: {e}"}

    # -----------------------------------------------------------------------
    # POST /v1/chat/completions
    # -----------------------------------------------------------------------

    @_router.post("/v1/chat/completions")
    async def proxy_chat_completions(
        request: Request,
        caller: CallerIdentity = Depends(_authenticator),
    ):
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from e

        raw_model = body.get("model", "")
        model = raw_model.removeprefix("openai/")

        if _use_globus and _models and model not in _models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model}' not found. Available: {list(_models.keys())}",
            )

        messages = validate_messages(body.get("messages", []))
        stream = bool(body.get("stream", False))

        # Forward every other OpenAI parameter to vLLM untouched. The proxy only
        # owns three fields: `model` (gateway alias → backend name), `messages`
        # (validated and size-capped downstream), and `stream` (controls which
        # routing path is taken). Everything else — temperature, max_tokens,
        # tools, tool_choice, chat_template_kwargs, top_p, seed, stop, logprobs,
        # response_format, … — passes through generically, so new sampling knobs
        # need no code change here.
        params = {k: v for k, v in body.items() if k not in _PROXY_OWNED_PARAMS}
        params.setdefault("temperature", 0.7)

        logger.info(
            f"Chat request: caller={caller.log_safe_id()}, model={model}, messages={len(messages)}, stream={stream}"
        )

        # Auto-detect SSH tunnel: if a tunnel URL is configured and reachable,
        # use it directly (low latency). Otherwise fall back to Globus Compute.
        if _tunnel_url and await _tunnel_alive(_tunnel_url, _tunnel_check_timeout):
            logger.info(f"Routing via SSH tunnel: {_tunnel_url}")

            # Build the Globus fallback coroutine once so it can be reused
            # whether the failure is detected before or mid-stream.
            async def _globus_fallback():
                return await _route_via_globus_compute(
                    model,
                    messages,
                    stream,
                    caller,
                    _client,
                    _relay_url,
                    _relay_secret,
                    _relay_enc_key,
                    _endpoint_id,
                    params,
                )

            try:
                return await _route_via_direct(
                    model,
                    messages,
                    stream,
                    _tunnel_url,
                    params,
                    globus_fallback_fn=_globus_fallback if _use_globus else None,
                )
            except HTTPException as e:
                # Tunnel failed before stream started (connect error / 503 / 504).
                if _use_globus and e.status_code in (503, 504):
                    logger.warning(f"SSH tunnel request failed ({e.status_code}); falling back to Globus Compute")
                else:
                    raise
        if _use_globus:
            return await _route_via_globus_compute(
                model,
                messages,
                stream,
                caller,
                _client,
                _relay_url,
                _relay_secret,
                _relay_enc_key,
                _endpoint_id,
                params,
            )
        else:
            return await _route_via_direct(model, messages, stream, _direct_url, params)

    fastapi_app.include_router(_router)
    return fastapi_app


# ---------------------------------------------------------------------------
# Tunnel health check
# ---------------------------------------------------------------------------


async def _tunnel_alive(tunnel_url: str, timeout: float = 1.0) -> bool:
    """Return True if the SSH tunnel endpoint is reachable and vLLM is healthy."""
    try:
        client = _get_http_client()
        resp = await client.get(f"{tunnel_url}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Internal routing helpers (pure functions — no module state)
# ---------------------------------------------------------------------------

# Request fields the proxy controls itself; everything else in the request body
# is forwarded to vLLM verbatim via the `params` dict.
_PROXY_OWNED_PARAMS = frozenset({"model", "messages", "stream"})


async def _route_via_globus_compute(
    model,
    messages,
    stream,
    caller,
    client_ref,
    relay_url,
    relay_secret,
    relay_enc_key,
    endpoint_id,
    params,
):
    client = client_ref[0]
    if not client or not client.is_available():
        raise HTTPException(status_code=503, detail="Globus Compute not configured")

    if stream and relay_url:
        try:
            return await _route_via_globus_compute_streaming(
                model,
                messages,
                caller,
                client,
                relay_url,
                relay_secret,
                relay_enc_key,
                params,
            )
        except Exception as e:
            logger.warning(f"Relay streaming failed — falling back to batch mode: {e}")

    try:
        logger.info(f"Submitting batch job to Globus endpoint: {endpoint_id}")
        result = await client.submit_inference(
            messages=messages,
            model=model,
            params=params,
        )

        if "error" in result:
            error_msg = result.get("error", "Unknown error")
            if result.get("error_type") == "AuthenticationError":
                raise HTTPException(
                    status_code=401,
                    detail=f"Globus Compute authentication required: {error_msg}",
                )
            raise HTTPException(status_code=503, detail=f"HPC inference failed: {error_msg}")

        logger.info("Batch inference completed successfully")
        if stream:
            return _convert_json_to_sse_stream(result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Globus Compute routing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal gateway error: {e}") from e


async def _route_via_globus_compute_streaming(
    model,
    messages,
    caller,
    client,
    relay_url,
    relay_secret,
    relay_enc_key,
    params,
):
    # Pass the caller's Globus token when available — gives per-user SLURM attribution.
    globus_token = caller.globus_token if caller.auth_mode == "globus" else None

    result = await client.submit_streaming_inference(
        messages=messages,
        model=model,
        relay_url=relay_url,
        globus_token=globus_token,
        params=params,
    )

    if "error" in result:
        error_msg = result.get("error", "Unknown error")
        if result.get("error_type") == "AuthenticationError":
            raise HTTPException(
                status_code=401,
                detail=f"Globus Compute authentication required: {error_msg}",
            )
        raise HTTPException(status_code=503, detail=f"HPC streaming failed: {error_msg}")

    channel_id = result["channel_id"]
    logger.info(f"Relay streaming: channel={channel_id[:8]}, relay={relay_url}")

    async def sse_generator():
        try:
            async with ws_connect(f"{relay_url}/consume/{channel_id}") as ws:
                if relay_secret:
                    await ws.send(json.dumps({"type": "auth", "secret": relay_secret}))

                async for msg_str in ws:
                    if relay_enc_key:
                        msg_str = decrypt_message(relay_enc_key, msg_str)

                    msg = json.loads(msg_str)

                    if msg["type"] == "chunk":
                        # Relay forwards vLLM's SSE chunk verbatim. Re-emit it
                        # untouched so tool_calls, logprobs, multi-choice, usage,
                        # finish_reason — anything vLLM produces — reaches the
                        # client without per-field reconstruction.
                        yield f"data: {json.dumps(msg['data'])}\n\n"

                    elif msg["type"] == "done":
                        yield "data: [DONE]\n\n"
                        break

                    elif msg["type"] == "error":
                        # Surface a generic OpenAI-shaped error event carrying a
                        # correlation ref. The full upstream message is logged
                        # server-side under the same ref so operators can match a
                        # client report to the log line without leaking internals.
                        ref = uuid.uuid4().hex[:12]
                        logger.error(
                            f"Relay upstream error on channel {channel_id[:8]} [ref={ref}]: {msg.get('message')}"
                        )
                        err = {"error": {"message": "upstream inference error", "type": "upstream_error", "ref": ref}}
                        yield f"data: {json.dumps(err)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

        except Exception as e:
            ref = uuid.uuid4().hex[:12]
            logger.error(f"Relay connection failed on channel {channel_id[:8]} [ref={ref}]: {e}", exc_info=True)
            err = {"error": {"message": "gateway streaming error", "type": "gateway_error", "ref": ref}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


async def _route_via_direct(model, messages, stream, direct_url, params, globus_fallback_fn=None):
    # Forward all client params verbatim; the proxy only overrides the three
    # fields it owns. max_tokens defaults to 2048 when the client omits it.
    payload = {
        **params,
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    payload.setdefault("max_tokens", 2048)
    target_url = f"{direct_url}/v1/chat/completions"
    logger.info(f"Direct vLLM request: {target_url}")

    try:
        if stream:

            async def stream_generator():
                job_id: str | None = None
                # seq counts complete SSE events (each "data: ...\n\n" block = 1 event).
                # The buffer proxy assigns seq per stored chunk; we mirror that count
                # so X-Resume-Job carries the right offset on reconnect.
                last_seq: int = 0
                done = False  # True only when we see "data: [DONE]" in the stream
                attempt = 0
                max_reconnect_attempts = 120  # up to ~120s: covers autossh reconnect time
                buf = b""  # accumulate bytes until we have complete SSE events

                while not done:
                    attempt += 1
                    headers: dict[str, str] = {}
                    if job_id is not None:
                        headers["X-Resume-Job"] = f"{job_id}:{last_seq}"
                        logger.info(f"Reconnecting to buffer proxy, resuming job {job_id[:8]} from seq {last_seq}")

                    try:
                        _client = _get_http_client()
                        async with _client.stream("POST", target_url, json=payload, headers=headers) as resp:
                            if resp.status_code != 200:
                                error_text = await resp.aread()
                                raise HTTPException(
                                    status_code=resp.status_code,
                                    detail=f"vLLM error: {error_text.decode()}",
                                )
                            if job_id is None:
                                job_id = resp.headers.get("X-Job-ID")

                            async for chunk in resp.aiter_bytes():
                                buf += chunk
                                # Flush complete SSE events (terminated by \n\n) to the client.
                                # Count each flushed event so last_seq matches buffer proxy's seq.
                                while b"\n\n" in buf:
                                    event, buf = buf.split(b"\n\n", 1)
                                    event_bytes = event + b"\n\n"
                                    if b"[DONE]" in event_bytes:
                                        done = True
                                    yield event_bytes
                                    if not done:
                                        last_seq += 1

                        # aiter_bytes ended — either [DONE] was seen (good) or tunnel
                        # dropped mid-stream (clean TCP close, no exception).
                        if not done:
                            raise httpx.ReadError("stream ended before [DONE]", request=None)

                    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
                        if attempt >= max_reconnect_attempts:
                            logger.error(f"Buffer proxy unreachable after {attempt} attempts: {e}")
                            if globus_fallback_fn is not None:
                                logger.warning("Falling back to Globus Compute after repeated tunnel failures")
                                try:
                                    fallback_resp = await globus_fallback_fn()
                                    if isinstance(fallback_resp, StreamingResponse):
                                        async for chunk in fallback_resp.body_iterator:
                                            yield chunk
                                    else:
                                        yield f"data: {json.dumps(fallback_resp)}\n\n"
                                        yield "data: [DONE]\n\n"
                                except Exception as fe:
                                    err = {
                                        "error": {
                                            "message": f"tunnel lost and fallback failed: {fe}",
                                            "type": "gateway_error",
                                        }
                                    }
                                    yield f"data: {json.dumps(err)}\n\n"
                                    yield "data: [DONE]\n\n"
                            else:
                                err = {
                                    "error": {
                                        "message": "tunnel lost; buffer proxy unreachable",
                                        "type": "gateway_error",
                                    }
                                }
                                yield f"data: {json.dumps(err)}\n\n"
                                yield "data: [DONE]\n\n"
                            return

                        # Tunnel is reconnecting — wait briefly then retry.
                        # The client SSE connection stays open the whole time.
                        wait = min(0.1 * attempt, 1.0)
                        logger.warning(f"Tunnel lost ({type(e).__name__}), attempt {attempt}, retrying in {wait:.2f}s")
                        await asyncio.sleep(wait)

                    except httpx.TimeoutException as e:
                        raise HTTPException(status_code=504, detail="vLLM request timed out") from e

                    else:
                        # Successful connection — reset attempt counter for next potential drop.
                        attempt = 0

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        client = _get_http_client()
        response = await client.post(target_url, json=payload)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"vLLM error: {response.text}",
            )
        return response.json()

    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to vLLM. Is the tunnel running? Error: {e}",
        ) from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail="vLLM request timed out") from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal gateway error: {e}") from e


def _convert_json_to_sse_stream(json_response: dict):
    """Simulate streaming by splitting a complete batch response into word-sized SSE chunks."""
    words_per_chunk = 2
    delay_between_chunks = 0.05

    async def sse_generator():
        choices = json_response.get("choices", [])
        if not choices:
            yield "data: [DONE]\n\n"
            return

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        role = message.get("role", "assistant")

        chunk_base = {
            "id": json_response.get("id", ""),
            "object": "chat.completion.chunk",
            "created": json_response.get("created", 0),
            "model": json_response.get("model", ""),
        }

        if role:
            role_chunk = {
                **chunk_base,
                "choices": [{"index": 0, "delta": {"role": role, "content": ""}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

        if content:
            words = content.split(" ")
            for i in range(0, len(words), words_per_chunk):
                word_group = words[i : i + words_per_chunk]
                text_chunk = " ".join(word_group) if i == 0 else " " + " ".join(word_group)
                text_chunk_data = {
                    **chunk_base,
                    "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(text_chunk_data)}\n\n"
                await asyncio.sleep(delay_between_chunks)

        finish_chunk = {
            **chunk_base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}],
            "usage": json_response.get("usage", {}),
        }
        yield f"data: {json.dumps(finish_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Standalone app (env-var config, used by `uvicorn hpc_as_api.app:app`)
# ---------------------------------------------------------------------------

app = make_app()
router = app.router


def main():
    import uvicorn

    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level=LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
