"""Tests for hpc_as_api.app — FastAPI routes, factory independence, and auth wiring."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared Globus mock fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_globus(monkeypatch):
    """Patch globus_compute_sdk and globus_sdk so tests run without the real SDKs."""
    for mod in [
        "globus_compute_sdk",
        "globus_compute_sdk.errors",
        "globus_compute_sdk.errors.error_types",
        "globus_compute_sdk.serialize",
        "globus_sdk",
        "globus_sdk.login_flows",
        "globus_sdk.login_flows.command_line_login_flow_manager",
        "globus_sdk.authorizers",
    ]:
        monkeypatch.setitem(__import__("sys").modules, mod, MagicMock())

    import sys

    for key in list(sys.modules.keys()):
        if key.startswith("hpc_as_api"):
            sys.modules.pop(key)


# ---------------------------------------------------------------------------
# Helper: build a TestClient from make_app() with direct (no-Globus) mode
# ---------------------------------------------------------------------------


def _direct_client(mock_globus, models=None, **kwargs):
    from hpc_as_api.app import make_app
    from hpc_as_api.auth import AuthConfig, Authenticator

    # Open authenticator — no credentials needed
    auth = Authenticator(
        AuthConfig(
            globus_client_id="",
            globus_client_secret="",
            allowed_domains=[],
            api_keys={"testkey": "test-service"},
        )
    )
    _models = models or {"test-model": {"hf_name": "org/TestModel", "url": "http://fake:8000"}}
    app = make_app(use_globus_compute=False, models=_models, auth=auth, **kwargs)
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_200(mock_globus):
    client = _direct_client(mock_globus)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["service"] == "HPC Gateway"


def test_health_lists_models(mock_globus):
    client = _direct_client(mock_globus)
    assert "test-model" in client.get("/health").json()["models"]


def test_health_mode_direct(mock_globus):
    client = _direct_client(mock_globus)
    assert client.get("/health").json()["mode"] == "direct"


# ---------------------------------------------------------------------------
# /v1/models — requires auth
# ---------------------------------------------------------------------------


def test_models_requires_auth(mock_globus):
    client = _direct_client(mock_globus)
    resp = client.get("/v1/models")
    assert resp.status_code in (401, 403, 422)


def test_models_returns_list_with_valid_key(mock_globus):
    client = _direct_client(mock_globus)
    resp = client.get("/v1/models", headers={"Authorization": "Bearer testkey"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert any(m["gateway_name"] == "test-model" for m in body["data"])


# ---------------------------------------------------------------------------
# Factory independence — two instances must not share state
# ---------------------------------------------------------------------------


def test_create_openai_app_independence(mock_globus):
    """Two calls to create_openai_app() must return independent FastAPI instances."""
    from hpc_as_api.auth import AuthConfig
    from hpc_as_api.presets.openai import create_openai_app

    auth = AuthConfig(globus_client_id="", globus_client_secret="", allowed_domains=[], api_keys={"k": "svc"})

    app_a = create_openai_app(
        models={"model-a": {"hf_name": "org/A", "url": "http://a:8000"}},
        auth=auth,
    )
    app_b = create_openai_app(
        models={"model-b": {"hf_name": "org/B", "url": "http://b:8000"}},
        auth=auth,
    )

    assert app_a is not app_b, "create_openai_app() must return new instances each call"

    with TestClient(app_a) as ca, TestClient(app_b) as cb:
        models_a = {
            m["gateway_name"] for m in ca.get("/v1/models", headers={"Authorization": "Bearer k"}).json()["data"]
        }
        models_b = {
            m["gateway_name"] for m in cb.get("/v1/models", headers={"Authorization": "Bearer k"}).json()["data"]
        }

    assert "model-a" in models_a
    assert "model-b" not in models_a
    assert "model-b" in models_b
    assert "model-a" not in models_b


# ---------------------------------------------------------------------------
# Model validation — unknown model → 404 (Globus path only)
# ---------------------------------------------------------------------------


def _globus_client(mock_globus, models=None):
    """Build a TestClient wired for Globus (use_globus_compute=True) with mocked submit."""

    from hpc_as_api.app import make_app
    from hpc_as_api.auth import AuthConfig, Authenticator

    auth = Authenticator(
        AuthConfig(
            globus_client_id="",
            globus_client_secret="",
            allowed_domains=[],
            api_keys={"testkey": "test-service"},
        )
    )
    _models = models or {"known-model": {"hf_name": "org/Known", "url": "http://fake:8000"}}
    app = make_app(use_globus_compute=True, models=_models, auth=auth, endpoint_id="fake-endpoint")
    return TestClient(app, raise_server_exceptions=False)


def test_unknown_model_returns_404(mock_globus):
    """Requesting a model not in HPC_MODELS on the Globus path must return 404."""
    client = _globus_client(mock_globus)
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer testkey"},
        json={"model": "nonexistent-xyz", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
    assert "nonexistent-xyz" in resp.json().get("detail", "")


def test_known_model_not_404(mock_globus):
    """A registered model must NOT be rejected at the model-validation gate."""

    client = _globus_client(mock_globus, models={"known-model": {"hf_name": "org/M", "url": "http://fake:8000"}})
    # The request will fail later (no real relay/Globus), but must not 404
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer testkey"},
        json={"model": "known-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code != 404


# ---------------------------------------------------------------------------
# /v1/chat/completions — direct mode
# ---------------------------------------------------------------------------


def test_chat_completions_requires_auth(mock_globus):
    client = _direct_client(mock_globus)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code in (401, 403, 422)


def test_chat_completions_direct_mode_calls_vllm(mock_globus):
    """In direct mode, the gateway should proxy the request to vLLM."""
    from hpc_as_api.app import make_app
    from hpc_as_api.auth import AuthConfig, Authenticator

    auth = Authenticator(
        AuthConfig(
            globus_client_id="",
            globus_client_secret="",
            allowed_domains=[],
            api_keys={"mykey": "test-service"},
        )
    )
    app = make_app(
        use_globus_compute=False,
        models={"m": {"hf_name": "org/M", "url": "http://fake:8000"}},
        direct_vllm_url="http://fake:8000",
        auth=auth,
    )

    fake_response = {
        "choices": [{"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }

    with patch("hpc_as_api.app._route_via_direct", new=AsyncMock(return_value=fake_response)):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer mykey"},
                json={"model": "m", "messages": [{"role": "user", "content": "Hello"}]},
            )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Globus token wiring — caller.globus_token forwarded to submit_streaming_inference
# ---------------------------------------------------------------------------


def test_globus_token_passed_to_submit_streaming(mock_globus):
    """When auth_mode=='globus', caller.globus_token must reach submit_streaming_inference."""
    import asyncio

    from hpc_as_api.app import _route_via_globus_compute_streaming
    from hpc_as_api.auth import CallerIdentity

    caller = CallerIdentity(
        name="user@example.com",
        auth_mode="globus",
        globus_token="test-globus-token-abc123",
    )

    mock_client = AsyncMock()
    mock_client.submit_streaming_inference = AsyncMock(
        return_value={"channel_id": "aaaaaaaa-0000-0000-0000-000000000000"}
    )

    # We only care that submit_streaming_inference was called with globus_token=...
    # The relay websocket connect will fail — that's fine for this unit test.
    with patch("hpc_as_api.app.ws_connect", side_effect=Exception("no relay in tests")):
        try:
            asyncio.get_event_loop().run_until_complete(
                _route_via_globus_compute_streaming(
                    model="m",
                    messages=[{"role": "user", "content": "hi"}],
                    caller=caller,
                    client=mock_client,
                    relay_url="ws://fake",
                    relay_secret="",
                    relay_enc_key="",
                    params={"temperature": 0.7, "max_tokens": 10},
                )
            )
        except Exception:
            pass

    mock_client.submit_streaming_inference.assert_called_once()
    call_kwargs = mock_client.submit_streaming_inference.call_args.kwargs
    assert call_kwargs.get("globus_token") == "test-globus-token-abc123"


def test_api_key_caller_sends_no_globus_token(mock_globus):
    """API-key callers (auth_mode='api_key') must pass globus_token=None."""
    import asyncio

    from hpc_as_api.app import _route_via_globus_compute_streaming
    from hpc_as_api.auth import CallerIdentity

    caller = CallerIdentity(name="myservice", auth_mode="api_key", globus_token=None)

    mock_client = AsyncMock()
    mock_client.submit_streaming_inference = AsyncMock(
        return_value={"channel_id": "bbbbbbbb-0000-0000-0000-000000000000"}
    )

    with patch("hpc_as_api.app.ws_connect", side_effect=Exception("no relay in tests")):
        try:
            asyncio.get_event_loop().run_until_complete(
                _route_via_globus_compute_streaming(
                    model="m",
                    messages=[{"role": "user", "content": "hi"}],
                    caller=caller,
                    client=mock_client,
                    relay_url="ws://fake",
                    relay_secret="",
                    relay_enc_key="",
                    params={"temperature": 0.7, "max_tokens": 10},
                )
            )
        except Exception:
            pass

    call_kwargs = mock_client.submit_streaming_inference.call_args.kwargs
    assert call_kwargs.get("globus_token") is None


# ---------------------------------------------------------------------------
# tools / tool_choice passthrough
# ---------------------------------------------------------------------------

_SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


def test_tools_forwarded_to_direct_route(mock_globus):
    """tools and tool_choice in the request body must reach _route_via_direct via params."""
    captured: dict = {}

    async def spy(model, messages, stream, direct_url, params):
        captured["params"] = params
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    with patch("hpc_as_api.app._route_via_direct", side_effect=spy):
        client = _direct_client(mock_globus)
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer testkey"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": _SAMPLE_TOOLS,
                "tool_choice": "auto",
            },
        )

    assert resp.status_code == 200
    assert captured["params"]["tools"] == _SAMPLE_TOOLS
    assert captured["params"]["tool_choice"] == "auto"


def test_tool_choice_required_forwarded(mock_globus):
    """tool_choice='required' must be preserved exactly."""
    captured: dict = {}

    async def spy(model, messages, stream, direct_url, params):
        captured["params"] = params
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "tool_calls"}]}

    with patch("hpc_as_api.app._route_via_direct", side_effect=spy):
        client = _direct_client(mock_globus)
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer testkey"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _SAMPLE_TOOLS,
                "tool_choice": "required",
            },
        )

    assert captured["params"]["tool_choice"] == "required"


def test_arbitrary_sampling_params_forwarded(mock_globus):
    """Params with no dedicated handling (top_p, seed, stop, …) must pass through untouched."""
    captured: dict = {}

    async def spy(model, messages, stream, direct_url, params):
        captured["params"] = params
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    with patch("hpc_as_api.app._route_via_direct", side_effect=spy):
        client = _direct_client(mock_globus)
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer testkey"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": 0.5,
                "seed": 42,
                "stop": ["\n\n"],
                "frequency_penalty": 0.3,
            },
        )

    p = captured["params"]
    assert p["top_p"] == 0.5
    assert p["seed"] == 42
    assert p["stop"] == ["\n\n"]
    assert p["frequency_penalty"] == 0.3


def test_no_tools_in_request_omits_key(mock_globus):
    """When tools are absent from the request body, the key is simply not in params."""
    captured: dict = {}

    async def spy(model, messages, stream, direct_url, params):
        captured["params"] = params
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    with patch("hpc_as_api.app._route_via_direct", side_effect=spy):
        client = _direct_client(mock_globus)
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer testkey"},
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert "tools" not in captured["params"]
    assert "tool_choice" not in captured["params"]


@pytest.mark.asyncio
async def test_route_via_direct_includes_tools_in_vllm_payload(mock_globus):
    """_route_via_direct must include tools/tool_choice in the forwarded httpx payload."""
    import hpc_as_api.app as app_module

    captured: dict = {}

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "tool_calls"}]
    }

    mock_aclient = AsyncMock()
    mock_aclient.__aenter__ = AsyncMock(return_value=mock_aclient)
    mock_aclient.__aexit__ = AsyncMock(return_value=None)

    async def capture_post(url, **kwargs):
        captured.update(kwargs.get("json") or {})
        return fake_response

    mock_aclient.post = capture_post

    with patch.object(app_module.httpx, "AsyncClient", return_value=mock_aclient):
        await app_module._route_via_direct(
            model="test-model",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            stream=False,
            direct_url="http://fake:8000",
            params={"temperature": 0.7, "max_tokens": 100, "tools": _SAMPLE_TOOLS, "tool_choice": "auto"},
        )

    assert captured.get("tools") == _SAMPLE_TOOLS
    assert captured.get("tool_choice") == "auto"


@pytest.mark.asyncio
async def test_route_via_direct_omits_tools_when_none(mock_globus):
    """_route_via_direct must not send tools/tool_choice keys when they are None."""
    import hpc_as_api.app as app_module

    captured: dict = {}

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
    }

    mock_aclient = AsyncMock()
    mock_aclient.__aenter__ = AsyncMock(return_value=mock_aclient)
    mock_aclient.__aexit__ = AsyncMock(return_value=None)

    async def capture_post(url, **kwargs):
        captured.update(kwargs.get("json") or {})
        return fake_response

    mock_aclient.post = capture_post

    with patch.object(app_module.httpx, "AsyncClient", return_value=mock_aclient):
        await app_module._route_via_direct(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            direct_url="http://fake:8000",
            params={"temperature": 0.7, "max_tokens": 100},
        )

    assert "tools" not in captured
    assert "tool_choice" not in captured


def test_tools_forwarded_to_globus_submit(mock_globus):
    """tools and tool_choice must reach submit_streaming_inference on the Globus path."""
    import asyncio

    from hpc_as_api.app import _route_via_globus_compute_streaming
    from hpc_as_api.auth import CallerIdentity

    caller = CallerIdentity(name="svc", auth_mode="api_key", globus_token=None)
    mock_client = AsyncMock()
    mock_client.submit_streaming_inference = AsyncMock(
        return_value={"channel_id": "cccccccc-0000-0000-0000-000000000000"}
    )

    with patch("hpc_as_api.app.ws_connect", side_effect=Exception("no relay in tests")):
        try:
            asyncio.get_event_loop().run_until_complete(
                _route_via_globus_compute_streaming(
                    model="m",
                    messages=[{"role": "user", "content": "hi"}],
                    caller=caller,
                    client=mock_client,
                    relay_url="ws://fake",
                    relay_secret="",
                    relay_enc_key="",
                    params={"tools": _SAMPLE_TOOLS, "tool_choice": "required"},
                )
            )
        except Exception:
            pass

    kw = mock_client.submit_streaming_inference.call_args.kwargs
    params = kw.get("params") or {}
    assert params.get("tools") == _SAMPLE_TOOLS
    assert params.get("tool_choice") == "required"


# ---------------------------------------------------------------------------
# validate_messages — tool-calling message shapes must be accepted
# ---------------------------------------------------------------------------


def test_validate_messages_accepts_tool_role(mock_globus):
    """A role:'tool' result message must pass validation (required for multi-turn)."""
    from hpc_as_api.auth import validate_messages

    messages = [
        {"role": "user", "content": "Weather in Berlin?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp":"9C"}'},
    ]
    # Must not raise and must preserve the messages (incl. tool_calls/tool_call_id) verbatim.
    assert validate_messages(messages) == messages


def test_validate_messages_allows_null_content_with_tool_calls(mock_globus):
    """An assistant tool-call turn has content=null; that must be allowed."""
    from hpc_as_api.auth import validate_messages

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"}}],
        },
    ]
    assert validate_messages(messages) == messages


def test_validate_messages_rejects_null_content_without_tool_calls(mock_globus):
    """content=null on a normal message is still an error."""
    from fastapi import HTTPException

    from hpc_as_api.auth import validate_messages

    with pytest.raises(HTTPException) as exc:
        validate_messages([{"role": "user", "content": None}])
    assert exc.value.status_code == 400


def test_validate_messages_still_rejects_unknown_role(mock_globus):
    """Genuinely invalid roles must still be rejected."""
    from fastapi import HTTPException

    from hpc_as_api.auth import validate_messages

    with pytest.raises(HTTPException) as exc:
        validate_messages([{"role": "robot", "content": "hi"}])
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Streaming relay: raw-chunk passthrough
# ---------------------------------------------------------------------------


class _FakeRelayConsumerWS:
    """Async relay consumer WebSocket that replays a fixed list of messages."""

    def __init__(self, messages):
        self._msgs = iter(messages)

    async def send(self, data):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._msgs)
        except StopIteration:
            raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass


async def _collect_sse(relay_messages):
    """Drive _route_via_globus_compute_streaming with the given relay messages, return SSE text."""
    from hpc_as_api.app import _route_via_globus_compute_streaming
    from hpc_as_api.auth import CallerIdentity

    caller = CallerIdentity(name="svc", auth_mode="api_key", globus_token=None)
    mock_client = AsyncMock()
    mock_client.submit_streaming_inference = AsyncMock(
        return_value={"channel_id": "dddddddd-0000-0000-0000-000000000000"}
    )

    sse_chunks: list = []
    with patch("hpc_as_api.app.ws_connect", return_value=_FakeRelayConsumerWS(relay_messages)):
        response = await _route_via_globus_compute_streaming(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            caller=caller,
            client=mock_client,
            relay_url="wss://fake",
            relay_secret="",
            relay_enc_key="",
            params={"max_tokens": 10},
        )
        async for chunk in response.body_iterator:
            if chunk.strip():
                sse_chunks.append(chunk)
    return "".join(sse_chunks)


def _sse_data_objects(sse_text):
    import json as json_mod

    return [json_mod.loads(ln[6:]) for ln in sse_text.splitlines() if ln.startswith("data:") and "[DONE]" not in ln]


def test_streaming_relays_chunk_verbatim(mock_globus):
    """A relay 'chunk' message must be re-emitted as the exact vLLM SSE chunk."""
    import asyncio
    import json as json_mod

    vllm_chunk = {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]}
    relay_messages = [
        json_mod.dumps({"type": "chunk", "data": vllm_chunk}),
        json_mod.dumps({"type": "done"}),
    ]

    sse_text = asyncio.get_event_loop().run_until_complete(_collect_sse(relay_messages))

    objs = _sse_data_objects(sse_text)
    assert objs == [vllm_chunk], "chunk must pass through byte-for-byte (as parsed JSON)"
    assert sse_text.rstrip().endswith("data: [DONE]")


def test_streaming_relays_tool_call_deltas(mock_globus):
    """Streaming tool-call deltas must survive the relay (the bug this refactor fixes)."""
    import asyncio
    import json as json_mod

    # vLLM streams a tool call as deltas with a tool_calls array, then a
    # finish_reason='tool_calls' chunk. The old relay dropped tool_calls entirely.
    tool_delta = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": '{"location":"Tokyo"}'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }
    finish_chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    relay_messages = [
        json_mod.dumps({"type": "chunk", "data": tool_delta}),
        json_mod.dumps({"type": "chunk", "data": finish_chunk}),
        json_mod.dumps({"type": "done"}),
    ]

    sse_text = asyncio.get_event_loop().run_until_complete(_collect_sse(relay_messages))
    objs = _sse_data_objects(sse_text)

    tool_calls = [tc for o in objs for c in o.get("choices", []) for tc in (c.get("delta", {}).get("tool_calls") or [])]
    assert tool_calls, "tool_calls deltas were dropped by the relay"
    assert tool_calls[0]["function"]["name"] == "get_weather"

    finish_reasons = [c.get("finish_reason") for o in objs for c in o.get("choices", []) if c.get("finish_reason")]
    assert finish_reasons[-1] == "tool_calls"


def test_streaming_relays_usage_chunk(mock_globus):
    """The include_usage final chunk (choices=[], cached_tokens) must pass through verbatim."""
    import asyncio
    import json as json_mod

    usage_chunk = {
        "choices": [],
        "usage": {
            "prompt_tokens": 600,
            "completion_tokens": 16,
            "total_tokens": 616,
            "prompt_tokens_details": {"cached_tokens": 576},
        },
    }
    relay_messages = [
        json_mod.dumps({"type": "chunk", "data": {"choices": [{"index": 0, "delta": {"content": "Hi"}}]}}),
        json_mod.dumps({"type": "chunk", "data": usage_chunk}),
        json_mod.dumps({"type": "done"}),
    ]

    sse_text = asyncio.get_event_loop().run_until_complete(_collect_sse(relay_messages))
    objs = _sse_data_objects(sse_text)

    usages = [o["usage"] for o in objs if o.get("usage")]
    assert usages, "usage chunk was dropped"
    assert usages[-1]["prompt_tokens_details"]["cached_tokens"] == 576


def test_streaming_upstream_error_emits_generic_event_with_ref(mock_globus):
    """An upstream relay error must become a generic OpenAI error event with a correlation ref."""
    import asyncio
    import json as json_mod

    secret_detail = "vLLM HTTP 500: torch CUDA OOM at /opt/secret/path/model.safetensors"
    relay_messages = [
        json_mod.dumps({"type": "error", "message": secret_detail}),
        json_mod.dumps({"type": "done"}),
    ]

    sse_text = asyncio.get_event_loop().run_until_complete(_collect_sse(relay_messages))
    objs = _sse_data_objects(sse_text)

    errors = [o["error"] for o in objs if o.get("error")]
    assert errors, "no error event emitted"
    err = errors[0]
    assert err["type"] == "upstream_error"
    assert err["message"] == "upstream inference error"  # generic, not the raw upstream text
    assert err["ref"] and len(err["ref"]) >= 8  # correlation nonce present
    assert secret_detail not in sse_text  # internal detail must NOT leak to the client
    assert sse_text.rstrip().endswith("data: [DONE]")
