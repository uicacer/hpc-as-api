"""Tests for hpc_as_api.app — FastAPI routes, factory independence, and auth wiring."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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
    auth = Authenticator(AuthConfig(
        globus_client_id="",
        globus_client_secret="",
        allowed_domains=[],
        api_keys={"testkey": "test-service"},
    ))
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
    from hpc_as_api.presets.openai import create_openai_app
    from hpc_as_api.auth import AuthConfig

    auth = AuthConfig(globus_client_id="", globus_client_secret="", allowed_domains=[],
                      api_keys={"k": "svc"})

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
        models_a = {m["gateway_name"] for m in ca.get("/v1/models", headers={"Authorization": "Bearer k"}).json()["data"]}
        models_b = {m["gateway_name"] for m in cb.get("/v1/models", headers={"Authorization": "Bearer k"}).json()["data"]}

    assert "model-a" in models_a
    assert "model-b" not in models_a
    assert "model-b" in models_b
    assert "model-a" not in models_b


# ---------------------------------------------------------------------------
# /v1/chat/completions — direct mode
# ---------------------------------------------------------------------------

def test_chat_completions_requires_auth(mock_globus):
    client = _direct_client(mock_globus)
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code in (401, 403, 422)


def test_chat_completions_direct_mode_calls_vllm(mock_globus):
    """In direct mode, the gateway should proxy the request to vLLM."""
    import httpx
    from hpc_as_api.app import make_app
    from hpc_as_api.auth import AuthConfig, Authenticator

    auth = Authenticator(AuthConfig(
        globus_client_id="", globus_client_secret="",
        allowed_domains=[], api_keys={"mykey": "test-service"},
    ))
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
                    model="m", messages=[{"role": "user", "content": "hi"}],
                    temperature=0.7, max_tokens=10, caller=caller,
                    client=mock_client, relay_url="ws://fake", relay_secret="", relay_enc_key="",
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
                    model="m", messages=[{"role": "user", "content": "hi"}],
                    temperature=0.7, max_tokens=10, caller=caller,
                    client=mock_client, relay_url="ws://fake", relay_secret="", relay_enc_key="",
                )
            )
        except Exception:
            pass

    call_kwargs = mock_client.submit_streaming_inference.call_args.kwargs
    assert call_kwargs.get("globus_token") is None
