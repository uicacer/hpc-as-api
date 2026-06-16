"""Tests for GlobusComputeClient — config, model resolution, payload size check."""

import json

# GlobusComputeClient imports globus_compute_sdk at module level — mock it before import
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_globus_modules(monkeypatch):
    """
    Patch globus_compute_sdk imports so tests run without the Globus SDK installed.
    Only the package-level symbols used in compute.py are stubbed.
    """
    fake_sdk = MagicMock()
    fake_sdk.Executor = MagicMock
    fake_sdk.errors.error_types.DeserializationError = Exception
    fake_sdk.errors.error_types.TaskExecutionFailed = Exception
    fake_sdk.serialize.AllCodeStrategies = MagicMock
    fake_sdk.serialize.ComputeSerializer = MagicMock

    fake_globus_sdk = MagicMock()
    fake_globus_sdk.GlobusAPIError = Exception
    fake_globus_sdk.login_flows.command_line_login_flow_manager.CommandLineLoginFlowEOFError = Exception

    monkeypatch.setitem(__import__("sys").modules, "globus_compute_sdk", fake_sdk)
    monkeypatch.setitem(__import__("sys").modules, "globus_compute_sdk.errors", fake_sdk.errors)
    monkeypatch.setitem(
        __import__("sys").modules,
        "globus_compute_sdk.errors.error_types",
        fake_sdk.errors.error_types,
    )
    monkeypatch.setitem(__import__("sys").modules, "globus_compute_sdk.serialize", fake_sdk.serialize)
    monkeypatch.setitem(__import__("sys").modules, "globus_sdk", fake_globus_sdk)
    monkeypatch.setitem(
        __import__("sys").modules,
        "globus_sdk.authorizers",
        fake_globus_sdk.authorizers,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "globus_sdk.login_flows",
        fake_globus_sdk.login_flows,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "globus_sdk.login_flows.command_line_login_flow_manager",
        fake_globus_sdk.login_flows.command_line_login_flow_manager,
    )
    return fake_sdk


def make_client(mock_globus_modules, **kwargs):
    """Import GlobusComputeClient after mocks are in place and instantiate it."""
    # Force re-import in case a cached version is already in sys.modules
    import sys

    sys.modules.pop("hpc_as_api.compute", None)
    from hpc_as_api.compute import GlobusComputeClient

    return GlobusComputeClient(**kwargs)


# ---------------------------------------------------------------------------
# Constructor / config resolution
# ---------------------------------------------------------------------------


def test_endpoint_from_arg(mock_globus_modules):
    client = make_client(mock_globus_modules, endpoint_id="test-uuid-123")
    assert client.endpoint_id == "test-uuid-123"


def test_endpoint_from_env(mock_globus_modules, monkeypatch):
    monkeypatch.setenv("GLOBUS_COMPUTE_ENDPOINT_ID", "env-uuid-456")
    client = make_client(mock_globus_modules)
    assert client.endpoint_id == "env-uuid-456"


def test_models_from_arg(mock_globus_modules):
    models = {"mymodel": {"hf_name": "org/Model", "url": "http://node:8000"}}
    client = make_client(mock_globus_modules, endpoint_id="x", models=models)
    assert client.models == models


def test_models_from_env(mock_globus_modules, monkeypatch):
    models = {"m1": {"hf_name": "org/M1", "url": "http://node:8000"}}
    monkeypatch.setenv("HPC_MODELS", json.dumps(models))
    client = make_client(mock_globus_modules, endpoint_id="x")
    assert client.models == models


def test_models_invalid_json_env(mock_globus_modules, monkeypatch):
    monkeypatch.setenv("HPC_MODELS", "not-json")
    client = make_client(mock_globus_modules, endpoint_id="x")
    assert client.models == {}


def test_is_available_true(mock_globus_modules):
    client = make_client(mock_globus_modules, endpoint_id="some-id")
    assert client.is_available() is True


def test_is_available_false(mock_globus_modules):
    client = make_client(mock_globus_modules, endpoint_id=None)
    assert client.is_available() is False


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------


def test_resolve_known_model(mock_globus_modules):
    models = {
        "qwen72b": {
            "hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
            "url": "http://ghi2-002:8000",
            "context_reserve_output": 4096,
        }
    }
    client = make_client(mock_globus_modules, endpoint_id="x", models=models)
    hf_name, url, max_tok = client._resolve_model("qwen72b")
    assert hf_name == "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    assert url == "http://ghi2-002:8000"
    assert max_tok == 4096


def test_resolve_unknown_model_uses_name_directly(mock_globus_modules, monkeypatch):
    monkeypatch.setenv("HPC_VLLM_URL", "http://fallback:8000")
    client = make_client(mock_globus_modules, endpoint_id="x", models={})
    hf_name, url, max_tok = client._resolve_model("some/Unknown-Model")
    assert hf_name == "some/Unknown-Model"
    assert url == "http://fallback:8000"
    assert max_tok == 2048


# ---------------------------------------------------------------------------
# _estimate_payload_size
# ---------------------------------------------------------------------------


def test_estimate_payload_size(mock_globus_modules):
    client = make_client(mock_globus_modules, endpoint_id="x")
    messages = [{"role": "user", "content": "Hello world"}]
    size = client._estimate_payload_size(messages)
    assert size > 0
    assert size < 1024  # small message, definitely under 1 KB


# ---------------------------------------------------------------------------
# remote_vllm_streaming: usage capture from include_usage chunk
# ---------------------------------------------------------------------------


class _FakeRelayWS:
    """Records messages sent by the producer; no real network."""

    def __init__(self):
        self.sent: list[str] = []

    def send(self, raw):
        self.sent.append(raw)

    def close(self):
        pass


class _FakeVLLMResponse:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200
        self.text = ""

    def iter_lines(self, decode_unicode=True):
        yield from self._lines


def test_streaming_relays_usage_with_cached_tokens(mock_globus_modules, monkeypatch):
    """
    vLLM emits its usage block (incl. prompt_tokens_details.cached_tokens) in a
    final SSE chunk whose ``choices`` is empty. The relay producer must capture
    that block and forward it in the ``done`` message — the empty-choices
    short-circuit must not drop it.
    """
    from hpc_as_api.compute import remote_vllm_streaming

    # No encryption → messages are sent as plaintext JSON we can parse.
    monkeypatch.delenv("RELAY_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("RELAY_SECRET", raising=False)

    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        # include_usage final chunk: empty choices, full usage block
        'data: {"choices":[],"usage":{"prompt_tokens":600,"completion_tokens":16,'
        '"total_tokens":616,"prompt_tokens_details":{"cached_tokens":576}}}',
        "data: [DONE]",
    ]

    fake_ws = _FakeRelayWS()

    # remote_vllm_streaming does `import requests` and
    # `from websockets.sync.client import connect` internally — neither is a
    # local dependency (they live on the HPC endpoint), so inject fakes.
    import sys
    import types

    fake_requests = types.ModuleType("requests")
    fake_requests.post = lambda *a, **k: _FakeVLLMResponse(sse_lines)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    fake_ws_client = types.ModuleType("websockets.sync.client")
    fake_ws_client.connect = lambda *a, **k: fake_ws
    fake_ws_pkg = types.ModuleType("websockets.sync")
    fake_ws_pkg.client = fake_ws_client
    fake_ws_root = types.ModuleType("websockets")
    fake_ws_root.sync = fake_ws_pkg
    monkeypatch.setitem(sys.modules, "websockets", fake_ws_root)
    monkeypatch.setitem(sys.modules, "websockets.sync", fake_ws_pkg)
    monkeypatch.setitem(sys.modules, "websockets.sync.client", fake_ws_client)

    result = remote_vllm_streaming(
        vllm_url="http://localhost:8000",
        model="gemma4-31b",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=16,
        relay_url="ws://localhost:9999",
        channel_id="chan-1",
    )

    assert result["ok"] is True

    done = [json.loads(m) for m in fake_ws.sent]
    done_msgs = [m for m in done if m.get("type") == "done"]
    assert done_msgs, "no done message sent"
    usage = done_msgs[-1].get("usage", {})
    assert usage.get("prompt_tokens_details", {}).get("cached_tokens") == 576
    assert done_msgs[-1].get("finish_reason") == "stop"
