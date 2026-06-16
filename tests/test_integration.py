"""
Integration tests against a real OpenAI-compatible inference server.

These tests run an hpc-as-api gateway in direct mode (USE_GLOBUS_COMPUTE=false)
pointed at a local LM Studio instance and verify end-to-end behaviour including
tool calling and streaming finish_reason.

Skip automatically when LM Studio is not reachable.

Run manually:
    uv run --extra dev pytest tests/test_integration.py -v

Tool-calling tests additionally require the loaded model to support tool-call
rendering via its chat template. They are skipped automatically if the model
rejects tool payloads with a template error.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Detect LM Studio and model capabilities
# ---------------------------------------------------------------------------

LM_STUDIO_URL = "http://localhost:1234"
LM_STUDIO_MODEL = "google/gemma-4-12b"

# Gemma 4 reasoning tokens count against max_tokens, so we need generous budgets.
# 27+ tokens can be consumed by <think> before any content appears.
REASONING_OVERHEAD = 512

try:
    _probe = httpx.get(f"{LM_STUDIO_URL}/v1/models", timeout=2)
    _model_ids = [m["id"] for m in _probe.json().get("data", [])]
    LM_STUDIO_AVAILABLE = LM_STUDIO_MODEL in _model_ids
except Exception:
    LM_STUDIO_AVAILABLE = False

# Probe whether tool calling works with the loaded model/template.
TOOL_CALLING_AVAILABLE = False
if LM_STUDIO_AVAILABLE:
    try:
        _tc_probe = httpx.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json={
                "model": LM_STUDIO_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}],
                "tool_choice": "auto",
                "max_tokens": 5,
            },
            timeout=15,
        )
        TOOL_CALLING_AVAILABLE = _tc_probe.status_code == 200
    except Exception:
        pass

lm_studio_only = pytest.mark.skipif(
    not LM_STUDIO_AVAILABLE,
    reason=f"LM Studio not running or model '{LM_STUDIO_MODEL}' not loaded",
)
tool_calling_only = pytest.mark.skipif(
    not TOOL_CALLING_AVAILABLE,
    reason=f"Model '{LM_STUDIO_MODEL}' does not support tool calling with its current chat template",
)

# ---------------------------------------------------------------------------
# Tools definition shared across tests
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Shared fixture: gateway in direct mode → LM Studio
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gateway():
    from hpc_as_api.app import make_app
    from hpc_as_api.auth import AuthConfig, Authenticator

    auth = Authenticator(
        AuthConfig(
            globus_client_id="",
            globus_client_secret="",
            allowed_domains=[],
            api_keys={"integration-test-key": "integration-test"},
        )
    )
    app = make_app(
        use_globus_compute=False,
        models={LM_STUDIO_MODEL: {"hf_name": LM_STUDIO_MODEL, "url": LM_STUDIO_URL}},
        direct_vllm_url=LM_STUDIO_URL,
        auth=auth,
    )
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


def _auth():
    return {"Authorization": "Bearer integration-test-key"}


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


@lm_studio_only
def test_health(gateway):
    resp = gateway.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@lm_studio_only
def test_basic_completion(gateway):
    """Non-streaming completion returns a message with a valid finish_reason."""
    resp = gateway.post(
        "/v1/chat/completions",
        headers=_auth(),
        json={
            "model": LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly the word: PONG"}],
            "max_tokens": REASONING_OVERHEAD + 20,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    choice = body["choices"][0]
    # Accept "stop" or "length" — reasoning tokens eat into the budget
    assert choice["finish_reason"] in ("stop", "length")
    assert body["choices"][0]["message"]["role"] == "assistant"


@lm_studio_only
def test_reasoning_content_present(gateway):
    """Gemma 4's reasoning_content should appear in the non-streaming response."""
    resp = gateway.post(
        "/v1/chat/completions",
        headers=_auth(),
        json={
            "model": LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "max_tokens": REASONING_OVERHEAD + 20,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    msg = body["choices"][0]["message"]
    # LM Studio surfaces reasoning tokens in the reasoning_content field
    assert "reasoning_content" in msg, "Expected reasoning_content from Gemma 4"
    assert msg["reasoning_content"]  # non-empty


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@lm_studio_only
def test_streaming_delivers_tokens_and_finish_reason(gateway):
    """Streaming response must deliver content or reasoning tokens and a finish_reason."""
    with gateway.stream(
        "POST",
        "/v1/chat/completions",
        headers=_auth(),
        json={
            "model": LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "Count to three."}],
            "max_tokens": REASONING_OVERHEAD + 50,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        any_tokens = False
        finish_reasons = []
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    any_tokens = True
                if choice.get("finish_reason"):
                    finish_reasons.append(choice["finish_reason"])

    assert any_tokens, "No tokens (content or reasoning) received"
    assert finish_reasons, "No finish_reason in stream"
    assert finish_reasons[-1] in ("stop", "length")


# ---------------------------------------------------------------------------
# Tool calling — skipped if the model's chat template doesn't support tools
# ---------------------------------------------------------------------------


@lm_studio_only
@tool_calling_only
def test_tool_choice_required_returns_tool_call(gateway):
    """With tool_choice='required', the model must issue at least one tool call."""
    resp = gateway.post(
        "/v1/chat/completions",
        headers=_auth(),
        json={
            "model": LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "What's the weather in Chicago?"}],
            "tools": _TOOLS,
            "tool_choice": "required",
            "max_tokens": REASONING_OVERHEAD + 256,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls", f"got: {choice['finish_reason']}"
    calls = choice["message"].get("tool_calls") or []
    assert len(calls) >= 1, "Expected at least one tool call"
    tc = calls[0]
    assert tc["function"]["name"] in ("get_weather", "calculate")
    args = json.loads(tc["function"]["arguments"])
    assert isinstance(args, dict)


@lm_studio_only
@tool_calling_only
def test_tool_choice_auto_calls_tool_for_weather(gateway):
    """With tool_choice='auto', a weather question should trigger get_weather."""
    resp = gateway.post(
        "/v1/chat/completions",
        headers=_auth(),
        json={
            "model": LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "What is the current weather in Paris?"}],
            "tools": _TOOLS,
            "tool_choice": "auto",
            "max_tokens": REASONING_OVERHEAD + 256,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"].get("tool_calls") or []
    assert any(c["function"]["name"] == "get_weather" for c in calls)


@lm_studio_only
@tool_calling_only
def test_tool_choice_auto_no_call_for_joke(gateway):
    """With tool_choice='auto', a joke request should NOT trigger any tool call."""
    resp = gateway.post(
        "/v1/chat/completions",
        headers=_auth(),
        json={
            "model": LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "Tell me a short joke."}],
            "tools": _TOOLS,
            "tool_choice": "auto",
            "max_tokens": REASONING_OVERHEAD + 128,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    calls = choice["message"].get("tool_calls") or []
    assert len(calls) == 0, f"Unexpected tool calls: {calls}"


@lm_studio_only
@tool_calling_only
def test_multi_turn_tool_use(gateway):
    """Inject a fake tool result and verify the model uses it in the follow-up."""
    resp1 = gateway.post(
        "/v1/chat/completions",
        headers=_auth(),
        json={
            "model": LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "What's the weather in Berlin? One sentence."}],
            "tools": _TOOLS,
            "tool_choice": "required",
            "max_tokens": REASONING_OVERHEAD + 256,
        },
    )
    assert resp1.status_code == 200, resp1.text
    msg1 = resp1.json()["choices"][0]["message"]
    calls = msg1.get("tool_calls") or []
    assert calls, "Turn 1 must issue a tool call"

    tc = calls[0]
    messages = [
        {"role": "user", "content": "What's the weather in Berlin? One sentence."},
        msg1,
        {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps({"temperature": "7°C", "condition": "rainy"}),
        },
    ]

    resp2 = gateway.post(
        "/v1/chat/completions",
        headers=_auth(),
        json={
            "model": LM_STUDIO_MODEL,
            "messages": messages,
            "tools": _TOOLS,
            "max_tokens": REASONING_OVERHEAD + 128,
        },
    )
    assert resp2.status_code == 200, resp2.text
    body2 = resp2.json()
    choice2 = body2["choices"][0]
    assert choice2["finish_reason"] in ("stop", "length")
    content2 = choice2["message"].get("content") or ""
    assert any(kw in content2.lower() for kw in ["7", "rain", "cold", "cool", "berlin"]), (
        f"Response doesn't reference tool result: {content2!r}"
    )


@lm_studio_only
@tool_calling_only
def test_streaming_tool_call_finish_reason(gateway):
    """Streaming with tool_choice='required' must emit finish_reason='tool_calls'."""
    with gateway.stream(
        "POST",
        "/v1/chat/completions",
        headers=_auth(),
        json={
            "model": LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "tools": _TOOLS,
            "tool_choice": "required",
            "max_tokens": REASONING_OVERHEAD + 256,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        finish_reasons = []
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            for choice in chunk.get("choices", []):
                if choice.get("finish_reason"):
                    finish_reasons.append(choice["finish_reason"])

    assert finish_reasons, "No finish_reason in stream"
    assert finish_reasons[-1] == "tool_calls", f"got: {finish_reasons[-1]}"


# ---------------------------------------------------------------------------
# Verify tools ARE forwarded even when the model rejects them (template issue)
# This test runs regardless of TOOL_CALLING_AVAILABLE and confirms the proxy
# forwards tools by observing LM Studio's response (200 with tool routing, or
# 400 with a template error — both prove the payload reached the backend).
# ---------------------------------------------------------------------------


@lm_studio_only
def test_tools_reach_backend(gateway):
    """
    Verify tools are forwarded to the backend by LM Studio.

    If the model's chat template supports tools: expect a 200 with finish_reason
    'tool_calls' or 'stop'. If the template doesn't support tools: expect a 400
    with a jinja/template error. In either case, LM Studio must have received
    the tools parameter — a plain 200 with finish_reason='stop' and NO tool
    mentions in the error would indicate tools were silently dropped.
    """
    resp = gateway.post(
        "/v1/chat/completions",
        headers=_auth(),
        json={
            "model": LM_STUDIO_MODEL,
            "messages": [{"role": "user", "content": "What's the weather in Chicago?"}],
            "tools": _TOOLS,
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    if resp.status_code == 200:
        # Template supports tools — must have made a tool call
        assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"
    else:
        # Template error proves tools reached the backend
        assert resp.status_code == 400
        detail = resp.json().get("detail", "")
        assert "template" in detail.lower() or "jinja" in detail.lower() or "tool" in detail.lower(), (
            f"Unexpected 400 error (not a template issue): {detail}"
        )
