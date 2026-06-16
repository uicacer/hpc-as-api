# Changelog

## 0.5.0 (2026-06-16)

### Transparent parameter & response passthrough

The proxy no longer maintains per-field allowlists for OpenAI request
parameters or streaming response fields. Previously, every new feature
(`tools`, `tool_choice`, `chat_template_kwargs`, …) required editing 3–4
signatures, and anything not explicitly enumerated — `top_p`, `seed`, `stop`,
`frequency_penalty`, `presence_penalty`, `response_format`, `logprobs`, `n`,
`logit_bias` — was silently dropped.

**Request side — forward everything except what the proxy owns.** The gateway
now captures all request-body parameters into a single `params` dict and
forwards them to vLLM verbatim. The proxy overrides only the three fields it
controls: `model` (gateway alias → backend name), `messages` (validated and
size-capped), and `stream` (selects the routing path). Adding a new sampling
knob now requires **zero code changes**.

**Response side — relay vLLM's SSE chunks verbatim.** The Globus Compute
streaming relay previously re-encoded each delta into a custom
`{"type": "token", content, reasoning_content}` schema and reconstructed an
OpenAI chunk on the consumer side. That normalization was an allowlist: it
**dropped streaming `tool_calls` deltas entirely** (a forced streaming tool
call arrived with `finish_reason=tool_calls` but no call name/arguments), and
also dropped `logprobs` and multi-choice (`n>1`) output. The relay now forwards
each vLLM chunk untouched as `{"type": "chunk", "data": <chunk>}`, and the
consumer re-emits it as-is. Tool calls, reasoning, `logprobs`, multiple
choices, and the `include_usage` final chunk (with
`prompt_tokens_details.cached_tokens`) all pass through with no per-field code.
This supersedes the empty-choices usage handling added in 0.4.1 — the usage
chunk now passes through like any other.

**Multi-turn tool calling unblocked in `validate_messages`.** The proxy's own
message validator rejected `role: "tool"` (allowed only user/assistant/system)
and rejected any message with `content: null`. Both shapes are required for
tool calling: a tool-result turn uses `role: "tool"`, and an assistant turn
that only issues tool calls has `content: null` with a `tool_calls` array. A
multi-turn tool exchange therefore failed at the gateway with HTTP 400
(`invalid role 'tool'`) **before ever reaching vLLM** — the `{"detail": …}`
FastAPI error envelope, not vLLM's. `validate_messages` now accepts the `tool`
and `developer` roles and allows `content: null` when `tool_calls` is present.
(This corrects an earlier mis-diagnosis that attributed the 400 to a server-side
chat-template; the model's tool-aware template was never the cause.)

**Breaking (internal API):** `GlobusComputeClient.submit_inference()` and
`submit_streaming_inference()` replace their `temperature` / `max_tokens` /
`chat_template_kwargs` / `tools` / `tool_choice` keyword arguments with a
single `params: dict`. The remote functions (`remote_vllm_inference`,
`remote_vllm_streaming`) take `params` instead of enumerated arguments. Public
top-level package API is unchanged.

**Server-side requirement (unchanged from 0.4.1):** `cached_tokens` reporting
still requires launching vLLM with `--enable-prompt-tokens-details`.

**Tests** — relay tests now assert verbatim chunk passthrough, including a
streaming tool-call-delta regression test, the `include_usage` chunk, and
arbitrary sampling params (`top_p`, `seed`, `stop`) reaching vLLM; plus
`validate_messages` tests for the `tool` role and null-content tool-call turns.
51 unit tests pass.

## 0.4.1 (2026-06-16)

### Fix: tool calling, streaming finish_reason, and direct-mode streaming lifecycle

**Tool calling passthrough** — `tools` and `tool_choice` were silently stripped
from every request. They are now forwarded on all three routing paths (direct
HTTP, Globus Compute non-streaming, Globus Compute streaming). The remote
functions serialised for Globus Compute (`_REMOTE_FN_SOURCE`,
`_REMOTE_STREAMING_FN_SOURCE`) also accept and forward both parameters to vLLM.

**Streaming `finish_reason`** — the SSE generator in the Globus Compute
streaming path only emitted `finish_reason` when vLLM also returned a `usage`
block; vLLM typically omits usage in streaming so clients never received
`finish_reason`. The final chunk is now always emitted with `finish_reason`
(defaulting to `"stop"`), with usage added when present.

**Streaming `usage` support** — the Globus Compute streaming remote function
now includes `"stream_options": {"include_usage": true}` in the vLLM payload
and passes the usage block through the relay `done` message.

**Streaming usage drop fix** — vLLM emits its usage block (including
`prompt_tokens_details.cached_tokens` when the server is launched with
`--enable-prompt-tokens-details`) in a final SSE chunk whose `choices` is empty.
The relay producer checked `if not choices: continue` *before* capturing usage,
so that chunk — and all usage stats — was silently dropped on the streaming
path. Usage is now captured before the empty-choices short-circuit, so
per-request prefix-cache stats reach the client.

**Direct-mode streaming lifecycle fix** — the `httpx.AsyncClient` and
`client.stream()` context were created outside the `StreamingResponse`
generator, causing the response stream to be closed before the generator
consumed it. Both are now scoped entirely inside `stream_generator()`.

**New tests** — 8 unit tests in `tests/test_app.py` cover tools forwarding and
`finish_reason` emission; a new test in `tests/test_compute.py` verifies the
streaming relay forwards the usage block (incl. `cached_tokens`) from vLLM's
empty-choices `include_usage` chunk.

## 0.3.4 (2026-06-10)

### Messaging: domain-agnostic positioning throughout

All public-facing materials now consistently present `hpc-as-api` as a
**domain-agnostic HTTP gateway for any HPC function** — not as an LLM-specific tool.

- **README**: Rewritten to lead with `HPCApp` and the general-purpose streaming
  gateway pattern. The simulation example appears before the LLM preset. "OpenAI-compatible
  gateway" is now framed as a built-in preset, not the product's identity.
- **paper/paper.md**: Title changed to *"A Domain-Agnostic HTTP Gateway for HPC Functions…"*.
  `HPCApp`, `make_app()`, and the framework architecture are now described as first-class
  subjects. The LLM preset is introduced as one application of the framework.
- **`hpc_as_api/__init__.py`**: Module docstring updated to lead with the domain-agnostic
  description and mark the LLM preset as "built-in application of the framework".
- **`pyproject.toml`**: Description updated to "Domain-agnostic HTTP gateway…"; keywords
  updated to add `domain-agnostic` and `scientific-computing`, remove `openai`.

No API or behavior changes.

## 0.3.3 (2026-06-10)

### Refactor: `make_app()` factory — multiple independent instances, no env-var injection

`app.py` now exposes a `make_app()` factory function.  Every call returns a
fresh, independent `FastAPI` instance that captures its configuration in closures
— there are no module-level globals involved in request handling.

**Before (0.3.x):** `create_openai_app()` in `presets/openai.py` worked by
injecting arguments into `os.environ` and then importing the cached module-level
singleton.  Calling it a second time with different arguments silently had no
effect.

**After:** `create_openai_app()` calls `make_app()` directly.  Each call returns
a genuinely independent app — two gateways with different endpoints, models, or
auth settings can live in the same process without interfering.

### Fix: per-user Globus token fully wired end-to-end

`app.py` now passes `caller.globus_token` to `submit_streaming_inference()` when
the caller authenticated with a Globus token.  In 0.3.1–0.3.2, `submit_streaming_inference()`
gained the `globus_token=` parameter but `app.py` never forwarded it.

### Fix: `HPCApp` payload size check

`HPCApp._add_route` now strips old images and enforces the Globus payload limit
before submitting any job with a `messages` field.  Previously only
`GlobusComputeClient.submit_inference()` did this check; the domain-agnostic
`HPCApp` path bypassed it entirely.

### Fix: version mismatch

`__init__.py.__version__` was stuck at `"0.2.0"` while `pyproject.toml` had
already advanced to `0.3.2`.  Both are now `"0.3.3"`.

### Fix: `globus_sdk.authorizers` mock missing in test_compute.py

The `mock_globus_modules` fixture now also stubs `globus_sdk.authorizers`, which
`compute.py` imports at module level via `AccessTokenAuthorizer`.

### Notes

- No breaking changes.  `app.py` still exposes a module-level `app` singleton
  (built from env vars) and a `router` for embedding; both continue to work.
- `make_app()` is now the recommended entry point for programmatic configuration.

## 0.3.2 (2026-06-04)

### New feature: programmatic auth configuration (`AuthConfig`)

All authentication settings are now configurable as Python arguments — no `.env`
file or environment variables required. Pass an `AuthConfig` instance to
`HPCApp` or `create_openai_app`:

```python
from hpc_as_api import AuthConfig
from hpc_as_api.presets.openai import create_openai_app

app = create_openai_app(
    endpoint_id="8d978809-...",
    models={"llama3": {"hf_name": "meta-llama/Llama-3-70b", "url": "http://gpu01:8000"}},
    relay_url="wss://relay.example.com",
    auth=AuthConfig(
        globus_client_id="your-client-id",
        globus_client_secret="your-client-secret",
        allowed_domains=["ornl.gov", "anl.gov"],   # [] = accept any Globus identity
        api_keys={"myservice": "sk-my-service-key"},
        rate_limit_requests=20,
        rate_limit_window=60,
    ),
)
```

Every `AuthConfig` field still falls back to its corresponding environment
variable when not supplied, so existing env-var-based deployments work without
any changes.

### Notes

- No breaking changes. `auth` defaults to `None`; existing code works without modification.
- `PROXY_ALLOWED_DOMAINS` now defaults to `""` (allow any valid Globus identity) instead
  of `"uic.edu"`. Deployments that relied on the old default should set
  `PROXY_ALLOWED_DOMAINS=uic.edu` explicitly.
- `AuthConfig` and `Authenticator` are now exported from the top-level package:
  `from hpc_as_api import AuthConfig`.

## 0.3.1 (2026-06-04)

### New feature: per-user Globus Compute job submission

`GlobusComputeClient.submit_streaming_inference()` now accepts an optional
`globus_token` parameter. When provided, the job is submitted under the
caller's own Globus identity rather than the client's stored credentials,
giving per-user SLURM-level attribution on the HPC cluster.

```python
result = await client.submit_streaming_inference(
    messages=messages,
    model="qwen25-vl-72b",
    relay_url="wss://relay.example.com",
    globus_token=caller_globus_access_token,  # optional
)
```

A short-lived `Executor` is created per request for token-authenticated calls
(the persistent executor is tied to stored credentials). The first request from
a Globus user pays the ~1.5 s AMQP setup cost; API-key callers are unaffected.

### Notes

- No breaking changes. `globus_token` defaults to `None`; existing code works without modification.
- When `globus_token` is `None`, behavior is identical to 0.3.0.

## 0.3.0 (2026-06-04)

### Security

- **Relay shared secret no longer travels via Globus Compute task arguments.**
  Previously, `RELAY_SECRET` was passed as a task argument when submitting the
  built-in `remote_vllm_streaming` function via Globus Compute. It is now read
  from `os.environ.get("RELAY_SECRET", "")` inside the remote function, matching
  the existing pattern for `RELAY_ENCRYPTION_KEY`. Neither credential traverses
  Globus Compute's AMQP channel anymore.

### Operational change required

Operators deploying the built-in vLLM preset must set `RELAY_SECRET` in the
Globus Compute endpoint's `worker_init` (in addition to `RELAY_ENCRYPTION_KEY`,
which was already required). Example:

```yaml
# ~/.globus_compute/<endpoint>/config.yaml
worker_init: |
  export RELAY_SECRET=<channel-access token>
  export RELAY_ENCRYPTION_KEY=<32-byte hex AES-256 key>
```

The submitter side (`app.py`) still reads `RELAY_SECRET` from its own
environment to authenticate its outbound consumer connection to the relay.

### Notes

- Public library API (`hpc_as_api.core.RelayProducer`) is unchanged. User-defined
  remote functions may still accept `relay_secret` as a parameter if they prefer.
  Only the built-in `remote_vllm_streaming` was updated.
- Requires `streamrelay >= 0.3.0` (unchanged).

## 0.2.0

- Initial public release.
