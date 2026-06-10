# Changelog

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
