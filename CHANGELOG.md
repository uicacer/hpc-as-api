# Changelog

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
