# Changelog

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
