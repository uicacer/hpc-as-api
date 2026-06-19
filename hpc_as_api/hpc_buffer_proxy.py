#!/usr/bin/env python3
"""
HPC Buffer Proxy — runs on the compute node, between the SSH tunnel and vLLM.

Why this exists:
  The SSH tunnel is a TCP connection between the HPC node and the relay VM.
  If that connection drops (relay restart, network blip), vLLM would normally
  see a disconnected client and cancel the running inference. This proxy
  sits between them: it holds an open connection to vLLM regardless of what
  the tunnel does, and buffers every SSE token with a sequence number so the
  relay can replay any tokens it missed after reconnecting.

Flow:
  Relay → SSH tunnel → this proxy (port 8002) → vLLM (port 8001)

On reconnect, relay sends X-Resume-Job: <job_id>:<last_seq> and the proxy
replays tokens from that offset, then continues live. The relay holds the
user's browser connection open the whole time — zero interruption.

Pure stdlib, Python 3.9+ compatible (runs inside SLURM job without any install).
"""

import http.server
import json
import logging
import os
import socketserver
import threading
import time
import urllib.request
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [hpc-buffer-proxy] %(levelname)s %(message)s",
)
log = logging.getLogger("hpc-buffer-proxy")

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
LISTEN_PORT = int(os.environ.get("BUFFER_PROXY_PORT", "8002"))
# How long to keep a completed/errored job's buffer in memory (seconds).
# Covers the case where relay reconnects just after [DONE] was sent.
JOB_TTL = int(os.environ.get("BUFFER_JOB_TTL", "300"))

# Global job registry  {job_id: JobBuffer}
_jobs: dict[str, "JobBuffer"] = {}
_jobs_lock = threading.Lock()


class JobBuffer:
    """
    Holds state for one streaming inference job.
    - tokens: deque of (seq, raw_bytes) pairs, kept in memory until TTL expires
    - finished: set when vLLM sends [DONE] or an error
    - error: set if something went wrong
    - waiters: asyncio Events for any relay connections blocked waiting for new tokens
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.tokens: list[tuple[int, bytes]] = []  # (seq, chunk_bytes)
        self.finished = threading.Event()
        self.error: str | None = None
        self.lock = threading.Lock()
        self._new_token = threading.Condition(self.lock)
        self.created_at = time.monotonic()

    def append(self, chunk: bytes):
        with self._new_token:
            seq = len(self.tokens)
            self.tokens.append((seq, chunk))
            self._new_token.notify_all()

    def mark_done(self, error: str | None = None):
        with self._new_token:
            self.error = error
            self.finished.set()
            self._new_token.notify_all()

    def iter_from(self, start_seq: int, timeout: float = 120.0):
        """
        Generator: yield (seq, chunk) starting from start_seq.
        Blocks until new tokens arrive or job finishes.
        Raises StopIteration when done.
        """
        idx = start_seq
        deadline = time.monotonic() + timeout
        while True:
            with self._new_token:
                while idx >= len(self.tokens) and not self.finished.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("job stream timed out")
                    self._new_token.wait(timeout=min(remaining, 1.0))

                # Drain whatever tokens arrived
                while idx < len(self.tokens):
                    seq, chunk = self.tokens[idx]
                    yield seq, chunk
                    idx += 1

                if self.finished.is_set() and idx >= len(self.tokens):
                    return


def _reap_old_jobs():
    """Background thread: remove jobs older than JOB_TTL seconds."""
    while True:
        time.sleep(60)
        now = time.monotonic()
        with _jobs_lock:
            expired = [jid for jid, j in _jobs.items() if j.finished.is_set() and (now - j.created_at) > JOB_TTL]
            for jid in expired:
                del _jobs[jid]
                log.info(f"Reaped job {jid[:8]}")


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(fmt % args)

    # ── /health ──────────────────────────────────────────────────────────────
    def _handle_health(self):
        try:
            with urllib.request.urlopen(f"{VLLM_URL}/health", timeout=2) as r:  # noqa: S310  # nosec B310
                vllm_ok = r.status == 200
        except Exception:
            vllm_ok = False

        body = json.dumps({"status": "ok", "vllm_healthy": vllm_ok, "active_jobs": len(_jobs)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── /v1/chat/completions ─────────────────────────────────────────────────
    def _handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length)

        try:
            body = json.loads(raw_body)
        except Exception:
            self._error(400, "invalid json")
            return

        stream = body.get("stream", False)
        job_id = str(uuid.uuid4())

        # Check if relay is resuming a previous job
        resume_header = self.headers.get("X-Resume-Job", "")
        resume_job_id = None
        resume_seq = 0
        if resume_header and ":" in resume_header:
            parts = resume_header.split(":", 1)
            resume_job_id = parts[0].strip()
            try:
                resume_seq = int(parts[1].strip())
            except ValueError:
                resume_seq = 0

            with _jobs_lock:
                existing = _jobs.get(resume_job_id)

            if existing is not None:
                log.info(f"Resuming job {resume_job_id[:8]} from seq {resume_seq}")
                self._stream_job(existing, resume_seq)
                return
            else:
                log.warning(f"Resume requested for unknown job {resume_job_id[:8]}, starting fresh")

        if not stream:
            # Non-streaming: just forward directly, no buffering needed
            self._forward_non_stream(raw_body)
            return

        # Start a new streaming job
        job = JobBuffer(job_id)
        with _jobs_lock:
            _jobs[job_id] = job

        log.info(f"New job {job_id[:8]}")

        # Fire off vLLM call in background thread
        threading.Thread(target=self._run_vllm_job, args=(job, raw_body), daemon=True).start()

        # Stream back to relay, seq 0 onwards
        self._stream_job(job, start_seq=0)

    def _run_vllm_job(self, job: JobBuffer, body: bytes):
        """Background thread: POST to vLLM, buffer every SSE chunk."""
        url = f"{VLLM_URL}/v1/chat/completions"
        req = urllib.request.Request(  # noqa: S310  # nosec B310
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310  # nosec B310
                if resp.status != 200:
                    err_body = resp.read().decode(errors="replace")
                    job.mark_done(error=f"vLLM {resp.status}: {err_body[:200]}")
                    return
                # Read SSE line by line; accumulate into proper chunk boundaries.
                # vLLM emits:  "data: {...}\n\n"
                # We store each complete "data: ...\n\n" unit as one token.
                buf = b""
                for raw_line in resp:
                    buf += raw_line
                    if buf.endswith(b"\n\n"):
                        job.append(buf)
                        buf = b""
                if buf:
                    job.append(buf)
        except Exception as e:
            log.error(f"vLLM call failed for job {job.job_id[:8]}: {e}")
            job.mark_done(error=str(e))
            return

        job.mark_done()
        log.info(f"Job {job.job_id[:8]} complete ({len(job.tokens)} chunks)")

    def _stream_job(self, job: JobBuffer, start_seq: int):
        """Send buffered + live tokens to the relay connection."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Job-ID", job.job_id)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            for _seq, chunk in job.iter_from(start_seq):
                # HTTP chunked transfer
                hex_len = format(len(chunk), "x").encode()
                self.wfile.write(hex_len + b"\r\n" + chunk + b"\r\n")
            # Final chunk
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Relay disconnected (tunnel drop). Job keeps buffering in background.
            log.info(f"Relay disconnected from job {job.job_id[:8]} at seq ~{start_seq}")
        except TimeoutError:
            log.warning(f"Job {job.job_id[:8]} stream timed out")
        except Exception as e:
            log.error(f"Stream error for job {job.job_id[:8]}: {e}")

    def _forward_non_stream(self, body: bytes):
        """Pass-through for non-streaming requests."""
        url = f"{VLLM_URL}/v1/chat/completions"
        req = urllib.request.Request(  # noqa: S310  # nosec B310
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310  # nosec B310
                resp_body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except Exception as e:
            self._error(503, str(e))

    def _error(self, code: int, msg: str):
        body = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._handle_health()
        else:
            self._error(404, "not found")

    def do_POST(self):
        if self.path.startswith("/v1/chat/completions"):
            self._handle_chat()
        else:
            self._error(404, "not found")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    threading.Thread(target=_reap_old_jobs, daemon=True).start()
    server = ThreadedHTTPServer(("127.0.0.1", LISTEN_PORT), ProxyHandler)
    log.info(f"Buffer proxy listening on 127.0.0.1:{LISTEN_PORT} → vLLM at {VLLM_URL}")
    server.serve_forever()


if __name__ == "__main__":
    main()
