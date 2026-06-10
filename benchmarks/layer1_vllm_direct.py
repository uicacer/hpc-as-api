"""
Layer 1: vLLM direct benchmark
Run FROM the Lakeshore login node (ghi2-002 is reachable on the internal LAN).

Tests the raw H100 + vLLM ceiling with no Globus Compute or relay overhead.
Sweeps concurrency levels and records TTFT, total latency, tokens/s per user.

Usage:
    python3 layer1_vllm_direct.py [--url http://ghi2-002:8000] [--output results/layer1.json]
"""

import argparse
import asyncio
import json
import statistics
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict

PROMPT = "Explain what a GPU is in exactly three sentences."
MODEL = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
MAX_TOKENS = 80
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64]
WARMUP_REQUESTS = 2


@dataclass
class RequestResult:
    concurrency: int
    user_id: int
    ttft_s: float        # time-to-first-token
    total_s: float       # wall time until [DONE]
    tokens: int
    tokens_per_s: float
    error: str = ""


def _build_payload():
    return json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "temperature": 0.0,
    }).encode()


def _count_tokens_in_chunk(chunk: bytes) -> int:
    """Count token fragments in one SSE chunk (multiple data: lines possible)."""
    count = 0
    for line in chunk.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            continue
        try:
            obj = json.loads(payload)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                count += 1
        except Exception:
            pass
    return count


async def _single_request(url: str, concurrency: int, user_id: int) -> RequestResult:
    """Fire one streaming request, measure TTFT and total latency."""
    import socket

    payload = _build_payload()
    t_start = time.perf_counter()
    ttft = None
    tokens = 0
    error = ""

    try:
        loop = asyncio.get_event_loop()

        def _do_request():
            nonlocal ttft, tokens
            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                for chunk in resp:
                    if chunk:
                        if ttft is None:
                            ttft = time.perf_counter() - t_start
                        tokens += _count_tokens_in_chunk(chunk)

        await loop.run_in_executor(None, _do_request)

    except Exception as exc:
        error = str(exc)[:120]

    total = time.perf_counter() - t_start
    ttft = ttft or total
    tps = tokens / total if total > 0 else 0.0

    return RequestResult(
        concurrency=concurrency,
        user_id=user_id,
        ttft_s=round(ttft, 3),
        total_s=round(total, 3),
        tokens=tokens,
        tokens_per_s=round(tps, 2),
        error=error,
    )


async def _run_level(url: str, concurrency: int) -> list[RequestResult]:
    tasks = [_single_request(url, concurrency, i) for i in range(concurrency)]
    return await asyncio.gather(*tasks)


def _print_summary(level: int, results: list[RequestResult]):
    ok = [r for r in results if not r.error]
    errors = [r for r in results if r.error]
    if not ok:
        print(f"  concurrency={level:3d}  ALL FAILED: {errors[0].error[:60]}")
        return
    ttfts = [r.ttft_s for r in ok]
    totals = [r.total_s for r in ok]
    tps_agg = sum(r.tokens for r in ok) / max(r.total_s for r in ok) if ok else 0
    print(
        f"  concurrency={level:3d}  "
        f"TTFT p50={statistics.median(ttfts):.2f}s p95={sorted(ttfts)[int(len(ttfts)*0.95)]:.2f}s  "
        f"total p50={statistics.median(totals):.2f}s  "
        f"agg_tok/s={tps_agg:.1f}  "
        f"errors={len(errors)}"
    )


async def main(url: str, output: str):
    print(f"\n=== Layer 1: vLLM direct  url={url} ===\n")

    # Sanity check
    try:
        urllib.request.urlopen(f"{url}/health", timeout=5)
        print("  vLLM /health: OK")
    except Exception as e:
        print(f"  WARNING: /health check failed: {e}  (continuing anyway)")

    # Warmup
    print(f"\n  Warmup ({WARMUP_REQUESTS} requests)...")
    await _run_level(url, WARMUP_REQUESTS)

    all_results: list[RequestResult] = []

    for level in CONCURRENCY_LEVELS:
        print(f"\n  → concurrency={level} ...")
        results = await _run_level(url, level)
        all_results.extend(results)
        _print_summary(level, results)
        # Brief pause between levels so vLLM drains
        await asyncio.sleep(2)

    # Save
    import os
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n  Results saved to {output}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://ghi2-002:8000")
    parser.add_argument("--output", default="results/layer1.json")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.output))
