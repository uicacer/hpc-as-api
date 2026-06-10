"""
Layer 2: end-to-end gateway load + stress test
Hits the public hpc-as-api proxy through Globus Compute + WebSocket relay.

Two test modes:
  --mode load    : sweep concurrency 1→2→4→8→16→32, validate 300-user target
  --mode stress  : ramp until failure to find the breaking point

Usage (from your laptop or anywhere with internet):
    python3 layer2_gateway.py \\
        --url https://relay.stream.acer.uic.edu:8001 \\
        --api-key sk-... \\
        --mode load \\
        --output results/layer2_load.json

    python3 layer2_gateway.py \\
        --url https://relay.stream.acer.uic.edu:8001 \\
        --api-key sk-... \\
        --mode stress \\
        --output results/layer2_stress.json

Dependencies: Python 3.9+ stdlib only (uses urllib + asyncio).
"""

import argparse
import asyncio
import json
import statistics
import time
import urllib.request
import urllib.error
import ssl
import os
from dataclasses import dataclass, asdict

PROMPT = "Explain what a GPU is in exactly three sentences."
MODEL = "qwen25-vl-72b"
MAX_TOKENS = 80

# Load test: fixed concurrency levels
LOAD_LEVELS = [1, 2, 4, 8, 16, 32]

# Stress test: keep doubling until >50% error rate or timeout
STRESS_START = 8
STRESS_MAX = 512
STRESS_ERROR_THRESHOLD = 0.5   # stop when >= 50% of requests fail

# Target SLA for 300-student class
TARGET_USERS = 300
TARGET_TTFT_P95_S = 10.0       # p95 TTFT ≤ 10 s is acceptable in a class
TARGET_TOTAL_P95_S = 60.0      # p95 total latency ≤ 60 s


@dataclass
class RequestResult:
    concurrency: int
    user_id: int
    ttft_s: float
    total_s: float
    tokens: int
    tokens_per_s: float
    error: str = ""
    http_status: int = 200


def _build_payload() -> bytes:
    return json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "temperature": 0.0,
    }).encode()


def _count_tokens_sse(data: bytes) -> int:
    count = 0
    for line in data.split(b"\n"):
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


async def _single_request(
    url: str, api_key: str, concurrency: int, user_id: int
) -> RequestResult:
    payload = _build_payload()
    t_start = time.perf_counter()
    ttft = None
    tokens = 0
    error = ""
    http_status = 0
    ctx = ssl.create_default_context()

    try:
        loop = asyncio.get_event_loop()

        def _do():
            nonlocal ttft, tokens, http_status
            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180, context=ctx) as resp:
                http_status = resp.status
                for chunk in resp:
                    if chunk:
                        if ttft is None:
                            ttft = time.perf_counter() - t_start
                        tokens += _count_tokens_sse(chunk)

        await loop.run_in_executor(None, _do)

    except urllib.error.HTTPError as exc:
        error = f"HTTP {exc.code}: {exc.reason}"
        http_status = exc.code
    except Exception as exc:
        error = str(exc)[:120]

    total = time.perf_counter() - t_start
    ttft = ttft or total

    return RequestResult(
        concurrency=concurrency,
        user_id=user_id,
        ttft_s=round(ttft, 3),
        total_s=round(total, 3),
        tokens=tokens,
        tokens_per_s=round(tokens / total, 2) if total > 0 else 0.0,
        error=error,
        http_status=http_status,
    )


async def _run_level(url: str, api_key: str, concurrency: int) -> list[RequestResult]:
    tasks = [_single_request(url, api_key, concurrency, i) for i in range(concurrency)]
    return await asyncio.gather(*tasks)


def _summarize(level: int, results: list[RequestResult]) -> dict:
    ok = [r for r in results if not r.error]
    errors = [r for r in results if r.error]
    error_rate = len(errors) / len(results) if results else 1.0

    if not ok:
        return {
            "concurrency": level,
            "n_ok": 0,
            "n_errors": len(errors),
            "error_rate": 1.0,
            "error_sample": errors[0].error if errors else "",
        }

    ttfts = sorted(r.ttft_s for r in ok)
    totals = sorted(r.total_s for r in ok)
    p95_idx = max(0, int(len(ok) * 0.95) - 1)
    agg_tps = sum(r.tokens for r in ok) / max(r.total_s for r in ok)

    return {
        "concurrency": level,
        "n_ok": len(ok),
        "n_errors": len(errors),
        "error_rate": round(error_rate, 3),
        "ttft_p50": round(statistics.median(ttfts), 3),
        "ttft_p95": round(ttfts[p95_idx], 3),
        "ttft_max": round(ttfts[-1], 3),
        "total_p50": round(statistics.median(totals), 3),
        "total_p95": round(totals[p95_idx], 3),
        "total_max": round(totals[-1], 3),
        "agg_tokens_per_s": round(agg_tps, 1),
        "tokens_per_user_p50": round(statistics.median(r.tokens for r in ok), 1),
        "error_sample": errors[0].error if errors else "",
    }


def _print_summary(s: dict):
    if s["n_ok"] == 0:
        print(f"  c={s['concurrency']:4d}  ALL FAILED  {s.get('error_sample','')[:60]}")
        return
    print(
        f"  c={s['concurrency']:4d}  "
        f"TTFT p50={s['ttft_p50']:.1f}s p95={s['ttft_p95']:.1f}s  "
        f"total p50={s['total_p50']:.1f}s p95={s['total_p95']:.1f}s  "
        f"agg={s['agg_tokens_per_s']:.0f} tok/s  "
        f"err={s['n_errors']}/{s['n_ok']+s['n_errors']}"
    )


async def run_load(url: str, api_key: str, output: str):
    print(f"\n=== Layer 2 LOAD TEST  url={url} ===")
    print(f"    SLA target: {TARGET_USERS} users, TTFT p95 ≤ {TARGET_TTFT_P95_S}s, total p95 ≤ {TARGET_TOTAL_P95_S}s\n")

    # Warmup
    print("  Warmup (1 request)...")
    await _run_level(url, api_key, 1)
    print("  Warmup done.\n")

    summaries = []
    raw_results = []

    for level in LOAD_LEVELS:
        print(f"  → concurrency={level} ...")
        results = await _run_level(url, api_key, level)
        raw_results.extend(results)
        s = _summarize(level, results)
        summaries.append(s)
        _print_summary(s)
        await asyncio.sleep(3)

    # Extrapolate to 300 users
    print("\n  ── 300-user projection ──")
    # Use the highest concurrency level that has < 10% errors
    usable = [s for s in summaries if s.get("error_rate", 1) < 0.1 and s["n_ok"] > 0]
    if usable:
        last = usable[-1]
        scale = TARGET_USERS / last["concurrency"]
        projected_ttft_p95 = last["ttft_p95"] * (scale ** 0.5)  # sublinear due to batching
        projected_tps = last["agg_tokens_per_s"]  # GPU-bound, doesn't scale
        print(f"  Extrapolating from c={last['concurrency']} (scale factor {scale:.1f}x):")
        print(f"    Projected TTFT p95 @ 300 users : {projected_ttft_p95:.1f} s")
        print(f"    Projected total throughput      : {projected_tps:.0f} tok/s  (GPU-bound, same)")
        print(f"    Tokens/user at 300 users        : {projected_tps/TARGET_USERS:.2f} tok/s/user")
        sla_ok = projected_ttft_p95 <= TARGET_TTFT_P95_S
        print(f"    SLA ({TARGET_TTFT_P95_S}s TTFT p95)         : {'✓ PASS' if sla_ok else '✗ FAIL — scale GPU or raise limit'}")

    _save(output, summaries, raw_results, mode="load")


async def run_stress(url: str, api_key: str, output: str):
    print(f"\n=== Layer 2 STRESS TEST  url={url} ===\n")

    # Warmup
    print("  Warmup (1 request)...")
    await _run_level(url, api_key, 1)

    summaries = []
    raw_results = []
    level = STRESS_START
    breaking_point = None

    while level <= STRESS_MAX:
        print(f"\n  → concurrency={level} ...")
        results = await _run_level(url, api_key, level)
        raw_results.extend(results)
        s = _summarize(level, results)
        summaries.append(s)
        _print_summary(s)

        if s["error_rate"] >= STRESS_ERROR_THRESHOLD:
            breaking_point = level
            print(f"\n  ✗ Breaking point reached at concurrency={level}  "
                  f"(error rate {s['error_rate']*100:.0f}%)")
            break

        await asyncio.sleep(5)
        level *= 2

    if breaking_point is None:
        print(f"\n  System survived up to concurrency={level//2} without breaking")

    _save(output, summaries, raw_results, mode="stress")


def _save(output: str, summaries: list, raw: list, mode: str):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    data = {
        "mode": mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_users": TARGET_USERS,
        "target_ttft_p95_s": TARGET_TTFT_P95_S,
        "summaries": summaries,
        "raw_results": [asdict(r) for r in raw],
    }
    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved → {output}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://relay.stream.acer.uic.edu:8001")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--mode", choices=["load", "stress"], default="load")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/layer2_{args.mode}.json"

    if args.mode == "load":
        asyncio.run(run_load(args.url, args.api_key, args.output))
    else:
        asyncio.run(run_stress(args.url, args.api_key, args.output))
