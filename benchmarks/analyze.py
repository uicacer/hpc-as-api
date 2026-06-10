"""
Analyze and compare Layer 1 + Layer 2 benchmark results.
Produces a text report and a CSV for plotting.

Usage:
    python3 analyze.py \\
        --layer1 results/layer1.json \\
        --layer2-load results/layer2_load.json \\
        --layer2-stress results/layer2_stress.json \\
        --output results/report.txt
"""

import argparse
import json
import os
import statistics


def load(path: str) -> dict | list | None:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarize_layer1(data: list) -> list[dict]:
    by_level = {}
    for r in data:
        c = r["concurrency"]
        by_level.setdefault(c, []).append(r)

    rows = []
    for c in sorted(by_level):
        ok = [r for r in by_level[c] if not r["error"]]
        errs = [r for r in by_level[c] if r["error"]]
        if not ok:
            rows.append({"concurrency": c, "layer": "L1_vllm_direct", "n_ok": 0, "n_err": len(errs)})
            continue
        ttfts = sorted(r["ttft_s"] for r in ok)
        totals = sorted(r["total_s"] for r in ok)
        p95 = lambda lst: lst[max(0, int(len(lst) * 0.95) - 1)]
        agg_tps = sum(r["tokens"] for r in ok) / max(r["total_s"] for r in ok)
        rows.append({
            "concurrency": c,
            "layer": "L1_vllm_direct",
            "n_ok": len(ok),
            "n_err": len(errs),
            "ttft_p50": round(statistics.median(ttfts), 3),
            "ttft_p95": round(p95(ttfts), 3),
            "total_p50": round(statistics.median(totals), 3),
            "total_p95": round(p95(totals), 3),
            "agg_tps": round(agg_tps, 1),
        })
    return rows


def format_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "  (no data)\n"
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  " + "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  " + "  ".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    for r in rows:
        lines.append("  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines) + "\n"


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def main(layer1_path, l2_load_path, l2_stress_path, output_path):
    l1_data = load(layer1_path)
    l2_load = load(l2_load_path)
    l2_stress = load(l2_stress_path)

    lines = []
    lines.append("=" * 72)
    lines.append("  hpc-as-api Benchmark Report")
    lines.append("=" * 72)
    lines.append("")

    all_rows = []

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    lines.append("LAYER 1: vLLM direct (ghi2-002:8000, no Globus / relay overhead)")
    lines.append("-" * 72)
    if l1_data:
        rows = summarize_layer1(l1_data)
        all_rows.extend(rows)
        cols = ["concurrency", "n_ok", "n_err", "ttft_p50", "ttft_p95", "total_p50", "total_p95", "agg_tps"]
        lines.append(format_table(rows, cols))
        best = [r for r in rows if r.get("n_err", 99) == 0]
        if best:
            peak = max(best, key=lambda r: r.get("agg_tps", 0))
            lines.append(f"  Peak throughput (no errors): {peak['agg_tps']} tok/s @ concurrency={peak['concurrency']}")
    else:
        lines.append("  (no data — run layer1_vllm_direct.py from the Lakeshore login node)\n")

    lines.append("")

    # ── Layer 2 load ─────────────────────────────────────────────────────────
    lines.append("LAYER 2: End-to-end gateway (Globus Compute + relay) — LOAD TEST")
    lines.append("-" * 72)
    if l2_load and "summaries" in l2_load:
        rows = l2_load["summaries"]
        for r in rows:
            r["layer"] = "L2_gateway_load"
        all_rows.extend(rows)
        cols = ["concurrency", "n_ok", "n_errors", "error_rate", "ttft_p50", "ttft_p95", "total_p50", "total_p95", "agg_tokens_per_s"]
        lines.append(format_table(rows, cols))

        # Gateway overhead (compare same concurrency levels)
        if l1_data:
            l1_rows = summarize_layer1(l1_data)
            l1_map = {r["concurrency"]: r for r in l1_rows}
            lines.append("  Gateway overhead (same concurrency, L2 vs L1):")
            for r in rows:
                c = r["concurrency"]
                if c in l1_map and r.get("ttft_p50") and l1_map[c].get("ttft_p50"):
                    overhead = r["ttft_p50"] - l1_map[c]["ttft_p50"]
                    lines.append(f"    c={c:3d}  TTFT overhead = {overhead:.2f}s  "
                                  f"(L2={r['ttft_p50']:.2f}s  L1={l1_map[c]['ttft_p50']:.2f}s)")
            lines.append("")

        # 300-user projection
        lines.append("  300-user class projection:")
        usable = [r for r in rows if r.get("error_rate", 1) < 0.1 and r.get("n_ok", 0) > 0]
        if usable:
            last = usable[-1]
            scale = 300 / last["concurrency"]
            proj_ttft = last["ttft_p95"] * (scale ** 0.5)
            proj_tps = last.get("agg_tokens_per_s", 0)
            lines.append(f"    Extrapolated from c={last['concurrency']} (×{scale:.1f})")
            lines.append(f"    Projected TTFT p95  @ 300 users : {proj_ttft:.1f} s")
            lines.append(f"    Projected throughput @ 300 users : {proj_tps:.0f} tok/s total")
            lines.append(f"    tok/s per student               : {proj_tps/300:.2f}")
            sla_ttft = l2_load.get("target_ttft_p95_s", 10.0)
            verdict = "✓ LIKELY OK" if proj_ttft <= sla_ttft else "✗ OVER SLA — action needed"
            lines.append(f"    300-user SLA verdict            : {verdict}")
    else:
        lines.append("  (no data — run layer2_gateway.py --mode load)\n")

    lines.append("")

    # ── Layer 2 stress ───────────────────────────────────────────────────────
    lines.append("LAYER 2: End-to-end gateway — STRESS TEST (breaking point)")
    lines.append("-" * 72)
    if l2_stress and "summaries" in l2_stress:
        rows = l2_stress["summaries"]
        for r in rows:
            r["layer"] = "L2_gateway_stress"
        all_rows.extend(rows)
        cols = ["concurrency", "n_ok", "n_errors", "error_rate", "ttft_p50", "ttft_p95", "total_p95", "agg_tokens_per_s"]
        lines.append(format_table(rows, cols))
        broken = [r for r in rows if r.get("error_rate", 0) >= 0.5]
        healthy = [r for r in rows if r.get("error_rate", 1) < 0.1]
        if broken:
            lines.append(f"  Breaking point: concurrency={broken[0]['concurrency']} (≥50% error rate)")
        if healthy:
            lines.append(f"  Safe maximum  : concurrency={healthy[-1]['concurrency']} (<10% error rate)")
    else:
        lines.append("  (no data — run layer2_gateway.py --mode stress)\n")

    lines.append("")
    lines.append("=" * 72)
    lines.append("  RECOMMENDATIONS")
    lines.append("=" * 72)
    lines.append("""
  1. If TTFT p95 at high concurrency is > 10 s:
       → Increase Globus endpoint max_blocks (more SLURM workers)
       → Enable tensor-parallel across all 4 H100s (--tensor-parallel-size 4)
         to quadruple throughput, reducing per-user wait time

  2. If breaking point < 300 users:
       → The bottleneck is likely Globus Compute max_blocks=4
       → Increase max_blocks to 16-32 in ~/.globus_compute/.../config.yaml
       → Consider a dedicated allocation for the class period

  3. For a class of 300 students (not all concurrent):
       → Realistic simultaneous active requests: ~30-60 (students think
         between prompts, read responses, take notes)
       → Plan for peak burst of ~60-100, not 300 simultaneous

  4. The relay server is NOT the bottleneck — it is a memory-copy
     forwarder and easily handles thousands of concurrent streams.
""")

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"  Report saved → {output_path}")

    csv_path = output_path.replace(".txt", ".csv")
    if all_rows:
        write_csv(csv_path, all_rows)
        print(f"  CSV saved    → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer1", default="results/layer1.json")
    parser.add_argument("--layer2-load", default="results/layer2_load.json")
    parser.add_argument("--layer2-stress", default="results/layer2_stress.json")
    parser.add_argument("--output", default="results/report.txt")
    args = parser.parse_args()
    main(args.layer1, args.layer2_load, args.layer2_stress, args.output)
