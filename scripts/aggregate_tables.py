#!/usr/bin/env python3
"""Regenerate aggregate result tables from per-seed run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmpt.benchmark import run_benchmark  # noqa: E402
from fmpt.results import write_aggregate_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Qwen3-0.6B run artifacts")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/qwen3_0.6b")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    args = parser.parse_args()

    summary = write_aggregate_summary(args.results_root, args.seeds)
    payload = run_benchmark(args.seeds, args.results_root)
    payload["main_results"] = summary

    benchmark_path = args.results_root / "benchmark.json"
    benchmark_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(benchmark_path), "seeds": args.seeds}, indent=2))


if __name__ == "__main__":
    main()
