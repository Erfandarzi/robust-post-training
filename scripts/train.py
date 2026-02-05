#!/usr/bin/env python3
"""Launch a frozen Qwen3-0.6B training configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen FMPT training config")
    parser.add_argument("config", type=Path, help="Path to configs/qwen3_0.6b/*.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config only")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    raise SystemExit(
        "Training entry point resolves configs only in this release; use the committed run "
        "artifacts under results/qwen3_0.6b/ for evaluation tables."
    )


if __name__ == "__main__":
    main()
