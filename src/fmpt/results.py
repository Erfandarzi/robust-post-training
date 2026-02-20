"""Load and aggregate Qwen3-0.6B run artifacts."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_ROOT = Path("results/qwen3_0.6b")
DEFAULT_SEEDS = [17, 29, 41]

OFFLINE_METHODS: dict[str, str] = {
    "sft": "SFT",
    "dpo": "DPO",
    "label_noise_robust_dpo": "Label-noise-robust DPO",
    "group_robust_dpo": "Group-Robust DPO",
    "dr_dpo": "Distributionally Robust DPO",
    "fm_dpo": "FM-DPO",
}

ONLINE_METHODS: dict[str, str] = {
    "sft": "SFT",
    "grpo": "GRPO",
    "prompt_groupdro": "Prompt/Rollout GroupDRO",
    "dr_regret": "Distributionally Robust Regret",
    "pessimistic_grpo": "Pessimistic reward GRPO",
    "noisy_corrected_grpo": "Noisy-reward-corrected GRPO",
    "fm_grpo": "FM-GRPO",
}

OFFLINE_METRIC_KEYS = [
    "clean_avg",
    "shifted_avg",
    "worst_domain",
    "worst_cell",
    "hack_rate",
    "ece",
    "kl",
]

ONLINE_METRIC_KEYS = [
    "clean_avg",
    "shifted_avg",
    "worst_domain",
    "worst_cell",
    "proxy_gold_gap",
    "hack_rate",
    "failed_runs",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_metrics_path(results_root: Path, seed: int) -> Path:
    return results_root / "seeds" / str(seed) / "metrics.json"


def load_seed_metrics(results_root: Path, seed: int) -> dict[str, Any]:
    return _load_json(seed_metrics_path(results_root, seed))


def load_run_manifest(results_root: Path) -> dict[str, Any]:
    return _load_json(results_root / "manifest.json")


def _mean_across_seeds(
    results_root: Path,
    seeds: list[int],
    lane: str,
    method_key: str,
    metric_key: str,
) -> float:
    values = []
    for seed in seeds:
        block = load_seed_metrics(results_root, seed)[lane][method_key]["metrics"]
        values.append(float(block[metric_key]))
    return float(statistics.mean(values))


def aggregate_lane(
    results_root: Path,
    seeds: list[int],
    lane: str,
    methods: dict[str, str],
    metric_keys: list[str],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method_key, display_name in methods.items():
        out[display_name] = {
            metric_key: round(_mean_across_seeds(results_root, seeds, lane, method_key, metric_key), 3)
            for metric_key in metric_keys
        }
    return out


def aggregate_main_results(
    results_root: Path | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    root = results_root or DEFAULT_RESULTS_ROOT
    seed_list = seeds or DEFAULT_SEEDS
    manifest = load_run_manifest(root)
    return {
        "model": manifest.get("model", "Qwen3-0.6B-Base"),
        "seeds": seed_list,
        "offline_dpo": aggregate_lane(root, seed_list, "offline_dpo", OFFLINE_METHODS, OFFLINE_METRIC_KEYS),
        "online_grpo": aggregate_lane(root, seed_list, "online_grpo", ONLINE_METHODS, ONLINE_METRIC_KEYS),
    }


def write_aggregate_summary(results_root: Path, seeds: list[int] | None = None) -> dict[str, Any]:
    seed_list = seeds or DEFAULT_SEEDS
    summary = aggregate_main_results(results_root, seed_list)
    out_path = results_root / "aggregate" / "main_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
