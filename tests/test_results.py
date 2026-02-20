"""Tests for Qwen result aggregation."""

from pathlib import Path

from fmpt.results import aggregate_main_results, load_run_manifest


def test_manifest_lists_all_runs():
    root = Path("results/qwen3_0.6b")
    manifest = load_run_manifest(root)
    assert manifest["model"] == "Qwen3-0.6B-Base"
    assert len(manifest["runs"]) == 39


def test_aggregate_main_results_reads_seed_files():
    summary = aggregate_main_results(Path("results/qwen3_0.6b"), [17, 29, 41])
    assert summary["offline_dpo"]["FM-DPO"]["worst_cell"] == 0.314
    assert summary["online_grpo"]["FM-GRPO"]["worst_domain"] == 0.371
