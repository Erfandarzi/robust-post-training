"""Crossed-shift benchmark for finite FMPT vs. single-axis baselines."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fmpt.objective import fmpt_objective
from fmpt.results import aggregate_main_results, write_aggregate_summary


@dataclass(frozen=True)
class Scenario:
    name: str
    ref_domains: tuple[float, ...]
    ref_evaluators: tuple[tuple[float, ...], ...]
    cell_losses: tuple[tuple[float, ...], ...]
    policy_kl: tuple[float, ...]
    rho_d: float
    rho_r: float
    lambda_kl: float


def _erm(cell_losses: tuple[tuple[float, ...], ...], ref_domains: tuple[float, ...], ref_evaluators: tuple[tuple[float, ...], ...]) -> float:
    total = 0.0
    for p_g, a_g, L_g in zip(ref_domains, ref_evaluators, cell_losses):
        total += p_g * sum(a_e * L_ge for a_e, L_ge in zip(a_g, L_g))
    return total


def _domain_only(cell_losses, ref_domains, ref_evaluators, lambda_kl, policy_kl, rho_d):
    psi = []
    for a_g, L_g, K_g in zip(ref_evaluators, cell_losses, policy_kl):
        psi.append(sum(a_e * L_ge for a_e, L_ge in zip(a_g, L_g)) + lambda_kl * K_g)
    return fmpt_objective(ref_domains, ref_evaluators, cell_losses, policy_kl, rho_d=rho_d, rho_r=0.0, lambda_kl=lambda_kl)


def _evaluator_only(cell_losses, ref_domains, ref_evaluators, lambda_kl, policy_kl, rho_r):
    return fmpt_objective(ref_domains, ref_evaluators, cell_losses, policy_kl, rho_d=0.0, rho_r=rho_r, lambda_kl=lambda_kl)


def _joint_dro(cell_losses, ref_domains, ref_evaluators, lambda_kl, policy_kl, rho_d, rho_r):
    # Conservative single-ball surrogate: one shared radius on the outer sum only.
    return fmpt_objective(
        ref_domains,
        ref_evaluators,
        cell_losses,
        policy_kl,
        rho_d=rho_d + rho_r,
        rho_r=0.0,
        lambda_kl=lambda_kl,
    )


def evaluate(scenario: Scenario) -> dict[str, float]:
    common = (
        scenario.cell_losses,
        scenario.ref_domains,
        scenario.ref_evaluators,
        scenario.lambda_kl,
        scenario.policy_kl,
    )
    return {
        "erm": _erm(scenario.cell_losses, scenario.ref_domains, scenario.ref_evaluators),
        "domain_only": _domain_only(*common, scenario.rho_d),
        "evaluator_only": _evaluator_only(*common, scenario.rho_r),
        "joint_dro": _joint_dro(*common, scenario.rho_d, scenario.rho_r),
        "fmpt": fmpt_objective(
            scenario.ref_domains,
            scenario.ref_evaluators,
            scenario.cell_losses,
            scenario.policy_kl,
            rho_d=scenario.rho_d,
            rho_r=scenario.rho_r,
            lambda_kl=scenario.lambda_kl,
        ),
    }


def _make_scenario(seed: int) -> Scenario:
    rng = random.Random(seed)
    n_domains = 4
    n_eval = 3

    ref_domains = tuple(rng.random() for _ in range(n_domains))
    s = sum(ref_domains)
    ref_domains = tuple(x / s for x in ref_domains)

    ref_evaluators = []
    cell_losses = []
    policy_kl = []
    for g in range(n_domains):
        masses = [rng.random() for _ in range(n_eval)]
        sm = sum(masses)
        ref_evaluators.append(tuple(m / sm for m in masses))
        base = 0.35 + 0.08 * g + rng.uniform(-0.03, 0.03)
        cell_losses.append(
            tuple(
                base + 0.05 * e + (0.12 if g == n_domains - 1 and e == 0 else 0.0) + rng.uniform(-0.02, 0.02)
                for e in range(n_eval)
            )
        )
        policy_kl.append(0.05 + 0.04 * g + (0.18 if g == n_domains - 1 else 0.0))

    return Scenario(
        name=f"seed_{seed}",
        ref_domains=ref_domains,
        ref_evaluators=tuple(ref_evaluators),
        cell_losses=tuple(cell_losses),
        policy_kl=tuple(policy_kl),
        rho_d=0.15,
        rho_r=0.10,
        lambda_kl=0.25,
    )


def run_benchmark(seeds: list[int], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    finite_rows = []
    for seed in seeds:
        scenario = _make_scenario(seed)
        scores = evaluate(scenario)
        row = {"seed": seed, **scores}
        finite_rows.append(row)

    summary = {}
    for method in ["erm", "domain_only", "evaluator_only", "joint_dro", "fmpt"]:
        vals = [row[method] for row in finite_rows]
        summary[method] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    payload = {
        "finite_kl_dro": {"seeds": seeds, "rows": finite_rows, "summary": summary},
        "main_results": aggregate_main_results(out_dir, seeds),
    }
    write_aggregate_summary(out_dir, seeds)
    (out_dir / "benchmark.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate FMPT experiment results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--out", type=Path, default=Path("results/qwen3_0.6b"))
    args = parser.parse_args()
    payload = run_benchmark(args.seeds, args.out)
    fmpt_mean = payload["finite_kl_dro"]["summary"]["fmpt"]["mean"]
    print(json.dumps({"wrote": str(args.out / "benchmark.json"), "fmpt_mean": round(fmpt_mean, 4)}, indent=2))


if __name__ == "__main__":
    main()
