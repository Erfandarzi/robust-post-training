"""Smoke tests for the finite FMPT dual."""

from __future__ import annotations

import math

from fmpt.benchmark import evaluate, _make_scenario
from fmpt.objective import entropic_dual, fmpt_objective, phi_g


def test_entropic_dual_recovers_empirical_mean_at_zero_radius():
    ref = [0.6, 0.4]
    losses = [0.2, 0.8]
    assert math.isclose(entropic_dual(ref, losses, 0.0), 0.6 * 0.2 + 0.4 * 0.8)


def test_phi_g_is_monotone_in_worst_cell_loss():
    ref = [0.5, 0.5]
    base = [0.30, 0.40]
    worse = [0.30, 1.20]
    assert phi_g(ref, worse, 0.1) > phi_g(ref, base, 0.1)


def test_fmpt_is_at_least_as_pessimistic_as_erm():
    scenario = _make_scenario(17)
    scores = evaluate(scenario)
    assert scores["fmpt"] >= scores["erm"] - 1e-6


def test_fmpt_runs_on_handcrafted_shift():
    ref_domains = [0.7, 0.3]
    ref_evaluators = [[0.5, 0.5], [0.5, 0.5]]
    cell_losses = [[0.4, 0.5], [0.6, 0.7]]
    policy_kl = [0.05, 0.40]
    value = fmpt_objective(
        ref_domains,
        ref_evaluators,
        cell_losses,
        policy_kl,
        rho_d=0.2,
        rho_r=0.1,
        lambda_kl=0.5,
    )
    assert value > 0.0
