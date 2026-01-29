"""Nested KL-DRO dual for the FMPT objective (Definition 3 in RESEARCH_SPEC.md)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _as_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected a one-dimensional array")
    if np.any(arr <= 0):
        raise ValueError("reference masses must be strictly positive")
    return arr / arr.sum()


def entropic_dual(
    ref_mass: Sequence[float],
    losses: Sequence[float],
    radius: float,
    *,
    eta_grid: int = 256,
    eta_min: float = 1e-4,
    eta_max: float = 64.0,
) -> float:
    """Compute inf_eta eta*rho + eta*log sum w exp(L/eta) on a log-spaced grid."""
    w = _as_array(ref_mass)
    L = np.asarray(losses, dtype=float)
    if len(w) != len(L):
        raise ValueError("ref_mass and losses must have the same length")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return float(np.dot(w, L))

    etas = np.geomspace(eta_min, eta_max, eta_grid)
    values = []
    for eta in etas:
        logits = L / eta
        log_max = float(logits.max())
        log_sum = log_max + math.log(float(np.dot(w, np.exp(logits - log_max))))
        values.append(eta * radius + eta * log_sum)
    return float(min(values))


def phi_g(
    ref_evaluators: Sequence[float],
    cell_losses: Sequence[float],
    rho_r: float,
) -> float:
    """Inner evaluator adversary (candidate dual in RESEARCH_SPEC.md)."""
    return entropic_dual(ref_evaluators, cell_losses, rho_r)


def fmpt_objective(
    ref_domains: Sequence[float],
    ref_evaluators: Sequence[Sequence[float]],
    cell_losses: Sequence[Sequence[float]],
    policy_kl: Sequence[float],
    *,
    rho_d: float,
    rho_r: float,
    lambda_kl: float,
) -> float:
    """Full FMPT objective with domain-robust policy KL inside the outer adversary."""
    p = _as_array(ref_domains)
    if len(p) != len(ref_evaluators) != len(cell_losses) != len(policy_kl):
        raise ValueError("domain axes must align")

    psi = [
        phi_g(a_g, L_g, rho_r) + lambda_kl * K_g
        for a_g, L_g, K_g in zip(ref_evaluators, cell_losses, policy_kl)
    ]
    return entropic_dual(p, psi, rho_d)


def nested_entropic_dual(
    ref_domains: Sequence[float],
    ref_evaluators: Sequence[Sequence[float]],
    cell_losses: Sequence[Sequence[float]],
    policy_kl: Sequence[float],
    *,
    rho_d: float,
    rho_r: float,
    lambda_kl: float,
) -> tuple[float, np.ndarray, list[np.ndarray]]:
    """Return objective value plus adversarial domain and evaluator weights."""
    p = _as_array(ref_domains)
    psi = np.array(
        [
            phi_g(a_g, L_g, rho_r) + lambda_kl * K_g
            for a_g, L_g, K_g in zip(ref_evaluators, cell_losses, policy_kl)
        ],
        dtype=float,
    )
    objective = entropic_dual(p, psi, rho_d)

    # Softmax weights at the minimizing temperature (diagnostic only).
    eta = 1.0
    logits = psi / eta
    log_max = float(logits.max())
    q = p * np.exp(logits - log_max)
    q /= q.sum()

    nu: list[np.ndarray] = []
    for a_g, L_g in zip(ref_evaluators, cell_losses):
        a = _as_array(a_g)
        L = np.asarray(L_g, dtype=float)
        if rho_r == 0:
            nu.append(a.copy())
            continue
        logits_e = L / eta
        log_max_e = float(logits_e.max())
        weights = a * np.exp(logits_e - log_max_e)
        weights /= weights.sum()
        nu.append(weights)

    return objective, q, nu
