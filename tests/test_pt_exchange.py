"""Tests for pt_exchange() — single-gap replica exchange decision (Step 1.5.1).

Metropolis criterion: A = min(1, exp((1/T_i - 1/T_j) * (E_i - E_j)))
"""

from __future__ import annotations

import math


def _make_rng(seed: int = 42):
    from pbc_datagen._core import Rng

    return Rng(seed)


def test_same_temp_always_accepts():
    """When T_i == T_j, Δ(1/T) = 0 → exp(0) = 1 → always accept."""
    from pbc_datagen._core import pt_exchange

    rng = _make_rng()
    T = 2.0
    for E_i, E_j in [(-8, 0), (0, -8), (4, 4), (-100, 100)]:
        assert pt_exchange(E_i, E_j, T, T, rng) is True


def test_known_acceptance_rate():
    """Measured acceptance rate matches analytical exp((1/T_i - 1/T_j)*(E_i - E_j))."""
    from pbc_datagen._core import pt_exchange

    rng = _make_rng(456)
    T_i, T_j = 1.0, 2.0
    E_i, E_j = -8.0, 0.0
    # exponent = (1/1 - 1/2) * (-8 - 0) = 0.5 * (-8) = -4
    expected_rate = math.exp(-4.0)  # ≈ 0.01832

    n_trials = 200_000
    accepts = sum(pt_exchange(E_i, E_j, T_i, T_j, rng) for _ in range(n_trials))
    measured_rate = accepts / n_trials

    sigma = math.sqrt(expected_rate * (1 - expected_rate) / n_trials)
    assert abs(measured_rate - expected_rate) < 4 * sigma


def test_extreme_gap_always_rejects():
    """Very large Δ(1/T) with unfavorable ΔE → exp(-999) ≈ 0 → always reject."""
    from pbc_datagen._core import pt_exchange

    rng = _make_rng(789)
    # (1/0.1 - 1/100) * (-100 - 0) = 9.99 * (-100) ≈ -999
    n_trials = 10_000
    accepts = sum(pt_exchange(-100.0, 0.0, 0.1, 100.0, rng) for _ in range(n_trials))
    assert accepts == 0
