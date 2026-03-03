"""Red-phase tests for Step 3.2: 2D parameter-space PT exchange criterion.

The new ``pt_exchange_param`` function handles exchanges along the parameter
direction (D for Blume-Capel, U for Ashkin-Teller) at fixed temperature.

    Δ = β × (param_i − param_j) × (dE/dp_i − dE/dp_j)

where dE/dp is the derivative of energy w.r.t. the Hamiltonian parameter:
  - BC:  dE/dD = Σ s_i²  (= quadrupole × N)
  - AT:  dE/dU = −Σ_{<ij>} σ_i σ_j τ_i τ_j  (cached_four_spin_)

Accept when Δ ≥ 0 deterministically, otherwise with prob exp(Δ).

The integration tests call C++ ``pt_rounds_2d_bc`` / ``pt_rounds_2d_at``
which run the full 2D PT loop (sweep + T-exchange + param-exchange +
observable collection) in a single C++ call — same design as the 1D
``pt_rounds``, no Python-level per-round overhead.

2D grid layout: n_T × n_param replicas, slot(i,j) = i*n_param + j.
Alternating T-direction and param-direction exchange sweeps each round.

BC parameter range: D < 1 only — high D drives a first-order transition
whose exponential barrier makes chi-squared testing impractical.

AT parameter range: includes U < 1 (physical basis Wolff) and U > 1
(remapped σ,s=στ basis) to exercise both codepaths.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.stats import chisquare

from tests.exact_2x2 import at_exact_probabilities, bc_exact_probabilities

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

L_TEST = 2

# Small 4-point T ladder for 2D tests — wide enough for good mixing,
# narrow enough to keep the test fast.
TEMPS_2D = [1.5, 2.0, 3.0, 5.0]
N_T = len(TEMPS_2D)

N_EQUIL_2D = 3_000
N_SAMPLE_2D = 300_000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_2d_pt_and_collect_histograms(
    model_type: str,
    temps: list[float],
    param_values: list[float],
    n_equil: int,
    n_sample: int,
    seed: int,
) -> list[dict[float, int]]:
    """Run pt_rounds_2d and return per-slot energy histograms.

    Parameters
    ----------
    model_type : "bc" or "at"
    temps : temperature ladder (length n_T)
    param_values : parameter ladder (length n_P) — D for BC, U for AT
    n_equil : equilibration rounds (no observable tracking)
    n_sample : sampling rounds (with observable tracking)
    seed : RNG seed
    """
    from pbc_datagen._core import Rng

    n_T_local = len(temps)
    n_P = len(param_values)
    M = n_T_local * n_P

    replicas: Any
    pt_rounds_fn: Any

    if model_type == "bc":
        from pbc_datagen._core import BlumeCapelModel, pt_rounds_2d_bc

        replicas = [BlumeCapelModel(L_TEST, seed + r) for r in range(M)]
        for i in range(n_T_local):
            for j in range(n_P):
                s = i * n_P + j
                replicas[s].set_crystal_field(param_values[j])
        pt_rounds_fn = pt_rounds_2d_bc
    elif model_type == "at":
        from pbc_datagen._core import AshkinTellerModel, pt_rounds_2d_at

        replicas = [AshkinTellerModel(L_TEST, seed + r) for r in range(M)]
        for i in range(n_T_local):
            for j in range(n_P):
                s = i * n_P + j
                replicas[s].set_four_spin_coupling(param_values[j])
        pt_rounds_fn = pt_rounds_2d_at
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Set initial temperatures
    for i in range(n_T_local):
        for j in range(n_P):
            s = i * n_P + j
            replicas[s].set_temperature(temps[i])

    # Identity permutation
    r2s = list(range(M))
    s2r = list(range(M))

    rng = Rng(seed)

    # Equilibrate (no tracking)
    pt_rounds_fn(replicas, temps, param_values, r2s, s2r, n_equil, rng, False)

    # Sample (with tracking)
    result = pt_rounds_fn(replicas, temps, param_values, r2s, s2r, n_sample, rng, True)

    # Build per-slot energy histograms from obs_streams
    histograms: list[dict[float, int]] = []
    for s in range(M):
        energy_stream = result["obs_streams"]["energy"][s]
        counts: dict[float, int] = {}
        for e in energy_stream:
            e_key = round(float(e), 8)
            counts[e_key] = counts.get(e_key, 0) + 1
        histograms.append(counts)

    return histograms


def _check_histograms_against_exact(
    histograms: list[dict[float, int]],
    exact_fn: Any,
    temps: list[float],
    param_values: list[float],
    n_sample: int,
    model_name: str,
    param_name: str,
) -> None:
    """Chi-squared test at every (T, param) slot against exact P(E).

    Skips slots with fewer than 2 valid bins (very low T where only the
    ground state is populated).
    """
    n_param = len(param_values)
    for i, T in enumerate(temps):
        for j, p in enumerate(param_values):
            s = i * n_param + j
            exact_probs = exact_fn(T, p)
            energy_levels = sorted(set(exact_probs.keys()) | set(histograms[s].keys()))

            observed = np.array([histograms[s].get(E, 0) for E in energy_levels], dtype=float)
            expected = np.array([exact_probs.get(E, 0.0) for E in energy_levels]) * n_sample

            # Drop bins with expected < 5 — chi-squared unreliable there
            keep = expected >= 5
            if np.sum(keep) < 2:
                continue

            obs_arr = observed[keep]
            exp_arr = expected[keep]
            # Rescale so sum(exp) == sum(obs) after dropping rare bins
            exp_arr = exp_arr * (obs_arr.sum() / exp_arr.sum())

            stat = chisquare(obs_arr, exp_arr)
            assert stat.pvalue > 0.001, (
                f"{model_name} 2D PT detailed balance violated at "
                f"T={T}, {param_name}={p} (slot {s}): "
                f"chi2={stat.statistic:.1f}, p={stat.pvalue:.6f}\n"
                f"  observed: {obs_arr}\n"
                f"  expected: {exp_arr}"
            )


# ---------------------------------------------------------------------------
# Unit tests for pt_exchange_param
# ---------------------------------------------------------------------------


class TestPtExchangeParam:
    """Unit tests for the param-direction exchange criterion."""

    def test_same_param_always_accepts(self) -> None:
        """When param_i == param_j, Δ = 0 → always accept.

        This is the trivial case: exchanging replicas at the same
        parameter value costs nothing energetically.
        """
        from pbc_datagen._core import Rng, pt_exchange_param

        rng = Rng(42)
        n_trials = 1000
        n_accept = sum(
            pt_exchange_param(dEdp_i=3.0, dEdp_j=7.0, T=2.0, param_i=0.5, param_j=0.5, rng=rng)
            for _ in range(n_trials)
        )
        assert n_accept == n_trials, (
            f"Expected 100% acceptance at same param, got {n_accept}/{n_trials}"
        )

    def test_deterministic_accept_large_favorable_delta(self) -> None:
        """When Δ is large and positive, acceptance is certain.

        Set up: dEdp_i > dEdp_j and param_i > param_j so
        Δ = β × (param_i − param_j) × (dEdp_i − dEdp_j) >> 0.
        """
        from pbc_datagen._core import Rng, pt_exchange_param

        rng = Rng(42)
        # β=1, (param_i - param_j)=10, (dEdp_i - dEdp_j)=10 → Δ=100
        n_trials = 1000
        n_accept = sum(
            pt_exchange_param(dEdp_i=15.0, dEdp_j=5.0, T=1.0, param_i=10.0, param_j=0.0, rng=rng)
            for _ in range(n_trials)
        )
        assert n_accept == n_trials, (
            f"Expected 100% acceptance for Δ=100, got {n_accept}/{n_trials}"
        )

    def test_acceptance_rate_matches_boltzmann(self) -> None:
        """Measured acceptance rate must match min(1, exp(Δ)) statistically.

        Choose dEdp and param values so Δ = −1.0 (mild rejection).
        Expected acceptance rate = exp(−1) ≈ 0.368.

        Use dEdp_i=0, dEdp_j=1, param_i=1, param_j=0, T=1 (β=1):
            Δ = 1 × (1−0) × (0−1) = −1 → rate ≈ exp(−1).
        """
        from pbc_datagen._core import Rng, pt_exchange_param

        rng = Rng(42)
        n_trials = 50_000
        n_accept = sum(
            pt_exchange_param(dEdp_i=0.0, dEdp_j=1.0, T=1.0, param_i=1.0, param_j=0.0, rng=rng)
            for _ in range(n_trials)
        )
        rate = n_accept / n_trials
        expected_rate = math.exp(-1.0)

        # Allow 3-sigma tolerance: σ = sqrt(p(1-p)/n) ≈ 0.0022
        tol = 3 * math.sqrt(expected_rate * (1 - expected_rate) / n_trials)
        assert abs(rate - expected_rate) < tol, (
            f"Acceptance rate {rate:.4f} deviates from expected "
            f"exp(-1)={expected_rate:.4f} by more than 3sigma ({tol:.4f})"
        )


# ---------------------------------------------------------------------------
# 2D PT detailed balance — Blume-Capel (D < 1)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPt2dDetailedBalanceBlumeCapel:
    """2×2 BC on a T×D grid: energy distribution must match exact P(E).

    D values restricted to D < 1 — high D drives a first-order
    transition that causes exponential mixing barriers even on 2×2.
    """

    @pytest.mark.parametrize(
        "D_values",
        [
            pytest.param([0.0, 0.5], id="D=0.0-0.5"),
            pytest.param([0.3, 0.8], id="D=0.3-0.8"),
        ],
    )
    def test_energy_distribution_matches_exact(self, D_values: list[float]) -> None:
        histograms = _run_2d_pt_and_collect_histograms(
            "bc", TEMPS_2D, D_values, N_EQUIL_2D, N_SAMPLE_2D, seed=100
        )
        _check_histograms_against_exact(
            histograms,
            bc_exact_probabilities,
            TEMPS_2D,
            D_values,
            N_SAMPLE_2D,
            "BlumeCapel",
            "D",
        )


# ---------------------------------------------------------------------------
# 2D PT detailed balance — Ashkin-Teller (U < 1 and U > 1)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPt2dDetailedBalanceAshkinTeller:
    """2×2 AT on a T×U grid: energy distribution must match exact P(E).

    Tests span both U < 1 (physical-basis Wolff) and U > 1 (remapped
    s=στ basis Wolff) to exercise both codepaths.
    """

    @pytest.mark.parametrize(
        "U_values",
        [
            pytest.param([0.0, 0.5], id="U=0.0-0.5"),
            pytest.param([0.5, 1.5], id="U=0.5-1.5"),
            pytest.param([1.0, 1.5], id="U=1.0-1.5"),
        ],
    )
    def test_energy_distribution_matches_exact(self, U_values: list[float]) -> None:
        histograms = _run_2d_pt_and_collect_histograms(
            "at", TEMPS_2D, U_values, N_EQUIL_2D, N_SAMPLE_2D, seed=200
        )
        _check_histograms_against_exact(
            histograms,
            at_exact_probabilities,
            TEMPS_2D,
            U_values,
            N_SAMPLE_2D,
            "AshkinTeller",
            "U",
        )
