"""PT detailed-balance tests on 2×2 lattices (Step 1.5.9).

Gold-standard physics test: does pt_rounds produce the exact Boltzmann
distribution at every temperature slot?  If the energy histogram at each
T slot matches the 2×2 exact partition function, the entire PT composition
(sweep + exchange + tracking) is correct.

This catches bugs that plumbing-only tests miss:
  - Temperature reassignment indexing (set_temperature(temps[r2t[r]]))
  - Exchange Metropolis criterion correctness
  - Fixed-temperature-slot tracking (t2r[t] lookup in pt_collect_obs)
  - D/U parameters surviving pt_rounds (not reset by set_temperature)
  - Sweep + exchange composition preserving canonical distribution
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
from scipy.stats import chisquare

from tests.exact_2x2 import (
    at_exact_probabilities,
    bc_exact_probabilities,
    ising_exact_probabilities,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Temperature ladder: 8-point geometric from T=1.0 to T=5.0
# ---------------------------------------------------------------------------
TEMPS = [round(1.0 * (5.0 / 1.0) ** (i / 7), 3) for i in range(8)]
M = len(TEMPS)

N_EQUIL = 5_000
N_SAMPLE = 500_000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_pt_and_collect_energy_histograms(
    model_type: str,
    temps: list[float],
    n_equil: int,
    n_sample: int,
    seed: int,
    *,
    D: float | None = None,
    U: float | None = None,
) -> list[dict[float, int]]:
    """Run pt_rounds and return per-slot energy histograms.

    Parameters
    ----------
    model_type : "ising", "bc", or "at"
    temps : temperature ladder
    n_equil : equilibration rounds (no observable tracking)
    n_sample : sampling rounds (with observable tracking)
    seed : RNG seed
    D : crystal field for Blume-Capel
    U : four-spin coupling for Ashkin-Teller
    """
    from pbc_datagen._core import (
        AshkinTellerModel,
        BlumeCapelModel,
        IsingModel,
        Rng,
    )

    M = len(temps)
    L = 2

    # Create replicas (Any because the 3 model types are unrelated to mypy)
    replicas: Any
    if model_type == "ising":
        replicas = [IsingModel(L, seed + i) for i in range(M)]
    elif model_type == "bc":
        replicas = [BlumeCapelModel(L, seed + i) for i in range(M)]
        for r in replicas:
            assert D is not None
            r.set_crystal_field(D)
    elif model_type == "at":
        replicas = [AshkinTellerModel(L, seed + i) for i in range(M)]
        for r in replicas:
            assert U is not None
            r.set_four_spin_coupling(U)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Set initial temperatures
    for i, r in enumerate(replicas):
        r.set_temperature(temps[i])

    # Identity permutation
    r2t = list(range(M))
    t2r = list(range(M))
    labels = [0] * M

    # Pick the right pt_rounds variant
    pt_rounds_fn: Any
    if model_type == "ising":
        from pbc_datagen._core import pt_rounds_ising

        pt_rounds_fn = pt_rounds_ising
    elif model_type == "bc":
        from pbc_datagen._core import pt_rounds_bc

        pt_rounds_fn = pt_rounds_bc
    else:
        from pbc_datagen._core import pt_rounds_at

        pt_rounds_fn = pt_rounds_at

    rng = Rng(seed)

    # Equilibrate (no tracking)
    pt_rounds_fn(replicas, temps, r2t, t2r, labels, n_equil, rng, False)

    # Sample (with tracking)
    result = pt_rounds_fn(replicas, temps, r2t, t2r, labels, n_sample, rng, True)

    # Build per-slot energy histograms
    histograms: list[dict[float, int]] = []
    for t in range(M):
        energy_stream = result["obs_streams"]["energy"][t]
        counts: dict[float, int] = {}
        for e in energy_stream:
            e_key = round(float(e), 8)
            counts[e_key] = counts.get(e_key, 0) + 1
        histograms.append(counts)

    return histograms


def _check_detailed_balance_all_slots(
    histograms: list[dict[float, int]],
    exact_fn: Callable[..., dict[Any, float]],
    temps: list[float],
    n_sample: int,
    model_name: str,
    **exact_kwargs: float,
) -> None:
    """Chi-squared test at every T slot against exact P(E).

    Skips slots with fewer than 2 valid bins (very low T where only the
    ground state is populated).
    """
    for t, T in enumerate(temps):
        exact_probs = exact_fn(T, **exact_kwargs)
        energy_levels = sorted(set(exact_probs.keys()) | set(histograms[t].keys()))

        observed = np.array([histograms[t].get(E, 0) for E in energy_levels], dtype=float)
        expected = np.array([exact_probs.get(E, 0.0) for E in energy_levels]) * n_sample

        # Drop bins with expected < 5 — chi-squared is unreliable there.
        keep = expected >= 5
        if np.sum(keep) < 2:
            continue

        obs_arr = observed[keep]
        exp_arr = expected[keep]
        # Rescale so sum(exp) == sum(obs); dropping rare bins removes a
        # tiny probability mass and scipy rejects mismatched sums.
        exp_arr = exp_arr * (obs_arr.sum() / exp_arr.sum())

        stat = chisquare(obs_arr, exp_arr)
        assert stat.pvalue > 0.001, (
            f"{model_name} PT detailed balance violated at T={T} (slot {t}): "
            f"chi2={stat.statistic:.1f}, p={stat.pvalue:.6f}\n"
            f"  observed: {obs_arr}\n"
            f"  expected: {exp_arr}"
        )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestPtDetailedBalanceIsing:
    """Ising 2×2: 3 energy levels, most sensitive baseline."""

    def test_energy_distribution_matches_exact(self) -> None:
        histograms = _run_pt_and_collect_energy_histograms(
            "ising", TEMPS, N_EQUIL, N_SAMPLE, seed=42
        )
        _check_detailed_balance_all_slots(
            histograms, ising_exact_probabilities, TEMPS, N_SAMPLE, "Ising"
        )


class TestPtDetailedBalanceBlumeCapel:
    """Blume-Capel 2×2 with D=1.0: vacancies break Ising degeneracy."""

    def test_energy_distribution_matches_exact(self) -> None:
        histograms = _run_pt_and_collect_energy_histograms(
            "bc", TEMPS, N_EQUIL, N_SAMPLE, seed=43, D=1.0
        )
        _check_detailed_balance_all_slots(
            histograms,
            bc_exact_probabilities,
            TEMPS,
            N_SAMPLE,
            "BlumeCapel",
            D=1.0,
        )


class TestPtDetailedBalanceATNonRemapped:
    """Ashkin-Teller 2×2 with U=0.5: coupled layers, non-remapped Wolff."""

    def test_energy_distribution_matches_exact(self) -> None:
        histograms = _run_pt_and_collect_energy_histograms(
            "at", TEMPS, N_EQUIL, N_SAMPLE, seed=44, U=0.5
        )
        _check_detailed_balance_all_slots(
            histograms,
            at_exact_probabilities,
            TEMPS,
            N_SAMPLE,
            "AT(U=0.5)",
            U=0.5,
        )


class TestPtDetailedBalanceATRemapped:
    """Ashkin-Teller 2×2 with U=1.5: remapped s=στ basis — most complex path."""

    def test_energy_distribution_matches_exact(self) -> None:
        histograms = _run_pt_and_collect_energy_histograms(
            "at", TEMPS, N_EQUIL, N_SAMPLE, seed=45, U=1.5
        )
        _check_detailed_balance_all_slots(
            histograms,
            at_exact_probabilities,
            TEMPS,
            N_SAMPLE,
            "AT(U=1.5)",
            U=1.5,
        )
