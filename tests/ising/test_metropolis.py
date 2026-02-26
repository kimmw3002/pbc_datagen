"""Red-phase tests for Step 1.1.3: Metropolis sweep with precomputed exp table.

The Metropolis algorithm (Metropolis et al., 1953) proposes single-spin
flips and accepts them with probability min(1, exp(-ΔE/T)).

We expose two internal functions for testability:
  - _delta_energy(site) -> int : local energy change if spin[site] were flipped
  - _metropolis_sweep() -> int : one full sweep (N proposals), returns # accepted

All imports are lazy so pytest can collect before the C++ binding exists.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _delta_energy: exact values on known configurations
# ---------------------------------------------------------------------------


def test_delta_energy_cold_start_equals_plus_eight() -> None:
    """On all-+1, flipping any spin costs ΔE = +8.

    The spin is +1, all 4 neighbors are +1.  Before flip: each bond
    contributes -1 to H.  After flip: each contributes +1.
    ΔE = 2 × s_i × Σ_neighbors = 2 × 1 × 4 = +8.
    """
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=8, seed=42)
    model.set_temperature(2.0)

    for site in range(model.L * model.L):
        assert model._delta_energy(site) == 8, (
            f"ΔE at site {site} on cold start should be +8, got {model._delta_energy(site)}"
        )


def test_delta_energy_checkerboard_equals_minus_eight() -> None:
    """On a 2×2 checkerboard, flipping any spin gains ΔE = -8.

    Every neighbor is antiparallel, so flipping aligns them all.
    ΔE = 2 × s_i × Σ_neighbors = 2 × 1 × (-4) = -8 (for a +1 site).
    Same magnitude for a -1 site by symmetry.
    """
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=2, seed=42)
    model.set_temperature(2.0)
    # Set checkerboard: (0,0)=+1, (0,1)=-1, (1,0)=-1, (1,1)=+1
    model.set_spin(1, -1)
    model.set_spin(2, -1)

    for site in range(4):
        assert model._delta_energy(site) == -8, (
            f"ΔE at site {site} on checkerboard should be -8, got {model._delta_energy(site)}"
        )


def test_delta_energy_independent_of_system_size() -> None:
    """ΔE depends only on the 4 neighbors, not on L.

    On a cold start, ΔE = +8 regardless of system size.
    """
    from pbc_datagen._core import IsingModel

    for L in [4, 16, 64]:
        model = IsingModel(L=L, seed=42)
        model.set_temperature(2.0)
        assert model._delta_energy(0) == 8


# ---------------------------------------------------------------------------
# _metropolis_sweep: acceptance rate and detailed balance
# ---------------------------------------------------------------------------


def test_metropolis_sweep_returns_accepted_count_in_valid_range() -> None:
    """_metropolis_sweep() returns an int in [0, N].

    Each sweep proposes N flips; between 0 and N can be accepted.
    """
    from pbc_datagen._core import IsingModel

    L = 8
    N = L * L
    model = IsingModel(L=L, seed=42)
    model.set_temperature(2.269)

    for _ in range(10):
        accepted = model._metropolis_sweep()
        assert isinstance(accepted, int)
        assert 0 <= accepted <= N, f"Accepted count {accepted} outside [0, {N}]"


def test_metropolis_cold_start_low_temp_near_zero_acceptance() -> None:
    """At T → 0 on a cold start, almost no flips are accepted.

    Every proposed flip costs ΔE = +8, and exp(-8/T) ≈ 0 at low T,
    so the acceptance rate should be near zero.
    """
    from pbc_datagen._core import IsingModel

    L = 16
    N = L * L
    model = IsingModel(L=L, seed=42)
    model.set_temperature(0.1)

    total_accepted = 0
    n_sweeps = 20
    for _ in range(n_sweeps):
        total_accepted += model._metropolis_sweep()

    acceptance_rate = total_accepted / (n_sweeps * N)
    assert acceptance_rate < 0.01, (
        f"Acceptance rate {acceptance_rate:.4f} too high at T=0.1 (expected ≈ 0)"
    )


def test_metropolis_high_temp_near_full_acceptance() -> None:
    """At T >> T_c, acceptance rate approaches 100%.

    At high temperature, exp(-ΔE/T) ≈ 1 for all ΔE, so nearly every
    proposed flip is accepted regardless of energy cost.
    At T=1000, the worst case is ΔE=+8 with exp(-8/1000) ≈ 0.992.
    """
    from pbc_datagen._core import IsingModel

    L = 16
    N = L * L
    model = IsingModel(L=L, seed=42)
    model.set_temperature(1000.0)

    total_accepted = 0
    n_sweeps = 50
    for _ in range(n_sweeps):
        total_accepted += model._metropolis_sweep()

    acceptance_rate = total_accepted / (n_sweeps * N)
    assert acceptance_rate > 0.95, (
        f"Acceptance rate {acceptance_rate:.4f} too low at T=1000 (expected > 0.95)"
    )


@pytest.mark.parametrize("T", np.linspace(0.5, 5.0, num=10))
def test_metropolis_detailed_balance_2x2(T: float) -> None:
    """Metropolis on 2×2 must reproduce exact Boltzmann distribution.

    The 2×2 Ising model has 16 states with 3 energy levels:
      E = -8 (degeneracy 2), E = 0 (degeneracy 12), E = +8 (degeneracy 2)

    Z(T) = 2 exp(8/T) + 12 + 2 exp(-8/T)

    We histogram the energy over many sweeps and chi-squared test
    against the exact probabilities.
    """
    from pbc_datagen._core import IsingModel
    from scipy.stats import chisquare

    Z = 2 * math.exp(8 / T) + 12 + 2 * math.exp(-8 / T)
    exact_probs = {
        -8: 2 * math.exp(8 / T) / Z,
        0: 12 / Z,
        8: 2 * math.exp(-8 / T) / Z,
    }

    model = IsingModel(L=2, seed=42)
    model.set_temperature(T)

    # Equilibrate
    for _ in range(500):
        model._metropolis_sweep()

    # Sample
    n_samples = 500_000
    energy_counts: dict[int, int] = {-8: 0, 0: 0, 8: 0}
    for _ in range(n_samples):
        model._metropolis_sweep()
        e = model.energy()
        energy_counts[e] += 1

    observed = np.array([energy_counts[-8], energy_counts[0], energy_counts[8]], dtype=float)
    expected = np.array([exact_probs[-8], exact_probs[0], exact_probs[8]]) * n_samples

    result = chisquare(observed, expected)
    assert result.pvalue > 0.001, (
        f"Detailed balance violated at T={T}: chi2={result.statistic:.1f}, "
        f"p={result.pvalue:.6f}\n"
        f"  observed: {observed}\n"
        f"  expected: {expected}"
    )
