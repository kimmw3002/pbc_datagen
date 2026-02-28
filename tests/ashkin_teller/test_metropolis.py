"""Red-phase tests for Step 1.3.3: Ashkin-Teller Metropolis sweep.

The AT Metropolis algorithm operates in the physical (σ, τ) basis
regardless of whether the embedded Wolff uses remapping.  It makes
2N random proposals per sweep: N for σ flips, N for τ flips.

ΔE for σ_i → -σ_i:
    ΔE = 2σ_i Σ_{j∈nbr(i)} σ_j (J + U τ_i τ_j)

ΔE for τ_i → -τ_i:
    ΔE = 2τ_i Σ_{j∈nbr(i)} τ_j (J + U σ_i σ_j)

Accept with probability min(1, exp(-ΔE / T)).

We expose:
  - _delta_energy_sigma(site) -> float : ΔE for flipping σ_i
  - _delta_energy_tau(site)   -> float : ΔE for flipping τ_i
  - _metropolis_sweep()       -> int   : 2N proposals, returns # accepted

All imports are lazy so pytest can collect before the C++ binding exists.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _delta_energy_sigma: exact values on known configurations
# ---------------------------------------------------------------------------


def test_delta_energy_sigma_cold_start() -> None:
    """On all-+1 (cold start), flipping any σ_i costs ΔE = 8(1 + U).

    ΔE = 2σ_i Σ_{j∈nbr(i)} σ_j (1 + U τ_i τ_j)
       = 2(+1) × 4 × (+1) × (1 + U × 1 × 1)
       = 8(1 + U).

    At U=0 (decoupled), this reduces to the Ising result ΔE = 8.
    At U=0.5, ΔE = 12.  At U=1.5, ΔE = 20.
    """
    from pbc_datagen._core import AshkinTellerModel

    for U, expected in [(0.0, 8.0), (0.5, 12.0), (1.0, 16.0), (1.5, 20.0)]:
        model = AshkinTellerModel(L=8, seed=42)
        model.set_temperature(2.0)
        model.set_four_spin_coupling(U)

        for site in range(model.L * model.L):
            de = model._delta_energy_sigma(site)
            assert de == pytest.approx(expected, abs=1e-12), (
                f"ΔE_σ at site {site} with U={U}: expected {expected}, got {de}"
            )


def test_delta_energy_tau_cold_start() -> None:
    """On all-+1, flipping any τ_i also costs ΔE = 8(1 + U).

    By symmetry of the Hamiltonian (J_σ = J_τ = 1), the formulas
    for σ-flip and τ-flip on a fully aligned lattice are identical.
    """
    from pbc_datagen._core import AshkinTellerModel

    for U, expected in [(0.0, 8.0), (0.5, 12.0), (1.5, 20.0)]:
        model = AshkinTellerModel(L=8, seed=42)
        model.set_temperature(2.0)
        model.set_four_spin_coupling(U)

        for site in range(model.L * model.L):
            de = model._delta_energy_tau(site)
            assert de == pytest.approx(expected, abs=1e-12), (
                f"ΔE_τ at site {site} with U={U}: expected {expected}, got {de}"
            )


def test_delta_energy_checkerboard_sigma_uniform_tau() -> None:
    """Checkerboard σ with uniform τ: σ-flip and τ-flip have different ΔE.

    On 2×2 with σ checkerboard = [+1, -1, -1, +1] and τ = [+1, +1, +1, +1]:

    σ-flip at site 0 (σ=+1, all σ neighbors = -1, all τ = +1):
      ΔE_σ = 2(+1) × [4 × (-1) × (1 + U × 1 × 1)] = -8(1 + U).
      Negative → always accepted (moves toward ground state).

    τ-flip at site 0 (τ=+1, all τ neighbors = +1, σ_0 × σ_j = +1 × -1 = -1):
      ΔE_τ = 2(+1) × [4 × (+1) × (1 + U × (+1)(−1))] = 8(1 - U).
      At U=0: costs +8 (pure Ising).  At U=1: costs 0 (four-spin
      coupling exactly cancels the two-spin cost).
    """
    from pbc_datagen._core import AshkinTellerModel

    cases = [
        # (U, expected_dE_sigma, expected_dE_tau)
        (0.0, -8.0, 8.0),  # decoupled: σ gains -8, τ costs +8
        (0.5, -12.0, 4.0),  # four-spin shifts both
        (1.0, -16.0, 0.0),  # at U=1, τ-flip is free!
    ]

    for U, expected_sigma, expected_tau in cases:
        model = AshkinTellerModel(L=2, seed=42)
        model.set_temperature(2.0)
        model.set_four_spin_coupling(U)

        # Set σ to checkerboard: [+1, -1, -1, +1]
        model.set_sigma(0, 1)
        model.set_sigma(1, -1)
        model.set_sigma(2, -1)
        model.set_sigma(3, 1)
        # τ stays all +1 (cold-start default)

        de_sigma = model._delta_energy_sigma(0)
        assert de_sigma == pytest.approx(expected_sigma, abs=1e-12), (
            f"ΔE_σ(0) with checkerboard σ, U={U}: expected {expected_sigma}, got {de_sigma}"
        )

        de_tau = model._delta_energy_tau(0)
        assert de_tau == pytest.approx(expected_tau, abs=1e-12), (
            f"ΔE_τ(0) with checkerboard σ, U={U}: expected {expected_tau}, got {de_tau}"
        )


# ---------------------------------------------------------------------------
# _metropolis_sweep: acceptance rate at extreme temperatures
# ---------------------------------------------------------------------------


def test_metropolis_sweep_returns_valid_count() -> None:
    """_metropolis_sweep() returns an int in [0, 2N].

    Each sweep proposes 2N spin changes (N for σ, N for τ).
    """
    from pbc_datagen._core import AshkinTellerModel

    L = 8
    N = L * L
    model = AshkinTellerModel(L=L, seed=42)
    model.set_temperature(2.0)
    model.set_four_spin_coupling(0.5)

    for _ in range(10):
        accepted = model._metropolis_sweep()
        assert isinstance(accepted, int)
        assert 0 <= accepted <= 2 * N, f"Accepted count {accepted} outside [0, {2 * N}]"


def test_metropolis_low_temp_near_zero_acceptance() -> None:
    """At T → 0, almost no proposals are accepted on a cold start.

    With all σ=+1 and all τ=+1, every flip costs ΔE = 8(1+U) > 0.
    At T=0.1 with U=0.5: ΔE = 12, exp(-12/0.1) ≈ 5×10⁻⁵³.
    Acceptance rate should be essentially zero.
    """
    from pbc_datagen._core import AshkinTellerModel

    L = 16
    N = L * L
    model = AshkinTellerModel(L=L, seed=42)
    model.set_temperature(0.1)
    model.set_four_spin_coupling(0.5)

    total_accepted = 0
    n_sweeps = 20
    for _ in range(n_sweeps):
        total_accepted += model._metropolis_sweep()

    acceptance_rate = total_accepted / (n_sweeps * 2 * N)
    assert acceptance_rate < 0.01, (
        f"Acceptance rate {acceptance_rate:.4f} too high at T=0.1 (expected ≈ 0)"
    )


def test_metropolis_high_temp_near_full_acceptance() -> None:
    """At T >> 1, acceptance rate approaches 100%.

    At high temperature, exp(-ΔE/T) ≈ 1 for all ΔE, so nearly every
    proposed spin change is accepted.
    """
    from pbc_datagen._core import AshkinTellerModel

    L = 16
    N = L * L
    model = AshkinTellerModel(L=L, seed=42)
    model.set_temperature(1000.0)
    model.set_four_spin_coupling(0.5)

    total_accepted = 0
    n_sweeps = 50
    for _ in range(n_sweeps):
        total_accepted += model._metropolis_sweep()

    acceptance_rate = total_accepted / (n_sweeps * 2 * N)
    assert acceptance_rate > 0.95, (
        f"Acceptance rate {acceptance_rate:.4f} too low at T=1000 (expected > 0.95)"
    )


# ---------------------------------------------------------------------------
# Detailed balance: 2×2 exact partition function (256 states)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("T", "U"),
    [
        (2.0, 0.0),  # decoupled: two independent Ising models
        (1.5, 0.5),  # lower T, weak four-spin coupling
        (2.0, 1.0),  # boundary: U = J
        (3.0, 0.5),  # high T, weak coupling
        (2.0, 1.5),  # remapped regime (U > 1)
    ],
)
def test_metropolis_detailed_balance_2x2(T: float, U: float) -> None:
    """Metropolis on 2×2 AT must reproduce exact Boltzmann distribution.

    The 2×2 Ashkin-Teller model has 2^8 = 256 microstates.  We enumerate
    all states, compute exact P(E) from the partition function, histogram
    the sampled energies, and chi-squared test against exact probabilities.

    Bins with expected count < 5 are dropped (chi-squared validity
    requirement; see LESSONS.md).

    Metropolis operates in the physical (σ, τ) basis at all U values —
    it does NOT use the remapped basis.  So unlike the Wolff test,
    the Metropolis detailed-balance test is the same code path for U ≤ 1
    and U > 1.
    """
    from pbc_datagen._core import AshkinTellerModel
    from scipy.stats import chisquare

    from tests.exact_2x2 import at_exact_probabilities

    exact_probs = at_exact_probabilities(T, U)
    energy_levels = sorted(exact_probs.keys())

    model = AshkinTellerModel(2, seed=42)
    model.set_temperature(T)
    model.set_four_spin_coupling(U)

    # Equilibrate
    for _ in range(1000):
        model._metropolis_sweep()

    # Sample
    n_samples = 500_000
    energy_counts: dict[float, int] = {E: 0 for E in energy_levels}

    for _ in range(n_samples):
        model._metropolis_sweep()
        E = round(model.energy(), 8)
        assert E in energy_counts, f"Unexpected energy level {E} (known levels: {energy_levels})"
        energy_counts[E] += 1

    observed = np.array([energy_counts[E] for E in energy_levels], dtype=float)
    expected = np.array([exact_probs[E] for E in energy_levels]) * n_samples

    result = chisquare(observed, expected)
    assert result.pvalue > 0.001, (
        f"Detailed balance violated at T={T}, U={U}: "
        f"chi2={result.statistic:.1f}, p={result.pvalue:.6f}\n"
        f"  observed: {observed}\n"
        f"  expected: {expected}"
    )
