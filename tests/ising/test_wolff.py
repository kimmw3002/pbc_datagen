"""Red-phase tests for Step 1.1.2: Wolff cluster update kernel.

The Wolff algorithm (Wolff, 1989) is a cluster Monte Carlo method:
  1. Pick a random seed spin.
  2. Grow a cluster by adding aligned neighbors with probability
     p_add = 1 - exp(-2J/T).
  3. Flip every spin in the cluster.

We expose it as model._wolff_step() -> int (returns cluster size)
so each building block is individually testable from pytest.

All imports are lazy (inside test functions) so pytest can *collect*
the tests even before the C++ binding exists.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_wolff_flips_at_least_one_spin() -> None:
    """After _wolff_step() on a cold start, at least one spin must flip.

    The Wolff algorithm always flips the seed spin at minimum (cluster
    size >= 1), so starting from all +1 we must see at least one -1.
    """
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=8, seed=42)
    model.set_temperature(2.269)  # near T_c

    model._wolff_step()

    spins = model.spins.ravel()
    assert np.any(spins == -1), "Wolff step did not flip any spins"


def test_wolff_returns_cluster_size_in_valid_range() -> None:
    """_wolff_step() must return an integer in [1, N].

    The cluster always contains at least the seed spin (size >= 1)
    and can contain at most all N spins (size <= N).
    """
    from pbc_datagen._core import IsingModel

    L = 8
    N = L * L
    model = IsingModel(L=L, seed=42)
    model.set_temperature(2.269)

    for _ in range(50):
        cluster_size = model._wolff_step()
        assert isinstance(cluster_size, int), (
            f"_wolff_step() should return int, got {type(cluster_size)}"
        )
        assert 1 <= cluster_size <= N, f"Cluster size {cluster_size} outside valid range [1, {N}]"


def test_wolff_magnetization_changes_by_twice_cluster_size() -> None:
    """Wolff flips a cluster: |m_before - m_after| must equal 2 * c / N.

    When a cluster of size c flips, every spin in it changes sign.
    The total spin sum changes by ±2c, so the intensive magnetization
    m = (1/N) Σ s_i changes by exactly 2c/N in absolute value.
    """
    from pbc_datagen._core import IsingModel

    L = 16
    N = L * L
    model = IsingModel(L=L, seed=77)
    model.set_temperature(2.269)

    for _ in range(30):
        m_before = model.magnetization()
        cluster_size = model._wolff_step()
        m_after = model.magnetization()

        delta_m = abs(m_before - m_after)
        expected = 2.0 * cluster_size / N

        assert delta_m == pytest.approx(expected, abs=1e-12), (
            f"|m_before - m_after| = {delta_m}, expected 2*{cluster_size}/{N} = {expected}"
        )


# ---------------------------------------------------------------------------
# Statistical / scaling tests
# ---------------------------------------------------------------------------


def test_wolff_high_temperature_small_clusters() -> None:
    """At T >> T_c, mean cluster size should be O(1).

    At high temperature the bond activation probability
    p_add = 1 - exp(-2J/T) is small, so clusters rarely grow beyond
    the seed spin.  We check that the mean cluster size is much
    smaller than the system size.
    """
    from pbc_datagen._core import IsingModel

    L = 32
    model = IsingModel(L=L, seed=42)
    model.set_temperature(100.0)  # T >> T_c ≈ 2.269

    sizes = []
    for _ in range(200):
        sizes.append(model._wolff_step())

    mean_size = np.mean(sizes)
    # At T=100, p_add ≈ 1 - exp(-0.02) ≈ 0.02.
    # Mean cluster size should be very small compared to N=1024.
    assert mean_size < 2, f"Mean cluster size {mean_size:.1f} is too large at high T (expected < 2)"


def test_wolff_low_temperature_large_clusters() -> None:
    """At T << T_c on a cold start, clusters should span most of the lattice.

    At low temperature the bond activation probability
    p_add = 1 - exp(-2J/T) ≈ 1, so aligned neighbors are almost
    certainly added.  On a cold start (all +1), the first cluster
    should encompass nearly all N spins.
    """
    from pbc_datagen._core import IsingModel

    L = 16
    N = L * L
    model = IsingModel(L=L, seed=42)
    model.set_temperature(0.1)  # T << T_c ≈ 2.269

    # On the very first step from all-+1, the cluster should be huge
    cluster_size = model._wolff_step()
    assert cluster_size > N * 0.9, (
        f"First cluster at low T has size {cluster_size}, "
        f"expected > {int(N * 0.9)} (≈ 90% of N={N})"
    )


# ---------------------------------------------------------------------------
# Detailed balance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", np.linspace(0.5, 5.0, num=10))
def test_wolff_detailed_balance_2x2(T: float) -> None:
    """Wolff on 2×2 must reproduce exact Boltzmann distribution.

    The 2×2 Ising model has 16 states with 3 energy levels:
      E = -8 (degeneracy 2), E = 0 (degeneracy 12), E = +8 (degeneracy 2)

    Z(T) = 2 exp(8/T) + 12 + 2 exp(-8/T)

    Wolff alone satisfies detailed balance for the Ising model, so
    the energy histogram must match the exact probabilities.
    """
    from pbc_datagen._core import IsingModel
    from scipy.stats import chisquare

    from tests.exact_2x2 import ising_exact_probabilities

    exact_probs = ising_exact_probabilities(T)

    model = IsingModel(L=2, seed=42)
    model.set_temperature(T)

    # Equilibrate
    for _ in range(500):
        model._wolff_step()

    # Sample
    n_samples = 500_000
    energy_counts: dict[int, int] = {-8: 0, 0: 0, 8: 0}
    for _ in range(n_samples):
        model._wolff_step()
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
