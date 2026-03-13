"""Red-phase tests for Step 6.1: XY Wolff cluster — O(2) reflection.

The O(2) Wolff algorithm (Wolff, 1989):
  1. Pick a random reflection axis φ ∈ [0, 2π).
  2. Grow a cluster by DFS: for each neighbor j of a cluster site i,
     add j with probability p_add = 1 − exp(min(0, −2β J (sᵢ·r̂)(sⱼ·r̂)))
     where (s·r̂) = cos(θ − φ) is the spin projection onto the axis.
  3. Reflect cluster spins perpendicular to r̂:
     s' = s − 2(s·r̂)r̂  →  θ → 2φ + π − θ  (mod 2π).

We expose it as model._wolff_step() -> int (returns cluster size)
so each building block is individually testable from pytest.

All imports are lazy (inside test functions) so pytest can *collect*
the tests even before the C++ binding exists.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers (pure Python, no C++ imports)
# ---------------------------------------------------------------------------

TWO_PI = 2.0 * math.pi


def _recompute_energy(thetas: np.ndarray, L: int) -> float:
    """Recompute E = -(1/2) Σ_{i,d} cos(θ_i − θ_{nbr[i*4+d]}) in Python.

    On an L×L PBC lattice, site (r, c) has neighbors:
      right = (r, (c+1)%L), left = (r, (c-1)%L),
      down  = ((r+1)%L, c), up   = ((r-1)%L, c).
    """
    flat = thetas.ravel()
    N = L * L
    total = 0.0
    for i in range(N):
        r, c = divmod(i, L)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            j = ((r + dr) % L) * L + (c + dc) % L
            total += math.cos(flat[i] - flat[j])
    return -total / 2.0


def _recompute_mag(thetas: np.ndarray) -> tuple[float, float]:
    """Recompute (mx, my) = (1/N)(Σ cos θ, Σ sin θ)."""
    flat = thetas.ravel()
    N = len(flat)
    mx = sum(math.cos(t) for t in flat) / N
    my = sum(math.sin(t) for t in flat) / N
    return mx, my


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_wolff_returns_cluster_size_in_valid_range() -> None:
    """_wolff_step() must return an integer in [1, N].

    The cluster always contains at least the seed spin (size >= 1)
    and can contain at most all N spins (size <= N).
    """
    from pbc_datagen._core import XYModel

    L = 8
    N = L * L
    model = XYModel(L=L, seed=42)
    model.set_temperature(0.8935)  # near T_BKT ≈ 0.893

    for _ in range(50):
        cluster_size = model._wolff_step()
        assert isinstance(cluster_size, int), (
            f"_wolff_step() should return int, got {type(cluster_size)}"
        )
        assert 1 <= cluster_size <= N, f"Cluster size {cluster_size} outside valid range [1, {N}]"


def test_wolff_changes_at_least_one_spin() -> None:
    """After _wolff_step() on a cold start, at least one spin must change.

    The seed spin is always reflected: θ → 2φ − θ.  For cold start
    (all θ = 0), the seed becomes 2φ which is ≠ 0 with probability 1
    (since φ is drawn uniformly from [0, 2π)).
    """
    from pbc_datagen._core import XYModel

    model = XYModel(L=8, seed=42)
    model.set_temperature(0.8935)

    before = model.spins.copy()
    model._wolff_step()
    after = model.spins

    assert not np.allclose(before, after), "Wolff step did not change any spins"


def test_wolff_only_cluster_size_spins_change() -> None:
    """Exactly cluster_size spins should have different angles after a step.

    The Wolff algorithm reflects only the cluster spins.  Non-cluster
    spins are untouched.  We verify by counting how many sites changed.
    """
    from pbc_datagen._core import XYModel

    L = 8
    model = XYModel(L=L, seed=42)
    model.set_temperature(0.8935)

    # Run a few steps to get away from cold start (where all θ=0
    # makes counting ambiguous when reflected angle is near 0).
    for _ in range(5):
        model._wolff_step()

    for _ in range(20):
        before = model.spins.ravel().copy()
        cluster_size = model._wolff_step()
        after = model.spins.ravel()

        n_changed = int(np.sum(~np.isclose(before, after, atol=1e-12)))
        assert n_changed == cluster_size, (
            f"Expected {cluster_size} spins to change, but {n_changed} did"
        )


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def test_wolff_high_temperature_small_clusters() -> None:
    """At T >> T_BKT, mean cluster size should be O(1).

    At high temperature the bond activation probability is small because
    the projection products (sᵢ·r̂)(sⱼ·r̂) are small on average for
    disordered spins.  Clusters rarely grow beyond the seed.
    """
    from pbc_datagen._core import XYModel

    L = 32
    model = XYModel(L=L, seed=42)
    model.set_temperature(100.0)  # T >> T_BKT ≈ 0.893

    sizes = [model._wolff_step() for _ in range(200)]
    mean_size = np.mean(sizes)
    assert mean_size < 2, f"Mean cluster size {mean_size:.1f} is too large at high T (expected < 2)"


def test_wolff_low_temperature_large_clusters() -> None:
    """At T << T_BKT on a cold start, clusters should span most of the lattice.

    At low temperature all spins are nearly aligned, so the projection
    products are large and p_add ≈ 1.  The first cluster from a cold
    start should encompass nearly all N spins.
    """
    from pbc_datagen._core import XYModel

    L = 16
    N = L * L
    model = XYModel(L=L, seed=42)
    model.set_temperature(0.1)  # T << T_BKT

    cluster_size = model._wolff_step()
    assert cluster_size > N * 0.9, (
        f"First cluster at low T has size {cluster_size}, "
        f"expected > {int(N * 0.9)} (≈ 90% of N={N})"
    )


# ---------------------------------------------------------------------------
# Angle normalization after reflection
# ---------------------------------------------------------------------------


def test_wolff_reflected_angles_normalized() -> None:
    """All spin angles must remain in [0, 2π) after many Wolff steps.

    The reflection θ → 2φ − θ can produce values outside [0, 2π).
    The implementation must normalize afterward.
    """
    from pbc_datagen._core import XYModel

    L = 8
    model = XYModel(L=L, seed=42)
    model.set_temperature(0.8935)

    for _ in range(100):
        model._wolff_step()

    angles = model.spins.ravel()
    assert np.all(angles >= 0.0), f"Found negative angle: {angles.min()}"
    assert np.all(angles < TWO_PI), f"Found angle >= 2π: {angles.max()}"


# ---------------------------------------------------------------------------
# Observable cache consistency
# ---------------------------------------------------------------------------


def test_wolff_energy_cache_consistent_after_step() -> None:
    """Cached energy must match a full Python recomputation after Wolff steps.

    The XY model caches energy incrementally.  Floating-point drift is
    expected with continuous spins, so we use a tolerance of 1e-6.
    """
    from pbc_datagen._core import XYModel

    L = 8
    model = XYModel(L=L, seed=42)
    model.set_temperature(0.8935)

    for step in range(50):
        model._wolff_step()

        cached_E = model.energy()
        recomputed_E = _recompute_energy(model.spins, L)
        assert cached_E == pytest.approx(recomputed_E, abs=1e-6), (
            f"Step {step}: cached energy {cached_E:.8f} != recomputed {recomputed_E:.8f}"
        )


def test_wolff_magnetization_cache_consistent_after_step() -> None:
    """Cached mx, my must match a full Python recomputation after Wolff steps."""
    from pbc_datagen._core import XYModel

    L = 8
    model = XYModel(L=L, seed=42)
    model.set_temperature(0.8935)

    for step in range(50):
        model._wolff_step()

        mx_recomp, my_recomp = _recompute_mag(model.spins)
        assert model.mx() == pytest.approx(mx_recomp, abs=1e-10), (
            f"Step {step}: cached mx {model.mx():.10f} != recomputed {mx_recomp:.10f}"
        )
        assert model.my() == pytest.approx(my_recomp, abs=1e-10), (
            f"Step {step}: cached my {model.my():.10f} != recomputed {my_recomp:.10f}"
        )


# ---------------------------------------------------------------------------
# Detailed balance — energy histogram chi-squared on 2×2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [0.5, 0.8, 1.0, 1.5, 3.0])
def test_wolff_detailed_balance_2x2(T: float) -> None:
    """Wolff on 2×2 must reproduce exact energy distribution P(E).

    The 2×2 XY model has continuous energy in [-8, +8].  We bin the
    energy range and compare the MCMC histogram against exact bin
    probabilities computed by numerical quadrature (256³ grid, fixing
    θ₀ = 0 by O(2) symmetry).

    Chi-squared test with p > 0.001, same threshold as the discrete
    model tests.  Bins with expected count < 5 are dropped per the
    chi-squared validity requirement (see docs/LESSONS.md).
    """
    from pbc_datagen._core import XYModel
    from scipy.stats import chisquare

    from tests.exact_2x2 import xy_2x2_exact_energy_histogram

    # Energy range for 2×2 XY: E ∈ [-8, +8].
    n_bins = 20
    bin_edges = np.linspace(-8.0, 8.0, n_bins + 1)

    exact_probs = xy_2x2_exact_energy_histogram(T, bin_edges, n_grid=256)

    model = XYModel(L=2, seed=42)
    model.set_temperature(T)

    # Equilibrate
    for _ in range(5000):
        model._wolff_step()

    # Sample with thinning.  On a 2×2 lattice the Wolff cluster is
    # often O(1) — especially at high T — so consecutive energies are
    # highly autocorrelated.  Thinning by 10 Wolff steps per sample
    # keeps the chi-squared test valid without needing millions of
    # independent samples.
    n_samples = 200_000
    thin = 20
    energies = np.empty(n_samples)
    for i in range(n_samples):
        for _ in range(thin):
            model._wolff_step()
        energies[i] = model.energy()

    # Build observed histogram (same bins as quadrature).
    observed, _ = np.histogram(energies, bins=bin_edges)

    # Expected counts.
    expected = exact_probs * n_samples

    # Drop bins with expected < 5 (chi-squared validity).
    mask = expected >= 5
    obs = observed[mask].astype(float)
    exp = expected[mask]

    # Rescale expected to match observed sum after dropping bins
    # (see docs/LESSONS.md — negligible correction).
    exp *= obs.sum() / exp.sum()

    result = chisquare(obs, exp)
    assert result.pvalue > 0.001, (
        f"Detailed balance violated at T={T}: chi2={result.statistic:.1f}, "
        f"p={result.pvalue:.6f}\n"
        f"  bins kept: {mask.sum()}/{n_bins}\n"
        f"  observed (kept): {obs}\n"
        f"  expected (kept): {exp}"
    )
