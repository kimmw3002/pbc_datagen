"""Red-phase tests for Step 6.2: XY Metropolis sweep.

The Metropolis sweep proposes N random-site updates per sweep, each with
a completely random new angle θ' ~ Uniform[0, 2π).  Accept with
probability min(1, exp(−βΔE)).

No tunable window parameter — the proposal is the full circle.  This is
simpler than a windowed perturbation and still satisfies detailed balance
because the proposal distribution q(θ→θ') = q(θ'→θ) = 1/(2π) is
symmetric.

We expose it as model._metropolis_sweep() -> int (returns acceptance count)
so each building block is individually testable from pytest.
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
    """Recompute E = -(1/2) Σ_{i,d} cos(θ_i − θ_{nbr}) in Python."""
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


def test_metropolis_returns_acceptance_count() -> None:
    """_metropolis_sweep() must return an int in [0, N].

    The return value is how many of the N proposals were accepted.
    """
    from pbc_datagen._core import XYModel

    L = 8
    N = L * L
    model = XYModel(L=L, seed=42)
    model.set_temperature(1.0)

    n_accepted = model._metropolis_sweep()
    assert isinstance(n_accepted, int), (
        f"_metropolis_sweep() should return int, got {type(n_accepted)}"
    )
    assert 0 <= n_accepted <= N, f"Acceptance count {n_accepted} outside valid range [0, {N}]"


# ---------------------------------------------------------------------------
# Temperature scaling of acceptance rate
# ---------------------------------------------------------------------------


def test_metropolis_high_T_high_acceptance() -> None:
    """At very high T, nearly all proposals should be accepted.

    When T → ∞, exp(−βΔE) → 1 for any ΔE, so every proposal is accepted.
    We check that the acceptance rate is > 90% at T=100.
    """
    from pbc_datagen._core import XYModel

    L = 16
    N = L * L
    model = XYModel(L=L, seed=42)
    model.set_temperature(100.0)

    total_accepted = 0
    n_sweeps = 50
    for _ in range(n_sweeps):
        total_accepted += model._metropolis_sweep()

    rate = total_accepted / (N * n_sweeps)
    assert rate > 0.90, f"Acceptance rate {rate:.3f} too low at T=100 (expected > 0.90)"


def test_metropolis_low_T_low_acceptance() -> None:
    """At very low T on a cold start, few proposals should be accepted.

    From the ground state (all θ=0, E=−2N), a random θ' almost certainly
    raises energy.  With β large, exp(−βΔE) ≈ 0 for ΔE > 0.
    We check that the acceptance rate on the first sweep is < 15%.
    """
    from pbc_datagen._core import XYModel

    L = 16
    N = L * L
    model = XYModel(L=L, seed=42)
    model.set_temperature(0.05)

    n_accepted = model._metropolis_sweep()
    rate = n_accepted / N
    assert rate < 0.05, (
        f"Acceptance rate {rate:.3f} too high at T=0.05 from cold start (expected < 0.05)"
    )


# ---------------------------------------------------------------------------
# Angle normalization
# ---------------------------------------------------------------------------


def test_metropolis_angles_normalized() -> None:
    """All angles must remain in [0, 2π) after many Metropolis sweeps."""
    from pbc_datagen._core import XYModel

    L = 8
    model = XYModel(L=L, seed=42)
    model.set_temperature(1.0)

    for _ in range(100):
        model._metropolis_sweep()

    angles = model.spins.ravel()
    assert np.all(angles >= 0.0), f"Found negative angle: {angles.min()}"
    assert np.all(angles < TWO_PI), f"Found angle >= 2π: {angles.max()}"


# ---------------------------------------------------------------------------
# Observable cache consistency
# ---------------------------------------------------------------------------


def test_metropolis_energy_cache_consistent() -> None:
    """Cached energy must match full recomputation after Metropolis sweeps.

    The Metropolis sweep updates the cache incrementally for each accepted
    proposal.  Floating-point drift is expected, so we use a tolerance.
    """
    from pbc_datagen._core import XYModel

    L = 8
    model = XYModel(L=L, seed=42)
    model.set_temperature(1.0)

    for step in range(50):
        model._metropolis_sweep()

        cached_E = model.energy()
        recomputed_E = _recompute_energy(model.spins, L)
        assert cached_E == pytest.approx(recomputed_E, abs=1e-6), (
            f"Step {step}: cached energy {cached_E:.8f} != recomputed {recomputed_E:.8f}"
        )


def test_metropolis_magnetization_cache_consistent() -> None:
    """Cached mx, my must match full recomputation after Metropolis sweeps."""
    from pbc_datagen._core import XYModel

    L = 8
    model = XYModel(L=L, seed=42)
    model.set_temperature(1.0)

    for step in range(50):
        model._metropolis_sweep()

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
def test_metropolis_detailed_balance_2x2(T: float) -> None:
    """Metropolis on 2×2 must reproduce exact energy distribution P(E).

    Same methodology as the Wolff detailed-balance test: bin energies,
    compare against exact probabilities from numerical quadrature,
    chi-squared with p > 0.001.

    Metropolis with uniform [0, 2π) proposals is slower to decorrelate
    than Wolff (especially at low T), so we use heavier thinning.
    """
    from pbc_datagen._core import XYModel
    from scipy.stats import chisquare

    from tests.exact_2x2 import xy_2x2_exact_energy_histogram

    n_bins = 20
    bin_edges = np.linspace(-8.0, 8.0, n_bins + 1)

    exact_probs = xy_2x2_exact_energy_histogram(T, bin_edges, n_grid=256)

    model = XYModel(L=2, seed=42)
    model.set_temperature(T)

    # Equilibrate — Metropolis-only, no Wolff.
    for _ in range(10_000):
        model._metropolis_sweep()

    # Sample with thinning.  Uniform-proposal Metropolis on a 2×2
    # lattice decorrelates slower than Wolff (single-site updates vs
    # non-local cluster flips).  Thin by 50 sweeps per sample.
    n_samples = 100_000
    thin = 50
    energies = np.empty(n_samples)
    for i in range(n_samples):
        for _ in range(thin):
            model._metropolis_sweep()
        energies[i] = model.energy()

    observed, _ = np.histogram(energies, bins=bin_edges)
    expected = exact_probs * n_samples

    # Drop bins with expected < 5 (chi-squared validity).
    mask = expected >= 5
    obs = observed[mask].astype(float)
    exp = expected[mask]

    # Rescale expected to match observed sum after dropping bins.
    exp *= obs.sum() / exp.sum()

    result = chisquare(obs, exp)
    assert result.pvalue > 0.001, (
        f"Detailed balance violated at T={T}: chi2={result.statistic:.1f}, "
        f"p={result.pvalue:.6f}\n"
        f"  bins kept: {mask.sum()}/{n_bins}\n"
        f"  observed (kept): {obs}\n"
        f"  expected (kept): {exp}"
    )
