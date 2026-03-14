"""Red-phase tests for Phase 6.3: sweep() on XYModel.

sweep(n_sweeps) performs n_sweeps iterations of:
    1. One full Metropolis sweep (N random-site proposals, θ' ~ Uniform[0, 2π))
    2. One Wolff cluster step (O(2) perpendicular reflection)

After each iteration it records the observables:
    - energy:  total energy (float)
    - mx:      x-component of intensive magnetization (float)
    - my:      y-component of intensive magnetization (float)
    - abs_m:   intensive absolute magnetization |m| = √(mx² + my²) (float)

Returns a dict {"energy": ndarray, "mx": ndarray, "my": ndarray, "abs_m": ndarray},
each of length n_sweeps.

All imports are lazy (inside test functions) so pytest can *collect*
the tests before the C++ methods exist.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pure-Python O(N) recomputation helpers for XY observables
# ---------------------------------------------------------------------------


def _xy_energy_python(theta: np.ndarray, nbr: np.ndarray, N: int) -> float:
    """H = -J Σ_{<ij>} cos(θ_i - θ_j), sum all 4 neighbors and divide by 2."""
    coupling = 0.0
    for i in range(N):
        for d in range(4):
            j = int(nbr[i, d])
            coupling += np.cos(theta[i] - theta[j])
    return -coupling / 2.0


def _xy_magnetization_python(theta: np.ndarray, N: int) -> tuple[float, float, float]:
    """Returns (mx, my, abs_m)."""
    mx_sum = np.sum(np.cos(theta))
    my_sum = np.sum(np.sin(theta))
    mx = mx_sum / N
    my = my_sum / N
    abs_m = np.sqrt(mx**2 + my**2)
    return mx, my, abs_m


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_sweep_returns_dict_with_observable_arrays() -> None:
    """sweep(n) must return a dict with keys 'energy', 'mx', 'my', 'abs_m'.

    Each value must be a numpy array of length n.
    """
    from pbc_datagen._core import XYModel

    model = XYModel(L=4, seed=42)
    model.set_temperature(1.0)

    n = 20
    result = model.sweep(n)

    assert isinstance(result, dict), f"sweep() should return dict, got {type(result)}"
    for key in ("energy", "mx", "my", "abs_m"):
        assert key in result, f"Missing key '{key}' in sweep result"
        assert isinstance(result[key], np.ndarray), (
            f"result['{key}'] should be ndarray, got {type(result[key])}"
        )
        assert len(result[key]) == n, f"result['{key}'] has length {len(result[key])}, expected {n}"


def test_sweep_requires_temperature_set() -> None:
    """sweep() must raise if temperature has not been set.

    Without a temperature, Boltzmann weights are undefined.
    """
    from pbc_datagen._core import XYModel

    model = XYModel(L=4, seed=42)
    with pytest.raises((ValueError, RuntimeError)):
        model.sweep(10)


# ---------------------------------------------------------------------------
# Observable tracking
# ---------------------------------------------------------------------------


def test_sweep_last_observable_matches_model_state() -> None:
    """The last recorded observable must match the model's current state.

    After sweep(n), model.energy() should equal result['energy'][-1],
    and similarly for mx, my, abs_magnetization.
    """
    from pbc_datagen._core import XYModel

    model = XYModel(L=8, seed=42)
    model.set_temperature(1.0)

    result = model.sweep(100)

    assert result["energy"][-1] == pytest.approx(model.energy(), abs=1e-10), (
        f"Last energy {result['energy'][-1]} != model.energy() {model.energy()}"
    )
    assert result["mx"][-1] == pytest.approx(model.mx(), abs=1e-14), (
        f"Last mx {result['mx'][-1]} != model.mx() {model.mx()}"
    )
    assert result["my"][-1] == pytest.approx(model.my(), abs=1e-14), (
        f"Last my {result['my'][-1]} != model.my() {model.my()}"
    )
    assert result["abs_m"][-1] == pytest.approx(model.abs_magnetization(), abs=1e-14), (
        f"Last |m| {result['abs_m'][-1]} != model.abs_magnetization() {model.abs_magnetization()}"
    )


def test_sweep_cache_matches_python_recompute() -> None:
    """After sweep(), cached observables must match full O(N) Python recomputation.

    This catches incremental cache bugs that accumulate over many sweeps.
    Run 200 sweeps then verify the final state against a from-scratch calculation.
    """
    from pbc_datagen._core import XYModel, make_neighbor_table

    L = 8
    N = L * L
    model = XYModel(L=L, seed=42)
    model.set_temperature(1.0)

    model.sweep(200)

    nbr = make_neighbor_table(L)
    # Read angles from the .spins view (L, L) of float64
    theta = model.spins.ravel().copy()

    e_py = _xy_energy_python(theta, nbr, N)
    mx_py, my_py, abs_m_py = _xy_magnetization_python(theta, N)

    assert model.energy() == pytest.approx(e_py, abs=1e-8), (
        f"C++ energy {model.energy()} != Python {e_py}"
    )
    assert model.mx() == pytest.approx(mx_py, abs=1e-12), f"C++ mx {model.mx()} != Python {mx_py}"
    assert model.my() == pytest.approx(my_py, abs=1e-12), f"C++ my {model.my()} != Python {my_py}"
    assert model.abs_magnetization() == pytest.approx(abs_m_py, abs=1e-12), (
        f"C++ |m| {model.abs_magnetization()} != Python {abs_m_py}"
    )


# ---------------------------------------------------------------------------
# Ergodicity — different initial conditions converge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [0.3, 0.5, 0.8, 1.5, 3.0])
def test_sweep_ergodicity_different_starts(T: float) -> None:
    """Cold start (all θ=0) and random start must converge to same ⟨E⟩.

    Ergodicity for energy holds at all temperatures — the BKT transition
    only affects the decay of spin-spin correlations, not the ability
    of the Metropolis + Wolff hybrid to explore the energy landscape.

    We use a blocked Welch t-test: estimate τ_int from each chain,
    block-average into approximately independent samples, then run
    ttest_ind on the block means (see docs/LESSONS.md).
    """
    import math

    from pbc_datagen._core import XYModel
    from pbc_datagen.autocorrelation import tau_int
    from scipy.stats import ttest_ind

    L = 8
    n_equil = 500
    n_measure = 5000

    # Run 1: cold start (all θ=0, the default)
    model_cold = XYModel(L=L, seed=123)
    model_cold.set_temperature(T)
    model_cold.sweep(n_equil)
    result_cold = model_cold.sweep(n_measure)

    # Run 2: random start
    model_hot = XYModel(L=L, seed=456)
    model_hot.set_temperature(T)
    model_hot.randomize()
    model_hot.sweep(n_equil)
    result_hot = model_hot.sweep(n_measure)

    e_cold = result_cold["energy"]
    e_hot = result_hot["energy"]

    # Estimate autocorrelation time and block into independent samples.
    tau_cold = tau_int(e_cold)
    tau_hot = tau_int(e_hot)
    block_size = max(math.ceil(3 * max(tau_cold, tau_hot)), 1)

    def _block_means(x: np.ndarray, bs: int) -> np.ndarray:
        n_blocks = len(x) // bs
        return x[: n_blocks * bs].reshape(n_blocks, bs).mean(axis=1)

    bm_cold = _block_means(e_cold, block_size)
    bm_hot = _block_means(e_hot, block_size)

    # Guard: if both series are near-constant, skip the t-test
    # (catastrophic cancellation — see docs/LESSONS.md).
    atol, rtol = 1e-12, 1e-8
    std_cold = np.std(bm_cold)
    std_hot = np.std(bm_hot)
    if std_cold < atol + rtol * abs(np.mean(bm_cold)) and std_hot < atol + rtol * abs(
        np.mean(bm_hot)
    ):
        return  # both constant → trivially ergodic

    stat = ttest_ind(bm_cold, bm_hot, equal_var=False)
    assert stat.pvalue > 0.001, (
        f"Ergodicity check failed at T={T}: blocked Welch t-test rejects.\n"
        f"  ⟨E⟩_cold={np.mean(e_cold):.2f}, ⟨E⟩_hot={np.mean(e_hot):.2f}\n"
        f"  τ_int_cold={tau_cold:.1f}, τ_int_hot={tau_hot:.1f}, "
        f"block_size={block_size}\n"
        f"  t={stat.statistic:.2f}, p={stat.pvalue:.6f}"
    )


# ---------------------------------------------------------------------------
# Detailed balance — energy histogram chi-squared on 2×2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [0.5, 0.8, 1.0, 1.5, 3.0])
def test_sweep_detailed_balance_2x2(T: float) -> None:
    """sweep() on 2×2 must reproduce exact energy distribution P(E).

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

    # Equilibrate with sweep()
    model.sweep(5000)

    # Sample with thinning.  Each sweep() call does 1 Metropolis sweep +
    # 1 Wolff step.  On a 2×2 lattice with only 4 sites, consecutive
    # energies are highly autocorrelated, so thin by 20 sweeps per sample.
    n_samples = 200_000
    thin = 20
    energies = np.empty(n_samples)
    for i in range(n_samples):
        model.sweep(thin)
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
