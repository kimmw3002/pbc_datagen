"""Red-phase tests for Step 1.1.4: sweep() = Wolff + Metropolis, observable tracking.

sweep(n_sweeps) performs n_sweeps iterations of:
    1. One full Metropolis sweep (N random single-spin proposals)
    2. One Wolff cluster step (grow + flip one cluster)

After each iteration it records the observables:
    - E:     total energy (int)
    - m:     intensive magnetization (float)
    - |m|:   intensive absolute magnetization (float)

Returns a dict {"energy": ndarray, "m": ndarray, "abs_m": ndarray}, each of
length n_sweeps.

All imports are lazy (inside test functions) so pytest can *collect*
the tests before the C++ binding exists.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_sweep_returns_dict_with_observable_arrays() -> None:
    """sweep(n) must return a dict with keys 'energy', 'm', 'abs_m'.

    Each value must be a numpy array of length n.  This defines the
    public API for the combined update + measurement step.
    """
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=4, seed=42)
    model.set_temperature(2.269)

    n = 20
    result = model.sweep(n)

    assert isinstance(result, dict), f"sweep() should return dict, got {type(result)}"
    for key in ("energy", "m", "abs_m"):
        assert key in result, f"Missing key '{key}' in sweep result"
        assert isinstance(result[key], np.ndarray), (
            f"result['{key}'] should be ndarray, got {type(result[key])}"
        )
        assert len(result[key]) == n, f"result['{key}'] has length {len(result[key])}, expected {n}"


def test_sweep_requires_temperature_set() -> None:
    """sweep() must raise if temperature has not been set.

    Without a temperature, Boltzmann weights are undefined.  The method
    should refuse to run rather than silently using T=0 or garbage.
    """
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=4, seed=42)
    with pytest.raises((ValueError, RuntimeError)):
        model.sweep(10)


# ---------------------------------------------------------------------------
# Observable tracking
# ---------------------------------------------------------------------------


def test_sweep_last_observable_matches_model_state() -> None:
    """The last recorded observable must match the model's current state.

    After sweep(n), model.energy() should equal result['energy'][-1],
    and similarly for magnetization and abs_magnetization.  This ensures
    the tracking stays in sync with the actual spin configuration.
    """
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=8, seed=42)
    model.set_temperature(2.269)

    result = model.sweep(100)

    assert result["energy"][-1] == model.energy(), (
        f"Last energy {result['energy'][-1]} != model.energy() {model.energy()}"
    )
    assert result["m"][-1] == pytest.approx(model.magnetization(), abs=1e-14), (
        f"Last m {result['m'][-1]} != model.magnetization() {model.magnetization()}"
    )
    assert result["abs_m"][-1] == pytest.approx(model.abs_magnetization(), abs=1e-14), (
        f"Last |m| {result['abs_m'][-1]} != model.abs_magnetization() {model.abs_magnetization()}"
    )


# ---------------------------------------------------------------------------
# Detailed balance (gold standard for combined sweep)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [0.5, 1.5, 2.269, 3.5, 5.0])
def test_sweep_detailed_balance_2x2(T: float) -> None:
    """Combined Wolff+Metropolis sweep on 2×2 must reproduce exact P(E).

    The 2×2 Ising model has 2⁴ = 16 states and only 3 energy levels:
      E = -8 (degeneracy 2), E = 0 (degeneracy 12), E = +8 (degeneracy 2)

    Z(T) = 2 exp(8/T) + 12 + 2 exp(-8/T)

    The hybrid (Metropolis + Wolff) must still satisfy detailed balance.
    We histogram the energy over many sweeps and chi-squared test against
    the exact probabilities.
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
    model.sweep(500)

    # Sample
    n_samples = 500_000
    result = model.sweep(n_samples)
    energies = result["energy"]

    energy_counts = {
        -8: int(np.sum(energies == -8)),
        0: int(np.sum(energies == 0)),
        8: int(np.sum(energies == 8)),
    }

    observed = np.array([energy_counts[-8], energy_counts[0], energy_counts[8]], dtype=float)
    expected = np.array([exact_probs[-8], exact_probs[0], exact_probs[8]]) * n_samples

    stat = chisquare(observed, expected)
    assert stat.pvalue > 0.001, (
        f"Detailed balance violated at T={T}: chi2={stat.statistic:.1f}, "
        f"p={stat.pvalue:.6f}\n"
        f"  observed: {observed}\n"
        f"  expected: {expected}"
    )


# ---------------------------------------------------------------------------
# Ergodicity
# ---------------------------------------------------------------------------


def test_sweep_ergodicity_opposite_starts() -> None:
    """Starting from all-+1 and all-−1, both must converge to same ⟨E⟩.

    At T = 5.0 >> T_c ≈ 2.269, the system is deep in the disordered phase.
    Both initial conditions must thermalize to the same equilibrium
    energy (within statistical fluctuations).
    """
    from pbc_datagen._core import IsingModel

    L = 8
    N = L * L
    T = 5.0
    n_equil = 500
    n_measure = 5000

    # Run 1: cold start (all +1, the default)
    model_up = IsingModel(L=L, seed=123)
    model_up.set_temperature(T)
    model_up.sweep(n_equil)
    result_up = model_up.sweep(n_measure)

    # Run 2: anti-aligned start (all -1)
    model_dn = IsingModel(L=L, seed=456)
    model_dn.set_temperature(T)
    for site in range(N):
        model_dn.set_spin(site, -1)
    model_dn.sweep(n_equil)
    result_dn = model_dn.sweep(n_measure)

    mean_e_up = np.mean(result_up["energy"])
    mean_e_dn = np.mean(result_dn["energy"])

    # Both should agree within a few percent of the energy scale.
    # At T=3.0 for L=8, ⟨E⟩ ≈ -50 to -70.  A tolerance of 5 is generous.
    assert abs(mean_e_up - mean_e_dn) < 5.0, (
        f"Ergodicity check failed: ⟨E⟩_up={mean_e_up:.1f}, ⟨E⟩_dn={mean_e_dn:.1f}, "
        f"difference={abs(mean_e_up - mean_e_dn):.1f}"
    )
