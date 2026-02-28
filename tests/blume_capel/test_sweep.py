"""Red-phase tests for Step 1.2.4: BC sweep() = Wolff + Metropolis, observable tracking.

sweep(n_sweeps) performs n_sweeps iterations of:
    1. One full Metropolis sweep (N random single-spin proposals over {-1,0,+1})
    2. One Wolff cluster step (grow + flip one cluster; vacancies block growth)

After each iteration it records the observables:
    - energy:  total energy H = -Σ s_i s_j + D Σ s_i² (float, D is continuous)
    - m:       intensive magnetization (float)
    - abs_m:   intensive absolute magnetization (float)
    - q:       quadrupole order parameter Q = (1/N) Σ s_i² (float)

Returns a dict {"energy": ndarray, "m": ndarray, "abs_m": ndarray, "q": ndarray},
each of length n_sweeps.

The combined update is essential for BC because:
  - Wolff alone is NOT ergodic — it can't create/destroy vacancies.
  - Metropolis alone suffers from critical slowing down near T_c.

All imports are lazy (inside test functions) so pytest can *collect*
the tests before the C++ binding exists.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_sweep_returns_dict_with_all_observable_arrays() -> None:
    """sweep(n) must return a dict with keys 'energy', 'm', 'abs_m', 'q'.

    Each value must be a numpy array of length n.  The 'q' key is what
    distinguishes BC from Ising — it tracks the quadrupole order parameter
    Q = (1/N) Σ s_i², which measures the fraction of magnetic (non-vacant) sites.
    """
    from pbc_datagen._core import BlumeCapelModel

    model = BlumeCapelModel(L=4, seed=42)
    model.set_temperature(2.0)
    model.set_crystal_field(0.5)

    n = 20
    result = model.sweep(n)

    assert isinstance(result, dict), f"sweep() should return dict, got {type(result)}"
    for key in ("energy", "m", "abs_m", "q"):
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
    from pbc_datagen._core import BlumeCapelModel

    model = BlumeCapelModel(L=4, seed=42)
    model.set_crystal_field(0.5)
    with pytest.raises((ValueError, RuntimeError)):
        model.sweep(10)


# ---------------------------------------------------------------------------
# Observable tracking
# ---------------------------------------------------------------------------


def test_sweep_last_observable_matches_model_state() -> None:
    """The last recorded observable must match the model's current state.

    After sweep(n), model.energy() should equal result['energy'][-1],
    and similarly for magnetization, abs_magnetization, and quadrupole.
    This ensures the tracking stays in sync with the actual spin state.
    """
    from pbc_datagen._core import BlumeCapelModel

    model = BlumeCapelModel(L=8, seed=42)
    model.set_temperature(2.0)
    model.set_crystal_field(0.5)

    result = model.sweep(100)

    assert result["energy"][-1] == pytest.approx(model.energy(), abs=1e-12), (
        f"Last energy {result['energy'][-1]} != model.energy() {model.energy()}"
    )
    assert result["m"][-1] == pytest.approx(model.magnetization(), abs=1e-14), (
        f"Last m {result['m'][-1]} != model.magnetization() {model.magnetization()}"
    )
    assert result["abs_m"][-1] == pytest.approx(model.abs_magnetization(), abs=1e-14), (
        f"Last |m| {result['abs_m'][-1]} != model.abs_magnetization() {model.abs_magnetization()}"
    )
    assert result["q"][-1] == pytest.approx(model.quadrupole(), abs=1e-14), (
        f"Last q {result['q'][-1]} != model.quadrupole() {model.quadrupole()}"
    )


# ---------------------------------------------------------------------------
# Detailed balance (gold standard for combined sweep)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("T", "D"),
    [
        (2.0, 0.0),  # Ising limit — 81 states collapse to 16 occupied
        (1.5, 0.5),  # moderate crystal field
        (2.0, 1.0),  # D > 0 — vacancies should appear in equilibrium
        (3.0, -0.5),  # D < 0 — vacancies are penalized
    ],
)
def test_sweep_detailed_balance_2x2(T: float, D: float) -> None:
    """Combined Wolff+Metropolis sweep on 2×2 must reproduce exact P(E).

    The 2×2 BC model has 3⁴ = 81 states.  We enumerate all states,
    compute exact P(E) from the partition function, and chi-squared test
    the energy histogram from sweep().

    This implicitly verifies that BOTH algorithms run correctly together:
    if Metropolis were missing, the system could never create vacancies
    and the D>0 cases would fail spectacularly.
    """
    from pbc_datagen._core import BlumeCapelModel
    from scipy.stats import chisquare

    from tests.exact_2x2 import bc_exact_probabilities

    exact_dist = bc_exact_probabilities(T, D)

    model = BlumeCapelModel(L=2, seed=42)
    model.set_temperature(T)
    model.set_crystal_field(D)

    # Equilibrate (longer burn-in needed at low T where autocorrelation is longer)
    model.sweep(2000)

    # Sample (1M to handle autocorrelation at lower T)
    n_samples = 1_000_000
    result = model.sweep(n_samples)
    energies = result["energy"]

    # Histogram the sampled energies
    energy_counts: dict[float, int] = {}
    for e in energies:
        e_key = round(float(e), 8)
        energy_counts[e_key] = energy_counts.get(e_key, 0) + 1

    # Build observed/expected arrays, sorted by energy
    all_energies = sorted(set(exact_dist.keys()) | set(energy_counts.keys()))
    observed = np.array([energy_counts.get(e, 0) for e in all_energies], dtype=float)
    expected = np.array([exact_dist.get(e, 0.0) for e in all_energies]) * n_samples

    # Drop bins with expected count < 5 (chi-squared validity requirement)
    keep = expected >= 5
    obs_arr = observed[keep]
    exp_arr = expected[keep]

    stat = chisquare(obs_arr, exp_arr)
    assert stat.pvalue > 0.001, (
        f"Detailed balance violated at T={T}, D={D}: "
        f"chi2={stat.statistic:.1f}, p={stat.pvalue:.6f}\n"
        f"  observed: {obs_arr}\n"
        f"  expected: {exp_arr}"
    )


# ---------------------------------------------------------------------------
# Ergodicity
# ---------------------------------------------------------------------------


def test_sweep_ergodicity_opposite_starts() -> None:
    """Starting from all-+1 and all-0, both must converge to same ⟨E⟩.

    At T=3.0 with D=0.5, the system is in the disordered regime.
    Starting from a fully magnetic state (Q=1) and a fully vacant state
    (Q=0), both must thermalize to the same equilibrium energy.

    We use Welch's t-test on the two energy time series to check whether
    their means are statistically indistinguishable (p > 0.001).

    This is the strongest ergodicity check for BC: it proves that the
    combined sweep can both CREATE vacancies (from all-+1) and FILL
    vacancies (from all-0), connecting the two sectors of configuration
    space that Wolff alone cannot bridge.
    """
    from pbc_datagen._core import BlumeCapelModel
    from scipy.stats import ttest_ind

    L = 8
    N = L * L
    T = 3.0
    D = 0.5
    n_equil = 500
    n_measure = 5000

    # Run 1: cold start (all +1, the default)
    model_mag = BlumeCapelModel(L=L, seed=123)
    model_mag.set_temperature(T)
    model_mag.set_crystal_field(D)
    model_mag.sweep(n_equil)
    result_mag = model_mag.sweep(n_measure)

    # Run 2: all-vacancy start (Q=0)
    model_vac = BlumeCapelModel(L=L, seed=456)
    model_vac.set_temperature(T)
    model_vac.set_crystal_field(D)
    for site in range(N):
        model_vac.set_spin(site, 0)
    model_vac.sweep(n_equil)
    result_vac = model_vac.sweep(n_measure)

    e_mag = result_mag["energy"]
    e_vac = result_vac["energy"]

    # Welch's t-test (unequal variance OK)
    stat = ttest_ind(e_mag, e_vac, equal_var=False)
    assert stat.pvalue > 0.001, (
        f"Ergodicity check failed: means differ significantly.\n"
        f"  ⟨E⟩_magnetic={np.mean(e_mag):.2f} ± {np.std(e_mag):.2f}\n"
        f"  ⟨E⟩_vacancy ={np.mean(e_vac):.2f} ± {np.std(e_vac):.2f}\n"
        f"  t={stat.statistic:.2f}, p={stat.pvalue:.6f}"
    )
