"""Red-phase tests for Step 1.3.4: AT sweep() = embedded Wolff + Metropolis, observable tracking.

sweep(n_sweeps) performs n_sweeps iterations of:
    1. One full Metropolis sweep (2N random proposals: N for sigma, N for tau)
    2. One embedded Wolff cluster step (grow + flip in projected Ising basis)

After each iteration it records all 7 observables:
    - energy:        full Hamiltonian H = -J Sigma_ij sigma_i sigma_j
                     - J Sigma_ij tau_i tau_j - U Sigma_ij sigma_i sigma_j tau_i tau_j
    - m_sigma:       (1/N) Sigma sigma_i
    - abs_m_sigma:   (1/N) |Sigma sigma_i|
    - m_tau:         (1/N) Sigma tau_i
    - abs_m_tau:     (1/N) |Sigma tau_i|
    - m_baxter:      (1/N) Sigma sigma_i tau_i  (Baxter order parameter)
    - abs_m_baxter:  (1/N) |Sigma sigma_i tau_i|

Returns a dict with these 7 keys, each mapping to an ndarray of length n_sweeps.

The combined update is essential for AT because:
  - Embedded Wolff kills critical slowing down near the phase transition.
  - Metropolis handles local decorrelation on both layers.
  - Together they guarantee ergodicity across the full 2^{2N} state space.

All imports are lazy (inside test functions) so pytest can *collect*
the tests before the C++ binding exists.
"""

from __future__ import annotations

import numpy as np
import pytest

# Expected keys in the sweep() result dict.
AT_OBSERVABLE_KEYS = (
    "energy",
    "m_sigma",
    "abs_m_sigma",
    "m_tau",
    "abs_m_tau",
    "m_baxter",
    "abs_m_baxter",
)

# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_sweep_returns_dict_with_all_observable_arrays() -> None:
    """sweep(n) must return a dict with all 7 AT observable keys.

    Each value must be a numpy array of length n.  The 7-key contract
    distinguishes AT from Ising (3 keys) and BC (4 keys), and matches
    the Model Interface in PLAN.md.
    """
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(L=4, seed=42)
    model.set_temperature(2.0)
    model.set_four_spin_coupling(0.5)

    n = 20
    result = model.sweep(n)

    assert isinstance(result, dict), f"sweep() should return dict, got {type(result)}"
    for key in AT_OBSERVABLE_KEYS:
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
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(L=4, seed=42)
    model.set_four_spin_coupling(0.5)
    with pytest.raises((ValueError, RuntimeError)):
        model.sweep(10)


# ---------------------------------------------------------------------------
# Observable tracking
# ---------------------------------------------------------------------------


def test_sweep_last_observable_matches_model_state() -> None:
    """The last recorded observable must match the model's current state.

    After sweep(n), each model observable method (energy, m_sigma, etc.)
    should return a value that exactly matches the last element of the
    corresponding array in the result dict.  This ensures the tracking
    stays in sync with the actual spin configuration through all 7
    observables.
    """
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(L=8, seed=42)
    model.set_temperature(2.0)
    model.set_four_spin_coupling(0.5)

    result = model.sweep(100)

    # Map dict keys to model methods
    key_to_method = {
        "energy": model.energy,
        "m_sigma": model.m_sigma,
        "abs_m_sigma": model.abs_m_sigma,
        "m_tau": model.m_tau,
        "abs_m_tau": model.abs_m_tau,
        "m_baxter": model.m_baxter,
        "abs_m_baxter": model.abs_m_baxter,
    }

    for key, method in key_to_method.items():
        expected = method()
        actual = result[key][-1]
        assert actual == pytest.approx(expected, abs=1e-12), (
            f"Last {key} = {actual} != model.{method.__name__}() = {expected}"
        )


# ---------------------------------------------------------------------------
# Detailed balance (gold standard for combined sweep)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("T", "U"),
    [
        (2.0, 0.0),  # decoupled: two independent Ising models
        (1.5, 0.5),  # lower T, weak four-spin coupling
        (2.0, 1.0),  # boundary: U = J
        (3.0, 0.5),  # high T, weak coupling
        (2.0, 1.5),  # remapped regime (U > 1)
        (6.0, 3.0),  # high U, baxter phase
    ],
)
def test_sweep_detailed_balance_2x2(T: float, U: float) -> None:
    """Combined Wolff+Metropolis sweep on 2x2 must reproduce exact P(E).

    The 2x2 Ashkin-Teller model has 2^8 = 256 microstates (sigma_i, tau_i
    in {-1, +1} for 4 sites).  We enumerate all states, compute exact P(E)
    from the partition function, histogram the sampled energies, and
    chi-squared test against exact probabilities.

    The U=1.5 case (remapped regime) is critical: it exercises the s=sigma*tau
    basis transformation inside the embedded Wolff step while Metropolis
    still operates in physical (sigma, tau) basis.  If the remapping
    translation back to physical spins is wrong, P(E) will be visibly off.
    """
    from pbc_datagen._core import AshkinTellerModel
    from scipy.stats import chisquare

    from tests.exact_2x2 import at_exact_probabilities

    exact_probs = at_exact_probabilities(T, U)
    energy_levels = sorted(exact_probs.keys())

    model = AshkinTellerModel(L=2, seed=42)
    model.set_temperature(T)
    model.set_four_spin_coupling(U)

    # Equilibrate
    model.sweep(2000)

    # Sample
    n_samples = 500_000
    result = model.sweep(n_samples)
    energies = result["energy"]

    # Histogram the sampled energies
    energy_counts: dict[float, int] = {E: 0 for E in energy_levels}
    for e in energies:
        e_key = round(float(e), 8)
        assert e_key in energy_counts, (
            f"Unexpected energy level {e_key} (known levels: {energy_levels})"
        )
        energy_counts[e_key] += 1

    observed = np.array([energy_counts[E] for E in energy_levels], dtype=float)
    expected = np.array([exact_probs[E] for E in energy_levels]) * n_samples

    stat = chisquare(observed, expected)
    assert stat.pvalue > 0.001, (
        f"Detailed balance violated at T={T}, U={U}: "
        f"chi2={stat.statistic:.1f}, p={stat.pvalue:.6f}\n"
        f"  observed: {observed}\n"
        f"  expected: {expected}"
    )


# ---------------------------------------------------------------------------
# Ergodicity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("T", "U"),
    [
        (5.0, 0.0),  # decoupled, disordered
        (5.0, 0.5),  # weak four-spin, disordered
        (3.0, 1.5),  # remapped regime, high T
        (6.0, 3.0),  # high U, baxter phase
    ],
)
def test_sweep_ergodicity_cold_vs_random(T: float, U: float) -> None:
    """Cold start and seeded-random start must converge to the same <E>.

    Run 1: cold start (all sigma=+1, all tau=+1 — the constructor default).
    Run 2: random start (each sigma_i, tau_i drawn uniformly from {-1, +1}
            using a seeded numpy RNG for reproducibility).

    At high T the system is deep in the disordered phase, so both must
    thermalize to the same equilibrium.  Welch's t-test checks that the
    energy means are statistically indistinguishable (p > 0.001).

    The U=1.5 case exercises the remapped Wolff codepath from a random
    initial condition — verifying that the s=sigma*tau projection doesn't
    need a special starting configuration to reach equilibrium.
    """
    from pbc_datagen._core import AshkinTellerModel
    from scipy.stats import ttest_ind

    L = 8
    N = L * L
    n_equil = 500
    n_measure = 5000

    # Run 1: cold start (all sigma=+1, all tau=+1, the default)
    model_cold = AshkinTellerModel(L=L, seed=123)
    model_cold.set_temperature(T)
    model_cold.set_four_spin_coupling(U)
    model_cold.sweep(n_equil)
    result_cold = model_cold.sweep(n_measure)

    # Run 2: seeded-random start
    init_rng = np.random.Generator(np.random.PCG64(seed=999))
    model_rand = AshkinTellerModel(L=L, seed=456)
    model_rand.set_temperature(T)
    model_rand.set_four_spin_coupling(U)
    for site in range(N):
        model_rand.set_sigma(site, int(init_rng.choice([-1, 1])))
        model_rand.set_tau(site, int(init_rng.choice([-1, 1])))
    model_rand.sweep(n_equil)
    result_rand = model_rand.sweep(n_measure)

    e_cold = result_cold["energy"]
    e_rand = result_rand["energy"]

    stat = ttest_ind(e_cold, e_rand, equal_var=False)
    assert stat.pvalue > 0.001, (
        f"Ergodicity check failed at T={T}, U={U}: means differ significantly.\n"
        f"  <E>_cold  = {np.mean(e_cold):.2f} +/- {np.std(e_cold):.2f}\n"
        f"  <E>_random= {np.mean(e_rand):.2f} +/- {np.std(e_rand):.2f}\n"
        f"  t={stat.statistic:.2f}, p={stat.pvalue:.6f}"
    )


# ---------------------------------------------------------------------------
# U > 1 remapped regime: sigma-tau symmetry check
# ---------------------------------------------------------------------------


def test_sweep_remapped_preserves_sigma_tau_symmetry() -> None:
    """At U > 1 the embedded Wolff uses s=sigma*tau remapping internally.

    The Hamiltonian has exact sigma <-> tau symmetry (J_sigma = J_tau = 1),
    so at equilibrium <|m_sigma|> must equal <|m_tau|> (up to statistics).

    If the remapping logic incorrectly breaks this symmetry — e.g., by
    always clustering sigma but never effectively updating tau — the
    absolute magnetizations will diverge.

    We use Welch's t-test to verify <|m_sigma|> ~ <|m_tau|> (p > 0.001).
    """
    from pbc_datagen._core import AshkinTellerModel
    from scipy.stats import ttest_ind

    L = 8
    T = 2.0
    U = 1.5  # remapped regime
    n_equil = 1000
    n_measure = 10_000

    model = AshkinTellerModel(L=L, seed=42)
    model.set_temperature(T)
    model.set_four_spin_coupling(U)

    # Equilibrate
    model.sweep(n_equil)

    # Sample
    result = model.sweep(n_measure)

    abs_m_sigma = result["abs_m_sigma"]
    abs_m_tau = result["abs_m_tau"]

    stat = ttest_ind(abs_m_sigma, abs_m_tau, equal_var=False)
    assert stat.pvalue > 0.001, (
        f"sigma-tau symmetry broken at U={U}: "
        f"<|m_sigma|>={np.mean(abs_m_sigma):.4f}, "
        f"<|m_tau|>={np.mean(abs_m_tau):.4f}, "
        f"t={stat.statistic:.2f}, p={stat.pvalue:.6f}"
    )
