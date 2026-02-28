"""Red-phase tests for Step 1.2.3: Blume-Capel Metropolis sweep.

The BC Metropolis algorithm proposes single-site spin changes:
  - Proposal: pick random site, propose new_spin from {-1, 0, +1} \\ {current}
    (symmetric proposal — no Hastings correction needed).
  - ΔE = -(s_new - s_old) × Σ_neighbors s_j  +  D × (s_new² - s_old²)
  - Accept if ΔE ≤ 0 or uniform() < exp(-ΔE/T).

Unlike Ising Metropolis (which only flips ±1 → ∓1), BC Metropolis can
propose transitions to/from the vacancy state (s = 0).  This is the
ONLY way vacancies are created or destroyed — Wolff can't do it.

We expose:
  - _delta_energy(site, new_spin) -> float : energy change for proposed transition
  - _metropolis_sweep() -> int : one sweep (N proposals), returns # accepted

All imports are lazy so pytest can collect before the C++ binding exists.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _delta_energy: exact values on known configurations
# ---------------------------------------------------------------------------


def test_delta_energy_magnetic_flip_no_crystal_field() -> None:
    """On all-+1 with D=0, flipping any site to -1 costs ΔE = +8.

    ΔE = -(s_new - s_old) × Σ_nbr + D × (s_new² - s_old²)
       = -((-1) - 1) × 4 + 0 × (1 - 1) = +8.

    Same as Ising — the crystal field plays no role in ±1 → ∓1 flips
    because s_new² = s_old² = 1.
    """
    from pbc_datagen._core import BlumeCapelModel

    model = BlumeCapelModel(L=8, seed=42)
    model.set_temperature(2.0)
    model.set_crystal_field(0.0)

    for site in range(model.L * model.L):
        de = model._delta_energy(site, -1)
        assert de == pytest.approx(8.0, abs=1e-12), (
            f"ΔE at site {site} for +1→-1 on cold start should be +8, got {de}"
        )


def test_delta_energy_to_vacancy() -> None:
    """On all-+1, proposing site → 0 gives ΔE = 4 - D.

    ΔE = -(0 - 1) × Σ_nbr + D × (0² - 1²)
       = 1 × 4 + D × (-1)  =  4 - D.

    At D=0: creating a vacancy costs +4 (breaks all aligned bonds).
    At D=2: costs +2 (crystal field partially offsets the bond loss).
    At D=5: actually GAINS energy (-1), vacancy creation is favorable.
    """
    from pbc_datagen._core import BlumeCapelModel

    for D, expected_dE in [(0.0, 4.0), (2.0, 2.0), (5.0, -1.0)]:
        model = BlumeCapelModel(L=8, seed=42)
        model.set_temperature(2.0)
        model.set_crystal_field(D)

        de = model._delta_energy(0, 0)
        assert de == pytest.approx(expected_dE, abs=1e-12), (
            f"ΔE for +1→0 at D={D}: expected {expected_dE}, got {de}"
        )


def test_delta_energy_from_vacancy() -> None:
    """On all-0 (vacancies), proposing site → +1 gives ΔE = +D.

    ΔE = -(1 - 0) × Σ_nbr + D × (1² - 0²)
       = -1 × 0 + D × 1  =  D.

    All neighbors are 0, so the coupling term vanishes entirely.
    The only cost is the crystal-field penalty for creating a magnetic site.
    """
    from pbc_datagen._core import BlumeCapelModel

    for D, expected_dE in [(0.0, 0.0), (2.0, 2.0), (-1.0, -1.0)]:
        L = 4
        model = BlumeCapelModel(L=L, seed=42)
        model.set_temperature(2.0)
        model.set_crystal_field(D)

        # Set all sites to vacancy
        for site in range(L * L):
            model.set_spin(site, 0)

        de = model._delta_energy(0, 1)
        assert de == pytest.approx(expected_dE, abs=1e-12), (
            f"ΔE for 0→+1 on all-vacancy at D={D}: expected {expected_dE}, got {de}"
        )


def test_delta_energy_crystal_field_only_affects_vacancy_transitions() -> None:
    """D shifts ΔE for transitions that change s², but NOT for ±1 → ∓1.

    For ±1 → ∓1: Δ(s²) = 1 - 1 = 0, so D cancels completely.
    For +1 → 0:  Δ(s²) = 0 - 1 = -1, so ΔE decreases by D per unit D.

    We verify by computing ΔE at D=0 and D=3, checking that the magnetic
    flip is unchanged while the vacancy transition shifts by exactly -3.
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 8
    # Compute ΔE at two different D values on the same configuration (cold start)
    dE_at_D: dict[float, dict[str, float]] = {}
    for D in [0.0, 3.0]:
        model = BlumeCapelModel(L=L, seed=42)
        model.set_temperature(2.0)
        model.set_crystal_field(D)

        dE_at_D[D] = {
            "flip": model._delta_energy(0, -1),  # +1 → -1
            "to_vac": model._delta_energy(0, 0),  # +1 → 0
        }

    # Magnetic flip: D must NOT matter (Δs² = 0)
    assert dE_at_D[0.0]["flip"] == pytest.approx(dE_at_D[3.0]["flip"], abs=1e-12), (
        f"D changed ΔE for ±1→∓1: D=0 gave {dE_at_D[0.0]['flip']}, D=3 gave {dE_at_D[3.0]['flip']}"
    )

    # Vacancy transition: ΔE should shift by D × Δ(s²) = 3 × (-1) = -3
    actual_shift = dE_at_D[3.0]["to_vac"] - dE_at_D[0.0]["to_vac"]
    assert actual_shift == pytest.approx(-3.0, abs=1e-12), (
        f"ΔE shift for +1→0 from D=0 to D=3: expected -3.0, got {actual_shift}"
    )


# ---------------------------------------------------------------------------
# _metropolis_sweep: acceptance rate and vacancy creation
# ---------------------------------------------------------------------------


def test_metropolis_sweep_returns_accepted_count_in_valid_range() -> None:
    """_metropolis_sweep() returns an int in [0, N].

    Each sweep proposes N spin changes; between 0 and N can be accepted.
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 8
    N = L * L
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(2.0)
    model.set_crystal_field(0.5)

    for _ in range(10):
        accepted = model._metropolis_sweep()
        assert isinstance(accepted, int)
        assert 0 <= accepted <= N, f"Accepted count {accepted} outside [0, {N}]"


def test_metropolis_cold_start_low_temp_near_zero_acceptance() -> None:
    """At T → 0 on a cold start with D=0, almost no changes are accepted.

    Every magnetic flip costs ΔE = +8, and every +1→0 costs ΔE = +4.
    At T=0.1, exp(-4/0.1) ≈ 2×10⁻¹⁸, so acceptance ≈ 0.
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 16
    N = L * L
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(0.1)
    model.set_crystal_field(0.0)

    total_accepted = 0
    n_sweeps = 20
    for _ in range(n_sweeps):
        total_accepted += model._metropolis_sweep()

    acceptance_rate = total_accepted / (n_sweeps * N)
    assert acceptance_rate < 0.01, (
        f"Acceptance rate {acceptance_rate:.4f} too high at T=0.1 (expected ≈ 0)"
    )


def test_metropolis_high_temp_near_full_acceptance() -> None:
    """At T >> 1, acceptance rate approaches 100%.

    At high temperature, exp(-ΔE/T) ≈ 1 for all ΔE, so nearly every
    proposed spin change is accepted.
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 16
    N = L * L
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(1000.0)
    model.set_crystal_field(0.5)

    total_accepted = 0
    n_sweeps = 50
    for _ in range(n_sweeps):
        total_accepted += model._metropolis_sweep()

    acceptance_rate = total_accepted / (n_sweeps * N)
    assert acceptance_rate > 0.95, (
        f"Acceptance rate {acceptance_rate:.4f} too low at T=1000 (expected > 0.95)"
    )


def test_metropolis_creates_vacancies_when_favorable() -> None:
    """At D > 0, Metropolis creates vacancies — something Wolff cannot do.

    Wolff only flips ±1 → ∓1 and never touches vacancies.  Metropolis is
    the sole mechanism for 0 ↔ ±1 transitions.  Starting from an all-+1
    cold state with large D (strongly favoring vacancies), after many
    Metropolis sweeps the quadrupole Q = (1/N)Σs² should drop well below 1.
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 8
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(2.0)
    model.set_crystal_field(3.0)  # strong vacancy preference

    for _ in range(200):
        model._metropolis_sweep()

    Q = model.quadrupole()
    assert Q < 0.9, (
        f"Q = {Q:.3f} after 200 Metropolis sweeps at D=3 — expected vacancies to appear (Q < 0.9)"
    )


# ---------------------------------------------------------------------------
# Detailed balance: 2×2 exact partition function (81 states)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("T", "D"),
    [
        (2.0, 0.0),  # Ising limit, moderate T
        (1.5, 0.0),  # Ising limit, lower T
        (2.0, 1.0),  # D > 0 favors vacancies
        (2.0, -1.0),  # D < 0 penalizes vacancies
        (3.0, 0.5),  # high T, weak crystal field
    ],
)
def test_metropolis_detailed_balance_2x2(T: float, D: float) -> None:
    """Metropolis on 2×2 BC must reproduce exact Boltzmann distribution.

    The 2×2 BC model has 3⁴ = 81 states.  We enumerate all states,
    compute exact P(E) from the partition function, histogram the sampled
    energies, and chi-squared test against the exact distribution.

    Bins with expected count < 5 are dropped (chi-squared requirement;
    see LESSONS.md).
    """
    from pbc_datagen._core import BlumeCapelModel
    from scipy.stats import chisquare

    from tests.exact_2x2 import bc_exact_probabilities

    exact_dist = bc_exact_probabilities(T, D)

    model = BlumeCapelModel(L=2, seed=42)
    model.set_temperature(T)
    model.set_crystal_field(D)

    # Equilibrate
    for _ in range(500):
        model._metropolis_sweep()

    # Sample
    n_samples = 500_000
    energy_counts: dict[float, int] = {}
    for _ in range(n_samples):
        model._metropolis_sweep()
        e = round(model.energy(), 8)
        energy_counts[e] = energy_counts.get(e, 0) + 1

    # Build observed/expected arrays, sorted by energy
    all_energies = sorted(set(exact_dist.keys()) | set(energy_counts.keys()))
    observed = np.array([energy_counts.get(e, 0) for e in all_energies], dtype=float)
    expected = np.array([exact_dist.get(e, 0.0) for e in all_energies]) * n_samples

    # Drop bins with expected count < 5 (chi-squared validity requirement)
    keep = expected >= 5
    obs_arr = observed[keep]
    exp_arr = expected[keep]

    result = chisquare(obs_arr, exp_arr)
    assert result.pvalue > 0.001, (
        f"Detailed balance violated at T={T}, D={D}: "
        f"chi2={result.statistic:.1f}, p={result.pvalue:.6f}\n"
        f"  observed: {obs_arr}\n"
        f"  expected: {exp_arr}"
    )
