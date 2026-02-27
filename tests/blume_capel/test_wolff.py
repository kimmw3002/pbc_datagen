"""Red-phase tests for Step 1.2.2: Blume-Capel Wolff cluster update.

BC Wolff is standard Wolff adapted for 3-state spins {-1, 0, +1}:
  - If spin[seed] == 0 → return immediately (cluster_size = 0).
    Vacancies can't seed clusters.
  - DFS grows through neighbors with spin[j] == seed_spin (±1).
    Spin-0 sites naturally fail this check and act as barriers.
  - Bond probability: p_add = 1 - exp(-2/T) — identical to Ising.
    D does NOT affect bond probabilities ((-s)² = s², cancels exactly).
  - Cluster flip: s → -s for all members.  Vacancies are never touched.

Key difference from Ising Wolff: cluster_size can be 0 (vacancy seed),
and vacancy sites fragment the lattice into disconnected magnetic islands.

All imports are lazy (inside test functions) so pytest can *collect*
the tests even before _wolff_step() is bound.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Basic mechanics (parallel to Ising Wolff tests)
# ---------------------------------------------------------------------------


def test_wolff_flips_at_least_one_spin() -> None:
    """On a cold start (all +1, no vacancies), at least one spin must flip.

    With all spins magnetic, the seed is guaranteed to be ±1, so the
    cluster must contain at least the seed.  Starting from all +1 we
    must see at least one -1 after a single Wolff step.
    """
    from pbc_datagen._core import BlumeCapelModel

    model = BlumeCapelModel(L=8, seed=42)
    model.set_temperature(2.269)
    model.set_crystal_field(0.0)

    model._wolff_step()

    spins = model.spins.ravel()
    assert np.any(spins == -1), "Wolff step did not flip any spins on all-+1 lattice"


def test_wolff_returns_cluster_size_in_valid_range() -> None:
    """_wolff_step() must return an int in [0, N].

    Unlike Ising (always >= 1), BC Wolff can return 0 when the seed
    lands on a vacancy.  The maximum is still N (every site in one cluster).
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 8
    N = L * L
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(2.269)
    model.set_crystal_field(0.5)

    for _ in range(50):
        cluster_size = model._wolff_step()
        assert isinstance(cluster_size, int), (
            f"_wolff_step() should return int, got {type(cluster_size)}"
        )
        assert 0 <= cluster_size <= N, f"Cluster size {cluster_size} outside valid range [0, {N}]"


def test_wolff_magnetization_changes_by_twice_cluster_size() -> None:
    """Flipping a cluster of size c changes m by exactly 2c/N.

    When c > 0: each flipped spin changes sign, so total spin sum
    changes by ±2c, and intensive m = (1/N)Σs changes by 2c/N.
    When c = 0 (vacancy seed): nothing changes, delta_m = 0.
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 16
    N = L * L
    model = BlumeCapelModel(L=L, seed=77)
    model.set_temperature(2.269)
    model.set_crystal_field(0.0)

    for _ in range(30):
        m_before = model.magnetization()
        cluster_size = model._wolff_step()
        m_after = model.magnetization()

        delta_m = abs(m_before - m_after)
        expected = 2.0 * cluster_size / N

        assert delta_m == pytest.approx(expected, abs=1e-12), (
            f"|Δm| = {delta_m}, expected 2×{cluster_size}/{N} = {expected}"
        )


# ---------------------------------------------------------------------------
# Vacancy-specific behavior
# ---------------------------------------------------------------------------


def test_wolff_all_vacancy_always_returns_zero() -> None:
    """On an all-vacancy lattice, every seed is spin-0 → cluster_size = 0.

    No spins should change because there are no magnetic sites to flip.
    This is the most direct test that vacancies can't seed clusters.
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 4
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(2.0)
    model.set_crystal_field(1.0)

    # Set every site to vacancy
    for site in range(L * L):
        model.set_spin(site, 0)

    spins_before = model.spins.copy()

    for _ in range(100):
        cluster_size = model._wolff_step()
        assert cluster_size == 0, f"Cluster size {cluster_size} on all-vacancy lattice (expected 0)"

    # Spins must be completely unchanged
    assert np.array_equal(model.spins, spins_before), (
        "Spins were modified on an all-vacancy lattice"
    )


def test_wolff_isolated_spin_max_cluster_one() -> None:
    """A lone +1 site surrounded by vacancies: cluster is at most size 1.

    If the seed lands on the lone +1, the cluster can't grow because all
    neighbors are spin-0 (fail the alignment check).  If the seed lands
    on any vacancy, cluster_size = 0.  So the only possible sizes are 0 and 1.
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 4
    N = L * L
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(0.1)  # very low T → p_add ≈ 1, cluster would be huge if it could grow
    model.set_crystal_field(0.0)

    # Set everything to 0, then place one +1
    for site in range(N):
        model.set_spin(site, 0)
    lone_site = 5  # site (1,1) on a 4×4
    model.set_spin(lone_site, 1)

    seen_sizes: set[int] = set()
    for _ in range(200):
        cluster_size = model._wolff_step()
        seen_sizes.add(cluster_size)
        assert cluster_size <= 1, (
            f"Cluster size {cluster_size} with isolated spin (max should be 1)"
        )

    # Over 200 steps, we should see both 0 (vacancy seed) and 1 (lone-spin seed)
    assert 0 in seen_sizes, "Never saw cluster_size=0 (vacancy seed) in 200 steps"
    assert 1 in seen_sizes, "Never saw cluster_size=1 (lone-spin seed) in 200 steps"


def test_wolff_vacancy_barrier_blocks_cluster_growth() -> None:
    """Two +1 islands separated by vacancy rows cannot share a cluster.

    Layout on a 4×4 PBC lattice:
      Row 0: sites 0-3   → all +1  (island A)
      Row 1: sites 4-7   → all 0   (barrier)
      Row 2: sites 8-11  → all +1  (island B)
      Row 3: sites 12-15 → all 0   (barrier — also blocks PBC wrap)

    At low T with p_add ≈ 1, a cluster in island A should flip ALL of
    row 0 but NONE of row 2 (and vice versa).  We verify that changed
    sites never span both islands.
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 4
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(0.1)  # p_add ≈ 1 — cluster fills entire island
    model.set_crystal_field(0.0)

    island_a = {0, 1, 2, 3}  # row 0
    island_b = {8, 9, 10, 11}  # row 2

    # Set barrier rows to vacancy
    for site in [4, 5, 6, 7, 12, 13, 14, 15]:
        model.set_spin(site, 0)

    for _ in range(100):
        spins_before = model.spins.ravel().copy()
        cluster_size = model._wolff_step()
        spins_after = model.spins.ravel()

        if cluster_size == 0:
            continue  # vacancy seed, nothing to check

        # Find which sites changed
        changed = set(np.where(spins_before != spins_after)[0])

        # Changed sites must be entirely within one island
        in_a = changed & island_a
        in_b = changed & island_b

        assert not (in_a and in_b), (
            f"Cluster crossed vacancy barrier! "
            f"Changed in island A: {in_a}, changed in island B: {in_b}"
        )


def test_wolff_vacancies_never_modified() -> None:
    """Wolff must never change the spin value of a vacancy site.

    Wolff only flips ±1 → ∓1 within the cluster.  Spin-0 sites should
    be completely inert: never added to the cluster, never modified.
    We sprinkle vacancies into the lattice and verify they survive
    many Wolff steps untouched.
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 8
    N = L * L
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(2.269)
    model.set_crystal_field(0.5)

    # Set every 4th site to vacancy
    vacancy_sites = list(range(0, N, 4))
    for site in vacancy_sites:
        model.set_spin(site, 0)

    for step in range(200):
        model._wolff_step()

        spins = model.spins.ravel()
        for site in vacancy_sites:
            assert spins[site] == 0, (
                f"Vacancy at site {site} was modified to {spins[site]} after step {step + 1}"
            )


# ---------------------------------------------------------------------------
# Detailed balance on pure ±1 sublattice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("T", "D"),
    [
        (0.5, 0.0),  # low T, no crystal field (pure Ising limit)
        (2.269, 0.0),  # critical T, pure Ising
        (5.0, 0.0),  # high T, pure Ising
        (2.269, 1.0),  # D > 0 should NOT affect Wolff bond probability
        (2.269, -1.0),  # D < 0 should NOT affect Wolff bond probability
    ],
)
def test_wolff_pure_magnetic_detailed_balance_2x2(T: float, D: float) -> None:
    """On a pure ±1 lattice (no vacancies), BC Wolff must reproduce Ising P(E).

    Since Wolff only flips ±1 → ∓1 and never creates vacancies, starting
    from all ±1 the system stays in the 2⁴ = 16 Ising states forever.
    The crystal field D adds D·Σs² = D·N to the energy of every state
    (because s² = 1 for all sites), so it shifts ALL energies by the same
    constant.  This doesn't affect the Boltzmann *ratios* — the sampling
    distribution is identical to Ising.

    We histogram the coupling energy E_coupling = -Σ s_i·s_j (which has
    the Ising spectrum: -8, 0, +8) and chi-squared test against
    Z(T) = 2 exp(8/T) + 12 + 2 exp(-8/T).

    Parametrizing over D values proves D doesn't leak into bond probabilities.
    """
    from pbc_datagen._core import BlumeCapelModel
    from scipy.stats import chisquare

    # Exact Ising probabilities for the coupling energy
    Z = 2 * math.exp(8 / T) + 12 + 2 * math.exp(-8 / T)
    exact_probs = {
        -8: 2 * math.exp(8 / T) / Z,
        0: 12 / Z,
        8: 2 * math.exp(-8 / T) / Z,
    }

    model = BlumeCapelModel(L=2, seed=42)
    model.set_temperature(T)
    model.set_crystal_field(D)

    # Equilibrate
    for _ in range(500):
        model._wolff_step()

    # Sample — measure coupling energy = total energy - D·N
    # (subtracting the constant crystal-field shift)
    n_samples = 500_000
    N = 4  # 2×2
    energy_counts: dict[int, int] = {-8: 0, 0: 0, 8: 0}

    for _ in range(n_samples):
        model._wolff_step()
        e_total = model.energy()
        e_coupling = round(e_total - D * N)  # remove D·Σs² = D·N for all-magnetic
        energy_counts[e_coupling] += 1

    observed = np.array([energy_counts[-8], energy_counts[0], energy_counts[8]], dtype=float)
    expected = np.array([exact_probs[-8], exact_probs[0], exact_probs[8]]) * n_samples

    result = chisquare(observed, expected)
    assert result.pvalue > 0.001, (
        f"Detailed balance violated at T={T}, D={D}: "
        f"chi2={result.statistic:.1f}, p={result.pvalue:.6f}\n"
        f"  observed: {observed}\n"
        f"  expected: {expected}"
    )
