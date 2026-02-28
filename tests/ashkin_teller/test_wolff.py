"""Red-phase tests for Step 1.3.2: Ashkin-Teller Embedded Wolff cluster update.

The Embedded Wolff algorithm (Wiseman & Domany, 1995) extends Wolff to the
two-layer AT model by projecting onto a single Ising-like variable per step:

  Case 1 (U ≤ 1, no remapping):
    1. Pick target layer: σ or τ (50/50 random).
    2. Hold the other layer fixed.
    3. Effective bond coupling: J_eff(j,k) = J + U × fixed_j × fixed_k.
    4. Grow Wolff cluster on the target layer using J_eff.
    5. Flip target-layer spins in the cluster.  Other layer unchanged.

  Case 2 (U > 1, remapping to σ, s=στ basis):
    1. Pick target: σ or s (50/50).
    2. Hold the other fixed.
    3. If clustering σ (s held): J_eff = 1 + 1 × s_j s_k.
       If clustering s (σ held): J_eff = U + 1 × σ_j σ_k.
    4. Grow cluster on the target variable.
    5. Flip back to physical basis:
       - Flipping σ with s held → both σ AND τ flip (since τ = s·σ).
       - Flipping s with σ held → only τ flips.

We expose it as model._wolff_step() -> int (returns cluster size).

All imports are lazy (inside test functions) so pytest can *collect*
the tests even before the C++ binding is updated.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. Basic correctness
# ---------------------------------------------------------------------------


def test_wolff_flips_at_least_one_spin() -> None:
    """After _wolff_step() on a cold start, at least one σ or τ spin must flip.

    The Embedded Wolff algorithm always picks a magnetic spin as seed
    (AT has no vacancies — all spins are ±1), so the cluster contains at
    least one site.  Starting from all +1, we must see at least one -1
    in either the σ or τ layer after one step.
    """
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(8, seed=42)
    model.set_temperature(2.0)
    model.set_four_spin_coupling(0.5)

    model._wolff_step()

    sigma = model.sigma.ravel()
    tau = model.tau.ravel()
    any_flipped = np.any(sigma == -1) or np.any(tau == -1)
    assert any_flipped, "Wolff step did not flip any spins on all-+1 lattice"


# ---------------------------------------------------------------------------
# 2. Cluster size in valid range
# ---------------------------------------------------------------------------


def test_wolff_returns_cluster_size_in_valid_range() -> None:
    """_wolff_step() must return an integer in [1, N].

    The cluster always contains at least the seed site (size ≥ 1) and
    can contain at most all N sites (size ≤ N).
    """
    from pbc_datagen._core import AshkinTellerModel

    L = 8
    N = L * L
    model = AshkinTellerModel(L, seed=42)
    model.set_temperature(2.0)
    model.set_four_spin_coupling(0.5)

    for _ in range(50):
        cluster_size = model._wolff_step()
        assert isinstance(cluster_size, int), (
            f"_wolff_step() should return int, got {type(cluster_size)}"
        )
        assert 1 <= cluster_size <= N, f"Cluster size {cluster_size} outside valid range [1, {N}]"


# ---------------------------------------------------------------------------
# 3. For U ≤ 1: exactly one layer changes per step
# ---------------------------------------------------------------------------


def test_wolff_only_one_layer_changes_when_not_remapped() -> None:
    """When U ≤ 1 (no remapping), each Wolff step targets σ OR τ, not both.

    The Embedded Wolff algorithm picks one of {σ, τ} at random, holds
    the other fixed, and grows a cluster in the chosen layer.  After
    the step, exactly one of {σ, τ} should have changed; the other
    must be bit-for-bit identical.

    Over many steps, both σ-only and τ-only changes should be observed
    (since the algorithm picks 50/50).
    """
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(8, seed=42)
    model.set_temperature(2.0)
    model.set_four_spin_coupling(0.5)  # U ≤ 1 → no remapping

    sigma_changed_count = 0
    tau_changed_count = 0
    n_steps = 200

    for _ in range(n_steps):
        sigma_before = model.sigma.ravel().copy()
        tau_before = model.tau.ravel().copy()

        model._wolff_step()

        sigma_after = model.sigma.ravel()
        tau_after = model.tau.ravel()

        sigma_changed = not np.array_equal(sigma_before, sigma_after)
        tau_changed = not np.array_equal(tau_before, tau_after)

        # Exactly one layer should change (not both, not neither)
        assert sigma_changed != tau_changed, (
            f"Expected exactly one layer to change, got "
            f"σ_changed={sigma_changed}, τ_changed={tau_changed}"
        )

        if sigma_changed:
            sigma_changed_count += 1
        if tau_changed:
            tau_changed_count += 1

    # Both layers should have been targeted at some point (50/50 chance).
    # With 200 steps, not seeing one layer is astronomically unlikely.
    assert sigma_changed_count > 0, f"σ was never targeted in {n_steps} steps"
    assert tau_changed_count > 0, f"τ was never targeted in {n_steps} steps"


# ---------------------------------------------------------------------------
# 4. For U > 1: both layers can change simultaneously
# ---------------------------------------------------------------------------


def test_wolff_both_layers_can_change_when_remapped() -> None:
    """When U > 1 (remapping active), some steps flip both σ AND τ.

    In the remapped (σ, s=στ) basis:
    - Clustering σ with s held fixed → flipping σ also flips τ (both change).
    - Clustering s with σ held fixed → only τ changes.

    So we should observe steps where both σ and τ change (σ-clustering),
    and steps where only τ changes (s-clustering).
    """
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(8, seed=42)
    model.set_temperature(2.0)
    model.set_four_spin_coupling(1.5)  # U > 1 → remapping active

    both_changed_count = 0
    only_tau_changed_count = 0
    n_steps = 200

    for _ in range(n_steps):
        sigma_before = model.sigma.ravel().copy()
        tau_before = model.tau.ravel().copy()

        model._wolff_step()

        sigma_after = model.sigma.ravel()
        tau_after = model.tau.ravel()

        sigma_changed = not np.array_equal(sigma_before, sigma_after)
        tau_changed = not np.array_equal(tau_before, tau_after)

        if sigma_changed and tau_changed:
            both_changed_count += 1
        elif tau_changed and not sigma_changed:
            only_tau_changed_count += 1
        else:
            # Only σ changed (impossible) or neither changed (impossible:
            # cluster always has ≥1 site and flipping changes ≥1 spin).
            pytest.fail(
                f"Invalid flip pattern: σ_changed={sigma_changed}, "
                f"τ_changed={tau_changed}. "
                f"Only two patterns are valid in the remapped basis: "
                f"both change (σ-clustering) or only τ changes (s-clustering)."
            )

    # Over 200 steps with 50/50 choice, both patterns must be observed
    assert both_changed_count > 0, (
        f"Never saw both σ and τ change simultaneously in {n_steps} steps "
        f"(expected from σ-clustering in remapped basis)"
    )
    assert only_tau_changed_count > 0, (
        f"Never saw only τ change in {n_steps} steps (expected from s-clustering in remapped basis)"
    )


# ---------------------------------------------------------------------------
# 5. High-temperature: small clusters
# ---------------------------------------------------------------------------


def test_wolff_high_temperature_small_clusters() -> None:
    """At T >> T_c, mean cluster size should be O(1).

    At high temperature the effective bond probability is small:
    p_add = 1 - exp(-2 J_eff / T).  When T is large, J_eff / T → 0 so
    p_add → 0 and clusters rarely grow beyond the seed spin.
    """
    from pbc_datagen._core import AshkinTellerModel

    L = 32
    model = AshkinTellerModel(L, seed=42)
    model.set_temperature(100.0)  # T >> any coupling
    model.set_four_spin_coupling(0.5)

    sizes = []
    for _ in range(200):
        sizes.append(model._wolff_step())

    mean_size = np.mean(sizes)
    assert mean_size < 2, f"Mean cluster size {mean_size:.1f} is too large at T=100 (expected < 2)"


# ---------------------------------------------------------------------------
# 6. Low-temperature: large clusters on cold start
# ---------------------------------------------------------------------------


def test_wolff_low_temperature_large_clusters() -> None:
    """At T << T_c on a cold start, clusters should span most of the lattice.

    At low temperature the effective bond probability p_add ≈ 1 for
    aligned spins (J_eff > 0 guaranteed on cold start where all spins
    agree).  The first cluster should encompass nearly all N sites.
    """
    from pbc_datagen._core import AshkinTellerModel

    L = 16
    N = L * L
    model = AshkinTellerModel(L, seed=42)
    model.set_temperature(0.1)  # T << T_c
    model.set_four_spin_coupling(0.5)

    cluster_size = model._wolff_step()
    assert cluster_size > N * 0.9, (
        f"First cluster at low T has size {cluster_size}, "
        f"expected > {int(N * 0.9)} (≈ 90% of N={N})"
    )


# ---------------------------------------------------------------------------
# 7. Detailed balance: 2×2 exact partition function (256 states)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("T", "U"),
    [
        (2.0, 0.0),  # decoupled: two independent Ising models
        (2.0, 0.5),  # weakly coupled, no remapping
        (2.0, 1.0),  # boundary U = J (strongest coupling before remap)
        (2.0, 1.5),  # remapped regime (U > 1, uses σ/s basis)
    ],
)
def test_wolff_detailed_balance_2x2(T: float, U: float) -> None:
    """Embedded Wolff on 2×2 must reproduce exact Boltzmann distribution.

    The 2×2 Ashkin-Teller model has 2^8 = 256 microstates.  We enumerate
    all of them, compute exact Boltzmann probabilities P(E) at temperature T
    with four-spin coupling U, and compare against an energy histogram from
    many Wolff steps via the chi-squared test.

    This is the gold-standard test for detailed balance.  Parametrizing over
    U values from 0 (decoupled Ising) through 1.5 (remapped regime) ensures
    both code paths produce the correct equilibrium distribution.
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
        model._wolff_step()

    # Sample
    n_samples = 500_000
    energy_counts: dict[float, int] = {E: 0 for E in energy_levels}

    for _ in range(n_samples):
        model._wolff_step()
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
