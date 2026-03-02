"""Tests for PT orchestration — Phase A/B ladder tuning & equilibration."""

from __future__ import annotations

import numpy as np
import pytest
from pbc_datagen.parallel_tempering import (
    PTEngine,
    kth_check_convergence,
    kth_redistribute,
)

# ---------------------------------------------------------------------------
# KTH redistribution — pure math, synthetic f(T) data
# ---------------------------------------------------------------------------


class TestKthRedistribute:
    """Test the KTH temperature redistribution algorithm.

    kth_redistribute(temps, f) takes current temperatures and the measured
    up-fraction f(T), returns *target* temperatures (before damping).
    Endpoints T_min and T_max are always fixed.
    """

    def test_linear_f_no_change(self) -> None:
        """If f(T) is already perfectly linear, temps shouldn't move.

        Linear f means uniform diffusion current — the ladder is already
        optimal.  The target temps should equal the input temps.
        """
        temps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Perfect linear: f=1 at cold end, f=0 at hot end
        f = np.linspace(1.0, 0.0, len(temps))

        target = kth_redistribute(temps, f)

        assert target[0] == pytest.approx(temps[0])  # T_min fixed
        assert target[-1] == pytest.approx(temps[-1])  # T_max fixed
        np.testing.assert_allclose(target, temps, atol=0.05)

    def test_bottleneck_concentrates_temps(self) -> None:
        """A steep df/dT region (bottleneck) should attract temperatures.

        Construct f(T) with a sharp drop in the middle — this is where
        replicas struggle to diffuse.  After redistribution, more temps
        should cluster near the steep region.
        """
        # 7 uniformly spaced temps
        temps = np.linspace(1.0, 4.0, 7)
        # f(T) with a steep drop between T=2.0 and T=2.5 (indices 2-3)
        f = np.array([1.0, 0.95, 0.85, 0.15, 0.05, 0.02, 0.0])

        target = kth_redistribute(temps, f)

        assert target[0] == pytest.approx(temps[0])
        assert target[-1] == pytest.approx(temps[-1])
        # The gap between the two middle temps should shrink
        # (temps concentrate where df/dT is large)
        gap_before = temps[3] - temps[2]  # 0.5 (uniform spacing)
        gap_after = target[3] - target[2]
        assert gap_after < gap_before

    def test_endpoints_fixed(self) -> None:
        """T_min and T_max must never change, regardless of f(T)."""
        temps = np.array([0.5, 1.5, 2.5, 3.5])
        f = np.array([1.0, 0.3, 0.1, 0.0])  # arbitrary

        target = kth_redistribute(temps, f)

        assert target[0] == pytest.approx(0.5)
        assert target[-1] == pytest.approx(3.5)

    def test_output_sorted_ascending(self) -> None:
        """Target temperatures must be sorted ascending."""
        temps = np.linspace(1.0, 5.0, 10)
        f = np.linspace(1.0, 0.0, 10)

        target = kth_redistribute(temps, f)

        assert np.all(np.diff(target) > 0)


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------


class TestConvergence:
    """Test the convergence criterion for KTH tuning.

    Convergence requires BOTH:
    1. Temperatures stable: max relative change < tol
    2. f(T) linear: R² of linear fit > 0.9
    """

    def test_converged(self) -> None:
        """Both conditions met → converged."""
        old_temps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        new_temps = np.array([1.0, 2.005, 3.002, 3.998, 5.0])  # < 1% change
        f = np.linspace(1.0, 0.0, 5)  # perfectly linear → R² = 1

        assert kth_check_convergence(old_temps, new_temps, f, tol=0.01)

    def test_not_converged_temps_unstable(self) -> None:
        """Temps moved too much → not converged, even if f is linear."""
        old_temps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        new_temps = np.array([1.0, 2.5, 3.0, 3.5, 5.0])  # > 1% change
        f = np.linspace(1.0, 0.0, 5)

        assert not kth_check_convergence(old_temps, new_temps, f, tol=0.01)

    def test_not_converged_f_nonlinear(self) -> None:
        """f(T) strongly nonlinear → not converged, even if temps stable."""
        old_temps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        new_temps = old_temps.copy()  # identical
        # Step function — strongly nonlinear (R² ≈ 0.6)
        f = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

        assert not kth_check_convergence(old_temps, new_temps, f, tol=0.01)


# ---------------------------------------------------------------------------
# Integration: tune_ladder actually converges on a real system
# ---------------------------------------------------------------------------


class TestTuneLadder:
    """Integration test: KTH tuning on small Ising system."""

    def test_tune_converges(self) -> None:
        """4×4 Ising with 5 replicas — tuning should converge."""
        engine = PTEngine(
            model_type="ising",
            L=4,
            param_value=1.0,
            T_range=(2.0, 2.5),
            n_replicas=5,
            seed=12345,
        )
        engine.tune_ladder()

        # After tuning, temps should still be sorted ascending
        assert np.all(np.diff(engine.temps) > 0)
        # Endpoints preserved
        assert engine.temps[0] == pytest.approx(2.0)
        assert engine.temps[-1] == pytest.approx(2.5)
        # Ladder is now locked
        assert engine.ladder_locked

    def test_acceptance_rate_safety(self) -> None:
        """If any gap acceptance rate < 10%, tune_ladder should abort.

        We force this by using an absurdly wide T range with few replicas.
        Either convergence fails or the post-tuning acceptance check fails —
        both are RuntimeError, both mean the ladder is broken.
        """
        engine = PTEngine(
            model_type="ising",
            L=4,
            param_value=1.0,
            T_range=(0.5, 50.0),  # absurd range
            n_replicas=3,  # way too few
            seed=42,
        )
        with pytest.raises(RuntimeError, match="(acceptance rate|converge)"):
            engine.tune_ladder(max_iterations=10)


# ---------------------------------------------------------------------------
# Phase B: Welch equilibration check — pure function, synthetic obs_streams
# ---------------------------------------------------------------------------


class TestWelchEquilibrationCheck:
    """Test the Welch t-test equilibration check.

    welch_equilibration_check(obs_streams, alpha) takes the obs_streams
    dict from PTResult (obs_streams[name][T_slot] = list of values)
    and returns True if all observables at all T slots pass the
    Welch t-test (first 20% vs last 20%), Bonferroni-corrected.
    """

    def test_stationary_passes(self) -> None:
        """IID Gaussian streams should pass — no drift."""
        from pbc_datagen.parallel_tempering import welch_equilibration_check

        rng = np.random.default_rng(42)
        M = 5  # temperature slots
        N = 10_000  # samples per slot
        # obs_streams format: dict[str, list[list[float]]]
        obs_streams: dict[str, list[list[float]]] = {
            "energy": [rng.normal(0, 1, N).tolist() for _ in range(M)],
            "abs_m": [rng.normal(0.5, 0.1, N).tolist() for _ in range(M)],
        }

        assert welch_equilibration_check(obs_streams)

    def test_drifting_fails(self) -> None:
        """A strong linear drift should fail — means are clearly different."""
        from pbc_datagen.parallel_tempering import welch_equilibration_check

        rng = np.random.default_rng(42)
        M = 5
        N = 10_000
        obs_streams: dict[str, list[list[float]]] = {}
        # Stationary observable — passes on its own
        obs_streams["energy"] = [rng.normal(0, 1, N).tolist() for _ in range(M)]
        # Drifting observable at slot 0 — first 20% mean ≠ last 20% mean
        drifting = np.linspace(-5, 5, N) + rng.normal(0, 0.1, N)
        slots = [rng.normal(0, 1, N).tolist() for _ in range(M)]
        slots[0] = drifting.tolist()
        obs_streams["abs_m"] = slots

        assert not welch_equilibration_check(obs_streams)

    def test_bonferroni_correction(self) -> None:
        """With many T slots × observables, threshold tightens.

        A marginal difference that would fail at α=0.05 might pass
        after Bonferroni correction with many tests.
        """
        from pbc_datagen.parallel_tempering import welch_equilibration_check

        rng = np.random.default_rng(123)
        M = 20  # many slots
        N = 10_000
        # All stationary — with 20 slots × 1 obs = 20 tests,
        # Bonferroni α = 0.05/20 = 0.0025. Should still pass.
        obs_streams: dict[str, list[list[float]]] = {
            "energy": [rng.normal(0, 1, N).tolist() for _ in range(M)],
        }

        assert welch_equilibration_check(obs_streams)


# ---------------------------------------------------------------------------
# Phase B: equilibrate() integration
# ---------------------------------------------------------------------------


class TestEquilibrate:
    """Integration tests for Phase B equilibration."""

    def test_requires_locked_ladder(self) -> None:
        """Cannot equilibrate before tuning — ladder must be locked."""
        engine = PTEngine(
            model_type="ising",
            L=4,
            param_value=1.0,
            T_range=(2.0, 2.5),
            n_replicas=5,
            seed=42,
        )
        assert not engine.ladder_locked
        with pytest.raises(RuntimeError, match="locked"):
            engine.equilibrate()

    def test_equilibrate_sets_tau_max(self) -> None:
        """After A→B, tau_max is a positive float."""
        engine = PTEngine(
            model_type="ising",
            L=4,
            param_value=1.0,
            T_range=(2.0, 2.5),
            n_replicas=5,
            seed=12345,
        )
        engine.tune_ladder()
        engine.equilibrate()

        assert engine.tau_max is not None
        assert engine.tau_max > 0

    def test_temps_unchanged_after_equilibrate(self) -> None:
        """Ladder must be immutable during Phase B."""
        engine = PTEngine(
            model_type="ising",
            L=4,
            param_value=1.0,
            T_range=(2.0, 2.5),
            n_replicas=5,
            seed=12345,
        )
        engine.tune_ladder()
        temps_before = engine.temps.copy()

        engine.equilibrate()

        np.testing.assert_array_equal(engine.temps, temps_before)
