"""Tests for PT orchestration — Phase A ladder tuning (KTH feedback)."""

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
    2. f(T) linear: R² of linear fit > 0.99
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
        # Steppy f — very nonlinear
        f = np.array([1.0, 0.99, 0.5, 0.01, 0.0])

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
            param_value=0.0,
            T_range=(1.5, 4.0),
            n_replicas=5,
            seed=12345,
        )
        engine.tune_ladder()

        # After tuning, temps should still be sorted ascending
        assert np.all(np.diff(engine.temps) > 0)
        # Endpoints preserved
        assert engine.temps[0] == pytest.approx(1.5)
        assert engine.temps[-1] == pytest.approx(4.0)
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
            param_value=0.0,
            T_range=(0.5, 50.0),  # absurd range
            n_replicas=3,  # way too few
            seed=42,
        )
        with pytest.raises(RuntimeError, match="(acceptance rate|converge)"):
            engine.tune_ladder(max_iterations=10)
