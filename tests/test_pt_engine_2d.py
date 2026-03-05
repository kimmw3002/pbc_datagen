"""Tests for PTEngine2D soft-failure behaviour (Phase B).

When convergence_check always returns converged=False, equilibrate()
must NOT raise — instead it should:
  - warn and set self.disagreement_slots to a non-empty list
  - set self.tau_max (not None)
  - hand warm replicas to Phase C so produce() can run
"""

from __future__ import annotations

from unittest.mock import patch

from pbc_datagen.convergence import ConvergenceResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _always_disagree(*args, **kwargs) -> ConvergenceResult:
    """Stub that makes every slot disagree on every observable."""
    # We need to know n_slots — pull it from the first observable stream.
    # args[0] is streams_a: dict[str, NDArray]
    streams_a = args[0]
    if not streams_a:
        return ConvergenceResult(converged=False, disagreement_map={})
    obs_names = list(streams_a.keys())
    n_slots = streams_a[obs_names[0]].shape[0]
    disagreement_map = {name: [True] * n_slots for name in obs_names}
    return ConvergenceResult(converged=False, disagreement_map=disagreement_map)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhaseB_SoftFailure:
    """Phase B soft-failure: no exception, disagreement_slots set, tau_max set."""

    def test_no_exception_on_convergence_failure(self) -> None:
        """equilibrate() must not raise even when convergence_check always fails."""
        from pbc_datagen.pt_engine_2d import PTEngine2D

        engine = PTEngine2D(
            model_type="blume_capel",
            L=4,
            T_range=(0.5, 2.0),
            param_range=(0.0, 1.0),
            n_T=3,
            n_P=2,
            seed=42,
        )
        engine.connectivity_checked = True  # skip Phase A

        with patch("pbc_datagen.pt_engine_2d.convergence_check", side_effect=_always_disagree):
            # Should NOT raise — soft failure
            engine.equilibrate(n_initial=10, n_max=10)

    def test_disagreement_slots_populated(self) -> None:
        """disagreement_slots must be non-empty after soft failure."""
        from pbc_datagen.pt_engine_2d import PTEngine2D

        engine = PTEngine2D(
            model_type="blume_capel",
            L=4,
            T_range=(0.5, 2.0),
            param_range=(0.0, 1.0),
            n_T=3,
            n_P=2,
            seed=42,
        )
        engine.connectivity_checked = True

        with patch("pbc_datagen.pt_engine_2d.convergence_check", side_effect=_always_disagree):
            engine.equilibrate(n_initial=10, n_max=10)

        assert len(engine.disagreement_slots) > 0, (
            "disagreement_slots should be non-empty after soft failure"
        )
        assert engine.disagreement_slots == sorted(engine.disagreement_slots), (
            "disagreement_slots should be sorted"
        )
        assert all(0 <= s < engine.M for s in engine.disagreement_slots), (
            "all disagreement slot indices must be in [0, M)"
        )

    def test_tau_max_set(self) -> None:
        """tau_max must be set (not None) after soft failure."""
        from pbc_datagen.pt_engine_2d import PTEngine2D

        engine = PTEngine2D(
            model_type="blume_capel",
            L=4,
            T_range=(0.5, 2.0),
            param_range=(0.0, 1.0),
            n_T=3,
            n_P=2,
            seed=42,
        )
        engine.connectivity_checked = True

        with patch("pbc_datagen.pt_engine_2d.convergence_check", side_effect=_always_disagree):
            engine.equilibrate(n_initial=10, n_max=10)

        assert engine.tau_max is not None, "tau_max must be set after soft failure"
        assert engine.tau_max >= 0.0, "tau_max must be non-negative"

    def test_disagreement_slots_empty_on_clean_convergence(self) -> None:
        """On clean convergence disagreement_slots stays empty."""
        from pbc_datagen.pt_engine_2d import PTEngine2D

        def _always_converge(*args, **kwargs) -> ConvergenceResult:
            return ConvergenceResult(converged=True, disagreement_map={})

        engine = PTEngine2D(
            model_type="blume_capel",
            L=4,
            T_range=(0.5, 2.0),
            param_range=(0.0, 1.0),
            n_T=3,
            n_P=2,
            seed=42,
        )
        engine.connectivity_checked = True

        with patch("pbc_datagen.pt_engine_2d.convergence_check", side_effect=_always_converge):
            engine.equilibrate(n_initial=10, n_max=10)

        assert engine.disagreement_slots == [], (
            "disagreement_slots should be empty on clean convergence"
        )
