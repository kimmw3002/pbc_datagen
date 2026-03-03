"""Red-phase tests for Step 3.4: Two-initialization convergence check.

The convergence module compares obs_streams from two independent 2D PT runs
(cold-start vs random-start) to detect metastable trapping.

The core function ``convergence_check`` takes two obs_streams dicts and:
1. Discards the first half of each stream (burn-in removal)
2. Compares the remaining samples at each (observable, slot) pair
3. Returns pass/fail per slot per observable, and overall convergence

obs_streams format: dict[str, list[list[float]]]
  obs_streams["energy"][slot] → list of sample values (one per round)

All tests use synthetic data — no C++ replicas needed.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_streams(
    n_slots: int,
    n_samples: int,
    obs_names: list[str],
    *,
    rng: np.random.Generator,
    mean: float = 0.0,
    std: float = 1.0,
) -> dict[str, list[list[float]]]:
    """Build a synthetic obs_streams dict: IID N(mean, std) at every slot."""
    return {
        obs: [rng.normal(mean, std, size=n_samples).tolist() for _ in range(n_slots)]
        for obs in obs_names
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConvergenceCheck:
    """Pure-Python convergence check on synthetic obs_streams."""

    def test_identical_streams_converge(self) -> None:
        """Passing the exact same samples from both runs must converge.

        This is the trivial case: if both initializations produced
        identical time series, there is zero between-chain variance.
        """
        from pbc_datagen.convergence import convergence_check

        rng = np.random.default_rng(42)
        streams = _make_streams(3, 2000, ["energy"], rng=rng)

        result = convergence_check(streams, streams)

        assert result.converged is True
        assert all(not d for d in result.disagreement_map["energy"])

    def test_iid_same_distribution_converges(self) -> None:
        """Two independent IID draws from N(0,1) → converged.

        Different samples but same generating distribution — the
        comparison must not false-reject.  2000 samples per stream,
        last 1000 used after burn-in discard.
        """
        from pbc_datagen.convergence import convergence_check

        rng_a = np.random.default_rng(100)
        rng_b = np.random.default_rng(200)
        streams_a = _make_streams(3, 2000, ["energy"], rng=rng_a)
        streams_b = _make_streams(3, 2000, ["energy"], rng=rng_b)

        result = convergence_check(streams_a, streams_b)

        assert result.converged is True

    def test_different_means_not_converged(self) -> None:
        """Run A ~ N(0,1), Run B ~ N(5,1) → not converged.

        A 5-sigma shift in the mean is unmistakable — the function
        must detect that the two runs sampled different distributions.
        """
        from pbc_datagen.convergence import convergence_check

        rng_a = np.random.default_rng(100)
        rng_b = np.random.default_rng(200)
        streams_a = _make_streams(3, 2000, ["energy"], rng=rng_a, mean=0.0)
        streams_b = _make_streams(3, 2000, ["energy"], rng=rng_b, mean=5.0)

        result = convergence_check(streams_a, streams_b)

        assert result.converged is False

    def test_disagreement_map_marks_correct_slots(self) -> None:
        """3 slots: slots 0,1 agree (same dist), slot 2 disagrees (shifted).

        The disagreement_map must flag only slot 2.
        """
        from pbc_datagen.convergence import convergence_check

        rng_a = np.random.default_rng(100)
        rng_b = np.random.default_rng(200)

        # Build streams where slot 2 has a big mean shift in run B
        streams_a = _make_streams(3, 2000, ["energy"], rng=rng_a, mean=0.0)
        streams_b = _make_streams(3, 2000, ["energy"], rng=rng_b, mean=0.0)
        # Overwrite slot 2 in run B with shifted distribution
        streams_b["energy"][2] = rng_b.normal(10.0, 1.0, size=2000).tolist()

        result = convergence_check(streams_a, streams_b)

        assert result.converged is False
        dmap = result.disagreement_map["energy"]
        assert dmap[0] is False, "Slot 0 should agree"
        assert dmap[1] is False, "Slot 1 should agree"
        assert dmap[2] is True, "Slot 2 should disagree"

    def test_multiple_observables_one_disagrees(self) -> None:
        """Energy agrees, magnetization disagrees → overall not converged.

        Any single observable disagreeing at any slot must fail the
        entire check — we can't trust the ensemble if one quantity
        hasn't equilibrated.
        """
        from pbc_datagen.convergence import convergence_check

        rng_a = np.random.default_rng(100)
        rng_b = np.random.default_rng(200)

        streams_a: dict[str, list[list[float]]] = {
            "energy": [rng_a.normal(0.0, 1.0, size=2000).tolist() for _ in range(2)],
            "mag": [rng_a.normal(0.0, 1.0, size=2000).tolist() for _ in range(2)],
        }
        streams_b: dict[str, list[list[float]]] = {
            "energy": [rng_b.normal(0.0, 1.0, size=2000).tolist() for _ in range(2)],
            "mag": [rng_b.normal(5.0, 1.0, size=2000).tolist() for _ in range(2)],
        }

        result = convergence_check(streams_a, streams_b)

        assert result.converged is False
        # Energy should be fine
        assert all(not d for d in result.disagreement_map["energy"])
        # Magnetization should disagree everywhere
        assert all(d for d in result.disagreement_map["mag"])

    def test_constant_observable_converges(self) -> None:
        """Both streams are constant 1.0 → converged.

        At low T, observables like |m| saturate to ~1.0 with zero
        variance.  The function must handle this without division by
        zero or NaN-induced false rejection.
        """
        from pbc_datagen.convergence import convergence_check

        streams: dict[str, list[list[float]]] = {
            "energy": [[1.0] * 2000 for _ in range(3)],
        }

        result = convergence_check(streams, streams)

        assert result.converged is True
