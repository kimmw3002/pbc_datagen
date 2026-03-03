"""Two-initialization convergence check for 2D PT grids.

Compares obs_streams from two independent runs (e.g. cold-start vs
random-start) to detect metastable trapping.  If both runs converge to
the same distribution at every (observable, slot) pair, the system is
genuinely equilibrated.  If they disagree, those slots are likely stuck
in different metastable basins.

Algorithm per (observable, slot):
  1. Discard the first half of each stream (burn-in removal).
  2. Variance-floor guard: if both halves are near-constant with the
     same value, the observable is trivially stationary → agree.
  3. Estimate τ_int from each stream's second half; use the max.
  4. Block into batch means of size ceil(3 × τ_int).
  5. Welch t-test between the two runs' batch means.
  6. Bonferroni correction across all (observable, slot) tests.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import stats

from pbc_datagen.autocorrelation import tau_int

_WELCH_ATOL: float = 1e-12
_WELCH_RTOL: float = 1e-10


@dataclass
class ConvergenceResult:
    """Result of two-initialization convergence check.

    Attributes
    ----------
    converged : bool
        True if all observables at all slots agree between the two runs.
    disagreement_map : dict[str, list[bool]]
        Per-observable, per-slot flag.  True means the two runs disagree
        at that (observable, slot) — i.e. the slot has NOT converged.
    """

    converged: bool
    disagreement_map: dict[str, list[bool]] = field(default_factory=dict)


def convergence_check(
    streams_a: dict[str, list[list[float]]],
    streams_b: dict[str, list[list[float]]],
    alpha: float = 0.05,
) -> ConvergenceResult:
    """Compare two independent PT runs for convergence.

    Parameters
    ----------
    streams_a, streams_b : dict[str, list[list[float]]]
        obs_streams from each run.  ``streams[name][slot]`` is a list
        of float values (one per PT round).
    alpha : float
        Family-wise significance level (default 0.05).

    Returns
    -------
    ConvergenceResult
        Contains overall pass/fail and per-(observable, slot) disagreement map.
    """
    if not streams_a:
        return ConvergenceResult(converged=True)

    obs_names = list(streams_a.keys())
    n_slots = len(streams_a[obs_names[0]])
    n_tests = len(obs_names) * n_slots
    threshold = alpha / n_tests  # Bonferroni

    disagreement_map: dict[str, list[bool]] = {name: [False] * n_slots for name in obs_names}

    for name in obs_names:
        for s in range(n_slots):
            series_a = streams_a[name][s]
            series_b = streams_b[name][s]

            if _slot_disagrees(series_a, series_b, threshold):
                disagreement_map[name][s] = True

    any_disagree = any(any(flags) for flags in disagreement_map.values())
    return ConvergenceResult(
        converged=not any_disagree,
        disagreement_map=disagreement_map,
    )


def _slot_disagrees(
    series_a: list[float],
    series_b: list[float],
    threshold: float,
) -> bool:
    """Test whether two streams at one slot have different means.

    Returns True if the two streams disagree (different distributions).
    """
    # Discard first half (burn-in)
    half_a = len(series_a) // 2
    half_b = len(series_b) // 2
    tail_a = np.asarray(series_a[half_a:], dtype=np.float64)
    tail_b = np.asarray(series_b[half_b:], dtype=np.float64)

    if len(tail_a) < 10 or len(tail_b) < 10:
        # Too few samples — can't tell, assume disagree
        return True

    # Variance-floor guard: skip t-test for trivially constant data
    std_a, std_b = float(np.std(tail_a)), float(np.std(tail_b))
    scale = max(abs(float(np.mean(tail_a))), abs(float(np.mean(tail_b))))
    tol = _WELCH_ATOL + _WELCH_RTOL * scale
    if std_a < tol and std_b < tol:
        return False  # both constant with same value → agree

    # Estimate τ_int from each tail; use the larger one for blocking
    tau_a = tau_int(tail_a)
    tau_b = tau_int(tail_b)
    block_size = max(1, math.ceil(3 * max(tau_a, tau_b)))

    # Block into approximately-independent batch means
    blocks_a = _block_means(tail_a, block_size)
    blocks_b = _block_means(tail_b, block_size)

    if len(blocks_a) < 5 or len(blocks_b) < 5:
        # Not enough independent blocks — can't tell, assume disagree
        return True

    # Welch t-test on batch means
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _, p_value = stats.ttest_ind(blocks_a, blocks_b, equal_var=False)

    p = float(p_value)  # type: ignore[arg-type]
    if math.isnan(p):
        return False  # NaN p-value → treat as agreeing

    return p < threshold


def _block_means(arr: npt.NDArray[np.float64], block_size: int) -> npt.NDArray[np.float64]:
    """Average contiguous blocks of ``block_size`` samples."""
    n_blocks = len(arr) // block_size
    if n_blocks == 0:
        return arr[:0]  # empty array
    return arr[: n_blocks * block_size].reshape(n_blocks, block_size).mean(axis=1)
