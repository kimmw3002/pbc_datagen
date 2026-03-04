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

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from pbc_datagen.autocorrelation import tau_int_batch

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

    disagreement_map: dict[str, list[bool]] = {}

    for name in obs_names:
        # --- 1. Convert list-of-lists → 2D ndarray ONCE (not per slot) ---
        raw_a = np.array(streams_a[name], dtype=np.float64)  # (n_slots, n_rounds)
        raw_b = np.array(streams_b[name], dtype=np.float64)

        # --- 2. Burn-in discard (first half) ---
        half = raw_a.shape[1] // 2
        tails_a = raw_a[:, half:]  # (n_slots, n_tail)
        tails_b = raw_b[:, half:]

        n_tail = tails_a.shape[1]
        if n_tail < 10:  # matches former _slot_disagrees guard
            disagreement_map[name] = [True] * n_slots
            continue

        # --- 3. Vectorised variance-floor guard ---
        std_a = tails_a.std(axis=1)  # (n_slots,)
        std_b = tails_b.std(axis=1)
        scale = np.maximum(np.abs(tails_a.mean(axis=1)), np.abs(tails_b.mean(axis=1)))
        tol = _WELCH_ATOL + _WELCH_RTOL * scale
        trivial = (std_a < tol) & (std_b < tol)  # agree without testing

        # --- 4. Batch tau_int for all non-trivial slots ---
        active = ~trivial
        disagrees = np.zeros(n_slots, dtype=bool)  # default: agree

        if active.any():
            tau_a = tau_int_batch(tails_a[active])  # (n_active,)
            tau_b = tau_int_batch(tails_b[active])
            block_sizes = np.maximum(1, np.ceil(3.0 * np.maximum(tau_a, tau_b))).astype(
                int
            )  # (n_active,)

            # --- 5. Group by block_size → batch ttest_ind per group ---
            da = np.zeros(active.sum(), dtype=bool)  # disagrees for active slots

            for bs in np.unique(block_sizes):
                grp = block_sizes == bs  # mask within active
                n_blocks = n_tail // bs
                if n_blocks < 5:  # too few blocks → disagree
                    da[grp] = True
                    continue
                # Block means: (n_grp, n_blocks)
                n_grp = int(grp.sum())
                ba = tails_a[active][grp, : n_blocks * bs].reshape(n_grp, n_blocks, bs).mean(axis=2)
                bb = tails_b[active][grp, : n_blocks * bs].reshape(n_grp, n_blocks, bs).mean(axis=2)
                # scipy ttest_ind broadcasts over rows when axis=1
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    _, p_raw = stats.ttest_ind(ba, bb, axis=1, equal_var=False)
                p_arr = np.asarray(p_raw, dtype=float)
                p_arr = np.where(np.isnan(p_arr), 1.0, p_arr)
                da[grp] = p_arr < threshold

            disagrees[active] = da

        disagreement_map[name] = disagrees.tolist()

    any_disagree = any(any(flags) for flags in disagreement_map.values())
    return ConvergenceResult(
        converged=not any_disagree,
        disagreement_map=disagreement_map,
    )
