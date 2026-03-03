"""Spectral connectivity check for 2D PT grids.

Analyses the random walk on the n_T × n_P replica grid to detect
disconnected or poorly-connected regions before expensive equilibration.

Grid layout matches pt_rounds_2d: slot(i,j) = i*n_P + j.

Edge indexing:
  T-direction:     t_accept_rates[j*(n_T-1) + i]  = edge (i,j)↔(i+1,j)
  Param-direction: p_accept_rates[i*(n_P-1) + j]  = edge (i,j)↔(i,j+1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import sparse


@dataclass
class ConnectivityResult:
    """Result of spectral connectivity check.

    Attributes
    ----------
    passed : bool
        True if spectral gap exceeds the threshold.
    gap : float
        Spectral gap (1 − λ₂) of the lazy random walk matrix.
    fiedler : ndarray or None
        Eigenvector of λ₂ when the check fails (sign pattern identifies
        the disconnected clusters).  None when the check passes.
    """

    passed: bool
    gap: float
    fiedler: npt.NDArray[np.float64] | None


def build_transition_matrix(
    n_T: int,
    n_P: int,
    t_accept_rates: npt.NDArray[np.float64],
    p_accept_rates: npt.NDArray[np.float64],
) -> sparse.csr_matrix:
    """Build row-stochastic lazy random walk matrix from edge acceptance rates.

    For each node s, the transition probability to neighbor s' is
    α(s,s') / d_max, where d_max = max weighted degree across all nodes.
    The self-loop probability absorbs the remainder: 1 − d(s)/d_max.

    The resulting matrix is symmetric (since edge weights are symmetric)
    and row-stochastic with all eigenvalues in [0, 1].
    """
    M = n_T * n_P
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # T-direction edges
    for j in range(n_P):
        for i in range(n_T - 1):
            rate = float(t_accept_rates[j * (n_T - 1) + i])
            if rate > 0:
                s_lo = i * n_P + j
                s_hi = (i + 1) * n_P + j
                rows.extend([s_lo, s_hi])
                cols.extend([s_hi, s_lo])
                vals.extend([rate, rate])

    # Param-direction edges
    for i in range(n_T):
        for j in range(n_P - 1):
            rate = float(p_accept_rates[i * (n_P - 1) + j])
            if rate > 0:
                s_lo = i * n_P + j
                s_hi = i * n_P + j + 1
                rows.extend([s_lo, s_hi])
                cols.extend([s_hi, s_lo])
                vals.extend([rate, rate])

    if not vals:
        # All edges dead — fully disconnected identity walk
        return sparse.eye(M, format="csr")  # type: ignore[return-value]

    W = sparse.coo_matrix((np.array(vals), (np.array(rows), np.array(cols))), shape=(M, M)).tocsr()
    d = np.asarray(W.sum(axis=1)).ravel()
    d_max = float(d.max())

    # P = W/d_max + diag(1 − d/d_max)  — lazy random walk
    P: sparse.csr_matrix = (W / d_max + sparse.diags(1.0 - d / d_max, format="csr")).tocsr()
    return P


def check_connectivity(
    n_T: int,
    n_P: int,
    t_accept_rates: npt.NDArray[np.float64],
    p_accept_rates: npt.NDArray[np.float64],
    min_gap: float = 0.01,
) -> ConnectivityResult:
    """Check if the 2D PT grid is well-connected via spectral gap analysis.

    Parameters
    ----------
    n_T, n_P : int
        Grid dimensions.
    t_accept_rates, p_accept_rates : ndarray
        Per-edge acceptance rates (see module docstring for indexing).
    min_gap : float
        Minimum spectral gap to pass.  Default 0.01.

    Returns
    -------
    ConnectivityResult
        Contains pass/fail, spectral gap, and Fiedler vector on failure.
    """
    P = build_transition_matrix(n_T, n_P, t_accept_rates, p_accept_rates)
    M = n_T * n_P

    # P is symmetric → real eigenvalues, use eigh for stability.
    # M is small (typically 10–200) so dense is fine.
    Pd = P.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(Pd)

    # eigh returns ascending order: eigenvalues[-1] = 1, eigenvalues[-2] = λ₂
    gap = float(1.0 - eigenvalues[-2]) if M > 1 else 1.0
    passed = gap >= min_gap

    fiedler: npt.NDArray[np.float64] | None = None
    if not passed:
        if gap < 1e-10:
            # Truly disconnected — degenerate eigenspace makes the
            # eigenvector unreliable.  Use connected components instead.
            n_comp, labels = sparse.csgraph.connected_components(P, directed=False)
            fiedler = np.where(labels == labels[0], 1.0, -1.0)
        else:
            fiedler = eigenvectors[:, -2].copy()

    return ConnectivityResult(passed=passed, gap=gap, fiedler=fiedler)
