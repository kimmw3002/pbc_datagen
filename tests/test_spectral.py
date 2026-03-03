"""Red-phase tests for Step 3.3: Spectral connectivity check for 2D PT grids.

The spectral module analyses the 2D parameter-space PT grid for connectivity.
Given per-edge acceptance rates (T-direction and param-direction), it:

1. Builds a row-stochastic Markov transition matrix for the random walk on
   the n_T × n_P grid, where transition probabilities are proportional to
   measured acceptance rates.
2. Computes the spectral gap (1 − λ₂) of that matrix.  A large gap means
   good mixing; a gap near zero means the grid has disconnected islands.
3. Extracts the Fiedler vector (eigenvector of λ₂) to identify which
   (T, param) slots belong to the isolated cluster.

Grid layout matches pt_rounds_2d: slot(i,j) = i*n_P + j.

Edge indexing:
  T-direction:     t_accept_rates[j*(n_T-1) + i]  = rate for edge (i,j)↔(i+1,j)
  Param-direction: p_accept_rates[i*(n_P-1) + j]  = rate for edge (i,j)↔(i,j+1)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_T = 3
N_P = 3
M = N_T * N_P  # 9 slots in a 3×3 grid
N_T_EDGES = N_P * (N_T - 1)  # 6 T-direction edges
N_P_EDGES = N_T * (N_P - 1)  # 6 param-direction edges


def _uniform_rates(
    rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """All edges have the same acceptance rate."""
    return (
        np.full(N_T_EDGES, rate),
        np.full(N_P_EDGES, rate),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildTransitionMatrix:
    """The transition matrix must be a valid row-stochastic matrix."""

    def test_transition_matrix_is_row_stochastic(self) -> None:
        """Rows sum to 1 and all entries are non-negative.

        With uniform acceptance rate 0.5 on a 3×3 grid, every node
        has 2–4 neighbors.  The transition matrix should be a valid
        probability matrix regardless of node degree.
        """
        from pbc_datagen.spectral import build_transition_matrix

        t_rates, p_rates = _uniform_rates(0.5)
        P = build_transition_matrix(N_T, N_P, t_rates, p_rates)

        # Convert to dense for easy inspection
        Pd = P.toarray()

        # All entries non-negative
        assert np.all(Pd >= -1e-15), f"Negative entries in P: {Pd.min()}"

        # Each row sums to 1
        row_sums = Pd.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


class TestCheckConnectivity:
    """End-to-end connectivity check: pass/fail + diagnostics."""

    def test_fully_connected_grid_passes(self) -> None:
        """Uniform non-zero acceptance rates → well-connected grid.

        A 3×3 grid with uniform rate 0.5 is a standard random walk on
        a grid graph — mixing is good, spectral gap should be well above
        the default threshold.
        """
        from pbc_datagen.spectral import check_connectivity

        t_rates, p_rates = _uniform_rates(0.5)
        result = check_connectivity(N_T, N_P, t_rates, p_rates)

        assert result.passed is True
        assert result.gap > 0.01, f"Gap {result.gap:.6f} too small for uniform grid"

    def test_dead_edge_disconnects_grid(self) -> None:
        """Zero acceptance on all T-edges at column j=1 splits the
        grid into left (j=0) and right (j=1,2) components.

        This creates a disconnected Markov chain → λ₂ = 1 → gap = 0.
        The check should fail.
        """
        from pbc_datagen.spectral import check_connectivity

        t_rates, p_rates = _uniform_rates(0.5)

        # Kill all param-direction edges between j=0 and j=1.
        # p_accept_rates[i*(N_P-1) + j] = rate for edge (i,j)↔(i,j+1)
        # Set j=0 edges to 0 for all T rows:
        for i in range(N_T):
            p_rates[i * (N_P - 1) + 0] = 0.0

        result = check_connectivity(N_T, N_P, t_rates, p_rates)

        assert result.passed is False
        assert result.gap < 1e-6, f"Expected gap ≈ 0 for disconnected grid, got {result.gap:.6f}"

    def test_fiedler_identifies_partition(self) -> None:
        """When the grid is split into two components, the Fiedler vector
        should have opposite signs on each side of the cut.

        Cut: all param-edges between j=0 and j=1 set to 0.
        Left component: slots with j=0  → {0, 3, 6}
        Right component: slots with j=1,2 → {1, 2, 4, 5, 7, 8}
        """
        from pbc_datagen.spectral import check_connectivity

        t_rates, p_rates = _uniform_rates(0.5)
        for i in range(N_T):
            p_rates[i * (N_P - 1) + 0] = 0.0

        result = check_connectivity(N_T, N_P, t_rates, p_rates)

        fiedler = result.fiedler
        assert fiedler is not None, "Fiedler vector should be returned on failure"

        left_slots = [0, 3, 6]  # j=0
        right_slots = [1, 2, 4, 5, 7, 8]  # j=1,2

        left_signs = {np.sign(fiedler[s]) for s in left_slots}
        right_signs = {np.sign(fiedler[s]) for s in right_slots}

        # All nodes in the same component should have the same sign
        assert len(left_signs) == 1, f"Left component should have uniform sign, got {left_signs}"
        assert len(right_signs) == 1, f"Right component should have uniform sign, got {right_signs}"
        # The two components should have opposite signs
        assert left_signs != right_signs, "Fiedler vector should have opposite signs across the cut"

    def test_full_column_bottleneck_fails(self) -> None:
        """When ALL param-edges between j=0 and j=1 are weak (0.01),
        there is no detour — every path from j=0 to j=1 must cross
        one of these bottlenecks.  The spectral gap should be tiny
        and the check should fail.
        """
        from pbc_datagen.spectral import check_connectivity

        t_rates, p_rates = _uniform_rates(0.5)
        for i in range(N_T):
            p_rates[i * (N_P - 1) + 0] = 0.01

        result = check_connectivity(N_T, N_P, t_rates, p_rates)

        assert result.passed is False, (
            f"Full-column bottleneck should fail, got gap={result.gap:.6f}"
        )

    def test_single_weak_edge_passes_via_detour(self) -> None:
        """When only ONE param-edge is weak but the other T rows have
        strong edges, replicas can detour: move in T, cross at a strong
        row, move back.  The gap is reduced but the grid stays connected.
        """
        from pbc_datagen.spectral import check_connectivity

        # Weaken only ONE param-edge: row i=0, between j=0 and j=1
        t_rates, p_rates = _uniform_rates(0.5)
        p_rates[0 * (N_P - 1) + 0] = 0.01  # only edge (0,0)↔(0,1)

        result = check_connectivity(N_T, N_P, t_rates, p_rates)

        assert result.passed is True, "Single weak edge with strong detour should pass"
