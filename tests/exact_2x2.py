"""Exact 2×2 partition functions for detailed-balance tests.

Each lattice model has a small enough state space on 2×2 that we can
enumerate every microstate, compute exact Boltzmann probabilities P(E),
and chi-squared test MCMC energy histograms against them.

This is the gold-standard verification for detailed balance — see
docs/LESSONS.md.

Shared across test_wolff.py, test_metropolis.py, and test_sweep.py
for each model to avoid code duplication.
"""

from __future__ import annotations

import itertools
import math

import numpy.typing as npt

# ---------------------------------------------------------------------------
# Ising: 2^4 = 16 states, 3 energy levels
# ---------------------------------------------------------------------------


def ising_exact_probabilities(T: float) -> dict[int, float]:
    """Exact P(E) for the 2×2 Ising model at temperature T.

    The 2×2 Ising model has only 3 energy levels:
      E = -8  (degeneracy 2:  all +1, all -1)
      E =  0  (degeneracy 12: states with mixed spins, not checkerboard)
      E = +8  (degeneracy 2:  checkerboard patterns)

    Z(T) = 2 exp(8/T) + 12 + 2 exp(-8/T)
    """
    Z = 2 * math.exp(8 / T) + 12 + 2 * math.exp(-8 / T)
    return {
        -8: 2 * math.exp(8 / T) / Z,
        0: 12 / Z,
        8: 2 * math.exp(-8 / T) / Z,
    }


# ---------------------------------------------------------------------------
# Blume-Capel: 3^4 = 81 states
# ---------------------------------------------------------------------------


def bc_exact_probabilities(T: float, D: float) -> dict[float, float]:
    """Exact P(E) for the 2×2 Blume-Capel model at temperature T, crystal field D.

    Enumerates all 3^4 = 81 states (spins ∈ {-1, 0, +1}).

    On the 2×2 PBC lattice, each site's 4 neighbors are only 2 distinct
    sites (each appearing twice due to wrapping).  The bonds are:
    (0,1), (0,2), (1,3), (2,3) — each counted twice in the neighbor sum.

    H = -(1/2) Σ_i Σ_{j∈nbr(i)} s_i s_j  +  D Σ_i s_i²
      = -2(s0·s1 + s0·s2 + s1·s3 + s2·s3)  +  D(s0² + s1² + s2² + s3²)
    """
    energy_weight: dict[float, float] = {}

    for s0, s1, s2, s3 in itertools.product([-1, 0, 1], repeat=4):
        coupling = -2 * (s0 * s1 + s0 * s2 + s1 * s3 + s2 * s3)
        crystal = D * (s0**2 + s1**2 + s2**2 + s3**2)
        e = coupling + crystal
        e_key = round(e, 8)

        w = math.exp(-e / T)
        energy_weight[e_key] = energy_weight.get(e_key, 0.0) + w

    Z = sum(energy_weight.values())
    return {e: w / Z for e, w in energy_weight.items()}


# ---------------------------------------------------------------------------
# Ashkin-Teller: 2^8 = 256 states (two ±1 layers)
# ---------------------------------------------------------------------------


def at_exact_energy(
    sigma: list[int],
    tau: list[int],
    U: float,
    nbr: npt.NDArray,
) -> float:
    """Compute AT energy on 2×2 PBC lattice, same convention as C++.

    C++ sums over all N×4 directed neighbor pairs, then divides by 2.

    H = -Σ σ_i σ_j - Σ τ_i τ_j - U Σ σ_i σ_j τ_i τ_j
    """
    sigma_sum = 0
    tau_sum = 0
    four_spin_sum = 0
    for i in range(4):
        for d in range(4):
            j = int(nbr[i, d])
            sigma_sum += sigma[i] * sigma[j]
            tau_sum += tau[i] * tau[j]
            four_spin_sum += sigma[i] * sigma[j] * tau[i] * tau[j]
    return -sigma_sum / 2.0 - tau_sum / 2.0 - U * four_spin_sum / 2.0


def at_exact_probabilities(T: float, U: float) -> dict[float, float]:
    """Exact P(E) for the 2×2 Ashkin-Teller model at temperature T, coupling U.

    Enumerates all 2^8 = 256 microstates (σ_i, τ_i ∈ {-1, +1} for 4 sites).

    Uses make_neighbor_table() from the C++ backend to ensure the energy
    calculation matches the C++ convention exactly.
    """
    from pbc_datagen._core import make_neighbor_table

    nbr = make_neighbor_table(2)
    beta = 1.0 / T
    energy_weights: dict[float, float] = {}

    all_configs = list(itertools.product([-1, 1], repeat=4))
    for sigma in all_configs:
        for tau in all_configs:
            E = at_exact_energy(list(sigma), list(tau), U, nbr)
            E_round = round(E, 8)
            weight = math.exp(-beta * E_round)
            energy_weights[E_round] = energy_weights.get(E_round, 0.0) + weight

    Z = sum(energy_weights.values())
    return {E: w / Z for E, w in energy_weights.items()}


# ---------------------------------------------------------------------------
# XY: continuous angles → numerical quadrature on 2×2
# ---------------------------------------------------------------------------


def xy_2x2_exact_energy_histogram(
    T: float,
    bin_edges: npt.NDArray,
    n_grid: int = 128,
) -> npt.NDArray:
    """Exact energy bin probabilities for the 2×2 XY model via quadrature.

    The XY model has continuous spins θ ∈ [0, 2π), so we discretize each
    angle into n_grid values and evaluate the Boltzmann-weighted energy
    histogram on a 3D grid (fixing θ₀ = 0 by O(2) symmetry).

    On the 2×2 PBC lattice, each undirected bond {i,j} appears 4 times
    in the directed neighbor sum (each site sees each neighbor twice due
    to wrapping).  With the factor of 1/2 in the energy:

        E = -2 [cos(θ₀-θ₁) + cos(θ₀-θ₂) + cos(θ₁-θ₃) + cos(θ₂-θ₃)]

    Parameters
    ----------
    T : float
        Temperature.
    bin_edges : 1-D array, length n_bins + 1
        Energy bin edges (same convention as np.histogram).
    n_grid : int
        Number of angle grid points per dimension.

    Returns
    -------
    probs : 1-D array, length n_bins
        Exact probability mass in each energy bin.
    """
    import numpy as np

    beta = 1.0 / T
    angles = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)

    # Fix θ₀ = 0 by O(2) symmetry.  Integrate over θ₁, θ₂, θ₃.
    t1, t2, t3 = np.meshgrid(angles, angles, angles, indexing="ij")
    t0 = 0.0

    # Energy: E = -2[cos(θ₀-θ₁) + cos(θ₀-θ₂) + cos(θ₁-θ₃) + cos(θ₂-θ₃)]
    E = -2.0 * (np.cos(t0 - t1) + np.cos(t0 - t2) + np.cos(t1 - t3) + np.cos(t2 - t3))

    boltz = np.exp(-beta * E)

    # Accumulate Boltzmann weight into energy bins.
    E_flat = E.ravel()
    boltz_flat = boltz.ravel()
    n_bins = len(bin_edges) - 1
    bin_weights = np.zeros(n_bins, dtype=np.float64)
    indices = np.digitize(E_flat, bin_edges) - 1  # bin index for each grid point
    for b in range(n_bins):
        mask = indices == b
        bin_weights[b] = np.sum(boltz_flat[mask])

    Z = np.sum(bin_weights)
    return bin_weights / Z
