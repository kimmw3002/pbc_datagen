"""Red-phase tests for Step 6.0: XYModel struct, constructor, observables.

The 2D XY model: H = -J Σ_{<ij>} cos(θ_i - θ_j),  J = 1,  θ ∈ [0, 2π).

These tests define the expected pybind11 API for construction and state
access.  They test energy() and magnetization() on known spin configs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Construction & state access
# ---------------------------------------------------------------------------


def test_xy_lattice_size_stored() -> None:
    """XYModel(L, seed) must store the lattice size, readable via .L."""
    from pbc_datagen._core import XYModel

    for L in [2, 4, 8, 16]:
        model = XYModel(L=L, seed=0)
        assert model.L == L, f"Expected L={L}, got {model.L}"


def test_xy_spins_shape_and_dtype() -> None:
    """model.spins must return an (L, L) array of float64 (angles)."""
    from pbc_datagen._core import XYModel

    L = 8
    model = XYModel(L=L, seed=42)
    spins = model.spins
    assert isinstance(spins, np.ndarray)
    assert spins.shape == (L, L)
    assert spins.dtype == np.float64


def test_xy_cold_start_all_zero() -> None:
    """Initial spin configuration must be all θ=0 (cold / ordered start).

    All spins point in the same direction (angle 0).  This is the
    ground state, analogous to all +1 in Ising.
    """
    from pbc_datagen._core import XYModel

    model = XYModel(L=8, seed=42)
    assert np.all(model.spins == 0.0)


# ---------------------------------------------------------------------------
# Temperature management
# ---------------------------------------------------------------------------


def test_xy_set_temperature_stores_value() -> None:
    """set_temperature(T) must update the temperature, readable via .T."""
    from pbc_datagen._core import XYModel

    model = XYModel(L=4, seed=42)
    model.set_temperature(0.8935)
    assert abs(model.T - 0.8935) < 1e-12


def test_xy_set_temperature_rejects_nonpositive() -> None:
    """set_temperature must raise for T <= 0."""
    from pbc_datagen._core import XYModel

    model = XYModel(L=4, seed=42)
    with pytest.raises((ValueError, RuntimeError)):
        model.set_temperature(0.0)
    with pytest.raises((ValueError, RuntimeError)):
        model.set_temperature(-1.5)


# ---------------------------------------------------------------------------
# Energy on known configurations
# ---------------------------------------------------------------------------


def test_xy_energy_cold_start() -> None:
    """Cold-start (all θ=0) energy must equal -2N.

    H = -J Σ cos(θ_i - θ_j).  On an L×L PBC lattice there are 2L²
    nearest-neighbor bonds.  All angles 0 → cos(0) = 1 for every bond.
    E = -1 × 2L² = -2N.
    """
    from pbc_datagen._core import XYModel

    for L in [2, 4, 8, 16]:
        model = XYModel(L=L, seed=42)
        model.set_temperature(1.0)
        N = L * L
        assert model.energy() == pytest.approx(-2.0 * N), (
            f"Wrong cold-start energy for L={L}: got {model.energy()}, expected {-2.0 * N}"
        )


def test_xy_energy_antiferro_2x2() -> None:
    """Checkerboard-like antiferro config on 2×2: alternating θ=0 and θ=π.

    Sites: 0=(0,0)→0, 1=(0,1)→π, 2=(1,0)→π, 3=(1,1)→0.
    Every bond connects θ=0 and θ=π: cos(0 - π) = cos(π) = -1.
    8 bonds × (-1) × (-J) = 8 bonds × (+1) = +8.  E = +8.
    """
    from pbc_datagen._core import XYModel

    model = XYModel(L=2, seed=42)
    model.set_temperature(1.0)
    model.set_spin(1, math.pi)
    model.set_spin(2, math.pi)
    assert model.energy() == pytest.approx(8.0, abs=1e-10), (
        f"Expected E=+8 for 2×2 antiferro, got {model.energy()}"
    )


def test_xy_energy_90_degree_config() -> None:
    """All θ=0 except site 0 at θ=π/2 on 2×2.

    On 2×2 PBC, each site has 4 neighbor slots but only 2 distinct
    neighbors (each counted twice due to wrapping).  So there are
    4 distinct bonds total (not 8) — each pair of distinct neighbors
    is connected by 2 directed edges, giving 8 directed bonds.

    The energy sums over directed bonds (each undirected bond counted
    once per endpoint): H = -J Σ_{<ij>} cos(θ_i - θ_j) where the sum
    is over 8 directed bonds (= 2L² for L=2).

    Site 0 (θ=π/2) has 4 directed bonds to sites 1 and 2 (both θ=0):
      4 × cos(π/2) = 4 × 0 = 0.
    Remaining 4 directed bonds connect sites 1↔2 and 3↔1, 3↔2 (all θ=0):
      4 × cos(0) = 4 × 1 = 4.
    E = -(0 + 4) = -4.
    """
    from pbc_datagen._core import XYModel

    model = XYModel(L=2, seed=42)
    model.set_temperature(1.0)
    model.set_spin(0, math.pi / 2)
    assert model.energy() == pytest.approx(-4.0, abs=1e-10), f"Expected E=-4, got {model.energy()}"


def test_xy_set_spin_wraps_angle() -> None:
    """set_spin should normalize angles to [0, 2π).

    Setting θ = -π/2 should be equivalent to θ = 3π/2.
    Setting θ = 3π should be equivalent to θ = π.
    """
    from pbc_datagen._core import XYModel

    model = XYModel(L=4, seed=42)
    model.set_spin(0, -math.pi / 2)
    assert model.spins.flat[0] == pytest.approx(3 * math.pi / 2, abs=1e-10)

    model.set_spin(0, 3 * math.pi)
    assert model.spins.flat[0] == pytest.approx(math.pi, abs=1e-10)


# ---------------------------------------------------------------------------
# Magnetization on known configurations
# ---------------------------------------------------------------------------


def test_xy_magnetization_cold_start() -> None:
    """Cold-start (all θ=0): m = (1/N)|Σ(cos θ, sin θ)| = 1.0.

    All spins point along x: mx = 1, my = 0 → |m| = 1.0.
    """
    from pbc_datagen._core import XYModel

    for L in [2, 4, 8]:
        model = XYModel(L=L, seed=42)
        assert model.abs_magnetization() == pytest.approx(1.0), (
            f"Wrong cold-start |m| for L={L}: got {model.abs_magnetization()}"
        )
        assert model.mx() == pytest.approx(1.0)
        assert model.my() == pytest.approx(0.0, abs=1e-10)


def test_xy_magnetization_antiferro_2x2() -> None:
    """2×2 antiferro (0, π, π, 0): mx = cos0+cosπ+cosπ+cos0 = 0,
    my = sin0+sinπ+sinπ+sin0 = 0 → |m| = 0.
    """
    from pbc_datagen._core import XYModel

    model = XYModel(L=2, seed=42)
    model.set_spin(1, math.pi)
    model.set_spin(2, math.pi)
    assert model.abs_magnetization() == pytest.approx(0.0, abs=1e-10)
    assert model.mx() == pytest.approx(0.0, abs=1e-10)
    assert model.my() == pytest.approx(0.0, abs=1e-10)


def test_xy_magnetization_90_degree() -> None:
    """All θ=π/2 → all spins point along y → |m| = 1.0, mx=0, my=1."""
    from pbc_datagen._core import XYModel

    L = 4
    model = XYModel(L=L, seed=42)
    for site in range(L * L):
        model.set_spin(site, math.pi / 2)
    assert model.abs_magnetization() == pytest.approx(1.0, abs=1e-10)
    assert model.mx() == pytest.approx(0.0, abs=1e-10)
    assert model.my() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Observables dict
# ---------------------------------------------------------------------------


def test_xy_observables_keys() -> None:
    """observables() must return a dict with the expected keys."""
    from pbc_datagen._core import XYModel

    model = XYModel(L=4, seed=42)
    model.set_temperature(1.0)
    obs = model.observables()
    expected_keys = {"energy", "mx", "my", "abs_m"}
    assert set(obs.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(obs.keys())}"


def test_xy_observables_values_cold_start() -> None:
    """observables() values must match individual accessors on cold start."""
    from pbc_datagen._core import XYModel

    model = XYModel(L=4, seed=42)
    model.set_temperature(1.0)
    obs = model.observables()
    assert obs["energy"] == pytest.approx(model.energy())
    assert obs["abs_m"] == pytest.approx(model.abs_magnetization())
    # Cold start: mx = 1.0, my = 0.0
    assert obs["mx"] == pytest.approx(1.0)
    assert obs["my"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Pure-Python O(N) recomputation helpers for XY observables
# ---------------------------------------------------------------------------


def _xy_energy_python(theta: np.ndarray, nbr: np.ndarray, N: int) -> float:
    """H = -J Σ_{<ij>} cos(θ_i - θ_j), sum all 4 neighbors and divide by 2."""
    coupling = 0.0
    for i in range(N):
        for d in range(4):
            j = int(nbr[i, d])
            coupling += np.cos(theta[i] - theta[j])
    return -coupling / 2.0


def _xy_magnetization_python(theta: np.ndarray, N: int) -> tuple[float, float, float]:
    """Returns (mx, my, abs_m)."""
    mx_sum = np.sum(np.cos(theta))
    my_sum = np.sum(np.sin(theta))
    mx = mx_sum / N
    my = my_sum / N
    abs_m = np.sqrt(mx**2 + my**2)
    return mx, my, abs_m


# ---------------------------------------------------------------------------
# snapshot() — shape, dtype, copy semantics
# ---------------------------------------------------------------------------


def test_snapshot_shape_and_dtype() -> None:
    """XYModel.snapshot() → (1, L, L) float64 array."""
    from pbc_datagen._core import XYModel

    L = 6
    model = XYModel(L=L, seed=42)
    snap = model.snapshot()
    assert isinstance(snap, np.ndarray)
    assert snap.shape == (1, L, L)
    assert snap.dtype == np.float64


def test_snapshot_matches_theta() -> None:
    """snapshot()[0] must equal the spin angle field after some evolution."""
    from pbc_datagen._core import XYModel

    model = XYModel(L=8, seed=99)
    model.set_temperature(1.5)
    # Evolve without sweep() — use individual building blocks
    for _ in range(50):
        model._metropolis_sweep()
        model._wolff_step()

    snap = model.snapshot()
    theta_view = model.spins  # (L, L) float64 view of theta
    np.testing.assert_array_equal(snap[0], theta_view)


def test_snapshot_is_copy_not_view() -> None:
    """snapshot() must return an owning copy — mutating spins must not affect it."""
    from pbc_datagen._core import XYModel

    model = XYModel(L=4, seed=42)
    snap_before = model.snapshot()
    # Cold start: all θ=0
    np.testing.assert_array_equal(snap_before, 0.0)

    model.set_temperature(5.0)
    # Mutate spins heavily with individual building blocks
    for _ in range(50):
        model._metropolis_sweep()
        model._wolff_step()
    snap_after = model.snapshot()
    # snap_before should still hold the cold-start zeros
    np.testing.assert_array_equal(snap_before[0], 0.0)
    # snap_after should differ (high-T scrambles the angles)
    assert not np.array_equal(snap_before, snap_after)


# ---------------------------------------------------------------------------
# randomize() — valid angles + cache consistency
# ---------------------------------------------------------------------------


def test_randomize_produces_valid_angles() -> None:
    """After randomize(), all θ ∈ [0, 2π) and not all the same."""
    from pbc_datagen._core import XYModel

    model = XYModel(L=16, seed=42)
    # Precondition: cold start, all θ=0
    assert np.all(model.spins == 0.0), "precondition: cold start"

    model.randomize()
    theta = model.spins.ravel()

    # All angles in [0, 2π)
    assert np.all(theta >= 0.0), "angles must be >= 0"
    assert np.all(theta < 2 * np.pi), "angles must be < 2π"
    # Not all the same (256 uniform draws should not all be identical)
    assert len(np.unique(theta)) > 1, "randomize() produced all-identical angles"
    # Should have good spread — at least 200 unique values on 256 sites
    assert len(np.unique(theta)) > 200, (
        f"expected diverse angles, got only {len(np.unique(theta))} unique values"
    )


def test_randomize_cache_matches_python_recompute() -> None:
    """After randomize(), cached E/mx/my/|m| must match Python recomputation."""
    from pbc_datagen._core import XYModel, make_neighbor_table

    L = 8
    N = L * L
    model = XYModel(L=L, seed=42)
    model.set_temperature(1.0)
    nbr = make_neighbor_table(L)

    for _ in range(10):
        model.randomize()
        theta = model.spins.ravel().copy()

        e_py = _xy_energy_python(theta, nbr, N)
        mx_py, my_py, abs_m_py = _xy_magnetization_python(theta, N)

        assert model.energy() == pytest.approx(e_py, abs=1e-8), (
            f"C++ energy {model.energy()} != Python {e_py}"
        )
        assert model.mx() == pytest.approx(mx_py, abs=1e-12)
        assert model.my() == pytest.approx(my_py, abs=1e-12)
        assert model.abs_magnetization() == pytest.approx(abs_m_py, abs=1e-12)
