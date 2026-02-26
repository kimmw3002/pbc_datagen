"""Red-phase tests for Step 1.1.1: IsingModel struct, constructor, set_temperature.

These tests define the expected pybind11 API for the Ising model's
construction and state access.  They also test energy() and
magnetization() on the known cold-start (all +1) configuration —
those are pure functions of the spin state, independent of update
kernels (Steps 1.1.2–1.1.4).

All imports are lazy (inside test functions) so pytest can *collect*
the tests even though IsingModel is not bound yet.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Construction & state access
# ---------------------------------------------------------------------------


def test_ising_lattice_size_stored() -> None:
    """IsingModel(L, seed) must store the lattice size, readable via .L."""
    from pbc_datagen._core import IsingModel

    for L in [2, 4, 8, 16]:
        model = IsingModel(L=L, seed=0)
        assert model.L == L, f"Expected L={L}, got {model.L}"


def test_ising_spins_shape_and_dtype() -> None:
    """model.spins must return an (L, L) array of int8."""
    from pbc_datagen._core import IsingModel

    L = 8
    model = IsingModel(L=L, seed=42)
    spins = model.spins
    assert isinstance(spins, np.ndarray)
    assert spins.shape == (L, L)
    assert spins.dtype == np.int8


def test_ising_cold_start_all_plus_one() -> None:
    """Initial spin configuration must be all +1 (cold / ordered start).

    A cold start avoids long equilibration at low T and is the standard
    default for production MC runs.
    """
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=8, seed=42)
    assert np.all(model.spins == 1)


# ---------------------------------------------------------------------------
# Temperature management
# ---------------------------------------------------------------------------


def test_ising_set_temperature_stores_value() -> None:
    """set_temperature(T) must update the temperature, readable via .T."""
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=4, seed=42)
    model.set_temperature(2.269)
    assert abs(model.T - 2.269) < 1e-12


def test_ising_set_temperature_rejects_nonpositive() -> None:
    """set_temperature must raise ValueError for T <= 0.

    A non-positive temperature is physically meaningless for Boltzmann
    weights: exp(-ΔE / T) blows up or is undefined.
    """
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=4, seed=42)
    with pytest.raises((ValueError, RuntimeError)):
        model.set_temperature(0.0)
    with pytest.raises((ValueError, RuntimeError)):
        model.set_temperature(-1.5)


# ---------------------------------------------------------------------------
# Energy & magnetization on known configurations
# ---------------------------------------------------------------------------


def test_ising_energy_cold_start() -> None:
    """Cold-start (all +1) energy must equal -2L².

    H = -J Σ_{<ij>} s_i s_j.  On an L×L square lattice with PBC there
    are 2L² nearest-neighbor bonds (L² horizontal + L² vertical).
    All spins +1 ⟹ every bond contributes -J = -1 ⟹ E = -2L².
    """
    from pbc_datagen._core import IsingModel

    for L in [2, 4, 8, 16]:
        model = IsingModel(L=L, seed=42)
        model.set_temperature(1.0)
        assert model.energy() == -2 * L * L, (
            f"Wrong cold-start energy for L={L}: got {model.energy()}, expected {-2 * L * L}"
        )


def test_ising_magnetization_cold_start() -> None:
    """Cold-start (all +1) intensive magnetization must equal 1.0.

    m = (1/N) Σ_i s_i.  All spins +1 ⟹ m = 1.0.
    """
    from pbc_datagen._core import IsingModel

    for L in [2, 4, 8]:
        model = IsingModel(L=L, seed=42)
        assert model.magnetization() == pytest.approx(1.0), (
            f"Wrong cold-start magnetization for L={L}: "
            f"got {model.magnetization()}, expected 1.0"
        )


def test_ising_abs_magnetization_cold_start() -> None:
    """Cold-start |m| must equal 1.0 (same as m for all-+1).

    |m| = (1/N) |Σ_i s_i|.  All spins +1 ⟹ |m| = 1.0.
    """
    from pbc_datagen._core import IsingModel

    for L in [2, 4, 8]:
        model = IsingModel(L=L, seed=42)
        assert model.abs_magnetization() == pytest.approx(1.0), (
            f"Wrong cold-start |m| for L={L}: "
            f"got {model.abs_magnetization()}, expected 1.0"
        )


def test_ising_2x2_checkerboard_observables() -> None:
    """2×2 checkerboard must have E = +8, m = 0, |m| = 0.

    On a 2×2 PBC lattice, the checkerboard pattern:
        +1 -1
        -1 +1
    Every bond connects opposite spins: s_i s_j = -1 for all 8 bonds.
    So E = -J Σ (-1) = +1 × 8 = +8.
    m = (1/4)(+1 -1 -1 +1) = 0.  |m| = 0.

    Also exercises set_spin() for constructing non-trivial configurations.
    """
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=2, seed=42)
    model.set_temperature(1.0)

    # Set checkerboard: sites 0=(0,0), 3=(1,1) get +1; sites 1=(0,1), 2=(1,0) get -1
    model.set_spin(1, -1)
    model.set_spin(2, -1)

    assert model.energy() == 8, f"Expected E=+8 for 2×2 checkerboard, got {model.energy()}"
    assert model.magnetization() == pytest.approx(0.0), (
        f"Expected m=0 for checkerboard, got {model.magnetization()}"
    )
    assert model.abs_magnetization() == pytest.approx(0.0), (
        f"Expected |m|=0 for checkerboard, got {model.abs_magnetization()}"
    )
