"""Red-phase tests for Step 1.2.1: BlumeCapelModel struct, constructor, observables.

These tests define the expected pybind11 API for the Blume-Capel model's
construction and state access.  They test energy(), magnetization(),
abs_magnetization(), and vacancy_density() on known spin configurations.

The Blume-Capel Hamiltonian is:

    H = -J Σ_{<ij>} s_i s_j  +  D Σ_i s_i²

where s_i ∈ {-1, 0, +1}, J = 1, and D is the crystal-field parameter.
The s = 0 state is a "vacancy" — it decouples from neighbors and doesn't
contribute to the crystal-field term either (0² = 0).

All imports are lazy (inside test functions) so pytest can *collect*
the tests even though BlumeCapelModel is not bound yet.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Construction & state access
# ---------------------------------------------------------------------------


def test_bc_lattice_size_stored() -> None:
    """BlumeCapelModel(L, seed) must store the lattice size, readable via .L."""
    from pbc_datagen._core import BlumeCapelModel

    for L in [2, 4, 8, 16]:
        model = BlumeCapelModel(L=L, seed=0)
        assert model.L == L, f"Expected L={L}, got {model.L}"


def test_bc_spins_shape_dtype_cold_start() -> None:
    """model.spins must be (L, L) int8, initialized to all +1 (cold start).

    Cold start (ordered, all +1) is the standard default — it avoids long
    equilibration at low T.  The spin values {-1, 0, +1} all fit in int8.
    """
    from pbc_datagen._core import BlumeCapelModel

    L = 8
    model = BlumeCapelModel(L=L, seed=42)
    spins = model.spins
    assert isinstance(spins, np.ndarray)
    assert spins.shape == (L, L)
    assert spins.dtype == np.int8
    assert np.all(spins == 1), "Cold start must be all +1"


# ---------------------------------------------------------------------------
# Temperature & crystal-field management
# ---------------------------------------------------------------------------


def test_bc_set_temperature() -> None:
    """set_temperature(T) must store T (readable via .T), reject T <= 0.

    A non-positive temperature is physically meaningless: exp(-ΔE / T)
    blows up or is undefined.
    """
    from pbc_datagen._core import BlumeCapelModel

    model = BlumeCapelModel(L=4, seed=42)
    model.set_temperature(2.5)
    assert abs(model.T - 2.5) < 1e-12

    with pytest.raises((ValueError, RuntimeError)):
        model.set_temperature(0.0)
    with pytest.raises((ValueError, RuntimeError)):
        model.set_temperature(-1.0)


def test_bc_set_crystal_field() -> None:
    """set_crystal_field(D) must store D (readable via .D).

    D can be any real number:
      D > 0 → penalizes magnetic sites (s² = 1), favors vacancies
      D < 0 → penalizes vacancies, favors ordering
      D = 0 → pure Ising limit (no crystal field)
    """
    from pbc_datagen._core import BlumeCapelModel

    model = BlumeCapelModel(L=4, seed=42)

    model.set_crystal_field(1.5)
    assert abs(model.D - 1.5) < 1e-12

    model.set_crystal_field(-0.5)
    assert abs(model.D - (-0.5)) < 1e-12

    model.set_crystal_field(0.0)
    assert abs(model.D - 0.0) < 1e-12


# ---------------------------------------------------------------------------
# set_spin — three-state spins
# ---------------------------------------------------------------------------


def test_bc_set_spin_accepts_valid_rejects_invalid() -> None:
    """set_spin must accept {-1, 0, +1} and reject anything else.

    Unlike Ising (±1 only), BC has three spin states.  The 0 state
    (vacancy) is the key distinction.
    """
    from pbc_datagen._core import BlumeCapelModel

    model = BlumeCapelModel(L=4, seed=42)

    # Valid values — should not raise
    model.set_spin(0, -1)
    model.set_spin(1, 0)
    model.set_spin(2, +1)

    # Read back via spins array (flat index → (row, col))
    assert model.spins[0, 0] == -1
    assert model.spins[0, 1] == 0
    assert model.spins[0, 2] == +1

    # Invalid values — must raise
    with pytest.raises((ValueError, RuntimeError)):
        model.set_spin(0, 2)
    with pytest.raises((ValueError, RuntimeError)):
        model.set_spin(0, -2)


# ---------------------------------------------------------------------------
# Energy on known configurations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("L", "D", "expected"),
    [
        (2, 0.0, -8.0),    # pure Ising limit: -2·4 = -8
        (2, 1.0, -4.0),    # -8 + 4 = -4
        (2, 3.0, 4.0),     # -8 + 12 = 4
        (4, 0.0, -32.0),   # -2·16 = -32
        (4, 0.5, -24.0),   # -32 + 0.5·16 = -24
        (8, 0.0, -128.0),  # -2·64 = -128
    ],
    ids=["L2_D0", "L2_D1", "L2_D3", "L4_D0", "L4_D0.5", "L8_D0"],
)
def test_bc_energy_cold_start(L: int, D: float, expected: float) -> None:
    """Cold-start (all +1) energy: E = -2L² + D·L².

    H = -J Σ_{<ij>} s_i s_j  +  D Σ_i s_i²

    All +1:
      - 2L² bonds, each s_i·s_j = +1 → coupling = -2L²
      - L² sites, each s_i² = 1     → crystal field = D·L²
      - Total: E = -2L² + D·L²
    """
    from pbc_datagen._core import BlumeCapelModel

    model = BlumeCapelModel(L=L, seed=42)
    model.set_crystal_field(D)
    assert model.energy() == pytest.approx(expected)


def test_bc_energy_all_vacancies() -> None:
    """All spins = 0 (vacancy state): E = 0 regardless of D.

    When every site is vacant:
      - s_i·s_j = 0 for all bonds → coupling = 0
      - s_i² = 0 for all sites    → crystal field = 0
    """
    from pbc_datagen._core import BlumeCapelModel

    for D in [0.0, 1.0, -2.0, 5.0]:
        model = BlumeCapelModel(L=4, seed=42)
        model.set_crystal_field(D)
        # Set all 16 sites to 0
        for site in range(16):
            model.set_spin(site, 0)

        assert model.energy() == pytest.approx(0.0), (
            f"All-vacancy energy must be 0, got {model.energy()} at D={D}"
        )


def test_bc_energy_2x2_checkerboard() -> None:
    """2×2 checkerboard (+1,-1,-1,+1): E = +2L² + D·L² = 8 + 4D.

    Every bond connects opposite spins: s_i·s_j = -1.
    On 2×2 PBC lattice there are 2·4 = 8 bonds → coupling = +8.
    All spins are ±1, so s_i² = 1 → crystal field = 4D.
    """
    from pbc_datagen._core import BlumeCapelModel

    for D in [0.0, 1.0, -1.5]:
        model = BlumeCapelModel(L=2, seed=42)
        model.set_crystal_field(D)

        # Checkerboard: sites 1=(0,1) and 2=(1,0) flipped to -1
        model.set_spin(1, -1)
        model.set_spin(2, -1)

        expected = 8.0 + 4.0 * D
        got = model.energy()
        assert got == pytest.approx(expected), (
            f"2×2 checkerboard at D={D}: expected E={expected}, got {got}"
        )


# ---------------------------------------------------------------------------
# Magnetization & vacancy density on known configurations
# ---------------------------------------------------------------------------


def test_bc_all_minus_one_observables() -> None:
    """All -1 state: E = -2L² + D·L² (same as all +1), m = -1.0, Q = 1.0.

    The all-down state is the other ordered ground state.  Energy is
    identical to all +1 because (-1)(-1) = +1 for coupling and
    (-1)² = 1 for the crystal field.  But m flips sign.
    This catches sign bugs in energy or magnetization.
    """
    from pbc_datagen._core import BlumeCapelModel

    L, D = 4, 1.0
    model = BlumeCapelModel(L=L, seed=42)
    model.set_crystal_field(D)
    for site in range(L * L):
        model.set_spin(site, -1)

    expected_E = -2.0 * L * L + D * L * L  # -32 + 16 = -16
    assert model.energy() == pytest.approx(expected_E), (
        f"All -1 energy: expected {expected_E}, got {model.energy()}"
    )
    assert model.magnetization() == pytest.approx(-1.0)
    assert model.abs_magnetization() == pytest.approx(1.0)
    assert model.quadrupole() == pytest.approx(1.0)


def test_bc_magnetization_known_states() -> None:
    """Magnetization on known configurations:

    m = (1/N) Σ_i s_i      (intensive, signed)
    |m| = (1/N) |Σ_i s_i|  (intensive, absolute)

    Cold start (all +1): m = 1.0, |m| = 1.0
    All vacancies (all 0): m = 0.0, |m| = 0.0
    Half +1 half -1:       m = 0.0, |m| = 0.0
    """
    from pbc_datagen._core import BlumeCapelModel

    # Cold start: all +1
    model = BlumeCapelModel(L=4, seed=42)
    assert model.magnetization() == pytest.approx(1.0)
    assert model.abs_magnetization() == pytest.approx(1.0)

    # All vacancies
    for site in range(16):
        model.set_spin(site, 0)
    assert model.magnetization() == pytest.approx(0.0)
    assert model.abs_magnetization() == pytest.approx(0.0)

    # 2×2 checkerboard: equal +1 and -1 → m = 0
    model2 = BlumeCapelModel(L=2, seed=42)
    model2.set_spin(1, -1)
    model2.set_spin(2, -1)
    assert model2.magnetization() == pytest.approx(0.0)
    assert model2.abs_magnetization() == pytest.approx(0.0)


def test_bc_quadrupole_known_states() -> None:
    """Quadrupole order parameter Q = (1/N) Σ_i s_i².

    Q measures the fraction of magnetically active sites:
      Q = 1 → all sites magnetic (s = ±1)
      Q = 0 → all sites vacant (s = 0)
    Equivalently, Q = 1 - ρ_vac.

    This is the standard BC order parameter alongside m — it jumps
    discontinuously at the tricritical point.

    Cold start (all +1): Q = 1.0
    All vacancies:        Q = 0.0
    Mixed (4 of 16 = 0):  Q = 0.75
    """
    from pbc_datagen._core import BlumeCapelModel

    # Cold start: all magnetic → Q = 1.0
    model = BlumeCapelModel(L=4, seed=42)
    assert model.quadrupole() == pytest.approx(1.0)

    # All vacancies → Q = 0.0
    for site in range(16):
        model.set_spin(site, 0)
    assert model.quadrupole() == pytest.approx(0.0)

    # 4 out of 16 sites vacant → Q = 12/16 = 0.75
    model2 = BlumeCapelModel(L=4, seed=42)
    model2.set_spin(0, 0)
    model2.set_spin(5, 0)
    model2.set_spin(10, 0)
    model2.set_spin(15, 0)
    assert model2.quadrupole() == pytest.approx(0.75)
