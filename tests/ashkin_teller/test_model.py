"""Tests for the Ashkin-Teller model: construction, observables, known configurations.

Step 1.3.1 of the implementation plan.

Hamiltonian:
    H = -J Σ_{<ij>} σ_i σ_j  -  J Σ_{<ij>} τ_i τ_j  -  U Σ_{<ij>} σ_i σ_j τ_i τ_j

J = 1 (fixed, not user-settable).  U is the four-spin coupling.
Both σ and τ are Ising-like: values in {-1, +1}.

Cold start: all σ = +1, all τ = +1.

Energy on cold start (L×L lattice with PBC, 2L² bonds):
    E = -(2 + U) × 2L²
    (all bond terms are +1, so each of the three sums equals 2L²)

Observables:
    m_sigma       = (1/N) Σ σ_i
    abs_m_sigma   = (1/N) |Σ σ_i|
    m_tau         = (1/N) Σ τ_i
    abs_m_tau     = (1/N) |Σ τ_i|
    m_baxter      = (1/N) Σ σ_i τ_i       (Baxter order parameter)
    abs_m_baxter  = (1/N) |Σ σ_i τ_i|
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Constructor stores lattice size
# ---------------------------------------------------------------------------
def test_constructor_stores_lattice_size() -> None:
    """AshkinTellerModel(L, seed) must expose L as a read-only attribute."""
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(8, seed=42)
    assert model.L == 8


# ---------------------------------------------------------------------------
# 2. Sigma spins: shape (L, L), dtype int8
# ---------------------------------------------------------------------------
def test_sigma_spins_shape_and_dtype() -> None:
    """The .sigma property returns an (L, L) int8 numpy view of the σ array."""
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(6, seed=42)
    sigma = model.sigma
    assert sigma.shape == (6, 6)
    assert sigma.dtype == np.int8


# ---------------------------------------------------------------------------
# 3. Tau spins: shape (L, L), dtype int8
# ---------------------------------------------------------------------------
def test_tau_spins_shape_and_dtype() -> None:
    """The .tau property returns an (L, L) int8 numpy view of the τ array."""
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(6, seed=42)
    tau = model.tau
    assert tau.shape == (6, 6)
    assert tau.dtype == np.int8


# ---------------------------------------------------------------------------
# 4. Cold start: all σ = +1, all τ = +1
# ---------------------------------------------------------------------------
def test_cold_start_all_plus_one() -> None:
    """Constructor initializes all σ and τ spins to +1 (cold/ferromagnetic start)."""
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(4, seed=42)
    assert np.all(model.sigma == 1)
    assert np.all(model.tau == 1)


# ---------------------------------------------------------------------------
# 5. Temperature: set/get and validation
# ---------------------------------------------------------------------------
def test_temperature_set_get_and_validation() -> None:
    """set_temperature(T) stores T (readable via .T), rejects T <= 0."""
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(4, seed=42)
    model.set_temperature(2.5)
    assert model.T == pytest.approx(2.5)

    # Change temperature
    model.set_temperature(0.1)
    assert model.T == pytest.approx(0.1)

    # T = 0 is invalid (division by zero in Boltzmann factors)
    with pytest.raises((ValueError, RuntimeError)):
        model.set_temperature(0.0)

    # Negative T is unphysical
    with pytest.raises((ValueError, RuntimeError)):
        model.set_temperature(-1.0)


# ---------------------------------------------------------------------------
# 6. Four-spin coupling: set/get, default is U = 0
# ---------------------------------------------------------------------------
def test_four_spin_coupling_set_get_and_default() -> None:
    """set_four_spin_coupling(U) stores U (readable via .U).  Default U = 0."""
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(4, seed=42)

    # Default: U = 0  (two decoupled Ising models)
    assert model.U == pytest.approx(0.0)

    model.set_four_spin_coupling(0.5)
    assert model.U == pytest.approx(0.5)

    # U > 1 triggers internal remapping (tested in test_wolff.py),
    # but the public U value should still read back correctly.
    model.set_four_spin_coupling(1.5)
    assert model.U == pytest.approx(1.5)

    # Negative U is not allowed
    with pytest.raises((ValueError, RuntimeError)):
        model.set_four_spin_coupling(-0.3)


# ---------------------------------------------------------------------------
# 7. Cold-start energy for various (L, U) combinations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "L, U, expected_energy",
    [
        # E = -(2 + U) * 2L²
        # L=2: 2L² = 8 bonds
        (2, 0.0, -16.0),  # two decoupled Ising models: 2 × (-8)
        (2, 0.5, -20.0),  # -(2.5) × 8
        (2, 1.0, -24.0),  # -(3) × 8
        (2, 1.5, -28.0),  # -(3.5) × 8  (U > 1, remapped regime)
        # L=4: 2L² = 32 bonds
        (4, 0.0, -64.0),  # -(2) × 32
        (4, 0.5, -80.0),  # -(2.5) × 32
    ],
)
def test_cold_start_energy(L: int, U: float, expected_energy: float) -> None:
    """Energy on the all-+1 cold start with various four-spin couplings."""
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(L, seed=42)
    model.set_four_spin_coupling(U)
    assert model.energy() == pytest.approx(expected_energy)


# ---------------------------------------------------------------------------
# 8. Cold-start magnetizations: all equal to 1.0
# ---------------------------------------------------------------------------
def test_cold_start_magnetizations() -> None:
    """On the cold start (all σ = all τ = +1), every magnetization is +1."""
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(4, seed=42)
    model.set_four_spin_coupling(0.5)  # U shouldn't affect magnetization values

    assert model.m_sigma() == pytest.approx(1.0)
    assert model.abs_m_sigma() == pytest.approx(1.0)
    assert model.m_tau() == pytest.approx(1.0)
    assert model.abs_m_tau() == pytest.approx(1.0)
    assert model.m_baxter() == pytest.approx(1.0)
    assert model.abs_m_baxter() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. Known configuration: σ checkerboard, τ all +1
# ---------------------------------------------------------------------------
def test_known_config_checkerboard_sigma() -> None:
    """Verify energy and magnetizations on a non-trivial known configuration.

    Setup (2×2):
        σ = [+1, -1, -1, +1]   (checkerboard)
        τ = [+1, +1, +1, +1]   (uniform)

    On a 2×2 PBC lattice, every neighbor pair in the checkerboard is
    anti-aligned, so all 8 directed bonds (4 unique × 2) give σ_i σ_j = -1.

    Bond sums:
        Σ σ_i σ_j       = -8   (all anti-aligned)
        Σ τ_i τ_j       = +8   (all aligned)
        Σ σ_i σ_j τ_i τ_j = -8   (= Σ σ_i σ_j since τ terms are +1)

    Energy = -1×(-8) - 1×(8) - U×(-8) = 8 - 8 + 8U = 8U

    Magnetizations:
        m_sigma     = (1+(-1)+(-1)+1)/4 = 0
        m_tau       = 4/4 = 1
        m_baxter    = ((+1)(+1) + (-1)(+1) + (-1)(+1) + (+1)(+1)) / 4 = 0
    """
    from pbc_datagen._core import AshkinTellerModel

    U = 0.5
    model = AshkinTellerModel(2, seed=42)
    model.set_four_spin_coupling(U)

    # Create checkerboard in σ (sites 1 and 2 flipped from cold start)
    model.set_sigma(1, -1)
    model.set_sigma(2, -1)
    # τ stays all +1

    # Energy
    assert model.energy() == pytest.approx(8.0 * U)  # 4.0

    # Magnetizations
    assert model.m_sigma() == pytest.approx(0.0)
    assert model.abs_m_sigma() == pytest.approx(0.0)
    assert model.m_tau() == pytest.approx(1.0)
    assert model.abs_m_tau() == pytest.approx(1.0)
    assert model.m_baxter() == pytest.approx(0.0)
    assert model.abs_m_baxter() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 10. set_sigma / set_tau validation
# ---------------------------------------------------------------------------
def test_set_spin_validation() -> None:
    """set_sigma and set_tau accept ±1 only.  Reject 0, 2, etc.

    Unlike Blume-Capel, AT spins are strictly Ising-like (no vacancies).
    """
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(4, seed=42)

    # Valid flips
    model.set_sigma(0, -1)
    assert model.sigma.flat[0] == -1
    model.set_tau(0, -1)
    assert model.tau.flat[0] == -1

    # Flip back
    model.set_sigma(0, 1)
    assert model.sigma.flat[0] == 1

    # Invalid: spin = 0 (not allowed in AT)
    with pytest.raises((ValueError, RuntimeError)):
        model.set_sigma(0, 0)

    with pytest.raises((ValueError, RuntimeError)):
        model.set_tau(0, 0)

    # Invalid: spin = 2 (out of range)
    with pytest.raises((ValueError, RuntimeError)):
        model.set_sigma(0, 2)

    with pytest.raises((ValueError, RuntimeError)):
        model.set_tau(0, 2)
