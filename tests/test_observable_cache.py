"""Tests for observable consistency: Python O(N) recomputation vs C++ methods.

These tests verify that model.energy(), model.magnetization(), etc. return
values consistent with the spin arrays.  Currently both use O(N) sums, so
these pass trivially.  When we switch to incremental caching (O(1) observable
updates on spin flip), these tests guard against cache desynchronization bugs.

Strategy for each model, for each mutation path (Metropolis, Wolff, set_spin):
  1. Warm up to get a non-trivial spin configuration
  2. Run the mutation 20 times
  3. After EACH mutation, recompute all observables from the spin array
     in pure Python and assert they match the C++ method calls

All imports are lazy (inside test functions) so pytest can collect
the tests before the C++ binding exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pbc_datagen._core import AshkinTellerModel, BlumeCapelModel


# ---------------------------------------------------------------------------
# Pure-Python O(N) observable recomputation helpers
# ---------------------------------------------------------------------------


def _ising_energy_python(spins: np.ndarray, nbr: np.ndarray, N: int) -> int:
    """H = -J Σ_{<ij>} s_i s_j  (each bond counted once)."""
    coupling = 0
    for i in range(N):
        si = int(spins[i])
        for d in range(4):
            j = int(nbr[i, d])
            coupling += si * int(spins[j])
    return -coupling // 2


def _magnetization_python(spins: np.ndarray, N: int) -> tuple[float, float]:
    """Returns (m, |m|) = ((1/N) Σ s_i, (1/N) |Σ s_i|)."""
    m_sum = int(np.sum(spins.astype(int)))
    return m_sum / N, abs(m_sum) / N


def _bc_energy_python(spins: np.ndarray, nbr: np.ndarray, N: int, D: float) -> float:
    """H = -J Σ_{<ij>} s_i s_j  +  D Σ_i s_i²."""
    coupling = 0
    sq_sum = 0
    for i in range(N):
        si = int(spins[i])
        sq_sum += si * si
        for d in range(4):
            j = int(nbr[i, d])
            coupling += si * int(spins[j])
    return -coupling / 2.0 + D * sq_sum


def _quadrupole_python(spins: np.ndarray, N: int) -> float:
    """Q = (1/N) Σ_i s_i²."""
    return int(np.sum(spins.astype(int) ** 2)) / N


def _at_energy_python(
    sigma: np.ndarray, tau: np.ndarray, nbr: np.ndarray, N: int, U: float
) -> float:
    """H = -Σ σ_i σ_j  -  Σ τ_i τ_j  -  U Σ σ_i σ_j τ_i τ_j."""
    sigma_sum = 0
    tau_sum = 0
    four_sum = 0
    for i in range(N):
        si, ti = int(sigma[i]), int(tau[i])
        for d in range(4):
            j = int(nbr[i, d])
            sj, tj = int(sigma[j]), int(tau[j])
            sigma_sum += si * sj
            tau_sum += ti * tj
            four_sum += si * sj * ti * tj
    return -sigma_sum / 2.0 - tau_sum / 2.0 - U * four_sum / 2.0


def _at_magnetizations_python(sigma: np.ndarray, tau: np.ndarray, N: int) -> dict[str, float]:
    """All 6 AT magnetization observables from flat σ, τ arrays."""
    s_sum = int(np.sum(sigma.astype(int)))
    t_sum = int(np.sum(tau.astype(int)))
    b_sum = int(np.sum(sigma.astype(int) * tau.astype(int)))
    return {
        "m_sigma": s_sum / N,
        "abs_m_sigma": abs(s_sum) / N,
        "m_tau": t_sum / N,
        "abs_m_tau": abs(t_sum) / N,
        "m_baxter": b_sum / N,
        "abs_m_baxter": abs(b_sum) / N,
    }


# ---------------------------------------------------------------------------
# Ising model — 3 observables: E (int), m, |m|
# ---------------------------------------------------------------------------


def test_ising_observables_after_metropolis() -> None:
    """After each Metropolis sweep, Python-recomputed E/m/|m| match C++."""
    from pbc_datagen._core import IsingModel, make_neighbor_table

    L, N = 8, 64
    model = IsingModel(L=L, seed=42)
    model.set_temperature(2.269)
    model.sweep(5)  # warm up to non-trivial state

    nbr = make_neighbor_table(L)

    for _ in range(20):
        model._metropolis_sweep()
        spins = model.spins.ravel()

        e_py = _ising_energy_python(spins, nbr, N)
        m_py, abs_m_py = _magnetization_python(spins, N)

        assert model.energy() == e_py
        assert model.magnetization() == pytest.approx(m_py, abs=1e-14)
        assert model.abs_magnetization() == pytest.approx(abs_m_py, abs=1e-14)


def test_ising_observables_after_wolff() -> None:
    """After each Wolff cluster flip, Python-recomputed E/m/|m| match C++."""
    from pbc_datagen._core import IsingModel, make_neighbor_table

    L, N = 8, 64
    model = IsingModel(L=L, seed=42)
    model.set_temperature(2.269)
    model.sweep(5)

    nbr = make_neighbor_table(L)

    for _ in range(20):
        model._wolff_step()
        spins = model.spins.ravel()

        e_py = _ising_energy_python(spins, nbr, N)
        m_py, abs_m_py = _magnetization_python(spins, N)

        assert model.energy() == e_py
        assert model.magnetization() == pytest.approx(m_py, abs=1e-14)
        assert model.abs_magnetization() == pytest.approx(abs_m_py, abs=1e-14)


def test_ising_observables_after_set_spin() -> None:
    """After manual set_spin() calls, Python-recomputed E/m/|m| match C++."""
    from pbc_datagen._core import IsingModel, make_neighbor_table

    L, N = 8, 64
    model = IsingModel(L=L, seed=42)
    model.set_temperature(2.269)
    model.sweep(5)

    nbr = make_neighbor_table(L)
    rng = np.random.default_rng(99)

    for _ in range(20):
        site = int(rng.integers(0, N))
        new_val = int(rng.choice([-1, 1]))
        model.set_spin(site, new_val)

        spins = model.spins.ravel()
        e_py = _ising_energy_python(spins, nbr, N)
        m_py, abs_m_py = _magnetization_python(spins, N)

        assert model.energy() == e_py
        assert model.magnetization() == pytest.approx(m_py, abs=1e-14)
        assert model.abs_magnetization() == pytest.approx(abs_m_py, abs=1e-14)


# ---------------------------------------------------------------------------
# Blume-Capel model — 4 observables: E (float), m, |m|, Q
#
# Uses D=1.5 so the crystal-field term is non-trivial.
# ---------------------------------------------------------------------------


def _assert_bc_observables(model: BlumeCapelModel, nbr: np.ndarray, N: int, D: float) -> None:
    """Recompute all BC observables in Python and assert match."""
    spins = model.spins.ravel()

    e_py = _bc_energy_python(spins, nbr, N, D)
    m_py, abs_m_py = _magnetization_python(spins, N)
    q_py = _quadrupole_python(spins, N)

    assert model.energy() == pytest.approx(e_py, abs=1e-10)
    assert model.magnetization() == pytest.approx(m_py, abs=1e-14)
    assert model.abs_magnetization() == pytest.approx(abs_m_py, abs=1e-14)
    assert model.quadrupole() == pytest.approx(q_py, abs=1e-14)


def test_bc_observables_after_metropolis() -> None:
    """After each Metropolis sweep, Python-recomputed E/m/|m|/Q match C++."""
    from pbc_datagen._core import BlumeCapelModel, make_neighbor_table

    L, N, D = 8, 64, 1.5
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(2.0)
    model.set_crystal_field(D)
    model.sweep(5)

    nbr = make_neighbor_table(L)

    for _ in range(20):
        model._metropolis_sweep()
        _assert_bc_observables(model, nbr, N, D)


def test_bc_observables_after_wolff() -> None:
    """After each Wolff cluster flip, Python-recomputed E/m/|m|/Q match C++."""
    from pbc_datagen._core import BlumeCapelModel, make_neighbor_table

    L, N, D = 8, 64, 1.5
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(2.0)
    model.set_crystal_field(D)
    model.sweep(5)

    nbr = make_neighbor_table(L)

    for _ in range(20):
        model._wolff_step()
        _assert_bc_observables(model, nbr, N, D)


def test_bc_observables_after_set_spin() -> None:
    """After manual set_spin() calls, Python-recomputed E/m/|m|/Q match C++."""
    from pbc_datagen._core import BlumeCapelModel, make_neighbor_table

    L, N, D = 8, 64, 1.5
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(2.0)
    model.set_crystal_field(D)
    model.sweep(5)

    nbr = make_neighbor_table(L)
    rng = np.random.default_rng(99)

    for _ in range(20):
        site = int(rng.integers(0, N))
        new_val = int(rng.choice([-1, 0, 1]))
        model.set_spin(site, new_val)
        _assert_bc_observables(model, nbr, N, D)


def test_bc_observables_after_set_crystal_field() -> None:
    """Changing D mid-simulation must update cached energy correctly.

    The crystal-field term D Σ s_i² changes when D changes, even though
    no spins moved.  Magnetization and quadrupole are D-independent.
    """
    from pbc_datagen._core import BlumeCapelModel, make_neighbor_table

    L, N = 8, 64
    model = BlumeCapelModel(L=L, seed=42)
    model.set_temperature(2.0)
    model.set_crystal_field(1.0)
    model.sweep(10)  # get a non-trivial mixed state

    nbr = make_neighbor_table(L)

    for D in [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, 3.0]:
        model.set_crystal_field(D)
        _assert_bc_observables(model, nbr, N, D)


# ---------------------------------------------------------------------------
# Ashkin-Teller model — 7 observables: E (float), m_σ, |m_σ|, m_τ, |m_τ|, m_B, |m_B|
#
# Two regimes tested:
#   U=0.7 (non-remapped): Wolff clusters on σ or τ directly
#   U=1.5 (remapped):     Wolff clusters on σ or s=στ (Wiseman-Domany basis)
# ---------------------------------------------------------------------------


def _assert_at_observables(model: AshkinTellerModel, nbr: np.ndarray, N: int, U: float) -> None:
    """Recompute all AT observables in Python and assert match."""
    sig = model.sigma.ravel()
    ta = model.tau.ravel()

    e_py = _at_energy_python(sig, ta, nbr, N, U)
    mags = _at_magnetizations_python(sig, ta, N)

    assert model.energy() == pytest.approx(e_py, abs=1e-10)
    assert model.m_sigma() == pytest.approx(mags["m_sigma"], abs=1e-14)
    assert model.abs_m_sigma() == pytest.approx(mags["abs_m_sigma"], abs=1e-14)
    assert model.m_tau() == pytest.approx(mags["m_tau"], abs=1e-14)
    assert model.abs_m_tau() == pytest.approx(mags["abs_m_tau"], abs=1e-14)
    assert model.m_baxter() == pytest.approx(mags["m_baxter"], abs=1e-14)
    assert model.abs_m_baxter() == pytest.approx(mags["abs_m_baxter"], abs=1e-14)


def test_at_observables_after_metropolis() -> None:
    """After each Metropolis sweep, Python-recomputed 7 observables match C++."""
    from pbc_datagen._core import AshkinTellerModel, make_neighbor_table

    L, N, U = 8, 64, 0.7
    model = AshkinTellerModel(L=L, seed=42)
    model.set_temperature(3.0)
    model.set_four_spin_coupling(U)
    model.sweep(5)

    nbr = make_neighbor_table(L)

    for _ in range(20):
        model._metropolis_sweep()
        _assert_at_observables(model, nbr, N, U)


def test_at_observables_after_wolff() -> None:
    """After each Wolff step, Python-recomputed 7 observables match C++."""
    from pbc_datagen._core import AshkinTellerModel, make_neighbor_table

    L, N, U = 8, 64, 0.7
    model = AshkinTellerModel(L=L, seed=42)
    model.set_temperature(3.0)
    model.set_four_spin_coupling(U)
    model.sweep(5)

    nbr = make_neighbor_table(L)

    for _ in range(20):
        model._wolff_step()
        _assert_at_observables(model, nbr, N, U)


def test_at_observables_after_set_spin() -> None:
    """After manual set_sigma/set_tau calls, Python-recomputed observables match C++."""
    from pbc_datagen._core import AshkinTellerModel, make_neighbor_table

    L, N, U = 8, 64, 0.7
    model = AshkinTellerModel(L=L, seed=42)
    model.set_temperature(3.0)
    model.set_four_spin_coupling(U)
    model.sweep(5)

    nbr = make_neighbor_table(L)
    rng = np.random.default_rng(99)

    for _ in range(20):
        site = int(rng.integers(0, N))
        model.set_sigma(site, int(rng.choice([-1, 1])))
        site = int(rng.integers(0, N))
        model.set_tau(site, int(rng.choice([-1, 1])))
        _assert_at_observables(model, nbr, N, U)


# ---------------------------------------------------------------------------
# Ashkin-Teller REMAPPED (U > 1) — Wolff uses (σ, s=στ) basis
#
# The Metropolis sweep is identical (always physical basis), but Wolff
# clustering is fundamentally different: modes 2 and 3 flip different
# combinations of σ and τ.  Caching bugs that only manifest under
# the remapped double-flip (mode 2: flip both σ and τ) would be missed
# without these tests.
# ---------------------------------------------------------------------------


def test_at_observables_after_set_four_spin_coupling() -> None:
    """Changing U mid-simulation must update cached energy correctly.

    The four-spin term U Σ σ_i σ_j τ_i τ_j changes when U changes,
    even though no spins moved.  All magnetizations are U-independent.
    """
    from pbc_datagen._core import AshkinTellerModel, make_neighbor_table

    L, N = 8, 64
    model = AshkinTellerModel(L=L, seed=42)
    model.set_temperature(3.0)
    model.set_four_spin_coupling(0.5)
    model.sweep(10)  # get a non-trivial state

    nbr = make_neighbor_table(L)

    for U in [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]:
        model.set_four_spin_coupling(U)
        _assert_at_observables(model, nbr, N, U)


def test_at_remapped_observables_after_metropolis() -> None:
    """U=1.5 (remapped): after Metropolis sweeps, observables match Python."""
    from pbc_datagen._core import AshkinTellerModel, make_neighbor_table

    L, N, U = 8, 64, 1.5
    model = AshkinTellerModel(L=L, seed=42)
    model.set_temperature(3.0)
    model.set_four_spin_coupling(U)
    model.sweep(5)

    nbr = make_neighbor_table(L)

    for _ in range(20):
        model._metropolis_sweep()
        _assert_at_observables(model, nbr, N, U)


def test_at_remapped_observables_after_wolff() -> None:
    """U=1.5 (remapped): after Wolff steps, observables match Python.

    This is the critical test: remapped Wolff mode 2 flips BOTH σ and τ
    simultaneously, which doubles the number of cache updates per cluster
    site compared to the non-remapped case.
    """
    from pbc_datagen._core import AshkinTellerModel, make_neighbor_table

    L, N, U = 8, 64, 1.5
    model = AshkinTellerModel(L=L, seed=42)
    model.set_temperature(3.0)
    model.set_four_spin_coupling(U)
    model.sweep(5)

    nbr = make_neighbor_table(L)

    for _ in range(20):
        model._wolff_step()
        _assert_at_observables(model, nbr, N, U)


def test_at_remapped_observables_after_set_spin() -> None:
    """U=1.5 (remapped): after set_sigma/set_tau, observables match Python."""
    from pbc_datagen._core import AshkinTellerModel, make_neighbor_table

    L, N, U = 8, 64, 1.5
    model = AshkinTellerModel(L=L, seed=42)
    model.set_temperature(3.0)
    model.set_four_spin_coupling(U)
    model.sweep(5)

    nbr = make_neighbor_table(L)
    rng = np.random.default_rng(99)

    for _ in range(20):
        site = int(rng.integers(0, N))
        model.set_sigma(site, int(rng.choice([-1, 1])))
        site = int(rng.integers(0, N))
        model.set_tau(site, int(rng.choice([-1, 1])))
        _assert_at_observables(model, nbr, N, U)
