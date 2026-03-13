"""Red-phase tests for Phase 5.0: snapshot() and randomize() on all models.

snapshot() returns a (C, L, L) numpy array — a point-in-time copy of the
model's spin configuration in the correct dtype.  This eliminates the
Python-side if-else that checks model_type to decide how to collect spins.

randomize() sets all spins to uniformly random valid values using the
model's internal RNG.  This eliminates the Python-side _randomize_all()
if-else that checks model_type to decide valid spin values.

All imports are lazy (inside test functions) so pytest can collect
the tests before the C++ methods are bound.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# snapshot() — shape, dtype, values
# ---------------------------------------------------------------------------


def test_ising_snapshot_shape_and_dtype() -> None:
    """IsingModel.snapshot() → (1, L, L) int8 array."""
    from pbc_datagen._core import IsingModel

    L = 6
    model = IsingModel(L=L, seed=42)
    snap = model.snapshot()
    assert isinstance(snap, np.ndarray)
    assert snap.shape == (1, L, L)
    assert snap.dtype == np.int8


def test_blume_capel_snapshot_shape_and_dtype() -> None:
    """BlumeCapelModel.snapshot() → (1, L, L) int8 array."""
    from pbc_datagen._core import BlumeCapelModel

    L = 6
    model = BlumeCapelModel(L=L, seed=42)
    snap = model.snapshot()
    assert isinstance(snap, np.ndarray)
    assert snap.shape == (1, L, L)
    assert snap.dtype == np.int8


def test_ashkin_teller_snapshot_shape_and_dtype() -> None:
    """AshkinTellerModel.snapshot() → (2, L, L) int8, channel 0=σ, 1=τ."""
    from pbc_datagen._core import AshkinTellerModel

    L = 6
    model = AshkinTellerModel(L=L, seed=42)
    snap = model.snapshot()
    assert isinstance(snap, np.ndarray)
    assert snap.shape == (2, L, L)
    assert snap.dtype == np.int8


def test_ising_snapshot_matches_spins() -> None:
    """snapshot() values must match spins[np.newaxis] on cold start."""
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=8, seed=99)
    snap = model.snapshot()
    expected = model.spins[np.newaxis].copy()
    np.testing.assert_array_equal(snap, expected)


def test_ashkin_teller_snapshot_matches_sigma_tau() -> None:
    """snapshot() must stack [sigma, tau] along channel axis."""
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(L=8, seed=99)
    model.set_temperature(2.0)
    model.sweep(10)  # get a non-trivial config
    snap = model.snapshot()
    expected = np.stack([model.sigma, model.tau])
    np.testing.assert_array_equal(snap, expected)


def test_snapshot_is_copy_not_view() -> None:
    """snapshot() must return an owning copy — mutating spins must not affect it."""
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=4, seed=42)
    snap_before = model.snapshot()
    model.set_temperature(5.0)
    model.sweep(50)  # mutate spins heavily
    snap_after = model.snapshot()
    # snap_before should still hold the cold-start values
    assert np.all(snap_before == 1), "snapshot was a view, not a copy"
    # snap_after should differ (high-T scrambles the lattice)
    assert not np.array_equal(snap_before, snap_after)


# ---------------------------------------------------------------------------
# randomize() — produces valid random configurations
# ---------------------------------------------------------------------------


def test_ising_randomize_produces_mixed_config() -> None:
    """After randomize(), Ising spins should contain both +1 and -1."""
    from pbc_datagen._core import IsingModel

    model = IsingModel(L=16, seed=42)
    assert np.all(model.spins == 1), "precondition: cold start"
    model.randomize()
    flat = model.spins.ravel()
    assert np.any(flat == 1) and np.any(flat == -1), "expected both ±1"
    assert set(np.unique(flat)) <= {-1, 1}, "Ising spins must be ±1 only"


def test_blume_capel_randomize_produces_three_values() -> None:
    """After randomize(), BC spins should contain -1, 0, and +1."""
    from pbc_datagen._core import BlumeCapelModel

    model = BlumeCapelModel(L=16, seed=42)
    model.randomize()
    flat = model.spins.ravel()
    unique = set(np.unique(flat))
    assert unique <= {-1, 0, 1}, f"BC spins must be in {{-1,0,+1}}, got {unique}"
    # With 256 sites, all three values should appear
    assert len(unique) == 3, f"expected all three values on 16×16, got {unique}"


def test_ashkin_teller_randomize_both_layers() -> None:
    """After randomize(), both σ and τ layers should be mixed ±1."""
    from pbc_datagen._core import AshkinTellerModel

    model = AshkinTellerModel(L=16, seed=42)
    assert np.all(model.sigma == 1), "precondition: cold start σ"
    assert np.all(model.tau == 1), "precondition: cold start τ"
    model.randomize()
    for name, arr in [("sigma", model.sigma), ("tau", model.tau)]:
        flat = arr.ravel()
        assert np.any(flat == 1) and np.any(flat == -1), f"{name} not mixed"
        assert set(np.unique(flat)) <= {-1, 1}, f"{name} must be ±1"
