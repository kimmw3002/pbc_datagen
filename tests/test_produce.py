"""Tests for Phase C: PT production snapshot harvesting.

PTEngine.produce() harvests decorrelated snapshots from all temperature
slots and streams them to an HDF5 file.  These tests verify the output
file layout, data integrity, metadata, and operational warnings.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

if TYPE_CHECKING:
    from pbc_datagen.parallel_tempering import PTEngine


# ---------------------------------------------------------------------------
# Helper: build a ready-to-produce engine (Phases A+B already done)
# ---------------------------------------------------------------------------


def _ready_engine(
    *,
    model_type: str = "ising",
    L: int = 4,
    param_value: float = 0.0,
    n_replicas: int = 5,
    seed: int = 12345,
) -> PTEngine:
    """Run Phases A+B on a small system, return engine ready for produce().

    Uses a 4x4 Ising lattice with 5 replicas — small enough to converge
    quickly, large enough that KTH + Welch checks are meaningful.
    """
    from pbc_datagen.parallel_tempering import PTEngine

    engine = PTEngine(
        model_type=model_type,
        L=L,
        param_value=param_value,
        T_range=(2.0, 2.5),
        n_replicas=n_replicas,
        seed=seed,
    )
    engine.tune_ladder()
    engine.equilibrate()
    return engine


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------


class TestProducePrecondition:
    """produce() requires tau_max to be set (Phase B must have completed)."""

    def test_requires_equilibration(self, tmp_path: Path) -> None:
        """Calling produce() before equilibrate() should raise RuntimeError.

        tau_max is only set after Phase B.  Without it, the thinning
        interval (3 * tau_max) is undefined, so production cannot proceed.
        """
        from pbc_datagen.parallel_tempering import PTEngine

        engine = PTEngine(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.5),
            n_replicas=5,
            seed=42,
        )
        assert engine.tau_max is None

        with pytest.raises(RuntimeError, match="tau_max|equilibrat"):
            engine.produce(tmp_path / "test.h5", n_snapshots=5)


# ---------------------------------------------------------------------------
# HDF5 output structure
# ---------------------------------------------------------------------------


class TestProduceHDF5Layout:
    """Verify produce() creates the correct HDF5 file structure."""

    def test_creates_all_temperature_groups(self, tmp_path: Path) -> None:
        """HDF5 file has one group per temperature in the locked ladder.

        The group names use the canonical "T={value}" format, and every
        temperature in the engine's ladder gets its own group.
        """
        engine = _ready_engine()
        path = tmp_path / "test.h5"

        engine.produce(path, n_snapshots=3)

        with h5py.File(path, "r") as f:
            groups = list(f.keys())
            assert len(groups) == engine.n_replicas
            # Each group name should match a T value in the ladder
            for T in engine.temps:
                assert f"T={T:.4f}" in groups

    def test_correct_snapshot_count(self, tmp_path: Path) -> None:
        """Each T slot has exactly n_snapshots snapshots in its dataset.

        PT production harvests one snapshot from *every* T slot per cycle,
        so all slots must have identical counts equal to n_snapshots.
        """
        engine = _ready_engine()
        path = tmp_path / "test.h5"
        n_snapshots = 5

        engine.produce(path, n_snapshots=n_snapshots)

        with h5py.File(path, "r") as f:
            for T in engine.temps:
                ds = f[f"T={T:.4f}"]["snapshots"]
                assert ds.shape[0] == n_snapshots

    def test_snapshot_dtype_and_shape(self, tmp_path: Path) -> None:
        """Ising snapshots are int8 with shape (n_snapshots, 1, L, L).

        Ising and BC models have C=1 (single spin channel).
        Shape convention: (N, C, L, L) where N = snapshot count.
        """
        L = 4
        engine = _ready_engine(L=L)
        path = tmp_path / "test.h5"
        n_snapshots = 3

        engine.produce(path, n_snapshots=n_snapshots)

        with h5py.File(path, "r") as f:
            ds = f[f"T={engine.temps[0]:.4f}"]["snapshots"]
            assert ds.dtype == np.int8
            assert ds.shape == (n_snapshots, 1, L, L)

    def test_observable_datasets_match_model(self, tmp_path: Path) -> None:
        """Observable datasets use the model's observables() keys, all float64.

        Ising produces: energy, m, abs_m.  Each should appear as a
        separate (N,) float64 dataset alongside the snapshots dataset.
        """
        engine = _ready_engine()
        path = tmp_path / "test.h5"
        n_snapshots = 3

        engine.produce(path, n_snapshots=n_snapshots)

        expected_obs = list(engine.replicas[0].observables().keys())

        with h5py.File(path, "r") as f:
            grp = f[f"T={engine.temps[0]:.4f}"]
            for name in expected_obs:
                assert name in grp, f"missing observable dataset '{name}'"
                ds = grp[name]
                assert ds.dtype == np.float64
                assert ds.shape == (n_snapshots,)


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


class TestProduceDataIntegrity:
    """Verify that stored snapshot data is physically valid."""

    def test_spins_are_valid_ising(self, tmp_path: Path) -> None:
        """All stored Ising spins are +/-1 (no zeros, no garbage).

        The Ising model only has spin values {-1, +1}.  Zeros or other
        values would indicate a bug in the spin-reading or HDF5 writing.
        """
        engine = _ready_engine()
        path = tmp_path / "test.h5"

        engine.produce(path, n_snapshots=3)

        with h5py.File(path, "r") as f:
            for T in engine.temps:
                data = f[f"T={T:.4f}"]["snapshots"][:]
                # Every element must be exactly +1 or -1
                assert np.all(np.isin(data, [-1, 1])), (
                    f"Invalid spin values at T={T}: unique={np.unique(data)}"
                )

    def test_observables_are_finite(self, tmp_path: Path) -> None:
        """All observable values must be finite (no NaN or Inf).

        NaN/Inf would indicate a corrupted model state or division by zero
        in the observable computation.
        """
        engine = _ready_engine()
        path = tmp_path / "test.h5"

        engine.produce(path, n_snapshots=3)

        expected_obs = list(engine.replicas[0].observables().keys())

        with h5py.File(path, "r") as f:
            for T in engine.temps:
                grp = f[f"T={T:.4f}"]
                for name in expected_obs:
                    values = grp[name][:]
                    assert np.all(np.isfinite(values)), f"Non-finite values in {name} at T={T}"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestProduceMetadata:
    """Verify campaign metadata is written to HDF5 root attributes."""

    def test_metadata_attrs_written(self, tmp_path: Path) -> None:
        """Root attrs include model_type, L, T_ladder, tau_max, seed.

        These are essential for resume and for downstream analysis to
        know what parameters produced the dataset.
        """
        engine = _ready_engine()
        path = tmp_path / "test.h5"

        engine.produce(path, n_snapshots=3)

        with h5py.File(path, "r") as f:
            assert f.attrs["model_type"] == "ising"
            assert f.attrs["L"] == engine.L
            assert f.attrs["tau_max"] == pytest.approx(engine.tau_max)
            assert f.attrs["seed"] == engine.seed
            np.testing.assert_allclose(f.attrs["T_ladder"], engine.temps)

    def test_address_maps_written(self, tmp_path: Path) -> None:
        """Root attrs include r2t and t2r address maps.

        These are needed for crash-resume: if we stop mid-production,
        the replica-to-temperature assignment must be restored.
        """
        engine = _ready_engine()
        path = tmp_path / "test.h5"

        engine.produce(path, n_snapshots=3)

        with h5py.File(path, "r") as f:
            r2t = f.attrs["r2t"]
            t2r = f.attrs["t2r"]
            assert len(r2t) == engine.n_replicas
            assert len(t2r) == engine.n_replicas

    def test_seed_history_present(self, tmp_path: Path) -> None:
        """seed_history attr exists with at least the initial (0, seed) entry.

        The seed history enables full reproducibility: each entry records
        which seed was used starting from which snapshot offset.
        """
        engine = _ready_engine()
        path = tmp_path / "test.h5"

        engine.produce(path, n_snapshots=3)

        with h5py.File(path, "r") as f:
            import json

            raw = json.loads(str(f.attrs["seed_history"]))
            assert len(raw) >= 1
            # First entry should reference the engine's seed
            assert raw[0][1] == engine.seed


# ---------------------------------------------------------------------------
# Resume — calling produce() on an existing HDF5 file
# ---------------------------------------------------------------------------


class TestProduceResume:
    """Verify produce() correctly resumes from a partially-filled HDF5 file.

    PT production writes one snapshot per cycle to every T slot in lockstep.
    If the process crashes mid-campaign, the HDF5 file already has some
    snapshots.  Calling produce() again with the same n_snapshots target
    must append only the missing ones — not crash, not duplicate.
    """

    def test_resume_does_not_crash_on_existing_file(self, tmp_path: Path) -> None:
        """Calling produce() twice must not raise on group-already-exists.

        h5py.create_group() raises ValueError if the group name is taken.
        produce() must skip group creation when T slots already exist.
        """
        engine = _ready_engine()
        path = tmp_path / "test.h5"

        engine.produce(path, n_snapshots=3)
        # Second call — must not raise
        engine.produce(path, n_snapshots=5)

    def test_resume_appends_remaining_snapshots(self, tmp_path: Path) -> None:
        """First call writes 3, second call targets 5 → file has exactly 5.

        n_snapshots is the *target total*, not "collect this many more".
        produce() counts existing snapshots on disk and only collects
        the remainder.
        """
        engine = _ready_engine()
        path = tmp_path / "test.h5"

        engine.produce(path, n_snapshots=3)
        engine.produce(path, n_snapshots=5)

        with h5py.File(path, "r") as f:
            for T in engine.temps:
                assert f[f"T={T:.4f}"]["snapshots"].shape[0] == 5

    def test_resume_already_complete_is_noop(self, tmp_path: Path) -> None:
        """If file already has n_snapshots, produce() adds nothing.

        This prevents accidental duplication when a completed campaign
        is restarted (e.g. the orchestrator re-runs all param values).
        """
        engine = _ready_engine()
        path = tmp_path / "test.h5"

        engine.produce(path, n_snapshots=3)
        engine.produce(path, n_snapshots=3)

        with h5py.File(path, "r") as f:
            for T in engine.temps:
                assert f[f"T={T:.4f}"]["snapshots"].shape[0] == 3

    def test_seed_history_from_caller(self, tmp_path: Path) -> None:
        """produce() uses caller-provided seed_history instead of default.

        On resume the orchestrator loads the old seed_history from
        read_resume_state(), appends a new (n_existing, new_seed) entry,
        and passes the extended list to produce().  produce() must store
        it verbatim, not overwrite with [(0, self.seed)].
        """
        import json

        engine = _ready_engine()
        path = tmp_path / "test.h5"
        history: list[tuple[int, int]] = [(0, 42), (3, 99999)]

        engine.produce(path, n_snapshots=3, seed_history=history)

        with h5py.File(path, "r") as f:
            raw = json.loads(str(f.attrs["seed_history"]))
            assert len(raw) == 2
            assert raw[0] == [0, 42]
            assert raw[1] == [3, 99999]
