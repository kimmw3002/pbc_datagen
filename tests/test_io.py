"""Tests for HDF5 snapshot I/O — flat-schema SnapshotWriter, attrs, resume."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ising_obs_names() -> list[str]:
    """Observable names for the Ising model."""
    return ["energy", "m", "abs_m"]


def _random_spins(M: int, C: int, L: int) -> np.ndarray:
    """Random ±1 spin array shaped (M, C, L, L), dtype int8."""
    rng = np.random.default_rng(42)
    return rng.choice([-1, 1], size=(M, C, L, L)).astype(np.int8)


# ---------------------------------------------------------------------------
# SnapshotWriter — flat dataset creation
# ---------------------------------------------------------------------------


class TestSnapshotWriterCreation:
    """Verify that create_datasets builds the correct flat HDF5 layout."""

    def test_create_datasets_layout(self, tmp_path: Path) -> None:
        """create_datasets produces root-level 'snapshots' + obs datasets."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        obs_names = _ising_obs_names()
        slot_keys = ["T=2.0000", "T=2.5000"]

        with SnapshotWriter(path) as w:
            w.create_datasets(slot_keys, n_snapshots=10, C=1, L=4, obs_names=obs_names)

        with h5py.File(path, "r") as f:
            assert "snapshots" in f
            for name in obs_names:
                assert name in f, f"missing observable dataset '{name}'"
            assert json.loads(str(f.attrs["slot_keys"])) == slot_keys
            assert json.loads(str(f.attrs["obs_names"])) == obs_names
            assert int(f.attrs["count"]) == 0

    def test_create_datasets_shapes(self, tmp_path: Path) -> None:
        """Datasets have correct shapes: (M, n_snap, C, L, L) and (M, n_snap)."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        L, C, M, n_snap = 4, 1, 3, 10
        obs_names = _ising_obs_names()
        slot_keys = [f"T={i:.4f}" for i in range(M)]

        with SnapshotWriter(path) as w:
            w.create_datasets(slot_keys, n_snap, C, L, obs_names)

        with h5py.File(path, "r") as f:
            assert f["snapshots"].shape == (M, n_snap, C, L, L)
            assert f["snapshots"].dtype == np.int8
            assert f["snapshots"].maxshape[1] is None  # resizable along axis 1
            for name in obs_names:
                assert f[name].shape == (M, n_snap)
                assert f[name].dtype == np.float64


# ---------------------------------------------------------------------------
# SnapshotWriter — write_round
# ---------------------------------------------------------------------------


class TestSnapshotWriterWriteRound:
    """Verify that write_round stores correct data."""

    def test_write_single_round(self, tmp_path: Path) -> None:
        """Writing one round stores spins and obs at [:, 0]."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        L, C, M = 4, 1, 2
        obs_names = _ising_obs_names()
        slot_keys = ["T=2.0000", "T=2.5000"]

        spins = _random_spins(M, C, L)
        obs = {
            "energy": np.array([-12.0, -8.0]),
            "m": np.array([0.5, 0.3]),
            "abs_m": np.array([0.5, 0.3]),
        }

        with SnapshotWriter(path) as w:
            w.create_datasets(slot_keys, n_snapshots=10, C=C, L=L, obs_names=obs_names)
            w.write_round(spins, obs)
            assert w.snapshot_count == 1

        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(f["snapshots"][:, 0], spins)
            assert f["energy"][0, 0] == pytest.approx(-12.0)
            assert f["energy"][1, 0] == pytest.approx(-8.0)

    def test_write_multiple_rounds(self, tmp_path: Path) -> None:
        """Three rounds produce count=3, each at correct index."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        L, C, M = 4, 1, 2
        obs_names = _ising_obs_names()
        slot_keys = ["T=2.0000", "T=2.5000"]

        rng = np.random.default_rng(99)
        all_data = []
        for _ in range(3):
            sp = rng.choice([-1, 1], size=(M, C, L, L)).astype(np.int8)
            ob = {name: rng.random(M) for name in obs_names}
            all_data.append((sp, ob))

        with SnapshotWriter(path) as w:
            w.create_datasets(slot_keys, n_snapshots=10, C=C, L=L, obs_names=obs_names)
            for sp, ob in all_data:
                w.write_round(sp, ob)
            assert w.snapshot_count == 3

        with h5py.File(path, "r") as f:
            assert int(f.attrs["count"]) == 3
            for i, (sp, ob) in enumerate(all_data):
                np.testing.assert_array_equal(f["snapshots"][:, i], sp)

    def test_two_channel_ashkin_teller(self, tmp_path: Path) -> None:
        """C=2 (Ashkin-Teller) snapshots have shape (M, n_snap, 2, L, L)."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        L, C, M = 4, 2, 2
        obs_names = ["energy", "m_sigma", "abs_m_sigma"]
        slot_keys = ["T=2.0000", "T=2.5000"]

        spins = _random_spins(M, C, L)

        with SnapshotWriter(path) as w:
            w.create_datasets(slot_keys, n_snapshots=5, C=C, L=L, obs_names=obs_names)
            obs = {name: np.zeros(M) for name in obs_names}
            w.write_round(spins, obs)

        with h5py.File(path, "r") as f:
            assert f["snapshots"].shape == (M, 5, 2, L, L)
            np.testing.assert_array_equal(f["snapshots"][:, 0], spins)


# ---------------------------------------------------------------------------
# SnapshotWriter — resume (open_datasets)
# ---------------------------------------------------------------------------


class TestSnapshotWriterResume:
    """Verify open_datasets loads state correctly for resume."""

    def test_open_datasets_resumes_count(self, tmp_path: Path) -> None:
        """After close + reopen, open_datasets restores the correct count."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        L, C, M = 4, 1, 2
        obs_names = _ising_obs_names()
        slot_keys = ["T=2.0000", "T=2.5000"]

        # Write 3 rounds, close
        with SnapshotWriter(path) as w:
            w.create_datasets(slot_keys, n_snapshots=10, C=C, L=L, obs_names=obs_names)
            for _ in range(3):
                w.write_round(
                    _random_spins(M, C, L),
                    {name: np.zeros(M) for name in obs_names},
                )

        # Reopen and resume
        with SnapshotWriter(path) as w:
            w.open_datasets()
            assert w.snapshot_count == 3

            # Write 2 more rounds
            for _ in range(2):
                w.write_round(
                    _random_spins(M, C, L),
                    {name: np.zeros(M) for name in obs_names},
                )
            assert w.snapshot_count == 5

        with h5py.File(path, "r") as f:
            assert int(f.attrs["count"]) == 5


# ---------------------------------------------------------------------------
# write_param_attrs — metadata round-trip
# ---------------------------------------------------------------------------


class TestWriteParamAttrs:
    """Verify campaign metadata survives an HDF5 attrs round-trip."""

    def test_roundtrip_all_fields(self, tmp_path: Path) -> None:
        """Every field written by write_param_attrs is recoverable."""
        from pbc_datagen.io import SnapshotWriter, write_param_attrs

        path = tmp_path / "test.h5"
        T_ladder = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
        r2t = [0, 1, 2, 3, 4]
        t2r = [0, 1, 2, 3, 4]
        seed = 123456789
        seed_history: list[tuple[int, int]] = [(0, 123456789)]

        # Create a minimal valid HDF5 file first
        with SnapshotWriter(path) as w:
            w.create_datasets(["T=2.0000"], n_snapshots=1, C=1, L=4, obs_names=_ising_obs_names())

        write_param_attrs(
            path,
            model_type="ising",
            L=4,
            param_value=0.0,
            T_ladder=T_ladder,
            tau_max=12.5,
            r2t=r2t,
            t2r=t2r,
            seed=seed,
            seed_history=seed_history,
        )

        with h5py.File(path, "r") as f:
            assert f.attrs["model_type"] == "ising"
            assert f.attrs["L"] == 4
            assert f.attrs["param_value"] == pytest.approx(0.0)
            assert f.attrs["tau_max"] == pytest.approx(12.5)
            assert f.attrs["seed"] == seed
            np.testing.assert_allclose(f.attrs["T_ladder"], T_ladder)
            np.testing.assert_array_equal(f.attrs["r2t"], r2t)
            np.testing.assert_array_equal(f.attrs["t2r"], t2r)

    def test_repeated_calls_overwrite_address_maps(self, tmp_path: Path) -> None:
        """Calling write_param_attrs again overwrites r2t/t2r."""
        from pbc_datagen.io import SnapshotWriter, write_param_attrs

        path = tmp_path / "test.h5"
        T_ladder = np.array([1.0, 2.0, 3.0])

        with SnapshotWriter(path) as w:
            w.create_datasets(["T=2.0000"], n_snapshots=1, C=1, L=4, obs_names=_ising_obs_names())

        write_param_attrs(
            path,
            model_type="ising",
            L=4,
            param_value=0.0,
            T_ladder=T_ladder,
            tau_max=5.0,
            r2t=[0, 1, 2],
            t2r=[0, 1, 2],
            seed=100,
            seed_history=[(0, 100)],
        )
        write_param_attrs(
            path,
            model_type="ising",
            L=4,
            param_value=0.0,
            T_ladder=T_ladder,
            tau_max=5.0,
            r2t=[2, 1, 0],
            t2r=[2, 1, 0],
            seed=100,
            seed_history=[(0, 100)],
        )

        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(f.attrs["r2t"], [2, 1, 0])
            np.testing.assert_array_equal(f.attrs["t2r"], [2, 1, 0])


# ---------------------------------------------------------------------------
# read_resume_state — resume from existing HDF5
# ---------------------------------------------------------------------------


class TestReadResumeState:
    """Verify read_resume_state extracts correct state from an HDF5 file."""

    def _make_populated_file(
        self,
        path: Path,
        *,
        n_snapshots: int = 5,
        r2t: list[int] | None = None,
        t2r: list[int] | None = None,
        seed_history: list[tuple[int, int]] | None = None,
    ) -> int:
        """Helper: create an HDF5 with uniform snapshot counts. Returns seed."""
        from pbc_datagen.io import SnapshotWriter, write_param_attrs

        seed = 42
        T_ladder = np.array([1.5, 2.0, 2.5])
        obs_names = _ising_obs_names()
        L, C, M = 4, 1, 3
        slot_keys = [f"T={T:.4f}" for T in T_ladder]

        with SnapshotWriter(path) as w:
            w.create_datasets(slot_keys, n_snapshots, C, L, obs_names)
            for _ in range(n_snapshots):
                w.write_round(
                    _random_spins(M, C, L),
                    {name: np.zeros(M) for name in obs_names},
                )

        write_param_attrs(
            path,
            model_type="ising",
            L=L,
            param_value=0.0,
            T_ladder=T_ladder,
            tau_max=10.0,
            r2t=r2t if r2t is not None else [0, 1, 2],
            t2r=t2r if t2r is not None else [0, 1, 2],
            seed=seed,
            seed_history=seed_history if seed_history is not None else [(0, seed)],
        )
        return seed

    def test_returns_seed_and_uniform_snapshot_count(self, tmp_path: Path) -> None:
        """Resume state has correct seed and equal counts across all slots."""
        from pbc_datagen.io import read_resume_state

        path = tmp_path / "test.h5"
        expected_seed = self._make_populated_file(path, n_snapshots=7)

        seed, state = read_resume_state(path)

        assert seed == expected_seed
        counts = state["snapshot_counts"]
        assert len(counts) == 3
        assert all(c == 7 for c in counts.values())

    def test_preserves_ladder_and_metadata(self, tmp_path: Path) -> None:
        """Resume state round-trips T_ladder, tau_max, and model_type."""
        from pbc_datagen.io import read_resume_state

        path = tmp_path / "test.h5"
        self._make_populated_file(path)

        _seed, state = read_resume_state(path)

        np.testing.assert_allclose(state["T_ladder"], [1.5, 2.0, 2.5])
        assert state["tau_max"] == pytest.approx(10.0)
        assert state["model_type"] == "ising"
        assert state["L"] == 4

    def test_resume_returns_latest_address_maps(self, tmp_path: Path) -> None:
        """read_resume_state returns the most recently saved r2t/t2r."""
        from pbc_datagen.io import read_resume_state

        path = tmp_path / "test.h5"
        self._make_populated_file(path, r2t=[2, 1, 0], t2r=[2, 1, 0])

        _seed, state = read_resume_state(path)

        np.testing.assert_array_equal(state["r2t"], [2, 1, 0])
        np.testing.assert_array_equal(state["t2r"], [2, 1, 0])

    def test_resume_returns_seed_history(self, tmp_path: Path) -> None:
        """seed_history must survive the round-trip."""
        from pbc_datagen.io import read_resume_state

        path = tmp_path / "test.h5"
        history: list[tuple[int, int]] = [(0, 42), (50, 99999)]
        self._make_populated_file(path, seed_history=history)

        _seed, state = read_resume_state(path)

        recovered = state["seed_history"]
        assert len(recovered) == 2
        assert recovered[0] == (0, 42)
        assert recovered[1] == (50, 99999)


# ---------------------------------------------------------------------------
# _snapshot_count — flat + old-format fallback
# ---------------------------------------------------------------------------


class TestSnapshotCount:
    """Verify _snapshot_count works for both flat and old formats."""

    def test_flat_format_uses_root_count(self, tmp_path: Path) -> None:
        """Flat format: _snapshot_count reads root-level count attr."""
        from pbc_datagen.io import _snapshot_count

        path = tmp_path / "test.h5"
        with h5py.File(path, "w") as f:
            f.attrs["count"] = 7
            assert _snapshot_count(f) == 7

    def test_old_format_group_with_count(self, tmp_path: Path) -> None:
        """Old format with count attr on group."""
        from pbc_datagen.io import _snapshot_count

        path = tmp_path / "test.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("T=2.0000")
            grp.create_dataset("snapshots", shape=(10, 1, 4, 4), dtype=np.int8)
            grp.attrs["count"] = 3
            assert _snapshot_count(f) == 3

    def test_old_format_group_no_count(self, tmp_path: Path) -> None:
        """Old format without count attr falls back to shape[0]."""
        from pbc_datagen.io import _snapshot_count

        path = tmp_path / "test.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("T=2.0000")
            grp.create_dataset("snapshots", shape=(5, 1, 4, 4), dtype=np.int8)
            assert _snapshot_count(f) == 5

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns 0."""
        from pbc_datagen.io import _snapshot_count

        path = tmp_path / "test.h5"
        with h5py.File(path, "w") as f:
            assert _snapshot_count(f) == 0
