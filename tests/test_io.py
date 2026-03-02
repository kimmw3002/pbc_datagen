"""Tests for HDF5 snapshot I/O — SnapshotWriter, attrs, resume."""

from __future__ import annotations

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


def _random_spins(L: int, C: int = 1) -> np.ndarray:
    """Random ±1 spin array shaped (C, L, L), dtype int8."""
    rng = np.random.default_rng(42)
    return rng.choice([-1, 1], size=(C, L, L)).astype(np.int8)


# ---------------------------------------------------------------------------
# SnapshotWriter — group hierarchy & dataset properties
# ---------------------------------------------------------------------------


class TestSnapshotWriterCreation:
    """Verify that create_temperature_slot builds the correct HDF5 layout."""

    def test_create_slot_group_hierarchy(self, tmp_path: Path) -> None:
        """Creating a T slot produces a group with 'snapshots' + one dataset per observable."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        obs_names = _ising_obs_names()

        with SnapshotWriter(path) as w:
            w.create_temperature_slot(T=2.269, L=4, C=1, obs_names=obs_names)

        with h5py.File(path, "r") as f:
            # Exactly one group should exist
            groups = list(f.keys())
            assert len(groups) == 1

            grp = f[groups[0]]
            assert "snapshots" in grp
            for name in obs_names:
                assert name in grp, f"missing observable dataset '{name}'"

    def test_create_slot_datasets_empty_and_resizable(self, tmp_path: Path) -> None:
        """Datasets start empty (axis-0 = 0) with unlimited maxshape on axis 0."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        L, C = 4, 1
        obs_names = _ising_obs_names()

        with SnapshotWriter(path) as w:
            w.create_temperature_slot(T=2.0, L=L, C=C, obs_names=obs_names)

        with h5py.File(path, "r") as f:
            grp = f[list(f.keys())[0]]

            snaps = grp["snapshots"]
            assert snaps.shape == (0, C, L, L)
            assert snaps.maxshape[0] is None  # resizable
            assert snaps.dtype == np.int8

            for name in obs_names:
                ds = grp[name]
                assert ds.shape == (0,)
                assert ds.maxshape[0] is None
                assert ds.dtype == np.float64


# ---------------------------------------------------------------------------
# SnapshotWriter — append_snapshot
# ---------------------------------------------------------------------------


class TestSnapshotWriterAppend:
    """Verify that append_snapshot grows datasets and stores correct data."""

    def test_append_single_snapshot_data_matches(self, tmp_path: Path) -> None:
        """Appending one snapshot gives shape (1, C, L, L) with exact data match."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        L, C = 4, 1
        obs_names = _ising_obs_names()
        spins = _random_spins(L, C)
        obs = {"energy": -12.0, "m": 0.5, "abs_m": 0.5}

        with SnapshotWriter(path) as w:
            w.create_temperature_slot(T=2.269, L=L, C=C, obs_names=obs_names)
            w.append_snapshot(T=2.269, spins=spins, obs_dict=obs)

        with h5py.File(path, "r") as f:
            grp = f[list(f.keys())[0]]
            snaps = grp["snapshots"]

            assert snaps.shape == (1, C, L, L)
            np.testing.assert_array_equal(snaps[0], spins)

    def test_append_multiple_grows_axis_zero(self, tmp_path: Path) -> None:
        """Three sequential appends produce shape (3, C, L, L), each slice correct."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        L, C = 4, 1
        obs_names = _ising_obs_names()

        rng = np.random.default_rng(99)
        snapshots = [rng.choice([-1, 1], size=(C, L, L)).astype(np.int8) for _ in range(3)]
        obs_vals = [{"energy": float(-8 + 2 * i), "m": 0.1 * i, "abs_m": 0.1 * i} for i in range(3)]

        with SnapshotWriter(path) as w:
            w.create_temperature_slot(T=2.0, L=L, C=C, obs_names=obs_names)
            for sp, ob in zip(snapshots, obs_vals):
                w.append_snapshot(T=2.0, spins=sp, obs_dict=ob)

        with h5py.File(path, "r") as f:
            grp = f[list(f.keys())[0]]
            snaps = grp["snapshots"]

            assert snaps.shape == (3, C, L, L)
            for i, expected in enumerate(snapshots):
                np.testing.assert_array_equal(snaps[i], expected)

    def test_append_stores_observable_values(self, tmp_path: Path) -> None:
        """Observable datasets grow in sync with snapshots, values are exact."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        L, C = 4, 1
        obs_names = _ising_obs_names()

        obs_a = {"energy": -16.0, "m": 1.0, "abs_m": 1.0}
        obs_b = {"energy": -8.0, "m": 0.0, "abs_m": 0.5}

        with SnapshotWriter(path) as w:
            w.create_temperature_slot(T=2.0, L=L, C=C, obs_names=obs_names)
            w.append_snapshot(T=2.0, spins=_random_spins(L, C), obs_dict=obs_a)
            w.append_snapshot(T=2.0, spins=_random_spins(L, C), obs_dict=obs_b)

        with h5py.File(path, "r") as f:
            grp = f[list(f.keys())[0]]

            assert grp["energy"].shape == (2,)
            assert grp["energy"][0] == pytest.approx(-16.0)
            assert grp["energy"][1] == pytest.approx(-8.0)

            assert grp["m"][0] == pytest.approx(1.0)
            assert grp["m"][1] == pytest.approx(0.0)

    def test_two_channel_snapshot_ashkin_teller(self, tmp_path: Path) -> None:
        """C=2 (Ashkin-Teller) snapshots have shape (N, 2, L, L)."""
        from pbc_datagen.io import SnapshotWriter

        path = tmp_path / "test.h5"
        L, C = 4, 2
        obs_names = [
            "energy",
            "m_sigma",
            "abs_m_sigma",
            "m_tau",
            "abs_m_tau",
            "m_baxter",
            "abs_m_baxter",
        ]
        spins = _random_spins(L, C)  # shape (2, 4, 4)
        obs = {name: float(i) for i, name in enumerate(obs_names)}

        with SnapshotWriter(path) as w:
            w.create_temperature_slot(T=3.0, L=L, C=C, obs_names=obs_names)
            w.append_snapshot(T=3.0, spins=spins, obs_dict=obs)

        with h5py.File(path, "r") as f:
            grp = f[list(f.keys())[0]]
            snaps = grp["snapshots"]

            assert snaps.shape == (1, 2, L, L)
            np.testing.assert_array_equal(snaps[0], spins)

            # All 7 observable datasets should exist
            for name in obs_names:
                assert name in grp
                assert grp[name].shape == (1,)


# ---------------------------------------------------------------------------
# write_param_attrs — metadata round-trip
# ---------------------------------------------------------------------------


class TestWriteParamAttrs:
    """Verify campaign metadata survives an HDF5 attrs round-trip."""

    def test_roundtrip_all_fields(self, tmp_path: Path) -> None:
        """Every field written by write_param_attrs is recoverable from HDF5 attrs."""
        from pbc_datagen.io import SnapshotWriter, write_param_attrs

        path = tmp_path / "test.h5"
        T_ladder = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
        r2t = [0, 1, 2, 3, 4]
        t2r = [0, 1, 2, 3, 4]
        seed = 123456789
        seed_history: list[tuple[int, int]] = [(0, 123456789)]

        # Create a minimal valid HDF5 file first
        with SnapshotWriter(path) as w:
            w.create_temperature_slot(T=2.0, L=4, C=1, obs_names=_ising_obs_names())

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
        """Calling write_param_attrs again overwrites r2t/t2r with latest values.

        During Phase C, replicas exchange temperatures every round.  The
        address maps (r2t, t2r) must be updated on disk so that a crash-
        resume restores the correct replica↔temperature assignment.
        """
        from pbc_datagen.io import SnapshotWriter, write_param_attrs

        path = tmp_path / "test.h5"
        T_ladder = np.array([1.0, 2.0, 3.0])

        with SnapshotWriter(path) as w:
            w.create_temperature_slot(T=2.0, L=4, C=1, obs_names=_ising_obs_names())

        # First write: identity map
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

        # Second write: replicas 0 and 2 swapped
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
        """Helper: create an HDF5 with uniform snapshot counts. Returns seed.

        PT production harvests one snapshot from *every* T slot per cycle,
        so all slots always have the same count.
        """
        from pbc_datagen.io import SnapshotWriter, write_param_attrs

        seed = 42
        T_ladder = np.array([1.5, 2.0, 2.5])
        obs_names = _ising_obs_names()
        L, C = 4, 1

        with SnapshotWriter(path) as w:
            for T in T_ladder:
                w.create_temperature_slot(T=T, L=L, C=C, obs_names=obs_names)

            # Uniform: n_snapshots per every T slot
            dummy_obs = {"energy": -8.0, "m": 0.5, "abs_m": 0.5}
            for _ in range(n_snapshots):
                for T in T_ladder:
                    w.append_snapshot(T=T, spins=_random_spins(L, C), obs_dict=dummy_obs)

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
        """Resume state has correct seed and equal counts across all T slots.

        PT production harvests all T slots in lockstep, so every slot
        must have the same snapshot count.
        """
        from pbc_datagen.io import read_resume_state

        path = tmp_path / "test.h5"
        expected_seed = self._make_populated_file(path, n_snapshots=7)

        seed, state = read_resume_state(path)

        assert seed == expected_seed
        counts = state["snapshot_counts"]
        assert len(counts) == 3  # one entry per T slot
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
        """read_resume_state returns the most recently saved r2t/t2r.

        During production, replicas swap temperatures each round.  If we
        crash and resume, we must restore the exact replica↔temperature
        assignment from the last write_param_attrs call.
        """
        from pbc_datagen.io import read_resume_state

        path = tmp_path / "test.h5"
        self._make_populated_file(path, r2t=[2, 1, 0], t2r=[2, 1, 0])

        _seed, state = read_resume_state(path)

        np.testing.assert_array_equal(state["r2t"], [2, 1, 0])
        np.testing.assert_array_equal(state["t2r"], [2, 1, 0])

    def test_resume_returns_seed_history(self, tmp_path: Path) -> None:
        """seed_history must survive the round-trip so the orchestrator can extend it.

        On resume the orchestrator appends (n_existing, new_seed) to the
        history.  If seed_history is lost, the replay audit trail breaks.
        """
        from pbc_datagen.io import read_resume_state

        path = tmp_path / "test.h5"
        history: list[tuple[int, int]] = [(0, 42), (50, 99999)]
        self._make_populated_file(path, seed_history=history)

        _seed, state = read_resume_state(path)

        recovered = state["seed_history"]
        assert len(recovered) == 2
        assert recovered[0] == (0, 42)
        assert recovered[1] == (50, 99999)
