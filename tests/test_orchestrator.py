"""Tests for orchestrator — file discovery, campaign execution, resume logic."""

from __future__ import annotations

from pathlib import Path

import h5py

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch_hdf5(path: Path) -> None:
    """Create a minimal valid HDF5 file (empty, but parseable)."""
    with h5py.File(path, "w"):
        pass


def _make_campaign_file(directory: Path, name: str, n_snapshots: int = 3) -> Path:
    """Create a small but real HDF5 campaign file via run_campaign.

    Uses 4×4 Ising, 3 replicas — fast enough for tests.
    """
    from pbc_datagen.orchestrator import run_campaign

    return run_campaign(
        model_type="ising",
        L=4,
        param_value=0.0,
        T_range=(2.0, 2.1),
        n_replicas=3,
        n_snapshots=n_snapshots,
        output_dir=directory,
    )


# ---------------------------------------------------------------------------
# find_existing_hdf5 — file discovery
# ---------------------------------------------------------------------------


class TestFindExistingHdf5:
    """Verify that find_existing_hdf5 locates the correct campaign file."""

    def test_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        """No matching files in the directory → returns None."""
        from pbc_datagen.orchestrator import find_existing_hdf5

        result = find_existing_hdf5(
            tmp_path, "ising", L=4, param_value=0.0, T_range=(2.0, 2.1), n_replicas=3
        )
        assert result is None

    def test_returns_newest_matching_file(self, tmp_path: Path) -> None:
        """Two files with different timestamps → returns the one with
        the larger (newer) timestamp suffix.

        Ising filename: ``ising_L4_T=2.0000-2.1000_R3_{timestamp}.h5``.
        """
        from pbc_datagen.orchestrator import find_existing_hdf5

        old = tmp_path / "ising_L4_T=2.0000-2.1000_R3_1000000000000.h5"
        new = tmp_path / "ising_L4_T=2.0000-2.1000_R3_2000000000000.h5"
        _touch_hdf5(old)
        _touch_hdf5(new)

        result = find_existing_hdf5(
            tmp_path, "ising", L=4, param_value=0.0, T_range=(2.0, 2.1), n_replicas=3
        )
        assert result is not None
        assert result.name == new.name

    def test_ignores_wrong_model_type(self, tmp_path: Path) -> None:
        """Files for a different model are not matched.

        A blume_capel file should not be returned when searching for ising.
        """
        from pbc_datagen.orchestrator import find_existing_hdf5

        wrong = tmp_path / "blume_capel_L4_D=0.0000_T=2.0000-2.1000_R3_9999999999999.h5"
        _touch_hdf5(wrong)

        result = find_existing_hdf5(
            tmp_path, "ising", L=4, param_value=0.0, T_range=(2.0, 2.1), n_replicas=3
        )
        assert result is None

    def test_ignores_wrong_lattice_size(self, tmp_path: Path) -> None:
        """Files for a different L are not matched.

        An L=8 file should not be returned when searching for L=4.
        """
        from pbc_datagen.orchestrator import find_existing_hdf5

        wrong = tmp_path / "ising_L8_T=2.0000-2.1000_R3_9999999999999.h5"
        _touch_hdf5(wrong)

        result = find_existing_hdf5(
            tmp_path, "ising", L=4, param_value=0.0, T_range=(2.0, 2.1), n_replicas=3
        )
        assert result is None

    def test_ignores_different_T_range(self, tmp_path: Path) -> None:
        """A file with a different T-range should not match."""
        from pbc_datagen.orchestrator import find_existing_hdf5

        wrong = tmp_path / "ising_L4_T=1.0000-3.0000_R3_9999999999999.h5"
        _touch_hdf5(wrong)

        result = find_existing_hdf5(
            tmp_path, "ising", L=4, param_value=0.0, T_range=(2.0, 2.1), n_replicas=3
        )
        assert result is None

    def test_ignores_different_n_replicas(self, tmp_path: Path) -> None:
        """A file with a different replica count should not match."""
        from pbc_datagen.orchestrator import find_existing_hdf5

        wrong = tmp_path / "ising_L4_T=2.0000-2.1000_R10_9999999999999.h5"
        _touch_hdf5(wrong)

        result = find_existing_hdf5(
            tmp_path, "ising", L=4, param_value=0.0, T_range=(2.0, 2.1), n_replicas=3
        )
        assert result is None


# ---------------------------------------------------------------------------
# _derive_seed — deterministic seed derivation for resume
# ---------------------------------------------------------------------------


class TestDeriveSeed:
    """Verify deterministic seed derivation for resume."""

    def test_deterministic(self) -> None:
        """Same inputs always produce the same output."""
        from pbc_datagen.orchestrator import _derive_seed

        a = _derive_seed(42, 10)
        b = _derive_seed(42, 10)
        assert a == b

    def test_differs_with_offset(self) -> None:
        """Different n_existing values produce different seeds.

        This ensures each resume point gets a unique PRNG stream,
        even if the original seed is the same.
        """
        from pbc_datagen.orchestrator import _derive_seed

        a = _derive_seed(42, 10)
        b = _derive_seed(42, 20)
        assert a != b


# ---------------------------------------------------------------------------
# run_campaign — single-param pipeline (A→B→C)
# ---------------------------------------------------------------------------


class TestRunCampaign:
    """Verify run_campaign executes the full PT pipeline and produces output."""

    def test_fresh_creates_hdf5_with_snapshots(self, tmp_path: Path) -> None:
        """A fresh campaign creates an HDF5 file with the requested
        number of snapshots in every T slot.
        """
        from pbc_datagen.orchestrator import run_campaign

        result_path = run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=3,
            output_dir=tmp_path,
        )

        assert result_path.exists()
        with h5py.File(result_path, "r") as f:
            # Flat schema: count=3, snapshots shape (M, 3, C, L, L)
            assert int(f.attrs["count"]) == 3
            assert f["snapshots"].shape[0] == 3  # M = n_replicas
            assert f["snapshots"].shape[1] == 3  # n_snapshots

    def test_output_filename_matches_convention(self, tmp_path: Path) -> None:
        """Ising output file encodes T-range and replica count.

        Format: ``ising_L4_T=2.0000-2.1000_R3_{ts}.h5``.
        No ``J=`` appears because Ising has no tunable parameter.
        """
        from pbc_datagen.orchestrator import run_campaign

        result_path = run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=3,
            output_dir=tmp_path,
        )

        name = result_path.name
        assert name.startswith("ising_L4_T=2.0000-2.1000_R3_")
        assert "J=" not in name
        assert name.endswith(".h5")

    def test_force_new_creates_separate_file(self, tmp_path: Path) -> None:
        """With force_new=True, a second campaign creates a NEW file
        even when a matching file already exists — does not resume.
        """
        from pbc_datagen.orchestrator import run_campaign

        path1 = run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=3,
            output_dir=tmp_path,
        )
        path2 = run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=3,
            output_dir=tmp_path,
            force_new=True,
        )

        # Two distinct files should exist
        assert path1 != path2
        assert path1.exists()
        assert path2.exists()


# ---------------------------------------------------------------------------
# run_campaign — resume from existing HDF5
# ---------------------------------------------------------------------------


class TestRunCampaignResume:
    """Verify run_campaign resume: find existing file, derive seed, extend history."""

    def test_resume_reuses_existing_file(self, tmp_path: Path) -> None:
        """A second run_campaign with higher n_snapshots resumes the SAME
        file instead of creating a new one.

        The orchestrator's default (force_new=False) should find the
        first file via find_existing_hdf5 and append to it.
        """
        from pbc_datagen.orchestrator import run_campaign

        path1 = run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=3,
            output_dir=tmp_path,
        )
        path2 = run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=5,
            output_dir=tmp_path,
        )

        assert path1 == path2  # same file, not a new one

    def test_resume_appends_to_target(self, tmp_path: Path) -> None:
        """After resume, the file has the new target count (5), not
        the old count (3) and not the sum (8).
        """
        from pbc_datagen.orchestrator import run_campaign

        run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=3,
            output_dir=tmp_path,
        )
        path = run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=5,
            output_dir=tmp_path,
        )

        with h5py.File(path, "r") as f:
            assert int(f.attrs["count"]) == 5

    def test_resume_extends_seed_history(self, tmp_path: Path) -> None:
        """After resume, seed_history has two entries: the original
        (0, initial_seed) and a new (n_existing, derived_seed).

        This audit trail records which PRNG stream produced which
        snapshots, enabling full replay.
        """
        import json

        from pbc_datagen.orchestrator import run_campaign

        run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=3,
            output_dir=tmp_path,
        )
        path = run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=5,
            output_dir=tmp_path,
        )

        with h5py.File(path, "r") as f:
            history = json.loads(str(f.attrs["seed_history"]))
            assert len(history) == 2
            # First entry starts at snapshot 0
            assert history[0][0] == 0
            # Second entry starts at snapshot 3 (where the first run stopped)
            assert history[1][0] == 3

    def test_resume_uses_derived_seed(self, tmp_path: Path) -> None:
        """The resumed campaign uses a DIFFERENT seed than the original.

        If the same seed were reused, the resumed snapshots would repeat
        the same PRNG stream from the start, breaking independence.
        """
        import json

        from pbc_datagen.orchestrator import run_campaign

        run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=3,
            output_dir=tmp_path,
        )
        path = run_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T_range=(2.0, 2.1),
            n_replicas=3,
            n_snapshots=5,
            output_dir=tmp_path,
        )

        with h5py.File(path, "r") as f:
            history = json.loads(str(f.attrs["seed_history"]))
            original_seed = history[0][1]
            resume_seed = history[1][1]
            assert original_seed != resume_seed
