"""Tests for single-chain MCMC runner — SingleChainEngine + campaign logic."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Step 2.5.1: SingleChainEngine.__init__ + model creation
# ---------------------------------------------------------------------------


class TestSingleChainInit:
    """Verify constructor creates a model with correct state."""

    def test_ising_model_created(self) -> None:
        """Constructor creates an Ising model with correct L and T."""
        from pbc_datagen.single_chain import SingleChainEngine

        engine = SingleChainEngine(model_type="ising", L=4, param_value=0.0, T=2.269, seed=42)
        assert engine.L == 4
        assert engine.T == 2.269
        assert engine.model_type == "ising"
        assert engine.tau_max is None

    def test_blume_capel_model_created(self) -> None:
        """Constructor creates a Blume-Capel model with correct param."""
        from pbc_datagen.single_chain import SingleChainEngine

        engine = SingleChainEngine(
            model_type="blume_capel", L=4, param_value=1.965, T=0.609, seed=42
        )
        assert engine.model_type == "blume_capel"
        assert engine.param_value == 1.965

    def test_ashkin_teller_model_created(self) -> None:
        """Constructor creates an Ashkin-Teller model with correct param."""
        from pbc_datagen.single_chain import SingleChainEngine

        engine = SingleChainEngine(model_type="ashkin_teller", L=4, param_value=0.5, T=1.0, seed=42)
        assert engine.model_type == "ashkin_teller"

    def test_unknown_model_raises(self) -> None:
        """Unknown model type raises ValueError."""
        import pytest
        from pbc_datagen.single_chain import SingleChainEngine

        with pytest.raises(ValueError, match="Unknown model"):
            SingleChainEngine(model_type="xyz", L=4, param_value=0.0, T=1.0, seed=42)


# ---------------------------------------------------------------------------
# Step 2.5.2: SingleChainEngine.equilibrate()
# ---------------------------------------------------------------------------


class TestEquilibrate:
    """Verify equilibrate sets tau_max via doubling Welch + tau_int."""

    def test_equilibrate_sets_positive_tau_max(self) -> None:
        """After equilibrate(), tau_max is a positive float."""
        from pbc_datagen.single_chain import SingleChainEngine

        engine = SingleChainEngine(model_type="ising", L=4, param_value=0.0, T=2.269, seed=42)
        engine.equilibrate()
        assert engine.tau_max is not None
        assert engine.tau_max > 0.0

    def test_produce_before_equilibrate_raises(self) -> None:
        """Calling produce() before equilibrate() raises RuntimeError."""
        import pytest
        from pbc_datagen.single_chain import SingleChainEngine

        engine = SingleChainEngine(model_type="ising", L=4, param_value=0.0, T=2.269, seed=42)
        with pytest.raises(RuntimeError, match="tau_max not set"):
            engine.produce(path="/dev/null", n_snapshots=1)


# ---------------------------------------------------------------------------
# Step 2.5.3: SingleChainEngine.produce()
# ---------------------------------------------------------------------------


class TestProduce:
    """Verify produce() writes correct HDF5 structure."""

    def test_hdf5_structure_after_produce(self, tmp_path: Path) -> None:
        """After equilibrate + produce(n=5), HDF5 has flat datasets with
        correct snapshot shape, observable datasets, and root attrs.
        """
        from pbc_datagen.single_chain import SingleChainEngine

        path = tmp_path / "test.h5"
        engine = SingleChainEngine(model_type="ising", L=4, param_value=0.0, T=2.269, seed=42)
        engine.equilibrate()
        engine.produce(path=path, n_snapshots=5)

        with h5py.File(path, "r") as f:
            # Root attrs
            assert str(f.attrs["model_type"]) == "ising"
            assert int(f.attrs["L"]) == 4
            assert float(f.attrs["tau_max"]) > 0
            assert int(f.attrs["seed"]) == 42
            assert int(f.attrs["count"]) == 5

            from pbc_datagen.registry import get_model_info

            # Flat snapshot dataset: (1, 5, 1, 4, 4) int8
            assert f["snapshots"].shape == (1, 5, 1, 4, 4)
            assert f["snapshots"].dtype == get_model_info("ising").snapshot_dtype

            # Observable datasets: shape (1, 5) each
            for obs_name in ("energy", "m", "abs_m"):
                assert f[obs_name].shape == (1, 5)

    def test_blume_capel_produce(self, tmp_path: Path) -> None:
        """Blume-Capel produce creates correct flat structure with 4 observables."""
        from pbc_datagen.single_chain import SingleChainEngine

        path = tmp_path / "bc_test.h5"
        engine = SingleChainEngine(
            model_type="blume_capel", L=4, param_value=1.965, T=0.609, seed=42
        )
        engine.equilibrate()
        engine.produce(path=path, n_snapshots=3)

        with h5py.File(path, "r") as f:
            assert int(f.attrs["count"]) == 3
            assert f["snapshots"].shape == (1, 3, 1, 4, 4)
            # BC has 4 observables: energy, m, abs_m, q
            for obs_name in ("energy", "m", "abs_m", "q"):
                assert f[obs_name].shape == (1, 3)

    def test_ashkin_teller_two_channels(self, tmp_path: Path) -> None:
        """Ashkin-Teller produce creates snapshots with C=2 channels."""
        from pbc_datagen.single_chain import SingleChainEngine

        path = tmp_path / "at_test.h5"
        engine = SingleChainEngine(model_type="ashkin_teller", L=4, param_value=0.5, T=1.0, seed=42)
        engine.equilibrate()
        engine.produce(path=path, n_snapshots=3)

        with h5py.File(path, "r") as f:
            # AT has C=2: (sigma, tau) → shape (1, 3, 2, 4, 4)
            assert f["snapshots"].shape == (1, 3, 2, 4, 4)

    def test_t_ladder_is_single_element(self, tmp_path: Path) -> None:
        """T_ladder attr is a 1-element array containing the target T."""
        from pbc_datagen.single_chain import SingleChainEngine

        path = tmp_path / "test.h5"
        engine = SingleChainEngine(model_type="ising", L=4, param_value=0.0, T=2.269, seed=42)
        engine.equilibrate()
        engine.produce(path=path, n_snapshots=2)

        with h5py.File(path, "r") as f:
            T_ladder = np.array(f.attrs["T_ladder"])
            assert len(T_ladder) == 1
            assert abs(T_ladder[0] - 2.269) < 1e-10


# ---------------------------------------------------------------------------
# Step 2.5.4: find_existing_single_hdf5 + run_single_campaign
# ---------------------------------------------------------------------------


def _touch_hdf5(path: Path) -> None:
    """Create a minimal valid HDF5 file."""
    with h5py.File(path, "w"):
        pass


class TestFindExistingSingleHdf5:
    """Verify file discovery for single-chain HDF5 files."""

    def test_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        from pbc_datagen.single_chain import find_existing_single_hdf5

        result = find_existing_single_hdf5(tmp_path, "ising", L=4, param_value=0.0, T=2.269)
        assert result is None

    def test_returns_newest_match(self, tmp_path: Path) -> None:
        """Two matching files → returns the one with larger timestamp."""
        from pbc_datagen.single_chain import find_existing_single_hdf5

        old = tmp_path / "ising_L4_T=2.2690_1000000000000.h5"
        new = tmp_path / "ising_L4_T=2.2690_2000000000000.h5"
        _touch_hdf5(old)
        _touch_hdf5(new)

        result = find_existing_single_hdf5(tmp_path, "ising", L=4, param_value=0.0, T=2.269)
        assert result is not None
        assert result.name == new.name

    def test_ignores_wrong_model(self, tmp_path: Path) -> None:
        from pbc_datagen.single_chain import find_existing_single_hdf5

        wrong = tmp_path / "blume_capel_L4_D=0.0000_T=2.2690_9999999999999.h5"
        _touch_hdf5(wrong)

        result = find_existing_single_hdf5(tmp_path, "ising", L=4, param_value=0.0, T=2.269)
        assert result is None

    def test_ignores_wrong_L(self, tmp_path: Path) -> None:
        from pbc_datagen.single_chain import find_existing_single_hdf5

        wrong = tmp_path / "ising_L8_T=2.2690_9999999999999.h5"
        _touch_hdf5(wrong)

        result = find_existing_single_hdf5(tmp_path, "ising", L=4, param_value=0.0, T=2.269)
        assert result is None

    def test_ignores_wrong_T(self, tmp_path: Path) -> None:
        from pbc_datagen.single_chain import find_existing_single_hdf5

        wrong = tmp_path / "ising_L4_T=3.0000_9999999999999.h5"
        _touch_hdf5(wrong)

        result = find_existing_single_hdf5(tmp_path, "ising", L=4, param_value=0.0, T=2.269)
        assert result is None

    def test_blume_capel_pattern_includes_param(self, tmp_path: Path) -> None:
        """Blume-Capel glob includes D= label and matches correctly."""
        from pbc_datagen.single_chain import find_existing_single_hdf5

        good = tmp_path / "blume_capel_L4_D=1.9650_T=0.6090_1000000000000.h5"
        _touch_hdf5(good)

        result = find_existing_single_hdf5(tmp_path, "blume_capel", L=4, param_value=1.965, T=0.609)
        assert result is not None
        assert result.name == good.name


class TestRunSingleCampaign:
    """Verify run_single_campaign creates correct HDF5 output."""

    def test_fresh_creates_hdf5(self, tmp_path: Path) -> None:
        """A fresh single-chain campaign creates HDF5 with correct layout."""
        from pbc_datagen.single_chain import run_single_campaign

        path = run_single_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T=2.269,
            n_snapshots=3,
            output_dir=tmp_path,
        )

        assert path.exists()
        with h5py.File(path, "r") as f:
            assert int(f.attrs["count"]) == 3
            assert f["snapshots"].shape[0] == 1  # M=1 for single chain

    def test_filename_convention_ising(self, tmp_path: Path) -> None:
        """Ising filename: ising_L4_T=2.2690_{ts}.h5 — no param label."""
        from pbc_datagen.single_chain import run_single_campaign

        path = run_single_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T=2.269,
            n_snapshots=2,
            output_dir=tmp_path,
        )

        name = path.name
        assert name.startswith("ising_L4_T=2.2690_")
        assert name.endswith(".h5")
        # No param label or replica count in the name
        assert "_R" not in name
        assert "J=" not in name

    def test_filename_convention_blume_capel(self, tmp_path: Path) -> None:
        """Blume-Capel filename includes D= label."""
        from pbc_datagen.single_chain import run_single_campaign

        path = run_single_campaign(
            model_type="blume_capel",
            L=4,
            param_value=1.965,
            T=0.609,
            n_snapshots=2,
            output_dir=tmp_path,
        )

        name = path.name
        assert name.startswith("blume_capel_L4_D=1.9650_T=0.6090_")
        assert name.endswith(".h5")

    def test_resume_reuses_file(self, tmp_path: Path) -> None:
        """A second run with higher n_snapshots resumes the same file."""
        from pbc_datagen.single_chain import run_single_campaign

        path1 = run_single_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T=2.269,
            n_snapshots=3,
            output_dir=tmp_path,
        )
        path2 = run_single_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T=2.269,
            n_snapshots=5,
            output_dir=tmp_path,
        )

        assert path1 == path2

    def test_resume_appends_to_target(self, tmp_path: Path) -> None:
        """After resume, file has the new target count, not old or sum."""
        from pbc_datagen.single_chain import run_single_campaign

        run_single_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T=2.269,
            n_snapshots=3,
            output_dir=tmp_path,
        )
        path = run_single_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T=2.269,
            n_snapshots=5,
            output_dir=tmp_path,
        )

        with h5py.File(path, "r") as f:
            assert int(f.attrs["count"]) == 5

    def test_resume_extends_seed_history(self, tmp_path: Path) -> None:
        """After resume, seed_history has two entries."""
        from pbc_datagen.single_chain import run_single_campaign

        run_single_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T=2.269,
            n_snapshots=3,
            output_dir=tmp_path,
        )
        path = run_single_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T=2.269,
            n_snapshots=5,
            output_dir=tmp_path,
        )

        with h5py.File(path, "r") as f:
            history = json.loads(str(f.attrs["seed_history"]))
            assert len(history) == 2
            assert history[0][0] == 0
            assert history[1][0] == 3

    def test_force_new_creates_separate_file(self, tmp_path: Path) -> None:
        """With force_new=True, a second run creates a new file."""
        from pbc_datagen.single_chain import run_single_campaign

        path1 = run_single_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T=2.269,
            n_snapshots=2,
            output_dir=tmp_path,
        )
        path2 = run_single_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T=2.269,
            n_snapshots=2,
            output_dir=tmp_path,
            force_new=True,
        )

        assert path1 != path2
        assert path1.exists()
        assert path2.exists()
