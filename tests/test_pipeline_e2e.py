"""Full pipeline integration tests: simulation -> HDF5 -> .pt -> plots.

Covers every valid (model, mode) combination:
  - Single chain: Ising, Blume-Capel, Ashkin-Teller
  - 1D PT: Ising, Blume-Capel, Ashkin-Teller
  - 2D PT: Blume-Capel, Ashkin-Teller (Ising has no param)

Each class runs 6 tests:
  1. test_h5_valid        — flat schema, channels, dtype, finite observables
  2. test_convert_to_pt   — .pt produced, correct record shape/dtype
  3. test_plot_obs_from_h5 — PNG produced from HDF5
  4. test_plot_snapshots_from_h5 — PNG(s) produced from HDF5
  5. test_plot_obs_from_pt — PNG produced from .pt
  6. test_plot_snapshots_from_pt — PNG(s) produced from .pt
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from pbc_datagen.registry import get_model_info  # noqa: E402

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Script import helper
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")


def _import_script(name: str) -> Any:
    """Import a script module from the scripts/ directory."""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PNG_MAGIC = b"\x89PNG"


def _assert_valid_png(path: Path) -> None:
    """Assert the file exists, is non-empty, and starts with PNG magic bytes."""
    assert path.exists(), f"PNG not found: {path}"
    assert path.stat().st_size > 0, f"PNG is empty: {path}"
    with open(path, "rb") as f:
        assert f.read(4) == PNG_MAGIC, f"Not a valid PNG: {path}"


def _assert_valid_pt(
    path: Path,
    C: int,
    L: int,
    dtype: torch.dtype = torch.int8,
) -> None:
    """Assert the .pt file is loadable with correct record structure."""
    assert path.exists(), f".pt not found: {path}"
    records = torch.load(path, weights_only=False)
    assert isinstance(records, list)
    assert len(records) > 0
    sample = records[0]
    assert "state" in sample
    assert "T" in sample
    state = sample["state"]
    assert state.shape == (C, L, L), f"Expected ({C},{L},{L}), got {state.shape}"
    assert state.dtype == dtype, f"Expected {dtype}, got {state.dtype}"


def _run_convert(h5_path: Path, pt_path: Path) -> None:
    """Run convert_to_pt.main() with monkeypatched sys.argv."""
    mod = _import_script("convert_to_pt")
    old_argv = sys.argv
    try:
        sys.argv = ["convert_to_pt", str(h5_path), "--output", str(pt_path)]
        mod.main()
    finally:
        sys.argv = old_argv


def _run_plot_obs(input_path: Path, output_path: Path) -> None:
    """Run plot_obs_vs_T.main() with --no-show."""
    mod = _import_script("plot_obs_vs_T")
    old_argv = sys.argv
    try:
        sys.argv = [
            "plot_obs_vs_T",
            str(input_path),
            "--no-show",
            "-o",
            str(output_path),
        ]
        mod.main()
    finally:
        sys.argv = old_argv
        plt.close("all")


def _run_plot_snapshots(input_path: Path, output_dir: Path) -> None:
    """Run plot_snapshots.main() with --no-show --n 3."""
    mod = _import_script("plot_snapshots")
    old_argv = sys.argv
    try:
        sys.argv = [
            "plot_snapshots",
            str(input_path),
            "--no-show",
            "--n",
            "3",
            "-o",
            str(output_dir),
        ]
        mod.main()
    finally:
        sys.argv = old_argv
        plt.close("all")


def _verify_h5_schema(
    path: Path,
    model_type: str,
    expected_M: int,
    expected_C: int,
) -> None:
    """Verify HDF5 flat schema, channel count, dtype, and finite observables."""
    info = get_model_info(model_type)
    with h5py.File(path, "r") as f:
        # Root attrs
        for attr in ("slot_keys", "obs_names", "count", "seed", "model_type"):
            assert attr in f.attrs, f"missing root attr '{attr}'"

        assert str(f.attrs["model_type"]) == model_type

        slot_keys = json.loads(str(f.attrs["slot_keys"]))
        obs_names = json.loads(str(f.attrs["obs_names"]))
        count = int(f.attrs["count"])

        assert len(slot_keys) == expected_M
        assert count > 0

        # Flat datasets
        assert "snapshots" in f
        snap = f["snapshots"]
        assert snap.shape[0] == expected_M
        assert snap.shape[2] == expected_C
        assert snap.dtype == info.snapshot_dtype

        # Observables finite
        for obs in obs_names:
            assert obs in f, f"missing observable dataset '{obs}'"
            assert np.all(np.isfinite(f[obs][:, :count]))

        # No per-group datasets
        groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        assert groups == [], f"unexpected HDF5 groups: {groups}"


# ---------------------------------------------------------------------------
# Single-Chain Tests
# ---------------------------------------------------------------------------


class TestPipelineSingleChainIsing:
    """Single-chain Ising: simulation -> HDF5 -> .pt -> plots."""

    h5_path: ClassVar[Path]
    pt_path: ClassVar[Path]
    base_dir: ClassVar[Path]

    @pytest.fixture(autouse=True, scope="class")
    def run_simulation(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        """Run a single-chain Ising simulation once for all tests."""
        from pbc_datagen.single_chain import run_single_campaign  # type: ignore[import-untyped]

        base = tmp_path_factory.mktemp("sc_ising")
        h5_path = run_single_campaign(
            model_type="ising",
            L=4,
            param_value=0.0,
            T=3.5,
            n_snapshots=5,
            output_dir=str(base / "sim"),
            force_new=True,
        )
        pt_path = base / "out.pt"

        cls = type(self)
        cls.h5_path = h5_path
        cls.pt_path = pt_path
        cls.base_dir = base

    def test_h5_valid(self) -> None:
        _verify_h5_schema(self.h5_path, "ising", expected_M=1, expected_C=1)

    def test_convert_to_pt(self) -> None:
        _run_convert(self.h5_path, self.pt_path)
        _assert_valid_pt(self.pt_path, C=1, L=4)

    def test_plot_obs_from_h5(self) -> None:
        out = self.base_dir / "obs_h5.png"
        _run_plot_obs(self.h5_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_h5(self) -> None:
        out_dir = self.base_dir / "snaps_h5"
        _run_plot_snapshots(self.h5_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_plot_obs_from_pt(self) -> None:
        out = self.base_dir / "obs_pt.png"
        _run_plot_obs(self.pt_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_pt(self) -> None:
        out_dir = self.base_dir / "snaps_pt"
        _run_plot_snapshots(self.pt_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1


class TestPipelineSingleChainBC:
    """Single-chain Blume-Capel: simulation -> HDF5 -> .pt -> plots."""

    h5_path: ClassVar[Path]
    pt_path: ClassVar[Path]
    base_dir: ClassVar[Path]

    @pytest.fixture(autouse=True, scope="class")
    def run_simulation(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        from pbc_datagen.single_chain import run_single_campaign  # type: ignore[import-untyped]

        base = tmp_path_factory.mktemp("sc_bc")
        h5_path = run_single_campaign(
            model_type="blume_capel",
            L=4,
            param_value=1.5,
            T=3.5,
            n_snapshots=5,
            output_dir=str(base / "sim"),
            force_new=True,
        )
        pt_path = base / "out.pt"

        cls = type(self)
        cls.h5_path = h5_path
        cls.pt_path = pt_path
        cls.base_dir = base

    def test_h5_valid(self) -> None:
        _verify_h5_schema(self.h5_path, "blume_capel", expected_M=1, expected_C=1)

    def test_convert_to_pt(self) -> None:
        _run_convert(self.h5_path, self.pt_path)
        _assert_valid_pt(self.pt_path, C=1, L=4)

    def test_plot_obs_from_h5(self) -> None:
        out = self.base_dir / "obs_h5.png"
        _run_plot_obs(self.h5_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_h5(self) -> None:
        out_dir = self.base_dir / "snaps_h5"
        _run_plot_snapshots(self.h5_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_plot_obs_from_pt(self) -> None:
        out = self.base_dir / "obs_pt.png"
        _run_plot_obs(self.pt_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_pt(self) -> None:
        out_dir = self.base_dir / "snaps_pt"
        _run_plot_snapshots(self.pt_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1


class TestPipelineSingleChainAT:
    """Single-chain Ashkin-Teller: simulation -> HDF5 -> .pt -> plots."""

    h5_path: ClassVar[Path]
    pt_path: ClassVar[Path]
    base_dir: ClassVar[Path]

    @pytest.fixture(autouse=True, scope="class")
    def run_simulation(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        from pbc_datagen.single_chain import run_single_campaign  # type: ignore[import-untyped]

        base = tmp_path_factory.mktemp("sc_at")
        h5_path = run_single_campaign(
            model_type="ashkin_teller",
            L=4,
            param_value=0.5,
            T=3.5,
            n_snapshots=5,
            output_dir=str(base / "sim"),
            force_new=True,
        )
        pt_path = base / "out.pt"

        cls = type(self)
        cls.h5_path = h5_path
        cls.pt_path = pt_path
        cls.base_dir = base

    def test_h5_valid(self) -> None:
        _verify_h5_schema(self.h5_path, "ashkin_teller", expected_M=1, expected_C=2)

    def test_convert_to_pt(self) -> None:
        _run_convert(self.h5_path, self.pt_path)
        _assert_valid_pt(self.pt_path, C=2, L=4)

    def test_plot_obs_from_h5(self) -> None:
        out = self.base_dir / "obs_h5.png"
        _run_plot_obs(self.h5_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_h5(self) -> None:
        out_dir = self.base_dir / "snaps_h5"
        _run_plot_snapshots(self.h5_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_plot_obs_from_pt(self) -> None:
        out = self.base_dir / "obs_pt.png"
        _run_plot_obs(self.pt_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_pt(self) -> None:
        out_dir = self.base_dir / "snaps_pt"
        _run_plot_snapshots(self.pt_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1


# ---------------------------------------------------------------------------
# 1D PT Tests
# ---------------------------------------------------------------------------


class TestPipeline1DPTIsing:
    """1D PT Ising: simulation -> HDF5 -> .pt -> plots."""

    h5_path: ClassVar[Path]
    pt_path: ClassVar[Path]
    base_dir: ClassVar[Path]

    @pytest.fixture(autouse=True, scope="class")
    def run_simulation(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        from pbc_datagen.orchestrator import generate_dataset  # type: ignore[import-untyped]

        base = tmp_path_factory.mktemp("pt1d_ising")
        sim_dir = base / "sim"
        generate_dataset(
            model_type="ising",
            L=4,
            param_values=[0.0],
            T_range=(3.0, 4.0),
            n_replicas=3,
            n_snapshots=5,
            output_dir=str(sim_dir),
            force_new=True,
        )
        h5_files = sorted(sim_dir.glob("ising_*.h5"))
        assert len(h5_files) == 1
        pt_path = base / "out.pt"

        cls = type(self)
        cls.h5_path = h5_files[0]
        cls.pt_path = pt_path
        cls.base_dir = base

    def test_h5_valid(self) -> None:
        _verify_h5_schema(self.h5_path, "ising", expected_M=3, expected_C=1)

    def test_convert_to_pt(self) -> None:
        _run_convert(self.h5_path, self.pt_path)
        _assert_valid_pt(self.pt_path, C=1, L=4)

    def test_plot_obs_from_h5(self) -> None:
        out = self.base_dir / "obs_h5.png"
        _run_plot_obs(self.h5_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_h5(self) -> None:
        out_dir = self.base_dir / "snaps_h5"
        _run_plot_snapshots(self.h5_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_plot_obs_from_pt(self) -> None:
        out = self.base_dir / "obs_pt.png"
        _run_plot_obs(self.pt_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_pt(self) -> None:
        out_dir = self.base_dir / "snaps_pt"
        _run_plot_snapshots(self.pt_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1


class TestPipeline1DPTBC:
    """1D PT Blume-Capel: simulation -> HDF5 -> .pt -> plots."""

    h5_path: ClassVar[Path]
    pt_path: ClassVar[Path]
    base_dir: ClassVar[Path]

    @pytest.fixture(autouse=True, scope="class")
    def run_simulation(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        from pbc_datagen.orchestrator import generate_dataset  # type: ignore[import-untyped]

        base = tmp_path_factory.mktemp("pt1d_bc")
        sim_dir = base / "sim"
        generate_dataset(
            model_type="blume_capel",
            L=4,
            param_values=[1.5],
            T_range=(3.0, 4.0),
            n_replicas=3,
            n_snapshots=5,
            output_dir=str(sim_dir),
            force_new=True,
        )
        h5_files = sorted(sim_dir.glob("blume_capel_*.h5"))
        assert len(h5_files) == 1
        pt_path = base / "out.pt"

        cls = type(self)
        cls.h5_path = h5_files[0]
        cls.pt_path = pt_path
        cls.base_dir = base

    def test_h5_valid(self) -> None:
        _verify_h5_schema(self.h5_path, "blume_capel", expected_M=3, expected_C=1)

    def test_convert_to_pt(self) -> None:
        _run_convert(self.h5_path, self.pt_path)
        _assert_valid_pt(self.pt_path, C=1, L=4)

    def test_plot_obs_from_h5(self) -> None:
        out = self.base_dir / "obs_h5.png"
        _run_plot_obs(self.h5_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_h5(self) -> None:
        out_dir = self.base_dir / "snaps_h5"
        _run_plot_snapshots(self.h5_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_plot_obs_from_pt(self) -> None:
        out = self.base_dir / "obs_pt.png"
        _run_plot_obs(self.pt_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_pt(self) -> None:
        out_dir = self.base_dir / "snaps_pt"
        _run_plot_snapshots(self.pt_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1


class TestPipeline1DPTAT:
    """1D PT Ashkin-Teller: simulation -> HDF5 -> .pt -> plots."""

    h5_path: ClassVar[Path]
    pt_path: ClassVar[Path]
    base_dir: ClassVar[Path]

    @pytest.fixture(autouse=True, scope="class")
    def run_simulation(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        from pbc_datagen.orchestrator import generate_dataset  # type: ignore[import-untyped]

        base = tmp_path_factory.mktemp("pt1d_at")
        sim_dir = base / "sim"
        generate_dataset(
            model_type="ashkin_teller",
            L=4,
            param_values=[0.5],
            T_range=(3.0, 4.0),
            n_replicas=3,
            n_snapshots=5,
            output_dir=str(sim_dir),
            force_new=True,
        )
        h5_files = sorted(sim_dir.glob("ashkin_teller_*.h5"))
        assert len(h5_files) == 1
        pt_path = base / "out.pt"

        cls = type(self)
        cls.h5_path = h5_files[0]
        cls.pt_path = pt_path
        cls.base_dir = base

    def test_h5_valid(self) -> None:
        _verify_h5_schema(self.h5_path, "ashkin_teller", expected_M=3, expected_C=2)

    def test_convert_to_pt(self) -> None:
        _run_convert(self.h5_path, self.pt_path)
        _assert_valid_pt(self.pt_path, C=2, L=4)

    def test_plot_obs_from_h5(self) -> None:
        out = self.base_dir / "obs_h5.png"
        _run_plot_obs(self.h5_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_h5(self) -> None:
        out_dir = self.base_dir / "snaps_h5"
        _run_plot_snapshots(self.h5_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_plot_obs_from_pt(self) -> None:
        out = self.base_dir / "obs_pt.png"
        _run_plot_obs(self.pt_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_pt(self) -> None:
        out_dir = self.base_dir / "snaps_pt"
        _run_plot_snapshots(self.pt_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1


# ---------------------------------------------------------------------------
# 2D PT Tests (BC and AT only — Ising has no param)
# ---------------------------------------------------------------------------


class TestPipeline2DPTBC:
    """2D PT Blume-Capel: simulation -> HDF5 -> .pt -> plots (incl. heatmaps)."""

    h5_path: ClassVar[Path]
    pt_path: ClassVar[Path]
    base_dir: ClassVar[Path]
    n_slots: ClassVar[int]

    @pytest.fixture(autouse=True, scope="class")
    def run_simulation(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        from pbc_datagen.pt_engine_2d import PTEngine2D  # type: ignore[import-untyped]

        base = tmp_path_factory.mktemp("pt2d_bc")
        n_T, n_P = 3, 2
        ts = int(time.time() * 1000)
        seed = ts % (2**63)
        h5_path = base / f"blume_capel_2d_{ts}.h5"

        engine = PTEngine2D(
            model_type="blume_capel",
            L=4,
            T_range=(3.5, 4.5),
            param_range=(1.4, 1.6),
            n_T=n_T,
            n_P=n_P,
            seed=seed,
        )
        engine.check_connectivity(n_rounds=200, min_gap=0.05)
        engine.equilibrate()
        engine.produce(h5_path, n_snapshots=5)

        pt_path = base / "out.pt"

        cls = type(self)
        cls.h5_path = h5_path
        cls.pt_path = pt_path
        cls.base_dir = base
        cls.n_slots = n_T * n_P

    def test_h5_valid(self) -> None:
        _verify_h5_schema(
            self.h5_path,
            "blume_capel",
            expected_M=self.n_slots,
            expected_C=1,
        )
        # 2D-specific attrs
        with h5py.File(self.h5_path, "r") as f:
            assert str(f.attrs.get("pt_mode", "")) == "2d"
            assert str(f.attrs.get("param_label", "")) == "D"

    def test_convert_to_pt(self) -> None:
        _run_convert(self.h5_path, self.pt_path)
        _assert_valid_pt(self.pt_path, C=1, L=4)

    def test_plot_obs_from_h5(self) -> None:
        out = self.base_dir / "obs_h5.png"
        _run_plot_obs(self.h5_path, out)
        _assert_valid_png(out)
        # 2D should also produce heatmap PNGs
        heatmaps = list(self.base_dir.glob("*_heatmap_*.png"))
        assert len(heatmaps) >= 1, "Expected heatmap PNGs for 2D PT"

    def test_plot_snapshots_from_h5(self) -> None:
        out_dir = self.base_dir / "snaps_h5"
        _run_plot_snapshots(self.h5_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_plot_obs_from_pt(self) -> None:
        out = self.base_dir / "obs_pt.png"
        _run_plot_obs(self.pt_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_pt(self) -> None:
        out_dir = self.base_dir / "snaps_pt"
        _run_plot_snapshots(self.pt_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1


class TestPipeline2DPTAT:
    """2D PT Ashkin-Teller: simulation -> HDF5 -> .pt -> plots (incl. heatmaps)."""

    h5_path: ClassVar[Path]
    pt_path: ClassVar[Path]
    base_dir: ClassVar[Path]
    n_slots: ClassVar[int]

    @pytest.fixture(autouse=True, scope="class")
    def run_simulation(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        from pbc_datagen.pt_engine_2d import PTEngine2D  # type: ignore[import-untyped]

        base = tmp_path_factory.mktemp("pt2d_at")
        n_T, n_P = 3, 2
        ts = int(time.time() * 1000)
        seed = ts % (2**63)
        h5_path = base / f"ashkin_teller_2d_{ts}.h5"

        engine = PTEngine2D(
            model_type="ashkin_teller",
            L=4,
            T_range=(3.5, 4.5),
            param_range=(0.4, 0.6),
            n_T=n_T,
            n_P=n_P,
            seed=seed,
        )
        engine.check_connectivity(n_rounds=200, min_gap=0.05)
        engine.equilibrate()
        engine.produce(h5_path, n_snapshots=5)

        pt_path = base / "out.pt"

        cls = type(self)
        cls.h5_path = h5_path
        cls.pt_path = pt_path
        cls.base_dir = base
        cls.n_slots = n_T * n_P

    def test_h5_valid(self) -> None:
        _verify_h5_schema(
            self.h5_path,
            "ashkin_teller",
            expected_M=self.n_slots,
            expected_C=2,
        )
        # 2D-specific attrs
        with h5py.File(self.h5_path, "r") as f:
            assert str(f.attrs.get("pt_mode", "")) == "2d"
            assert str(f.attrs.get("param_label", "")) == "U"

    def test_convert_to_pt(self) -> None:
        _run_convert(self.h5_path, self.pt_path)
        _assert_valid_pt(self.pt_path, C=2, L=4)

    def test_plot_obs_from_h5(self) -> None:
        out = self.base_dir / "obs_h5.png"
        _run_plot_obs(self.h5_path, out)
        _assert_valid_png(out)
        # 2D should also produce heatmap PNGs
        heatmaps = list(self.base_dir.glob("*_heatmap_*.png"))
        assert len(heatmaps) >= 1, "Expected heatmap PNGs for 2D PT"

    def test_plot_obs_from_pt(self) -> None:
        out = self.base_dir / "obs_pt.png"
        _run_plot_obs(self.pt_path, out)
        _assert_valid_png(out)

    def test_plot_snapshots_from_h5(self) -> None:
        out_dir = self.base_dir / "snaps_h5"
        _run_plot_snapshots(self.h5_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_plot_snapshots_from_pt(self) -> None:
        out_dir = self.base_dir / "snaps_pt"
        _run_plot_snapshots(self.pt_path, out_dir)
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) >= 1
