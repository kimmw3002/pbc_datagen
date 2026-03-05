"""End-to-end tests: single-chain, 1D PT, 2D PT with Ashkin-Teller.

Runs small campaigns, verifies HDF5 flat schema, then resumes and
verifies appended snapshots.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _verify_flat_schema(
    path: Path,
    expected_M: int,
    expected_count: int,
) -> None:
    """Assert HDF5 file has correct flat schema."""
    with h5py.File(path, "r") as f:
        # Root attrs
        required = (
            "slot_keys",
            "obs_names",
            "count",
            "seed",
            "seed_history",
            "tau_max",
            "model_type",
        )
        for attr in required:
            assert attr in f.attrs, f"missing root attr '{attr}'"

        slot_keys = json.loads(str(f.attrs["slot_keys"]))
        obs_names = json.loads(str(f.attrs["obs_names"]))
        count = int(f.attrs["count"])

        assert str(f.attrs["model_type"]) == "ashkin_teller"
        assert len(slot_keys) == expected_M
        assert count == expected_count

        # Flat datasets
        assert "snapshots" in f
        snap = f["snapshots"]
        assert snap.shape[0] == expected_M
        assert snap.shape[2] == 2  # C=2 for AT
        assert snap.dtype == np.int8

        for obs in obs_names:
            assert obs in f, f"missing observable dataset '{obs}'"
            assert f[obs].shape[0] == expected_M

        # Data integrity
        data = snap[:, :count]
        assert set(np.unique(data).tolist()) <= {-1, 1}
        for obs in obs_names:
            assert np.all(np.isfinite(f[obs][:, :count]))

        # No per-group datasets
        groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        assert groups == [], f"unexpected HDF5 groups: {groups}"

        sh = json.loads(str(f.attrs["seed_history"]))
        assert len(sh) >= 1


# ---------------------------------------------------------------------------
# Single Chain
# ---------------------------------------------------------------------------


class TestE2ESingleChain:
    """Single-chain MCMC: fresh run + resume."""

    def test_fresh_and_resume(self, tmp_path: Path) -> None:
        from pbc_datagen.single_chain import run_single_campaign

        out = tmp_path / "single"

        # Fresh: 5 snapshots
        path = run_single_campaign(
            model_type="ashkin_teller",
            L=4,
            param_value=0.5,
            T=3.0,
            n_snapshots=5,
            output_dir=str(out),
            force_new=True,
        )
        _verify_flat_schema(path, expected_M=1, expected_count=5)

        # Resume: extend to 10
        path2 = run_single_campaign(
            model_type="ashkin_teller",
            L=4,
            param_value=0.5,
            T=3.0,
            n_snapshots=10,
            output_dir=str(out),
        )
        assert path2 == path, "resume should reuse the same file"
        _verify_flat_schema(path, expected_M=1, expected_count=10)

        with h5py.File(path, "r") as f:
            sh = json.loads(str(f.attrs["seed_history"]))
            assert len(sh) == 2


# ---------------------------------------------------------------------------
# 1D PT
# ---------------------------------------------------------------------------


class TestE2E1DPT:
    """1D parallel tempering: fresh run + resume."""

    def test_fresh_and_resume(self, tmp_path: Path) -> None:
        from pbc_datagen.orchestrator import generate_dataset

        out = tmp_path / "pt1d"

        # Fresh: 5 snapshots
        generate_dataset(
            model_type="ashkin_teller",
            L=4,
            param_values=[0.5],
            T_range=(2.5, 4.0),
            n_replicas=4,
            n_snapshots=5,
            output_dir=str(out),
            force_new=True,
        )

        h5_files = sorted(out.glob("ashkin_teller_*.h5"))
        assert len(h5_files) == 1
        path = h5_files[0]
        _verify_flat_schema(path, expected_M=4, expected_count=5)

        # 1D-specific attrs
        with h5py.File(path, "r") as f:
            for attr in ("T_ladder", "r2t", "t2r"):
                assert attr in f.attrs, f"missing 1D attr '{attr}'"
            assert len(np.array(f.attrs["T_ladder"])) == 4

        # Resume: extend to 8
        generate_dataset(
            model_type="ashkin_teller",
            L=4,
            param_values=[0.5],
            T_range=(2.5, 4.0),
            n_replicas=4,
            n_snapshots=8,
            output_dir=str(out),
        )
        _verify_flat_schema(path, expected_M=4, expected_count=8)

        with h5py.File(path, "r") as f:
            sh = json.loads(str(f.attrs["seed_history"]))
            assert len(sh) == 2


# ---------------------------------------------------------------------------
# 2D PT
# ---------------------------------------------------------------------------


class TestE2E2DPT:
    """2D parameter-space parallel tempering: fresh run + resume."""

    def test_fresh_and_resume(self, tmp_path: Path) -> None:
        from pbc_datagen.io import read_resume_state_2d
        from pbc_datagen.orchestrator import _derive_seed
        from pbc_datagen.pt_engine_2d import PTEngine2D

        out = tmp_path / "pt2d"
        out.mkdir(parents=True, exist_ok=True)
        n_T, n_P = 4, 3
        M = n_T * n_P

        ts = int(time.time() * 1000)
        seed = ts % (2**63)
        path = out / f"ashkin_teller_2d_{ts}.h5"

        # Fresh: 3 snapshots
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
        engine.produce(path, n_snapshots=3)

        _verify_flat_schema(path, expected_M=M, expected_count=3)

        # 2D-specific attrs
        with h5py.File(path, "r") as f:
            assert str(f.attrs.get("pt_mode", "")) == "2d"
            for attr in ("temps", "params", "r2s", "s2r", "param_label"):
                assert attr in f.attrs, f"missing 2D attr '{attr}'"
            assert len(np.array(f.attrs["temps"])) == n_T
            assert len(np.array(f.attrs["params"])) == n_P

            slot_keys = json.loads(str(f.attrs["slot_keys"]))
            assert all("U=" in k for k in slot_keys)
            assert all("T=" in k for k in slot_keys)

        # Resume: extend to 6
        old_seed, state = read_resume_state_2d(path)
        counts = state["snapshot_counts"]
        n_existing = min(counts.values()) if counts else 0
        new_seed = _derive_seed(old_seed, n_existing)
        seed_history: list[tuple[int, int]] = state["seed_history"]
        seed_history.append((n_existing, new_seed))

        engine2 = PTEngine2D(
            model_type="ashkin_teller",
            L=4,
            T_range=(3.5, 4.5),
            param_range=(0.4, 0.6),
            n_T=n_T,
            n_P=n_P,
            seed=new_seed,
        )
        engine2.tau_max = state["tau_max"]
        engine2.connectivity_checked = True
        engine2.produce(path, n_snapshots=6, seed_history=seed_history)

        _verify_flat_schema(path, expected_M=M, expected_count=6)

        with h5py.File(path, "r") as f:
            sh = json.loads(str(f.attrs["seed_history"]))
            assert len(sh) == 2
