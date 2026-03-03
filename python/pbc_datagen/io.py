# Disk I/O for snapshots (HDF5).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
from loguru import logger


def _t_group_name(T: float) -> str:
    """Canonical HDF5 group name for a temperature slot."""
    return f"T={T:.4f}"


def _slot_group_name_2d(T: float, param: float, param_label: str) -> str:
    """Canonical HDF5 group name for a 2D PT slot."""
    return f"T={T:.4f}_{param_label}={param:.4f}"


class SnapshotWriter:
    """Streaming HDF5 writer for PT snapshot data.

    Opens (or creates) an HDF5 file at *path*.  Temperature slots are
    created once, then snapshots are appended one at a time with an
    automatic ``flush()`` after each write.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._file: h5py.File = h5py.File(self._path, "a")
        logger.debug("Opened HDF5 file: {}", self._path.name)

    def create_temperature_slot(self, T: float, L: int, C: int, obs_names: list[str]) -> None:
        """Create a temperature group with resizable datasets.

        Datasets created:
        - ``snapshots``: shape ``(0, C, L, L)``, int8, resizable along axis 0
        - One ``(0,)`` float64 dataset per name in *obs_names*
        """
        name = _t_group_name(T)
        logger.debug("Creating T slot: {} (C={}, L={})", name, C, L)
        grp = self._file.create_group(name)
        grp.create_dataset(
            "snapshots",
            shape=(0, C, L, L),
            maxshape=(None, C, L, L),
            dtype=np.int8,
            chunks=(1, C, L, L),
        )
        for name in obs_names:
            grp.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=np.float64,
                chunks=(1,),
            )

    def create_slot_2d(
        self,
        T: float,
        param: float,
        param_label: str,
        L: int,
        C: int,
        obs_names: list[str],
    ) -> None:
        """Create a 2D PT slot group with resizable datasets."""
        name = _slot_group_name_2d(T, param, param_label)
        logger.debug("Creating 2D slot: {} (C={}, L={})", name, C, L)
        grp = self._file.create_group(name)
        grp.create_dataset(
            "snapshots",
            shape=(0, C, L, L),
            maxshape=(None, C, L, L),
            dtype=np.int8,
            chunks=(1, C, L, L),
        )
        for obs_name in obs_names:
            grp.create_dataset(
                obs_name,
                shape=(0,),
                maxshape=(None,),
                dtype=np.float64,
                chunks=(1,),
            )

    def append_snapshot_2d(
        self,
        T: float,
        param: float,
        param_label: str,
        spins: npt.NDArray[np.int8],
        obs_dict: dict[str, float],
    ) -> None:
        """Append one snapshot to a 2D slot, then flush."""
        key = _slot_group_name_2d(T, param, param_label)
        grp = self._file[key]

        ds = grp["snapshots"]
        n = ds.shape[0]
        ds.resize(n + 1, axis=0)
        ds[n] = spins

        for name, value in obs_dict.items():
            obs_ds = grp[name]
            m = obs_ds.shape[0]
            obs_ds.resize(m + 1, axis=0)
            obs_ds[m] = value

        self._file.flush()

    def append_snapshot(
        self,
        T: float,
        spins: npt.NDArray[np.int8],
        obs_dict: dict[str, float],
    ) -> None:
        """Append one snapshot + observable values, then flush."""
        grp = self._file[_t_group_name(T)]

        # Grow snapshots dataset by 1 along axis 0
        ds = grp["snapshots"]
        n = ds.shape[0]
        ds.resize(n + 1, axis=0)
        ds[n] = spins

        # Grow each observable dataset
        for name, value in obs_dict.items():
            obs_ds = grp[name]
            m = obs_ds.shape[0]
            obs_ds.resize(m + 1, axis=0)
            obs_ds[m] = value

        self._file.flush()

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        self._file.close()

    def __enter__(self) -> SnapshotWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def write_param_attrs(
    path: str | Path,
    *,
    model_type: str,
    L: int,
    param_value: float,
    T_ladder: npt.NDArray[np.float64],
    tau_max: float,
    r2t: list[int],
    t2r: list[int],
    seed: int,
    seed_history: list[tuple[int, int]],
) -> None:
    """Write campaign metadata as HDF5 root-level attributes."""
    with h5py.File(path, "a") as f:
        f.attrs["model_type"] = model_type
        f.attrs["L"] = L
        f.attrs["param_value"] = param_value
        f.attrs["T_ladder"] = np.asarray(T_ladder, dtype=np.float64)
        f.attrs["tau_max"] = tau_max
        f.attrs["r2t"] = np.asarray(r2t, dtype=np.int64)
        f.attrs["t2r"] = np.asarray(t2r, dtype=np.int64)
        f.attrs["seed"] = seed
        # seed_history is list[tuple[int,int]] — store as JSON string
        f.attrs["seed_history"] = json.dumps(seed_history)


def read_resume_state(path: str | Path) -> tuple[int, dict[str, Any]]:
    """Load campaign state from an HDF5 file for resumption.

    Returns ``(seed, state)`` where *seed* is the last-used PRNG seed
    and *state* is a dict with keys: ``model_type``, ``L``,
    ``param_value``, ``T_ladder``, ``tau_max``, ``r2t``, ``t2r``,
    ``seed_history``, ``snapshot_counts``.
    """
    with h5py.File(path, "r") as f:
        seed = int(f.attrs["seed"])
        model_type = str(f.attrs["model_type"])
        L = int(f.attrs["L"])
        param_value = float(f.attrs["param_value"])
        T_ladder = np.array(f.attrs["T_ladder"], dtype=np.float64)
        tau_max = float(f.attrs["tau_max"])
        r2t = np.array(f.attrs["r2t"], dtype=np.int64)
        t2r = np.array(f.attrs["t2r"], dtype=np.int64)
        seed_history_raw = json.loads(str(f.attrs["seed_history"]))
        seed_history: list[tuple[int, int]] = [
            (int(pair[0]), int(pair[1])) for pair in seed_history_raw
        ]

        # Count snapshots per temperature slot
        snapshot_counts: dict[str, int] = {}
        for key in f.keys():
            grp = f[key]
            if "snapshots" in grp:
                snapshot_counts[key] = grp["snapshots"].shape[0]

    state: dict[str, Any] = {
        "model_type": model_type,
        "L": L,
        "param_value": param_value,
        "T_ladder": T_ladder,
        "tau_max": tau_max,
        "r2t": r2t,
        "t2r": t2r,
        "seed_history": seed_history,
        "snapshot_counts": snapshot_counts,
    }

    total = sum(snapshot_counts.values())
    n_slots = len(snapshot_counts)
    logger.debug(
        "Resume state loaded: {} T slots, {} total snapshots, tau_max={:.1f}",
        n_slots,
        total,
        tau_max,
    )

    return seed, state


def write_param_attrs_2d(
    path: str | Path,
    *,
    model_type: str,
    L: int,
    param_label: str,
    temps: npt.NDArray[np.float64],
    params: npt.NDArray[np.float64],
    tau_max: float,
    r2s: list[int],
    s2r: list[int],
    seed: int,
    seed_history: list[tuple[int, int]],
) -> None:
    """Write 2D campaign metadata as HDF5 root-level attributes."""
    with h5py.File(path, "a") as f:
        f.attrs["model_type"] = model_type
        f.attrs["L"] = L
        f.attrs["param_label"] = param_label
        f.attrs["temps"] = np.asarray(temps, dtype=np.float64)
        f.attrs["params"] = np.asarray(params, dtype=np.float64)
        f.attrs["tau_max"] = tau_max
        f.attrs["r2s"] = np.asarray(r2s, dtype=np.int64)
        f.attrs["s2r"] = np.asarray(s2r, dtype=np.int64)
        f.attrs["seed"] = seed
        f.attrs["seed_history"] = json.dumps(seed_history)
        f.attrs["pt_mode"] = "2d"


def read_resume_state_2d(path: str | Path) -> tuple[int, dict[str, Any]]:
    """Load 2D campaign state from an HDF5 file for resumption.

    Returns ``(seed, state)`` where *state* has keys: ``model_type``,
    ``L``, ``param_label``, ``temps``, ``params``, ``tau_max``,
    ``r2s``, ``s2r``, ``seed_history``, ``snapshot_counts``.
    """
    with h5py.File(path, "r") as f:
        seed = int(f.attrs["seed"])
        model_type = str(f.attrs["model_type"])
        L = int(f.attrs["L"])
        param_label = str(f.attrs["param_label"])
        temps = np.array(f.attrs["temps"], dtype=np.float64)
        params = np.array(f.attrs["params"], dtype=np.float64)
        tau_max = float(f.attrs["tau_max"])
        r2s = list(np.array(f.attrs["r2s"], dtype=np.int64))
        s2r = list(np.array(f.attrs["s2r"], dtype=np.int64))
        seed_history_raw = json.loads(str(f.attrs["seed_history"]))
        seed_history: list[tuple[int, int]] = [
            (int(pair[0]), int(pair[1])) for pair in seed_history_raw
        ]

        snapshot_counts: dict[str, int] = {}
        for key in f.keys():
            grp = f[key]
            if "snapshots" in grp:
                snapshot_counts[key] = grp["snapshots"].shape[0]

    state: dict[str, Any] = {
        "model_type": model_type,
        "L": L,
        "param_label": param_label,
        "temps": temps,
        "params": params,
        "tau_max": tau_max,
        "r2s": r2s,
        "s2r": s2r,
        "seed_history": seed_history,
        "snapshot_counts": snapshot_counts,
    }

    total = sum(snapshot_counts.values())
    n_slots = len(snapshot_counts)
    logger.debug(
        "2D resume state loaded: {} slots, {} total snapshots, tau_max={:.1f}",
        n_slots,
        total,
        tau_max,
    )

    return seed, state
