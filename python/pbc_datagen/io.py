# Disk I/O for snapshots (HDF5).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
from loguru import logger


def _snapshot_count(f: h5py.File) -> int:
    """Read actual snapshot count, backward-compatible.

    New flat format stores ``count`` as a root attribute.
    Old per-group format: fall back to the first group's ``snapshots`` shape[0].
    """
    if "count" in f.attrs:
        return int(f.attrs["count"])
    # Old-format fallback (per-group)
    for key in f.keys():
        obj = f[key]
        if isinstance(obj, h5py.Group) and "snapshots" in obj:
            if "count" in obj.attrs:
                return int(obj.attrs["count"])
            return int(obj["snapshots"].shape[0])
    return 0


_ATTR_ELEM_LIMIT = 512  # arrays larger than this become root-level datasets


def _t_group_name(T: float) -> str:
    """Canonical HDF5 group name for a temperature slot."""
    return f"T={T:.4f}"


def _slot_group_name_2d(T: float, param: float, param_label: str) -> str:
    """Canonical HDF5 group name for a 2D PT slot."""
    return f"T={T:.4f}_{param_label}={param:.4f}"


class SnapshotWriter:
    """Streaming HDF5 writer for PT snapshot data — flat schema.

    All snapshots are stored in root-level datasets:
    - ``snapshots``: shape ``(M, n_snapshots, C, L, L)``, int8
    - One ``(M, n_snapshots)`` float64 dataset per observable

    ``M`` is the number of slots (temperature or 2D grid points).
    ``count`` (root attr) tracks the actual number of written rounds.

    The caller is responsible for calling ``flush()`` at appropriate
    intervals (e.g. once per production round) to persist buffered
    writes to disk.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._file: h5py.File = h5py.File(self._path, "a")
        self._count: int = 0
        self._snap_ds: h5py.Dataset | None = None
        self._obs_ds: dict[str, h5py.Dataset] = {}
        logger.debug("Opened HDF5 file: {}", self._path.name)

    def create_datasets(
        self,
        slot_keys: list[str],
        n_snapshots: int,
        C: int,
        L: int,
        obs_names: list[str],
    ) -> None:
        """Fresh run: pre-allocate flat datasets.

        Creates root-level datasets and stores slot_keys / obs_names
        as JSON attributes for later discovery.
        """
        M = len(slot_keys)
        self._file.attrs["slot_keys"] = json.dumps(slot_keys)
        self._file.attrs["obs_names"] = json.dumps(obs_names)
        self._file.attrs["count"] = 0

        self._snap_ds = self._file.create_dataset(
            "snapshots",
            shape=(M, n_snapshots, C, L, L),
            maxshape=(M, None, C, L, L),
            dtype=np.int8,
            chunks=(1, 1, C, L, L),
        )
        for name in obs_names:
            self._obs_ds[name] = self._file.create_dataset(
                name,
                shape=(M, n_snapshots),
                maxshape=(M, None),
                dtype=np.float64,
                chunks=(1, 1),
            )
        self._count = 0
        logger.debug(
            "Created flat datasets: M={}, n_snap={}, C={}, L={}, obs={}",
            M,
            n_snapshots,
            C,
            L,
            obs_names,
        )

    def open_datasets(self) -> None:
        """Resume: cache dataset refs and load count."""
        self._snap_ds = self._file["snapshots"]
        obs_names = json.loads(str(self._file.attrs["obs_names"]))
        for name in obs_names:
            self._obs_ds[name] = self._file[name]
        self._count = int(self._file.attrs["count"])
        logger.debug("Opened existing datasets, count={}", self._count)

    def write_round(
        self,
        all_spins: npt.NDArray[np.int8],
        obs_arrays: dict[str, npt.NDArray[np.float64]],
    ) -> None:
        """Batch-write one round.

        Args:
            all_spins: Shape ``(M, C, L, L)`` — one snapshot per slot.
            obs_arrays: ``{name: (M,)}`` — one value per slot per observable.

        If the pre-allocated size is exhausted, datasets are extended by
        doubling (on the snapshot axis) to amortise resize overhead.
        """
        assert self._snap_ds is not None
        n = self._count
        # Auto-extend if pre-allocated size is exhausted
        if n >= self._snap_ds.shape[1]:
            new_size = max(n + 1, self._snap_ds.shape[1] * 2)
            self._snap_ds.resize(new_size, axis=1)
            for ds in self._obs_ds.values():
                ds.resize(new_size, axis=1)
        self._snap_ds[:, n] = all_spins  # 1 h5py call
        for name, vals in obs_arrays.items():
            self._obs_ds[name][:, n] = vals  # 1 call per obs
        self._count = n + 1

    @property
    def snapshot_count(self) -> int:
        """Return the number of actually written snapshot rounds."""
        return self._count

    def write_metadata(self, attrs: dict[str, object]) -> None:
        """Write arbitrary key-value pairs to root-level HDF5 attributes.

        Called alongside ``flush()`` so that every flushed round carries
        up-to-date campaign metadata (model_type, address maps, seed, etc.).
        A crash between flushes loses at most one round of snapshots *and*
        the metadata stays consistent with the last flushed count.

        Values that are numpy arrays are written directly; lists of ints
        are converted to int64 arrays; everything else is written as-is.

        Arrays with more than ``_ATTR_ELEM_LIMIT`` elements are stored as
        root-level datasets instead of attributes to avoid the HDF5 ~64 KB
        per-attribute object-header limit.
        """
        for key, value in attrs.items():
            arr = np.asarray(value) if isinstance(value, (list, np.ndarray)) else None
            if arr is not None and arr.ndim >= 1 and arr.size > _ATTR_ELEM_LIMIT:
                # Too large for an HDF5 attribute — store as a root-level dataset.
                if key in self._file:
                    self._file[key][()] = arr  # overwrite in-place
                else:
                    self._file.create_dataset(key, data=arr)
            else:
                self._file.attrs[key] = value

    def flush(self) -> None:
        """Persist in-memory count to HDF5 attr, then flush to disk."""
        self._file.attrs["count"] = self._count
        self._file.flush()

    def close(self) -> None:
        """Persist count and close the underlying HDF5 file."""
        self._file.attrs["count"] = self._count
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
        if "slot_keys" in f.attrs:
            # New flat format
            slot_keys = json.loads(str(f.attrs["slot_keys"]))
            count = int(f.attrs["count"])
            snapshot_counts = {k: count for k in slot_keys}
        else:
            # Old per-group fallback
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Group) and "snapshots" in obj:
                    if "count" in obj.attrs:
                        snapshot_counts[key] = int(obj.attrs["count"])
                    else:
                        snapshot_counts[key] = int(obj["snapshots"].shape[0])

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
        r2s_arr = np.asarray(r2s, dtype=np.int64)
        s2r_arr = np.asarray(s2r, dtype=np.int64)
        for key, arr in [("r2s", r2s_arr), ("s2r", s2r_arr)]:
            if key in f:
                f[key][()] = arr
            else:
                f.create_dataset(key, data=arr)
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
        # Support both storage forms: dataset (large grids) and attribute (small grids).
        r2s = list(np.array(f["r2s"] if "r2s" in f else f.attrs["r2s"], dtype=np.int64))
        s2r = list(np.array(f["s2r"] if "s2r" in f else f.attrs["s2r"], dtype=np.int64))
        seed_history_raw = json.loads(str(f.attrs["seed_history"]))
        seed_history: list[tuple[int, int]] = [
            (int(pair[0]), int(pair[1])) for pair in seed_history_raw
        ]

        snapshot_counts: dict[str, int] = {}
        if "slot_keys" in f.attrs:
            # New flat format
            slot_keys = json.loads(str(f.attrs["slot_keys"]))
            count = int(f.attrs["count"])
            snapshot_counts = {k: count for k in slot_keys}
        else:
            # Old per-group fallback
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Group) and "snapshots" in obj:
                    if "count" in obj.attrs:
                        snapshot_counts[key] = int(obj.attrs["count"])
                    else:
                        snapshot_counts[key] = int(obj["snapshots"].shape[0])

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
