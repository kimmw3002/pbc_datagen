#!/usr/bin/env python
"""Convert HDF5 snapshot datasets to .pt (PyTorch) format.

Supports two input types:
  1. Single .h5 file  → one .pt file
  2. Directory of .h5 files → one aggregated .pt file

Usage:
    python scripts/convert_to_pt.py INPUT [--output OUTPUT]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch

# Model → Hamiltonian parameter label
_PARAM_LABELS: dict[str, str] = {
    "blume_capel": "D",
    "ashkin_teller": "U",
}


def parse_slot_key(key: str) -> dict[str, float]:
    """Parse a slot key like 'T=0.5000_D=1.9000' into {'T': 0.5, 'D': 1.9}."""
    result: dict[str, float] = {}
    for part in key.split("_"):
        m = re.match(r"([A-Za-z_]+)=([\d.eE+\-]+)", part)
        if m:
            result[m.group(1)] = float(m.group(2))
    return result


def _read_flat_schema(f: h5py.File, model_type: str) -> list[dict[str, object]]:
    """Read HDF5 file with flat schema (slot_keys + root-level datasets)."""
    records: list[dict[str, object]] = []

    count = int(f.attrs.get("count", 0))
    if count == 0:
        return []

    slot_keys: list[str] = json.loads(str(f.attrs["slot_keys"]))
    obs_names: list[str] = json.loads(str(f.attrs["obs_names"]))
    M = len(slot_keys)

    snapshots = f["snapshots"][:, :count]  # (M, count, C, L, L)
    obs_data: dict[str, np.ndarray] = {}
    for name in obs_names:
        obs_data[name] = f[name][:, :count]  # (M, count)

    is_2d = str(f.attrs.get("pt_mode", "")) == "2d"
    param_label = _PARAM_LABELS.get(model_type)

    param_value: float | None = None
    if not is_2d and param_label is not None:
        param_value = float(f.attrs.get("param_value", 0.0))

    for s in range(M):
        parsed = parse_slot_key(slot_keys[s])
        T_val = round(parsed["T"], 4)

        for n in range(count):
            record: dict[str, object] = {
                "state": torch.tensor(snapshots[s, n], dtype=torch.int8),
                "T": T_val,
            }
            for obs_name in obs_names:
                record[obs_name] = float(obs_data[obs_name][s, n])
            if param_label is not None:
                if is_2d:
                    record[param_label] = round(parsed[param_label], 4)
                elif param_value is not None:
                    record[param_label] = round(param_value, 4)
            records.append(record)

    return records


def read_hdf5(path: Path) -> list[dict[str, object]]:
    """Read a single HDF5 file and return a list of record dicts.

    Each record: {"state": Tensor(C,L,L), "T": float, <obs>: float, ...}
    Only the flat schema (slot_keys attr) is supported.
    """
    with h5py.File(path, "r") as f:
        model_type = str(f.attrs["model_type"])
        records = _read_flat_schema(f, model_type)

    if not records:
        warnings.warn(f"Skipping {path.name}: no records found", stacklevel=2)

    return records


def convert_file(path: Path) -> list[dict[str, object]]:
    """Convert a single HDF5 file to a list of record dicts."""
    return read_hdf5(path)


def convert_directory(dir_path: Path) -> list[dict[str, object]]:
    """Convert all HDF5 files in a directory to a single list of record dicts."""
    h5_files = sorted(dir_path.rglob("*.h5"))
    if not h5_files:
        print(f"Error: no .h5 files found in {dir_path}", file=sys.stderr)
        raise SystemExit(1)

    # Validate consistency
    model_types: set[str] = set()
    L_values: set[int] = set()
    for p in h5_files:
        with h5py.File(p, "r") as f:
            model_types.add(str(f.attrs["model_type"]))
            L_values.add(int(f.attrs["L"]))

    if len(model_types) > 1:
        print(f"Error: mixed model types in directory: {model_types}", file=sys.stderr)
        raise SystemExit(1)
    if len(L_values) > 1:
        print(f"Error: mixed L values in directory: {L_values}", file=sys.stderr)
        raise SystemExit(1)

    all_records: list[dict[str, object]] = []
    for p in h5_files:
        recs = read_hdf5(p)
        all_records.extend(recs)
        if recs:
            print(f"  {p.name}: {len(recs)} records")
        else:
            print(f"  {p.name}: skipped (empty)")

    return all_records


def _extract_timestamp(filename: str) -> int:
    """Extract trailing digits from an HDF5 filename as a timestamp."""
    m = re.search(r"(\d+)\.h5$", filename)
    return int(m.group(1)) if m else 0


def _default_output_file(input_path: Path) -> Path:
    """Generate default output .pt filename from input path."""
    if input_path.is_file():
        return input_path.with_suffix(".pt")

    # Directory: construct from aggregated metadata
    h5_files = sorted(input_path.rglob("*.h5"))
    if not h5_files:
        return input_path.parent / f"{input_path.name}.pt"

    with h5py.File(h5_files[0], "r") as f:
        model_type = str(f.attrs["model_type"])
        L = int(f.attrs["L"])

    param_label = _PARAM_LABELS.get(model_type)

    # Collect all T and param values across files
    all_T: set[float] = set()
    all_param: set[float] = set()
    for p in h5_files:
        with h5py.File(p, "r") as f:
            slot_keys = json.loads(str(f.attrs["slot_keys"]))
            is_2d = str(f.attrs.get("pt_mode", "")) == "2d"
            for key in slot_keys:
                parsed = parse_slot_key(key)
                all_T.add(parsed["T"])
                if param_label and param_label in parsed:
                    all_param.add(parsed[param_label])
            # For 1D/single-chain, param from attr
            if not is_2d and param_label:
                pv = float(f.attrs.get("param_value", 0.0))
                all_param.add(pv)

    Tmin, Tmax = min(all_T), max(all_T)
    nT = len(all_T)
    latest_ts = max(_extract_timestamp(p.name) for p in h5_files)

    if model_type == "ising":
        name = f"ising_L{L}_T={Tmin:.4f}-{Tmax:.4f}_{nT}x1_{latest_ts}.pt"
    elif param_label:
        Pmin, Pmax = min(all_param), max(all_param)
        nP = len(all_param)
        name = (
            f"{model_type}_L{L}_T={Tmin:.4f}-{Tmax:.4f}"
            f"_{param_label}={Pmin:.4f}-{Pmax:.4f}_{nT}x{nP}_{latest_ts}.pt"
        )
    else:
        name = f"{model_type}_L{L}_T={Tmin:.4f}-{Tmax:.4f}_{nT}x1_{latest_ts}.pt"

    return input_path.parent / name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HDF5 snapshot datasets to .pt (PyTorch) format"
    )
    parser.add_argument("input", type=Path, help="HDF5 file or directory of HDF5 files")
    parser.add_argument("--output", type=Path, default=None, help="Output .pt path (default: auto)")
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        raise SystemExit(1)

    # Auto-detect input type
    if input_path.is_file():
        print(f"Converting single file: {input_path}")
        records = convert_file(input_path)
    elif input_path.is_dir():
        print(f"Converting directory: {input_path}")
        records = convert_directory(input_path)
    else:
        print(f"Error: {input_path} is neither a file nor a directory", file=sys.stderr)
        raise SystemExit(1)

    if not records:
        print("Error: no records produced", file=sys.stderr)
        raise SystemExit(1)

    # Determine output path
    output_path = args.output or _default_output_file(input_path)
    if output_path.exists():
        print(f"Error: output file already exists: {output_path}", file=sys.stderr)
        raise SystemExit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, output_path)

    # Summary
    sample = records[0]
    state_tensor = sample["state"]
    assert hasattr(state_tensor, "shape") and hasattr(state_tensor, "dtype")
    state_shape = state_tensor.shape
    keys = [k for k in sample if k != "state"]
    print(f"\nSaved {len(records)} records to {output_path}")
    print(f"  state shape: {tuple(state_shape)}, dtype: {state_tensor.dtype}")
    print(f"  keys: state, {', '.join(keys)}")
    print(f"  file size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
