#!/usr/bin/env python
"""Plot ⟨O⟩ vs T curves from an HDF5 dataset file."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_temperature(group_name: str) -> float:
    """Extract temperature value from group name like 'T=0.5'."""
    m = re.match(r"T=([\d.eE+\-]+)", group_name)
    if m is None:
        raise ValueError(f"Cannot parse temperature from group name: {group_name}")
    return float(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ⟨O⟩ vs T from HDF5 dataset")
    parser.add_argument("hdf5", type=Path, help="Path to HDF5 file")
    parser.add_argument("--obs", nargs="+", help="Observable names to plot (default: all)")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path (default: auto)")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show()")
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        model_type = f.attrs["model_type"]
        L = f.attrs["L"]
        param_value = f.attrs["param_value"]

        # Collect temperature groups sorted by T
        t_groups = sorted(
            [(parse_temperature(name), name) for name in f if name.startswith("T=")],
            key=lambda x: x[0],
        )

        # Auto-detect observable names from first group
        first_group = f[t_groups[0][1]]
        all_obs = [k for k in first_group if k != "snapshots"]
        obs_names = args.obs if args.obs else all_obs

        # Collect data: {obs_name: (T_array, mean_array, stderr_array)}
        temps = np.array([t for t, _ in t_groups])
        data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in obs_names:
            means = []
            stderrs = []
            for _, gname in t_groups:
                vals = f[gname][name][:]
                means.append(np.mean(vals))
                stderrs.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            data[name] = (np.array(means), np.array(stderrs))

    # Plot: subplot grid with ceil(n/2) rows × 2 cols
    n_obs = len(obs_names)
    ncols = 2
    nrows = math.ceil(n_obs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False)

    for idx, name in enumerate(obs_names):
        ax = axes[idx // ncols, idx % ncols]
        mean, stderr = data[name]
        ax.errorbar(temps, mean, yerr=stderr, fmt="o-", capsize=3, markersize=4)
        ax.set_xlabel("T")
        ax.set_ylabel(f"⟨{name}⟩")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_obs, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"{model_type}  L={L}  param={param_value}", fontsize=14)
    fig.tight_layout()

    out_path = args.output or Path(f"images/{args.hdf5.stem}_obs_vs_T.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
