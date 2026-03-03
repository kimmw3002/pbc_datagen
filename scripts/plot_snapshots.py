#!/usr/bin/env python
"""Plot random snapshot samples from each temperature in an HDF5 dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def parse_temperature(group_name: str) -> float:
    """Extract temperature value from group name like 'T=0.5'."""
    m = re.match(r"T=([\d.eE+\-]+)", group_name)
    if m is None:
        raise ValueError(f"Cannot parse temperature from group name: {group_name}")
    return float(m.group(1))


def parse_slot_2d(group_name: str, param_label: str) -> tuple[float, float]:
    """Extract (T, param) from a 2D group name like 'T=0.5_U=1.2'."""
    m = re.match(rf"T=([\d.eE+\-]+)_{re.escape(param_label)}=([\d.eE+\-]+)", group_name)
    if m is None:
        raise ValueError(f"Cannot parse 2D slot from group name: {group_name}")
    return float(m.group(1)), float(m.group(2))


# Discrete colormap for Ising/Blume-Capel: -1=blue, 0=white, +1=red
CMAP_3STATE = ListedColormap(["#2166ac", "#f7f7f7", "#b2182b"])
NORM_3STATE = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], CMAP_3STATE.N)

# Discrete colormap for Ashkin-Teller layers: -1=blue, +1=red
CMAP_2STATE = ListedColormap(["#2166ac", "#b2182b"])
NORM_2STATE = BoundaryNorm([-1.5, 0, 1.5], CMAP_2STATE.N)


def plot_single_channel(axes: np.ndarray, snapshots: np.ndarray, indices: list[int]) -> None:
    """Plot C=1 snapshots (Ising/BC) as a single row of images."""
    for col, idx in enumerate(indices):
        ax = axes[col] if axes.ndim == 1 else axes[0, col]
        ax.imshow(snapshots[idx, 0], cmap=CMAP_3STATE, norm=NORM_3STATE, interpolation="nearest")
        ax.set_title(f"#{idx}", fontsize=8)
        ax.axis("off")


def plot_two_channel(axes: np.ndarray, snapshots: np.ndarray, indices: list[int]) -> None:
    """Plot C=2 snapshots (AT) as three rows: sigma, tau, sigma*tau."""
    for col, idx in enumerate(indices):
        ax_s = axes[0, col]
        ax_t = axes[1, col]
        ax_st = axes[2, col]
        sigma = snapshots[idx, 0]
        tau = snapshots[idx, 1]
        ax_s.imshow(sigma, cmap=CMAP_2STATE, norm=NORM_2STATE, interpolation="nearest")
        ax_t.imshow(tau, cmap=CMAP_2STATE, norm=NORM_2STATE, interpolation="nearest")
        ax_st.imshow(sigma * tau, cmap=CMAP_2STATE, norm=NORM_2STATE, interpolation="nearest")
        ax_s.set_title(f"#{idx}", fontsize=8)
        ax_s.axis("off")
        ax_t.axis("off")
        ax_st.axis("off")
        if col == 0:
            ax_s.set_ylabel("σ", fontsize=10)
            ax_t.set_ylabel("τ", fontsize=10)
            ax_st.set_ylabel("στ", fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot snapshot samples from HDF5 dataset")
    parser.add_argument("hdf5", type=Path, help="Path to HDF5 file")
    parser.add_argument("--n", type=int, default=10, help="Snapshots per temperature (default: 10)")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("images"), help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show()")
    parser.add_argument("--T", nargs="*", type=float, help="Temperature values to include")
    parser.add_argument("--param", nargs="*", type=float, help="Parameter values to include")
    parser.add_argument(
        "--list", action="store_true", help="List available T/param values and exit"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.hdf5, "r") as f:
        model_type = f.attrs["model_type"]
        L = f.attrs["L"]
        is_2d = str(f.attrs.get("pt_mode", "")) == "2d"
        param_label = str(f.attrs["param_label"]) if is_2d else None

        if is_2d:
            assert param_label is not None
            slots = sorted(
                [(*parse_slot_2d(name, param_label), name) for name in f if name.startswith("T=")],
                key=lambda x: (x[1], x[0]),  # param, then T
            )
            slot_iter = [(T_val, pv, gname) for T_val, pv, gname in slots]
        else:
            param_value = f.attrs["param_value"]
            t_groups = sorted(
                [(parse_temperature(name), name) for name in f if name.startswith("T=")],
                key=lambda x: x[0],
            )
            slot_iter = [(T_val, param_value, gname) for T_val, gname in t_groups]

        # --list: print available values and exit
        if args.list:
            temps = sorted({T_val for T_val, _, _ in slot_iter})
            params = sorted({pv for _, pv, _ in slot_iter})
            print(f"T values ({len(temps)}):  {', '.join(f'{t:.4f}' for t in temps)}")
            if is_2d:
                assert param_label is not None
                print(
                    f"{param_label} values ({len(params)}):  "
                    f"{', '.join(f'{p:.4f}' for p in params)}"
                )
            else:
                print(f"param value: {params[0]}")
            return

        # Apply optional filters (group names are already 4 d.p.)
        if args.T is not None:
            t_set = set(args.T)
            slot_iter = [(T_val, pv, gname) for T_val, pv, gname in slot_iter if T_val in t_set]
        if args.param is not None:
            p_set = set(args.param)
            slot_iter = [(T_val, pv, gname) for T_val, pv, gname in slot_iter if pv in p_set]

        for T_val, pv, gname in slot_iter:
            snaps = f[gname]["snapshots"]
            N, C, _, _ = snaps.shape
            if N == 0:
                continue
            n_pick = min(args.n, N)
            indices = sorted(rng.choice(N, size=n_pick, replace=False).tolist())

            snap_data = snaps[:]

            if C == 1:
                ncols = n_pick
                fig, axes = plt.subplots(1, ncols, figsize=(1.5 * ncols, 2), squeeze=False)
                plot_single_channel(axes[0], snap_data, indices)
            else:
                ncols = n_pick
                fig, axes = plt.subplots(3, ncols, figsize=(1.5 * ncols, 4.5), squeeze=False)
                plot_two_channel(axes, snap_data, indices)

            if is_2d:
                suptitle = f"{model_type} L={L}  T={T_val:.4f}  {param_label}={pv:.4f}"
            else:
                suptitle = f"{model_type} L={L} param={pv}  T={T_val:.4f}"
            fig.suptitle(suptitle, fontsize=11)
            fig.tight_layout()

            if is_2d:
                out_name = f"{args.hdf5.stem}_T={T_val:.4f}_{param_label}={pv:.4f}.png"
            else:
                out_name = f"{args.hdf5.stem}_T={T_val:.4f}.png"
            out_path = args.output_dir / out_name
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {out_path}")
            plt.close(fig)

    if not args.no_show:
        # Re-show isn't practical after close; just inform user
        print(f"\nAll snapshots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
