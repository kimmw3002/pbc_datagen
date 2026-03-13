#!/usr/bin/env python
"""Plot random snapshot samples from each temperature in an HDF5 or .pt dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
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


def _plot_pt(args: argparse.Namespace) -> None:
    """Handle .pt input."""
    import torch

    rng = np.random.default_rng(args.seed)
    records = torch.load(args.input, weights_only=False)
    if not records:
        print("Empty .pt file")
        return

    sample = records[0]
    C_val = sample["state"].shape[0]
    L = sample["state"].shape[-1]

    # Infer param
    if "D" in sample:
        param_label: str | None = "D"
    elif "U" in sample:
        param_label = "U"
    else:
        param_label = None

    # Only treat as 2D if there are multiple distinct param values
    if param_label is not None:
        distinct_params = {round(r[param_label], 4) for r in records}
        is_2d = len(distinct_params) > 1
    else:
        is_2d = False
    model_type = args.input.stem

    # Group by (T, param)
    groups: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for rec in records:
        pv = round(rec[param_label], 4) if param_label else 0.0
        groups[(round(rec["T"], 4), pv)].append(rec)

    # --list
    if args.list:
        temps = sorted({t for t, _ in groups})
        params = sorted({p for _, p in groups})
        print(f"T values ({len(temps)}):  {', '.join(f'{t:.4f}' for t in temps)}")
        if is_2d:
            print(f"{param_label} values ({len(params)}):  {', '.join(f'{p:.4f}' for p in params)}")
        return

    # Apply filters
    slot_iter = sorted(groups.keys(), key=lambda x: (x[1], x[0]))
    if args.T is not None:
        t_set = {round(t, 4) for t in args.T}
        slot_iter = [(t, p) for t, p in slot_iter if t in t_set]
    if args.param is not None:
        p_set = {round(p, 4) for p in args.param}
        slot_iter = [(t, p) for t, p in slot_iter if p in p_set]

    for T_val, pv in slot_iter:
        recs = groups[(T_val, pv)]
        N = len(recs)
        if N == 0:
            continue
        n_pick = min(args.n, N)
        indices = sorted(rng.choice(N, size=n_pick, replace=False).tolist())

        # Stack states into (N, C, L, L)
        snap_data = np.stack([recs[i]["state"].numpy() for i in range(N)])

        if C_val == 1:
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
            suptitle = f"{model_type} L={L}  T={T_val:.4f}"
        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout()

        if is_2d:
            out_name = f"{args.input.stem}_T={T_val:.4f}_{param_label}={pv:.4f}.png"
        else:
            out_name = f"{args.input.stem}_T={T_val:.4f}.png"
        out_path = args.output_dir / out_name
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot snapshot samples from HDF5 or .pt dataset")
    parser.add_argument("input", type=Path, help="Path to HDF5 or .pt file")
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
    parser.add_argument(
        "--rigorous",
        action="store_true",
        help="Exclude unconverged slots flagged in HDF5 from plots.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input.suffix == ".pt":
        _plot_pt(args)
        if not args.no_show:
            print(f"\nAll snapshots saved to {args.output_dir}/")
        return

    # --- HDF5 path (original code) ---
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.input, "r") as f:
        model_type = f.attrs["model_type"]
        L = f.attrs["L"]
        is_2d = str(f.attrs.get("pt_mode", "")) == "2d"
        param_label = str(f.attrs["param_label"]) if is_2d else None

        # Only flat schema is supported
        is_flat = "slot_keys" in f.attrs
        if not is_flat:
            raise ValueError(
                f"Unsupported HDF5 format in {args.input}: missing 'slot_keys' attr. "
                "Only the flat schema is supported."
            )
        _ds = f.get("disagreement_slots") or f.attrs.get("disagreement_slots", [])
        _disagree_indices: list[int] = list(np.asarray(_ds, dtype=int).tolist())
        if is_flat and _disagree_indices:
            _all_keys: list[str] = json.loads(str(f.attrs["slot_keys"]))
            disagree_slot_keys: list[str] = [_all_keys[i] for i in _disagree_indices]
        else:
            disagree_slot_keys = []
        if disagree_slot_keys:
            import warnings

            warnings.warn(
                f"HDF5 contains {len(disagree_slot_keys)} unconverged slot(s) "
                f"(Phase B soft failure). Pass --rigorous to exclude them.",
                stacklevel=2,
            )

        if is_flat:
            slot_keys = json.loads(str(f.attrs["slot_keys"]))
            count = int(f.attrs["count"])
            snaps_ds = f["snapshots"]  # (M, n_snap, C, L, L)

            if is_2d:
                assert param_label is not None
                slot_iter_raw = []
                for i, key in enumerate(slot_keys):
                    T_val, pv = parse_slot_2d(key, param_label)
                    slot_iter_raw.append((round(T_val, 4), round(pv, 4), i))
                slot_iter_raw.sort(key=lambda x: (x[1], x[0]))
            else:
                param_value = f.attrs.get("param_value", 0.0)
                slot_iter_raw = []
                for i, key in enumerate(slot_keys):
                    T_val = parse_temperature(key)
                    slot_iter_raw.append((round(T_val, 4), float(param_value), i))
                slot_iter_raw.sort(key=lambda x: x[0])

            # --list
            if args.list:
                temps = sorted({T_val for T_val, _, _ in slot_iter_raw})
                params = sorted({pv for _, pv, _ in slot_iter_raw})
                print(f"T values ({len(temps)}):  {', '.join(f'{t:.4f}' for t in temps)}")
                if is_2d:
                    print(
                        f"{param_label} values ({len(params)}):  "
                        f"{', '.join(f'{p:.4f}' for p in params)}"
                    )
                else:
                    print(f"param value: {params[0]}")
                return

            # Apply filters
            if args.T is not None:
                t_set = {round(t, 4) for t in args.T}
                slot_iter_raw = [(T, pv, i) for T, pv, i in slot_iter_raw if T in t_set]
            if args.param is not None:
                p_set = {round(p, 4) for p in args.param}
                slot_iter_raw = [(T, pv, i) for T, pv, i in slot_iter_raw if pv in p_set]
            if args.rigorous and disagree_slot_keys:
                _excl = set(disagree_slot_keys)
                slot_iter_raw = [
                    (T, pv, i) for T, pv, i in slot_iter_raw if slot_keys[i] not in _excl
                ]

            for T_val, pv, slot_idx in slot_iter_raw:
                snap_data = snaps_ds[slot_idx, :count]  # (count, C, L, L)
                N = snap_data.shape[0]
                if N == 0:
                    continue
                C_val = snap_data.shape[1]
                n_pick = min(args.n, N)
                indices = sorted(rng.choice(N, size=n_pick, replace=False).tolist())

                if C_val == 1:
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
                    out_name = f"{args.input.stem}_T={T_val:.4f}_{param_label}={pv:.4f}.png"
                else:
                    out_name = f"{args.input.stem}_T={T_val:.4f}.png"
                out_path = args.output_dir / out_name
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"Saved → {out_path}")
                plt.close(fig)

    if not args.no_show:
        print(f"\nAll snapshots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
