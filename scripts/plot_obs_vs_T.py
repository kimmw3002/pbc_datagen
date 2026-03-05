#!/usr/bin/env python
"""Plot ⟨O⟩ vs T curves from an HDF5 dataset file."""

from __future__ import annotations

import argparse
import json
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


def parse_slot_2d(group_name: str, param_label: str) -> tuple[float, float]:
    """Extract (T, param) from a 2D group name like 'T=0.5_U=1.2'."""
    m = re.match(rf"T=([\d.eE+\-]+)_{re.escape(param_label)}=([\d.eE+\-]+)", group_name)
    if m is None:
        raise ValueError(f"Cannot parse 2D slot from group name: {group_name}")
    return float(m.group(1)), float(m.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ⟨O⟩ vs T from HDF5 dataset")
    parser.add_argument("hdf5", type=Path, help="Path to HDF5 file")
    parser.add_argument("--obs", nargs="+", help="Observable names to plot (default: all)")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path (default: auto)")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show()")
    parser.add_argument(
        "--rigorous",
        action="store_true",
        help="Exclude unconverged slots flagged in HDF5 from plots.",
    )
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        model_type = f.attrs["model_type"]
        L = f.attrs["L"]
        is_2d = str(f.attrs.get("pt_mode", "")) == "2d"
        is_flat = "slot_keys" in f.attrs
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
            obs_names_all = json.loads(str(f.attrs["obs_names"]))
            obs_names = args.obs if args.obs else obs_names_all

            if is_2d:
                param_label = str(f.attrs["param_label"])
                # Parse (T, param, slot_idx) from slot_keys
                slots = []
                for i, key in enumerate(slot_keys):
                    T_val, pv = parse_slot_2d(key, param_label)
                    slots.append((T_val, pv, i))
                slots.sort(key=lambda x: (x[1], x[0]))
                if args.rigorous and disagree_slot_keys:
                    _excl = set(disagree_slot_keys)
                    slots = [(t, pv, i) for t, pv, i in slots if slot_keys[i] not in _excl]

                param_vals = sorted(set(s[1] for s in slots))
                data_2d: dict[str, dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
                for obs in obs_names:
                    data_2d[obs] = {}
                    for pv in param_vals:
                        pv_slots = [(t, idx) for t, p, idx in slots if p == pv]
                        ts = np.array([t for t, _ in pv_slots])
                        means = []
                        stderrs = []
                        for _, idx in pv_slots:
                            vals = f[obs][idx, :count]
                            means.append(np.mean(vals))
                            stderrs.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                        data_2d[obs][pv] = (ts, np.array(means), np.array(stderrs))
            else:
                param_value = f.attrs.get("param_value", 0.0)
                # Parse T from slot_keys
                t_slots = []
                for i, key in enumerate(slot_keys):
                    T_val = parse_temperature(key)
                    t_slots.append((T_val, i))
                t_slots.sort(key=lambda x: x[0])
                if args.rigorous and disagree_slot_keys:
                    _excl = set(disagree_slot_keys)
                    t_slots = [(t, i) for t, i in t_slots if slot_keys[i] not in _excl]

                temps = np.array([t for t, _ in t_slots])
                data_1d: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for name in obs_names:
                    means = []
                    stderrs = []
                    for _, idx in t_slots:
                        vals = f[name][idx, :count]
                        means.append(np.mean(vals))
                        stderrs.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                    data_1d[name] = (np.array(means), np.array(stderrs))
        else:
            # Old per-group schema fallback
            if is_2d:
                param_label = str(f.attrs["param_label"])
                slots_old = sorted(
                    [
                        (*parse_slot_2d(name, param_label), name)
                        for name in f
                        if name.startswith("T=")
                    ],
                    key=lambda x: (x[1], x[0]),
                )

                first_group = f[slots_old[0][2]]
                all_obs = [k for k in first_group if k != "snapshots"]
                obs_names = args.obs if args.obs else all_obs

                param_vals = sorted(set(s[1] for s in slots_old))
                data_2d = {}
                for obs in obs_names:
                    data_2d[obs] = {}
                    for pv in param_vals:
                        pv_slots = [(t, gn) for t, p, gn in slots_old if p == pv]
                        ts = np.array([t for t, _ in pv_slots])
                        means = []
                        stderrs = []
                        for _, gname in pv_slots:
                            vals = f[gname][obs][:]
                            means.append(np.mean(vals))
                            stderrs.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                        data_2d[obs][pv] = (ts, np.array(means), np.array(stderrs))
            else:
                param_value = f.attrs["param_value"]
                t_groups = sorted(
                    [(parse_temperature(name), name) for name in f if name.startswith("T=")],
                    key=lambda x: x[0],
                )

                first_group = f[t_groups[0][1]]
                all_obs = [k for k in first_group if k != "snapshots"]
                obs_names = args.obs if args.obs else all_obs

                temps = np.array([t for t, _ in t_groups])
                data_1d = {}
                for name in obs_names:
                    means = []
                    stderrs = []
                    for _, gname in t_groups:
                        vals = f[gname][name][:]
                        means.append(np.mean(vals))
                        stderrs.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                    data_1d[name] = (np.array(means), np.array(stderrs))

    # Plot: subplot grid with ceil(n/2) rows × 2 cols
    n_obs = len(obs_names)
    ncols = 2
    nrows = math.ceil(n_obs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False)

    if is_2d:
        cmap = plt.get_cmap("viridis", len(param_vals))
        for idx, obs in enumerate(obs_names):
            ax = axes[idx // ncols, idx % ncols]
            for ci, pv in enumerate(param_vals):
                ts, mean, stderr = data_2d[obs][pv]
                ax.errorbar(
                    ts,
                    mean,
                    yerr=stderr,
                    fmt="o-",
                    capsize=3,
                    markersize=3,
                    color=cmap(ci),
                    label=f"{param_label}={pv:.3f}",
                )
            ax.set_xlabel("T")
            ax.set_ylabel(f"⟨{obs}⟩")
            ax.set_title(obs)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, ncol=2)
        title = f"{model_type}  L={L}  2D PT"
    else:
        for idx, name in enumerate(obs_names):
            ax = axes[idx // ncols, idx % ncols]
            mean, stderr = data_1d[name]
            ax.errorbar(temps, mean, yerr=stderr, fmt="o-", capsize=3, markersize=4)
            ax.set_xlabel("T")
            ax.set_ylabel(f"⟨{name}⟩")
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
        title = f"{model_type}  L={L}  param={param_value}"

    # Hide unused subplots
    for idx in range(n_obs, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    out_path = args.output or Path(f"images/{args.hdf5.stem}_obs_vs_T.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # --- 2D heatmap (separate PNG per observable) ---
    if is_2d:
        source = slots if is_flat else slots_old  # type: ignore[possibly-undefined]
        t_vals = sorted(set(s[0] for s in source))
        for obs in obs_names:
            grid = np.zeros((len(param_vals), len(t_vals)))
            for pi, pv in enumerate(param_vals):
                ts_arr, mean, _ = data_2d[obs][pv]
                for ti, t in enumerate(t_vals):
                    ix = int(np.argmin(np.abs(ts_arr - t)))
                    grid[pi, ti] = mean[ix]

            fig_h, ax_h = plt.subplots(figsize=(8, 6))
            abs_obs = ("abs_m", "q", "abs_m_sigma", "abs_m_tau", "abs_m_baxter")
            vmin: float | None = 0 if obs in abs_obs else None
            vmax: float | None = 1 if obs in abs_obs else None
            im = ax_h.imshow(
                grid,
                aspect="auto",
                origin="lower",
                extent=(t_vals[0], t_vals[-1], param_vals[0], param_vals[-1]),
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
            )
            ax_h.set_xlabel("T")
            ax_h.set_ylabel(param_label)
            ax_h.set_title(f"⟨{obs}⟩  —  {model_type} L={L}")
            fig_h.colorbar(im, ax=ax_h, label=f"⟨{obs}⟩")
            fig_h.tight_layout()

            hm_path = out_path.parent / f"{args.hdf5.stem}_heatmap_{obs}.png"
            fig_h.savefig(hm_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {hm_path}")
            plt.close(fig_h)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
