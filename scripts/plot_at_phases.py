#!/usr/bin/env python
"""Phase-diagram heatmap for Ashkin-Teller 2D PT datasets.

Classifies each (T, U) slot into one of three phases based on
⟨|M_σ|⟩ and ⟨|M_baxter|⟩ thresholds, produces a coloured heatmap,
and prints per-phase counts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

# Phase labels
ORDERED = 0  # |M_σ| high, |M_bax| high
BAXTER = 1  # |M_σ| low,  |M_bax| high
DISORDERED = 2  # |M_σ| low,  |M_bax| low
UNCLASSIFIED = 3  # intermediate

PHASE_NAMES = {
    ORDERED: "Ordered",
    BAXTER: "Baxter",
    DISORDERED: "Disordered",
    UNCLASSIFIED: "Unclassified",
}
PHASE_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#999999"]  # red, blue, green, grey


def parse_slot_2d(group_name: str, param_label: str) -> tuple[float, float]:
    """Extract (T, param) from a 2D group name like 'T=0.5_U=1.2'."""
    m = re.match(rf"T=([\d.eE+\-]+)_{re.escape(param_label)}=([\d.eE+\-]+)", group_name)
    if m is None:
        raise ValueError(f"Cannot parse 2D slot from group name: {group_name}")
    return float(m.group(1)), float(m.group(2))


def classify(
    mean_sigma: float,
    mean_baxter: float,
    sigma_hi: float,
    sigma_lo: float,
    baxter_hi: float,
    baxter_lo: float,
) -> int:
    """Return phase label for a single (T, U) point."""
    if mean_sigma > sigma_hi and mean_baxter > baxter_hi:
        return ORDERED
    if mean_sigma < sigma_lo and mean_baxter > baxter_hi:
        return BAXTER
    if mean_sigma < sigma_lo and mean_baxter < baxter_lo:
        return DISORDERED
    return UNCLASSIFIED


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

SlotData = dict[tuple[float, float], tuple[float, float]]  # (T,U) -> (mean_sigma, mean_baxter)


def load_hdf5(path: Path) -> tuple[str, int, SlotData]:
    import h5py

    with h5py.File(path, "r") as f:
        model_type = str(f.attrs["model_type"])
        L = int(f.attrs["L"])
        slot_keys: list[str] = json.loads(str(f.attrs["slot_keys"]))
        count = int(f.attrs["count"])
        param_label = str(f.attrs.get("param_label", "U"))

        data: SlotData = {}
        for i, key in enumerate(slot_keys):
            T_val, U_val = parse_slot_2d(key, param_label)
            sigma_vals = np.asarray(f["abs_m_sigma"][i, :count])
            baxter_vals = np.asarray(f["abs_m_baxter"][i, :count])
            data[(T_val, U_val)] = (float(np.mean(sigma_vals)), float(np.mean(baxter_vals)))

    return model_type, L, data


def load_pt(path: Path) -> tuple[str, int, SlotData]:
    import torch

    records = torch.load(path, weights_only=False)
    if not records:
        raise ValueError(f"Empty .pt file: {path}")

    sample = records[0]
    L = int(sample["state"].shape[-1])
    model = path.stem.split("_L")[0]

    # Group by (T, U)
    groups: dict[tuple[float, float], list[dict]] = defaultdict(list)  # type: ignore[assignment]
    for rec in records:
        key = (round(rec["T"], 4), round(rec["U"], 4))
        groups[key].append(rec)

    data: SlotData = {}
    for (T_val, U_val), recs in groups.items():
        sigma_mean = float(np.mean([r["abs_m_sigma"] for r in recs]))
        baxter_mean = float(np.mean([r["abs_m_baxter"] for r in recs]))
        data[(T_val, U_val)] = (sigma_mean, baxter_mean)

    return model, L, data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AT phase-diagram heatmap from HDF5 or .pt dataset"
    )
    parser.add_argument("input", type=Path, help="Path to HDF5 or .pt file")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path (default: auto)")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show()")
    parser.add_argument(
        "--sigma-hi", type=float, default=0.9, help="⟨|M_σ|⟩ threshold for ordered (default: 0.9)"
    )
    parser.add_argument(
        "--sigma-lo",
        type=float,
        default=0.1,
        help="⟨|M_σ|⟩ threshold for disordered (default: 0.1)",
    )
    parser.add_argument(
        "--baxter-hi",
        type=float,
        default=0.9,
        help="⟨|M_bax|⟩ threshold for ordered/Baxter (default: 0.9)",
    )
    parser.add_argument(
        "--baxter-lo",
        type=float,
        default=0.1,
        help="⟨|M_bax|⟩ threshold for disordered (default: 0.1)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if input_path.suffix == ".pt":
        model_type, L, slot_data = load_pt(input_path)
    else:
        model_type, L, slot_data = load_hdf5(input_path)

    # --- Classify each slot ---
    phase_map: dict[tuple[float, float], int] = {}
    for (T, U), (ms, mb) in slot_data.items():
        phase_map[(T, U)] = classify(
            ms, mb, args.sigma_hi, args.sigma_lo, args.baxter_hi, args.baxter_lo
        )

    # --- Phase counts ---
    counts: dict[int, int] = {ORDERED: 0, BAXTER: 0, DISORDERED: 0, UNCLASSIFIED: 0}
    for ph in phase_map.values():
        counts[ph] += 1

    print(f"\n{'Phase':<16} {'Count':>6}")
    print("-" * 24)
    for ph_id in [ORDERED, BAXTER, DISORDERED, UNCLASSIFIED]:
        print(f"{PHASE_NAMES[ph_id]:<16} {counts[ph_id]:>6}")
    print(f"{'TOTAL':<16} {sum(counts.values()):>6}")

    # --- Build grid for pcolormesh ---
    T_unique = np.array(sorted({t for t, _ in phase_map}))
    U_unique = np.array(sorted({u for _, u in phase_map}))
    nT, nU = len(T_unique), len(U_unique)
    T_idx = {t: i for i, t in enumerate(T_unique)}
    U_idx = {u: i for i, u in enumerate(U_unique)}

    # Phase matrix: rows = U, cols = T  (so y-axis = U, x-axis = T)
    Z = np.full((nU, nT), UNCLASSIFIED, dtype=int)
    for (T, U), ph in phase_map.items():
        Z[U_idx[U], T_idx[T]] = ph

    # Build cell edges for pcolormesh (handles non-uniform spacing)
    def cell_edges(centers: np.ndarray) -> np.ndarray:
        """Compute N+1 cell edges from N non-uniform cell centres."""
        edges = np.empty(len(centers) + 1)
        mids = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - (mids[0] - centers[0])
        edges[-1] = centers[-1] + (centers[-1] - mids[-1])
        edges[1:-1] = mids
        return edges

    T_edges = cell_edges(T_unique)
    U_edges = cell_edges(U_unique)

    # --- Plot ---
    cmap = ListedColormap(PHASE_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pcolormesh(T_edges, U_edges, Z, cmap=cmap, norm=norm, edgecolors="face", linewidth=0)
    ax.set_xlabel("T", fontsize=12)
    ax.set_ylabel("U", fontsize=12)
    ax.set_title(f"AT Phase Diagram — {model_type}  L={L}", fontsize=14)

    # Legend
    patches = [
        mpatches.Patch(color=PHASE_COLORS[i], label=f"{PHASE_NAMES[i]} ({counts[i]})")
        for i in range(4)
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=10)

    fig.tight_layout()

    out_path = args.output or Path(f"images/{input_path.stem}_phase_diagram.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
