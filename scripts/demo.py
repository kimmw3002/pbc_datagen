#!/usr/bin/env python
"""Unified demo: equilibrate a model, then sweep forever with live lattice + observables.

Usage:
    python scripts/demo.py ising --T 2.269
    python scripts/demo.py blume_capel --T 0.609 --param 1.966
    python scripts/demo.py ashkin_teller --T 2.269 --param 0.0

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
from pbc_datagen.registry import get_model_info, make_cmap_norm, valid_model_names

# Models that support the demo loop (need snapshot() + sweep() + observables())
_DEMO_MODELS = [n for n in valid_model_names() if n != "xy"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Live lattice demo for any registered model")
    parser.add_argument("model", choices=_DEMO_MODELS, help="Model name")
    parser.add_argument("--L", type=int, default=64, help="Lattice side length (default: 64)")
    parser.add_argument("--T", type=float, required=True, help="Temperature")
    parser.add_argument("--param", type=float, default=None, help="Model parameter (D, U, ...)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    info = get_model_info(args.model)

    # Validate --param
    if info.param_label is not None and args.param is None:
        parser.error(f"Model {args.model!r} requires --param ({info.param_label})")
    if info.param_label is None and args.param is not None:
        parser.error(f"Model {args.model!r} has no parameter, but --param was given")

    # Construct model
    model = info.constructor(args.L, seed=args.seed)
    model.set_temperature(args.T)
    if info.set_param is not None and args.param is not None:
        info.set_param(model, args.param)

    # Output path: scripts/<model>_L<L>_T<T>[_<param_label><param>].png
    if info.param_label is not None:
        fname = f"{args.model}_L{args.L}_T{args.T}_{info.param_label}{args.param}.png"
        out_path = Path("scripts") / fname
    else:
        out_path = Path("scripts") / f"{args.model}_L{args.L}_T{args.T}.png"

    # Equilibrate
    print(f"Equilibrating {args.model} L={args.L} T={args.T}", end="")
    if info.param_label is not None:
        print(f" {info.param_label}={args.param}", end="")
    print(" ...")
    model.sweep(1000)
    print("Equilibrated.")

    # Setup figure
    cmap, norm = make_cmap_norm(info.viz)
    viz = info.viz
    is_at = args.model == "ashkin_teller"

    if info.n_channels == 1:
        n_panels = 1
    elif is_at:
        n_panels = 3  # sigma, tau, sigma*tau (Baxter order parameter)
    else:
        n_panels = info.n_channels

    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3), squeeze=False)
    axes_flat = axes[0]

    # Initial snapshot
    snap = model.snapshot()
    imgs = []
    for i in range(min(info.n_channels, n_panels)):
        im = axes_flat[i].imshow(snap[i], cmap=cmap, norm=norm, interpolation="nearest")
        cb = fig.colorbar(im, ax=axes_flat[i], fraction=0.046, pad=0.04)
        if viz.tick_values is not None and viz.tick_labels is not None:
            cb.set_ticks(list(viz.tick_values))
            cb.set_ticklabels(list(viz.tick_labels))
        axes_flat[i].axis("off")
        imgs.append(im)

    # AT-specific: sigma*tau panel
    if is_at and n_panels == 3:
        product = snap[0] * snap[1]
        im = axes_flat[2].imshow(product, cmap=cmap, norm=norm, interpolation="nearest")
        cb = fig.colorbar(im, ax=axes_flat[2], fraction=0.046, pad=0.04)
        if viz.tick_values is not None and viz.tick_labels is not None:
            cb.set_ticks(list(viz.tick_values))
            cb.set_ticklabels(list(viz.tick_labels))
        axes_flat[2].axis("off")
        axes_flat[2].set_title("\u03c3\u03c4")
        imgs.append(im)

    # Panel labels
    if is_at:
        axes_flat[0].set_title("\u03c3")
        axes_flat[1].set_title("\u03c4")

    title = f"{args.model}  L={args.L}  T={args.T}"
    if info.param_label is not None:
        title += f"  {info.param_label}={args.param}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    first_iter = True

    # Infinite sweep loop
    try:
        while True:
            t0 = time.perf_counter()
            model.sweep(100)

            # Print observables
            obs = model.observables()
            obs_line = "  ".join(f"{k}={v:+.4f}" for k, v in obs.items())
            print(f"\r{obs_line}", end="", flush=True)

            # Update lattice display
            snap = model.snapshot()
            for i in range(min(info.n_channels, len(imgs))):
                imgs[i].set_data(snap[i])
            if is_at and len(imgs) > 2:
                imgs[2].set_data(snap[0] * snap[1])

            fig.savefig(out_path, dpi=150, bbox_inches="tight")

            if first_iter:
                print(f"\nImage: {out_path}")
                first_iter = False

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, 0.5 - elapsed))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
