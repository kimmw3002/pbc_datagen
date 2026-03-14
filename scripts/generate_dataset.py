#!/usr/bin/env python
"""CLI entry point for dataset generation.

Wraps ``pbc_datagen.orchestrator.generate_dataset()`` with argparse,
rich display, and loguru logging (stdout + per-run log file).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from loguru import logger
from pbc_datagen.registry import get_model_info, valid_model_names
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

VALID_MODELS = valid_model_names()

console = Console()


def _setup_logging(output_dir: Path) -> Path:
    """Configure loguru: stdout + timestamped log file in logs/.

    Returns the log file path.
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{ts}.log"

    # Remove default stderr handler
    logger.remove()

    fmt_plain = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}"
    fmt_color = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"

    # Stdout — INFO and above, with colors
    logger.add(sys.stdout, format=fmt_color, level="INFO", colorize=True)

    # File — DEBUG and above, no color tags
    logger.add(str(log_file), format=fmt_plain, level="DEBUG")

    return log_file


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-quality 2D lattice model snapshot datasets.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=VALID_MODELS,
        help="Model type: ising, blume_capel, ashkin_teller, xy",
    )
    parser.add_argument("--L", type=int, required=True, help="Lattice side length")
    parser.add_argument(
        "--params",
        type=float,
        nargs="+",
        default=None,
        help="Hamiltonian parameter values (D for Blume-Capel, U for Ashkin-Teller). "
        "Not used for Ising or XY (J=1 is fixed).",
    )
    parser.add_argument(
        "--T-range",
        type=float,
        nargs=2,
        required=True,
        metavar=("T_MIN", "T_MAX"),
        help="Temperature range [T_min, T_max]",
    )
    parser.add_argument(
        "--n-replicas", type=int, default=20, help="PT replicas for 1D PT (default: 20)"
    )
    parser.add_argument(
        "--n-snapshots", type=int, default=100, help="Snapshots per slot (default: 100)"
    )
    parser.add_argument(
        "--param-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("P_MIN", "P_MAX"),
        help="2D PT: parameter range [min, max]. Enables 2D PT mode for BC/AT.",
    )
    parser.add_argument(
        "--n-T", type=int, default=10, help="2D PT: number of temperature points (default: 10)"
    )
    parser.add_argument(
        "--n-P", type=int, default=10, help="2D PT: number of parameter points (default: 10)"
    )
    parser.add_argument(
        "--connectivity-rounds",
        type=int,
        default=100,
        help="2D PT Phase A: exchange rounds for connectivity check (default: 100)",
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="OpenMP threads for C++ sweep loop (default: 4)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/", help="Output directory (default: output/)"
    )
    parser.add_argument(
        "--new", action="store_true", help="Force fresh start, ignore existing files"
    )

    return parser.parse_args(argv)


@logger.catch
def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # --- Set OMP_NUM_THREADS BEFORE importing _core ---
    # The OpenMP thread pool is created on first use and never resized.
    # Importing pbc_datagen._core (via orchestrator) can trigger pool
    # creation, so the env var must be set first.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    from pbc_datagen.orchestrator import (  # noqa: E402
        generate_dataset,
        generate_dataset_2d,
        set_omp_threads,
    )

    # --- Determine 1D vs 2D mode ---
    use_2d = args.param_range is not None

    # --- Validate arguments ---
    info = get_model_info(args.model)
    if info.param_label is None:
        if args.params is not None or use_2d:
            console.print(
                f"[bold red]Error:[/] {args.model} has no tunable Hamiltonian parameter. "
                "Do not pass --params or --param-range."
            )
            raise SystemExit(1)
        # Dummy value so the loop runs once — param-less models ignore param_value
        args.params = [0.0]
    elif use_2d:
        if args.params is not None:
            console.print(
                "[bold red]Error:[/] --params and --param-range are mutually exclusive. "
                "Use --param-range for 2D PT, --params for 1D PT."
            )
            raise SystemExit(1)
        args.param_range = [round(p, 4) for p in args.param_range]
    else:
        if args.params is None:
            console.print(
                f"[bold red]Error:[/] --params or --param-range required for {args.model}."
            )
            raise SystemExit(1)

    # Snap T values to 4 decimal places so filename and simulation always
    # agree.  Sub-0.0001 temperature precision is never physically meaningful.
    args.T_range = [round(t, 4) for t in args.T_range]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = _setup_logging(output_dir)

    # Enable library logging (disabled by default in pbc_datagen/__init__.py)
    logger.enable("pbc_datagen")

    # --- Summary panel ---
    pl = get_model_info(args.model).param_label

    table = Table(show_header=False, border_style="dim", pad_edge=False, box=None)
    table.add_column("key", style="bold", min_width=16)
    table.add_column("value")
    table.add_row("Model", args.model)
    table.add_row("L", str(args.L))
    table.add_row("PT mode", "2D grid" if use_2d else "1D ladder")
    if use_2d:
        assert pl is not None
        table.add_row(f"{pl} range", f"[{args.param_range[0]}, {args.param_range[1]}]")
        table.add_row("Grid", f"{args.n_T} T × {args.n_P} {pl}")
    elif pl is not None:
        assert args.params is not None
        params_str = ", ".join(f"{p:.4f}" for p in args.params)
        table.add_row(f"{pl} values", params_str)
        table.add_row("Replicas", str(args.n_replicas))
    else:
        table.add_row("J", "1 (fixed)")
        table.add_row("Replicas", str(args.n_replicas))
    table.add_row("T range", f"[{args.T_range[0]}, {args.T_range[1]}]")
    table.add_row("Snapshots/slot", str(args.n_snapshots))
    table.add_row("Threads", str(args.threads))
    table.add_row("Output", str(output_dir.resolve()))
    table.add_row("Mode", "fresh" if getattr(args, "new") else "resume")
    table.add_row("Log file", str(log_file))

    console.print(Panel(table, title="[bold cyan]pbc_datagen[/]", border_style="cyan"))

    logger.info("Starting dataset generation")
    t0 = time.perf_counter()

    # Also call set_omp_threads for the log message
    set_omp_threads(args.threads)

    if use_2d:
        logger.info(
            "model={} L={} T_range={} param_range={} grid={}x{} n_snapshots={}",
            args.model,
            args.L,
            tuple(args.T_range),
            tuple(args.param_range),
            args.n_T,
            args.n_P,
            args.n_snapshots,
        )
        generate_dataset_2d(
            model_type=args.model,
            L=args.L,
            T_range=tuple(args.T_range),
            param_range=tuple(args.param_range),
            n_T=args.n_T,
            n_P=args.n_P,
            n_snapshots=args.n_snapshots,
            output_dir=str(output_dir),
            force_new=args.new,
            connectivity_rounds=args.connectivity_rounds,
        )
    else:
        logger.info(
            "model={} L={} params={} T_range={} n_replicas={} n_snapshots={}",
            args.model,
            args.L,
            args.params,
            tuple(args.T_range),
            args.n_replicas,
            args.n_snapshots,
        )
        assert args.params is not None
        generate_dataset(
            model_type=args.model,
            L=args.L,
            param_values=args.params,
            T_range=tuple(args.T_range),
            n_replicas=args.n_replicas,
            n_snapshots=args.n_snapshots,
            output_dir=str(output_dir),
            force_new=args.new,
        )

    elapsed = time.perf_counter() - t0
    logger.info("Done in {:.1f}s", elapsed)
    console.print(f"\n[bold green]Complete[/] in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
