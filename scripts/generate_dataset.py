#!/usr/bin/env python
"""CLI entry point for dataset generation.

Wraps ``pbc_datagen.orchestrator.generate_dataset()`` with argparse,
rich display, and loguru logging (stdout + per-run log file).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger
from pbc_datagen.orchestrator import generate_dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

VALID_MODELS = ("ising", "blume_capel", "ashkin_teller")

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
        help="Model type: ising, blume_capel, ashkin_teller",
    )
    parser.add_argument("--L", type=int, required=True, help="Lattice side length")
    parser.add_argument(
        "--params",
        type=float,
        nargs="+",
        required=True,
        help="Hamiltonian parameter values (D for BC, U for AT, J for Ising)",
    )
    parser.add_argument(
        "--T-range",
        type=float,
        nargs=2,
        required=True,
        metavar=("T_MIN", "T_MAX"),
        help="Temperature range [T_min, T_max]",
    )
    parser.add_argument("--n-replicas", type=int, default=20, help="PT replicas (default: 20)")
    parser.add_argument(
        "--n-snapshots", type=int, default=100, help="Snapshots per T slot (default: 100)"
    )
    parser.add_argument("--workers", type=int, default=4, help="Max parallel workers (default: 4)")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = _setup_logging(output_dir)

    # Enable library logging (disabled by default in pbc_datagen/__init__.py)
    logger.enable("pbc_datagen")

    # --- Summary panel ---
    param_label = {"ising": "J", "blume_capel": "D", "ashkin_teller": "U"}[args.model]
    params_str = ", ".join(f"{p:.4f}" for p in args.params)

    table = Table(show_header=False, border_style="dim", pad_edge=False, box=None)
    table.add_column("key", style="bold", min_width=16)
    table.add_column("value")
    table.add_row("Model", args.model)
    table.add_row("L", str(args.L))
    table.add_row(f"{param_label} values", params_str)
    table.add_row("T range", f"[{args.T_range[0]}, {args.T_range[1]}]")
    table.add_row("Replicas", str(args.n_replicas))
    table.add_row("Snapshots/T", str(args.n_snapshots))
    table.add_row("Workers", str(args.workers))
    table.add_row("Output", str(output_dir.resolve()))
    table.add_row("Mode", "fresh" if getattr(args, "new") else "resume")
    table.add_row("Log file", str(log_file))

    console.print(Panel(table, title="[bold cyan]pbc_datagen[/]", border_style="cyan"))

    logger.info("Starting dataset generation")
    logger.info(
        "model={} L={} params={} T_range={} n_replicas={} n_snapshots={} workers={}",
        args.model,
        args.L,
        args.params,
        tuple(args.T_range),
        args.n_replicas,
        args.n_snapshots,
        args.workers,
    )

    t0 = time.perf_counter()

    generate_dataset(
        model_type=args.model,
        L=args.L,
        param_values=args.params,
        T_range=tuple(args.T_range),
        n_replicas=args.n_replicas,
        n_snapshots=args.n_snapshots,
        max_workers=args.workers,
        output_dir=str(output_dir),
        force_new=args.new,
        log_file=str(log_file),
    )

    elapsed = time.perf_counter() - t0
    logger.info("Done in {:.1f}s", elapsed)
    console.print(f"\n[bold green]Complete[/] in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
