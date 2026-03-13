#!/usr/bin/env python
"""CLI entry point for single-chain MCMC dataset generation.

Wraps ``pbc_datagen.single_chain.run_single_campaign()`` with argparse,
rich display, and loguru logging.  Simpler than ``generate_dataset.py``:
one temperature, no replicas, no ladder tuning.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger
from pbc_datagen.registry import valid_model_names
from pbc_datagen.single_chain import run_single_campaign
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
    log_file = log_dir / f"single_{ts}.log"

    logger.remove()

    fmt_plain = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}"
    fmt_color = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"

    logger.add(sys.stdout, format=fmt_color, level="INFO", colorize=True)
    logger.add(str(log_file), format=fmt_plain, level="DEBUG")

    return log_file


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate snapshots via single-chain MCMC at one (param, T) point.",
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
        default=None,
        help="Hamiltonian parameter value (D for Blume-Capel, U for Ashkin-Teller). "
        "Not used for Ising.",
    )
    parser.add_argument(
        "--T",
        type=float,
        required=True,
        help="Temperature (single value)",
    )
    parser.add_argument(
        "--n-snapshots",
        type=int,
        default=100,
        help="Number of snapshots to collect (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Force fresh start, ignore existing files",
    )

    return parser.parse_args(argv)


@logger.catch
def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # --- Validate --params vs model ---
    if args.model == "ising":
        if args.params is not None:
            console.print(
                "[bold red]Error:[/] Ising has no tunable Hamiltonian parameter "
                "(J=1 is fixed in C++). Do not pass --params for Ising."
            )
            raise SystemExit(1)
        param_value = 0.0
    else:
        if args.params is None:
            console.print(f"[bold red]Error:[/] --params is required for {args.model}.")
            raise SystemExit(1)
        param_value = args.params

    # Snap T to 4 decimal places
    T = round(args.T, 4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = _setup_logging(output_dir)
    logger.enable("pbc_datagen")

    # --- Summary panel ---
    param_label: dict[str, str] = {"blume_capel": "D", "ashkin_teller": "U"}

    table = Table(show_header=False, border_style="dim", pad_edge=False, box=None)
    table.add_column("key", style="bold", min_width=16)
    table.add_column("value")
    table.add_row("Model", args.model)
    table.add_row("L", str(args.L))
    if args.model in param_label:
        table.add_row(f"{param_label[args.model]}", f"{param_value:.4f}")
    else:
        table.add_row("J", "1 (fixed)")
    table.add_row("T", f"{T:.4f}")
    table.add_row("Snapshots", str(args.n_snapshots))
    table.add_row("Output", str(output_dir.resolve()))
    table.add_row("Mode", "fresh" if args.new else "resume")
    table.add_row("Log file", str(log_file))

    console.print(
        Panel(table, title="[bold cyan]pbc_datagen — single chain[/]", border_style="cyan")
    )

    logger.info("Starting single-chain generation")
    logger.info(
        "model={} L={} param={} T={:.4f} n_snapshots={}",
        args.model,
        args.L,
        param_value,
        T,
        args.n_snapshots,
    )

    t0 = time.perf_counter()

    run_single_campaign(
        model_type=args.model,
        L=args.L,
        param_value=param_value,
        T=T,
        n_snapshots=args.n_snapshots,
        output_dir=str(output_dir),
        force_new=args.new,
    )

    elapsed = time.perf_counter() - t0
    logger.info("Done in {:.1f}s", elapsed)
    console.print(f"\n[bold green]Complete[/] in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
