#!/usr/bin/env python
"""Parallel sweep of a (T, param) grid using single-chain MCMC.

Drives ``pbc_datagen.single_chain.run_single_campaign()`` across a Cartesian
product of temperatures (np.geomspace) × param values (np.linspace) using
``ProcessPoolExecutor`` for parallel execution.

Each worker runs one (T, param) campaign in its own process.  A shared log
file captures DEBUG output from all workers; INFO output goes to stdout in
the orchestrating process only.  A single failure does not kill the run —
the summary table shows OK / FAIL for every task.
"""

from __future__ import annotations

import argparse
import collections
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger
from pbc_datagen.registry import get_model_info, valid_model_names
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

VALID_MODELS = valid_model_names()

console = Console()

_FMT_PLAIN = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}"


# ---------------------------------------------------------------------------
# Logging setup (orchestrator process)
# ---------------------------------------------------------------------------


# Ring buffer for the last N log lines shown in the Live display.
_LOG_LINES: collections.deque[str] = collections.deque(maxlen=20)


def _deque_sink(message: str) -> None:
    """Loguru sink that appends formatted lines to the shared deque."""
    _LOG_LINES.append(message.rstrip("\n"))


def _setup_logging(output_dir: Path) -> Path:
    """Configure loguru: in-memory deque (INFO) + timestamped log file (DEBUG).

    Returns the log file path.
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"parallel_{ts}.log"

    logger.remove()
    # Live-display sink: INFO lines are captured into _LOG_LINES deque
    logger.add(_deque_sink, format=_FMT_PLAIN, level="INFO", colorize=False)
    # enqueue=True: background thread serialises writes → safe across processes
    logger.add(str(log_file), format=_FMT_PLAIN, level="DEBUG", enqueue=True)

    return log_file


# ---------------------------------------------------------------------------
# Worker (runs in a subprocess per task)
# ---------------------------------------------------------------------------


def _run_one(
    T: float,
    param: float,
    model: str,
    L: int,
    n_snapshots: int,
    output_dir: str,
    force_new: bool,
    log_file: str,
) -> tuple[float, float, str | None, str | None]:
    """Run a single (T, param) campaign.  Executed inside a worker process.

    Sets ``OMP_NUM_THREADS=1`` **before** any C++ import to avoid
    oversubscription (each worker is already its own Python process; letting
    OpenMP spawn N threads inside M workers would give M×N threads competing
    for the same cores).

    The loguru file sink is opened with ``enqueue=True``: each worker has its
    own internal queue + background thread.  Linux ``O_APPEND`` writes are
    atomic for small records, so concurrent workers writing to the same file
    are safe.

    Returns
    -------
    (T, param, path_str, None)   on success
    (T, param, None, error_str)  on failure
    """
    # Must precede any import that touches pbc_datagen._core — the OpenMP
    # thread pool is created on first use and cannot be resized afterwards.
    os.environ["OMP_NUM_THREADS"] = "1"

    # Re-configure loguru in this subprocess (the parent's handlers are not
    # inherited when using "spawn" start method; even with "fork" we want a
    # clean per-worker config that writes only to the shared file).
    from loguru import logger as _log  # noqa: PLC0415 (intentional late import)

    _log.remove()
    _log.add(log_file, format=_FMT_PLAIN, level="DEBUG", enqueue=True)

    # pbc_datagen/__init__.py calls logger.disable("pbc_datagen") so library
    # messages are suppressed by default.  Re-enable after import.
    from pbc_datagen.single_chain import run_single_campaign  # noqa: PLC0415

    _log.enable("pbc_datagen")

    # Structured subdirectory so HDF5 files don't pile up flat in output_dir:
    #   ising       →  {root}/ising/L{L}/
    #   blume_capel →  {root}/blume_capel/L{L}/D={p:.4f}/
    #   ashkin_teller → {root}/ashkin_teller/L{L}/U={p:.4f}/
    label = get_model_info(model).param_label
    if label is not None:
        task_dir = os.path.join(output_dir, model, f"L{L}", f"{label}={param:.4f}")
    else:
        task_dir = os.path.join(output_dir, model, f"L{L}")

    try:
        path = run_single_campaign(
            model_type=model,
            L=L,
            param_value=param,
            T=T,
            n_snapshots=n_snapshots,
            output_dir=task_dir,
            force_new=force_new,
        )
        return (T, param, str(path), None)
    except Exception as exc:  # noqa: BLE001
        _log.error("FAILED T={:.4f} param={:.4f}: {}", T, param, exc)
        return (T, param, None, str(exc))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep a (T, param) grid using parallel single-chain MCMC. "
            "T points are generated via np.geomspace; param points via np.linspace."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=VALID_MODELS,
        help="Model: ising / blume_capel / ashkin_teller",
    )
    parser.add_argument("--L", type=int, required=True, help="Lattice side length")
    parser.add_argument(
        "--T-min",
        type=float,
        required=True,
        dest="T_min",
        help="Low end of temperature range",
    )
    parser.add_argument(
        "--T-max",
        type=float,
        required=True,
        dest="T_max",
        help="High end of temperature range",
    )
    parser.add_argument(
        "--n-T",
        type=int,
        default=10,
        dest="n_T",
        help="Number of T points via np.geomspace (default: 10)",
    )
    parser.add_argument(
        "--param-min",
        type=float,
        default=None,
        dest="param_min",
        help="Low end of param range (required for blume_capel / ashkin_teller)",
    )
    parser.add_argument(
        "--param-max",
        type=float,
        default=None,
        dest="param_max",
        help="High end of param range (required for blume_capel / ashkin_teller)",
    )
    parser.add_argument(
        "--n-param",
        type=int,
        default=1,
        dest="n_param",
        help="Number of param points via np.linspace (default: 1)",
    )
    parser.add_argument(
        "--n-snapshots",
        type=int,
        default=100,
        dest="n_snapshots",
        help="Snapshots per (T, param) point (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/",
        dest="output_dir",
        help="Root output directory (default: output/)",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Force fresh start, ignore existing HDF5 files",
    )
    default_threads = os.cpu_count() or 1
    parser.add_argument(
        "--threads",
        type=int,
        default=default_threads,
        help=f"Max parallel Python workers (default: {default_threads})",
    )
    return parser.parse_args(argv)


@logger.catch
def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # --- Validate model vs param flags ---
    info = get_model_info(args.model)
    if info.param_label is None:
        if args.param_min is not None or args.param_max is not None:
            console.print(
                f"[bold red]Error:[/] {args.model} has no tunable Hamiltonian parameter. "
                "Do not pass --param-min / --param-max."
            )
            raise SystemExit(1)
        param_values = np.array([0.0])
    else:
        if args.param_min is None or args.param_max is None:
            console.print(
                f"[bold red]Error:[/] --param-min and --param-max are required for {args.model}."
            )
            raise SystemExit(1)
        param_values = np.linspace(args.param_min, args.param_max, args.n_param)

    # Round to 4 d.p. — sub-0.0001 precision is never physically meaningful,
    # and rounding ensures filenames match the simulation values exactly.
    T_values = np.round(np.geomspace(args.T_min, args.T_max, args.n_T), 4)
    param_values = np.round(param_values, 4)

    # param-outer, T-inner: sweeps all T at a fixed param before advancing
    tasks = [(float(T), float(p)) for p in param_values for T in T_values]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = _setup_logging(output_dir)
    logger.enable("pbc_datagen")

    # --- Summary panel ---
    param_label: dict[str, str] = {"blume_capel": "D", "ashkin_teller": "U"}

    table = Table(show_header=False, border_style="dim", pad_edge=False, box=None)
    table.add_column("key", style="bold", min_width=18)
    table.add_column("value")
    table.add_row("Model", args.model)
    table.add_row("L", str(args.L))
    table.add_row(
        "T range",
        f"[{T_values[0]:.4f}, {T_values[-1]:.4f}] × {args.n_T} pts (geomspace)",
    )
    if args.model in param_label:
        pl = param_label[args.model]
        table.add_row(
            f"{pl} range",
            f"[{param_values[0]:.4f}, {param_values[-1]:.4f}] × {args.n_param} pts (linspace)",
        )
    else:
        table.add_row("J", "1 (fixed)")
    table.add_row(
        "Tasks",
        f"{len(tasks)} total ({args.n_T} T × {len(param_values)} param)",
    )
    table.add_row("Snapshots/task", str(args.n_snapshots))
    table.add_row("Workers", str(args.threads))
    table.add_row("Mode", "fresh" if args.new else "resume")
    table.add_row("Output", str(output_dir.resolve()))
    table.add_row("Log file", str(log_file))

    console.print(
        Panel(
            table,
            title="[bold cyan]pbc_datagen — parallel single-chain sweep[/]",
            border_style="cyan",
        )
    )

    logger.info(
        "Starting parallel sweep: model={} L={} n_T={} n_param={} tasks={} workers={}",
        args.model,
        args.L,
        args.n_T,
        len(param_values),
        len(tasks),
        args.threads,
    )

    t0 = time.perf_counter()
    results: list[tuple[float, float, str | None, str | None]] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    bar = progress.add_task("Running tasks", total=len(tasks))

    def _build_display() -> Group:
        """Build a Rich renderable: log panel on top, progress bar below."""
        if _LOG_LINES:
            log_text = Text("\n".join(_LOG_LINES))
        else:
            log_text = Text("(waiting for output...)", style="dim")
        log_panel = Panel(
            log_text,
            title="[bold]Log (last 20 lines)[/]",
            border_style="dim",
            height=22,
        )
        return Group(log_panel, progress)

    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(
                _run_one,
                T,
                p,
                args.model,
                args.L,
                args.n_snapshots,
                str(output_dir),
                args.new,
                str(log_file),
            ): (T, p)
            for T, p in tasks
        }

        with Live(_build_display(), console=console, refresh_per_second=4) as live:
            for fut in as_completed(futures):
                T_key, p_key = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    # Pool-level failure (e.g. worker killed by OOM killer)
                    results.append((T_key, p_key, None, str(exc)))
                    logger.error("Worker crashed T={:.4f} param={:.4f}: {}", T_key, p_key, exc)
                else:
                    results.append(result)
                    _, _, path, err = result
                    if path is not None:
                        logger.info(
                            "OK   T={:.4f} param={:.4f} → {}",
                            result[0],
                            result[1],
                            path,
                        )
                    else:
                        logger.warning(
                            "SKIP T={:.4f} param={:.4f}: {}",
                            result[0],
                            result[1],
                            err,
                        )
                progress.advance(bar)
                live.update(_build_display())

    elapsed = time.perf_counter() - t0

    # --- Summary ---
    n_ok = sum(1 for r in results if r[2] is not None)
    n_fail = len(results) - n_ok

    if n_fail:
        fail_table = Table(title="Failed Tasks", show_header=True, header_style="bold")
        fail_table.add_column("T", style="cyan", justify="right")
        fail_table.add_column("param", style="cyan", justify="right")
        fail_table.add_column("error")
        for T_r, p_r, _, err_r in sorted(results, key=lambda r: (r[0], r[1])):
            if err_r is not None:
                fail_table.add_row(f"{T_r:.4f}", f"{p_r:.4f}", err_r)
        console.print(fail_table)

    console.print(
        f"\n[bold]Done[/] in {elapsed:.1f}s — "
        f"[green]{n_ok} OK[/] / [red]{n_fail} FAILED[/] of {len(tasks)} tasks"
    )
    logger.info("Sweep complete in {:.1f}s: {} OK, {} FAILED", elapsed, n_ok, n_fail)

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
