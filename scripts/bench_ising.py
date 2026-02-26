#!/usr/bin/env python
"""Benchmark Ising model sweep performance."""

import time

import numpy as np
import numpy.typing as npt
from pbc_datagen._core import IsingModel
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

L = 64
N_SWEEPS = 1_000_000
CHUNK = 10_000  # sweep in chunks so we can update the display
T_CRITICAL = 2.269

console.print(
    Panel(
        f"[bold]L={L}[/]  |  [bold]N={L * L:,}[/] spins  |  "
        f"[bold]{N_SWEEPS:,}[/] sweeps  |  T={T_CRITICAL} (critical point)",
        title="[bold cyan]Ising Benchmark[/]",
        border_style="cyan",
    )
)

model = IsingModel(L, seed=42)
model.set_temperature(T_CRITICAL)

# --- warmup ---
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    transient=True,
) as progress:
    progress.add_task("Warming up (1,000 sweeps)...", total=None)
    model.sweep(1000)

console.print("[green]Warmup complete.[/]\n")

# --- timed run with live display ---
n_chunks = N_SWEEPS // CHUNK

progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=40),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
)
task = progress.add_task("Sweeping...", total=n_chunks)


def make_display(energy: int = 0, mag: float = 0.0, abs_mag: float = 0.0) -> Table:
    """Build the live display table."""
    outer = Table.grid(padding=(0, 0))
    outer.add_row(progress)
    obs = Table(show_header=False, border_style="dim", pad_edge=False, box=None)
    obs.add_column("label", style="bold", min_width=10)
    obs.add_column("value", justify="right", min_width=12)
    obs.add_row("  Energy", f"[yellow]{energy:,}[/]")
    obs.add_row("  m", f"[yellow]{mag:+.4f}[/]")
    obs.add_row("  |m|", f"[yellow]{abs_mag:.4f}[/]")
    outer.add_row(obs)
    return outer


result: dict[str, npt.NDArray[np.int32 | np.float64]] = {}
t0 = time.perf_counter()

with Live(make_display(), console=console, refresh_per_second=10) as live:
    for _ in range(n_chunks):
        result = model.sweep(CHUNK)
        progress.advance(task)
        live.update(
            make_display(
                energy=int(result["energy"][-1]),
                mag=float(result["m"][-1]),
                abs_mag=float(result["abs_m"][-1]),
            )
        )

elapsed = time.perf_counter() - t0

# --- results table ---
sweeps_per_sec = N_SWEEPS / elapsed
spin_updates_per_sec = N_SWEEPS * L * L / elapsed

table = Table(title="Results", border_style="green", show_header=False)
table.add_column("Metric", style="bold")
table.add_column("Value", justify="right")

table.add_row("Elapsed", f"{elapsed:.2f} s")
table.add_row("Sweeps/sec", f"{sweeps_per_sec:,.0f}")
table.add_row("Spin-updates/sec", f"{spin_updates_per_sec:,.0f}")
table.add_row("ns/spin-update", f"{1e9 / spin_updates_per_sec:.1f}")
table.add_row("", "")
table.add_row("Final energy", f"{result['energy'][-1]}")
table.add_row("Final |m|", f"{result['abs_m'][-1]:.4f}")

console.print()
console.print(table)
