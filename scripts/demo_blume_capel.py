#!/usr/bin/env python
"""Demo: Blume-Capel model at the tricritical point with observable time series."""

import time

import matplotlib.pyplot as plt
import numpy as np
from pbc_datagen._core import BlumeCapelModel
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

# --- simulation parameters ---
L = 64
N_SWEEPS = 1_000_000
CHUNK = 10_000  # sweep in chunks so we can update the display

# Tricritical point of the 2D Blume-Capel model (square lattice, J=1)
# H = -J Σ s_i s_j + D Σ s_i²
# Values from Kwak et al. (2015) / Silva et al. (2006)
T_TRI = 0.609
D_TRI = 1.966

console.print(
    Panel(
        f"[bold]L={L}[/]  |  [bold]N={L * L:,}[/] spins  |  "
        f"[bold]{N_SWEEPS:,}[/] sweeps\n"
        f"T={T_TRI} , D={D_TRI}  (tricritical point)",
        title="[bold cyan]Blume-Capel Demo[/]",
        border_style="cyan",
    )
)

model = BlumeCapelModel(L, seed=42)
model.set_temperature(T_TRI)
model.set_crystal_field(D_TRI)

# --- warmup ---
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    transient=True,
) as progress:
    progress.add_task("Warming up (10,000 sweeps)...", total=None)
    model.sweep(10_000)

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

# Preallocate arrays for all observables
energy = np.empty(N_SWEEPS)
mag = np.empty(N_SWEEPS)
abs_mag = np.empty(N_SWEEPS)
quad = np.empty(N_SWEEPS)


def make_display(e: float = 0.0, m: float = 0.0, am: float = 0.0, q: float = 0.0) -> Table:
    """Build the live display table."""
    outer = Table.grid(padding=(0, 0))
    outer.add_row(progress)
    obs = Table(show_header=False, border_style="dim", pad_edge=False, box=None)
    obs.add_column("label", style="bold", min_width=10)
    obs.add_column("value", justify="right", min_width=12)
    obs.add_row("  Energy", f"[yellow]{e:+.4f}[/]")
    obs.add_row("  m", f"[yellow]{m:+.4f}[/]")
    obs.add_row("  |m|", f"[yellow]{am:.4f}[/]")
    obs.add_row("  Q", f"[yellow]{q:.4f}[/]")
    outer.add_row(obs)
    return outer


t0 = time.perf_counter()

with Live(make_display(), console=console, refresh_per_second=10) as live:
    for i in range(n_chunks):
        result = model.sweep(CHUNK)
        lo = i * CHUNK
        hi = lo + CHUNK
        energy[lo:hi] = result["energy"]
        mag[lo:hi] = result["m"]
        abs_mag[lo:hi] = result["abs_m"]
        quad[lo:hi] = result["q"]
        progress.advance(task)
        live.update(
            make_display(
                e=float(result["energy"][-1]),
                m=float(result["m"][-1]),
                am=float(result["abs_m"][-1]),
                q=float(result["q"][-1]),
            )
        )

elapsed = time.perf_counter() - t0

# --- results summary ---
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
table.add_row("⟨E⟩", f"{energy.mean():+.4f}")
table.add_row("⟨m⟩", f"{mag.mean():+.4f}")
table.add_row("⟨|m|⟩", f"{abs_mag.mean():.4f}")
table.add_row("⟨Q⟩", f"{quad.mean():.4f}")

console.print()
console.print(table)

# --- four-panel observable time series ---
console.print("\n[bold cyan]Generating plots...[/]")

# Thin to ~10k points for plotting (every 100th sweep)
thin = 100
t_axis = np.arange(0, N_SWEEPS, thin)
e_thin = energy[::thin]
m_thin = mag[::thin]
am_thin = abs_mag[::thin]
q_thin = quad[::thin]

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
fig.suptitle(
    f"Blume-Capel  L={L}  T={T_TRI}  D={D_TRI}  (tricritical point)",
    fontsize=13,
    fontweight="bold",
)

sweep_label = "sweep"

# Energy
ax = axes[0, 0]
ax.plot(t_axis, e_thin, linewidth=0.3, color="tab:red", alpha=0.7)
e_mean = energy.mean()
ax.axhline(e_mean, color="black", ls="--", lw=0.8, label=f"⟨E⟩={e_mean:+.4f}")
ax.set_ylabel("Energy  E / N")
ax.legend(fontsize=8)
ax.set_title("Energy density")

# Magnetization
ax = axes[0, 1]
ax.plot(t_axis, m_thin, linewidth=0.3, color="tab:blue", alpha=0.7)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_ylabel("Magnetization  m")
ax.set_title("Magnetization density")

# |m|
ax = axes[1, 0]
ax.plot(t_axis, am_thin, linewidth=0.3, color="tab:green", alpha=0.7)
am_mean = abs_mag.mean()
ax.axhline(am_mean, color="black", ls="--", lw=0.8, label=f"⟨|m|⟩={am_mean:.4f}")
ax.set_ylabel("|m|")
ax.set_xlabel(sweep_label)
ax.legend(fontsize=8)
ax.set_title("Absolute magnetization")

# Quadrupole
ax = axes[1, 1]
ax.plot(t_axis, q_thin, linewidth=0.3, color="tab:purple", alpha=0.7)
q_mean = quad.mean()
ax.axhline(q_mean, color="black", ls="--", lw=0.8, label=f"⟨Q⟩={q_mean:.4f}")
ax.set_ylabel("Q = (1/N) Σ sᵢ²")
ax.set_xlabel(sweep_label)
ax.legend(fontsize=8)
ax.set_title("Quadrupole (vacancy order parameter)")

fig.tight_layout()
out_path = "blume_capel_tricritical.png"
fig.savefig(out_path, dpi=150)
console.print(f"[green]Saved → {out_path}[/]")
plt.show()
