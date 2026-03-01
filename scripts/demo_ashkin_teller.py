#!/usr/bin/env python
"""Demo: Ashkin-Teller model with observable time series.

Defaults to the Ising critical temperature at the decoupled point (U=0).
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from pbc_datagen._core import AshkinTellerModel
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# --- CLI ---
parser = argparse.ArgumentParser(description="Ashkin-Teller demo with observable plots")
parser.add_argument(
    "--T", type=float, default=2.269, help="Temperature (default: 2.269, Ising critical)"
)
parser.add_argument(
    "--U", type=float, default=0.0, help="Four-spin coupling (default: 0.0, decoupled)"
)
parser.add_argument("--L", type=int, default=64, help="Lattice side length (default: 64)")
parser.add_argument(
    "--sweeps", type=int, default=1_000_000, help="Number of sweeps (default: 1000000)"
)
parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
parser.add_argument("--random", action="store_true", help="Hot-start (random) initialization")
args = parser.parse_args()

console = Console()

# --- simulation parameters ---
L = args.L
N_SWEEPS = args.sweeps
CHUNK = 10_000  # sweep in chunks so we can update the display
T = args.T
U = args.U

is_decoupled = U == 0.0
point_label = "(decoupled → two Ising)" if is_decoupled else ""
start_label = "hot start" if args.random else "cold start"

console.print(
    Panel(
        f"[bold]L={L}[/]  |  [bold]N={L * L:,}[/] spins  |  "
        f"[bold]{N_SWEEPS:,}[/] sweeps\n"
        f"T={T} , U={U}  {point_label}  ({start_label})",
        title="[bold cyan]Ashkin-Teller Demo[/]",
        border_style="cyan",
    )
)

model = AshkinTellerModel(L, seed=args.seed)
model.set_temperature(T)
model.set_four_spin_coupling(U)

if args.random:
    rng = np.random.default_rng(args.seed + 1)
    for site in range(L * L):
        model.set_sigma(site, int(rng.choice([-1, 1])))
        model.set_tau(site, int(rng.choice([-1, 1])))

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

# Preallocate arrays for all 7 observables
energy = np.empty(N_SWEEPS)
m_sigma = np.empty(N_SWEEPS)
abs_m_sigma = np.empty(N_SWEEPS)
m_tau = np.empty(N_SWEEPS)
abs_m_tau = np.empty(N_SWEEPS)
m_baxter = np.empty(N_SWEEPS)
abs_m_baxter = np.empty(N_SWEEPS)


def make_display(
    e: float = 0.0,
    ams: float = 0.0,
    amt: float = 0.0,
    amb: float = 0.0,
) -> Table:
    """Build the live display table."""
    outer = Table.grid(padding=(0, 0))
    outer.add_row(progress)
    obs = Table(show_header=False, border_style="dim", pad_edge=False, box=None)
    obs.add_column("label", style="bold", min_width=10)
    obs.add_column("value", justify="right", min_width=12)
    obs.add_row("  Energy", f"[yellow]{e:+.4f}[/]")
    obs.add_row("  |m_σ|", f"[yellow]{ams:.4f}[/]")
    obs.add_row("  |m_τ|", f"[yellow]{amt:.4f}[/]")
    obs.add_row("  |m_B|", f"[yellow]{amb:.4f}[/]")
    outer.add_row(obs)
    return outer


t0 = time.perf_counter()

with Live(make_display(), console=console, refresh_per_second=10) as live:
    for i in range(n_chunks):
        result = model.sweep(CHUNK)
        lo = i * CHUNK
        hi = lo + CHUNK
        energy[lo:hi] = result["energy"]
        m_sigma[lo:hi] = result["m_sigma"]
        abs_m_sigma[lo:hi] = result["abs_m_sigma"]
        m_tau[lo:hi] = result["m_tau"]
        abs_m_tau[lo:hi] = result["abs_m_tau"]
        m_baxter[lo:hi] = result["m_baxter"]
        abs_m_baxter[lo:hi] = result["abs_m_baxter"]
        progress.advance(task)
        live.update(
            make_display(
                e=float(result["energy"][-1]),
                ams=float(result["abs_m_sigma"][-1]),
                amt=float(result["abs_m_tau"][-1]),
                amb=float(result["abs_m_baxter"][-1]),
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
table.add_row("⟨m_σ⟩", f"{m_sigma.mean():+.4f}")
table.add_row("⟨|m_σ|⟩", f"{abs_m_sigma.mean():.4f}")
table.add_row("⟨m_τ⟩", f"{m_tau.mean():+.4f}")
table.add_row("⟨|m_τ|⟩", f"{abs_m_tau.mean():.4f}")
table.add_row("⟨m_B⟩", f"{m_baxter.mean():+.4f}")
table.add_row("⟨|m_B|⟩", f"{abs_m_baxter.mean():.4f}")

console.print()
console.print(table)

# --- 3x2 observable time series ---
console.print("\n[bold cyan]Generating plots...[/]")

# Thin to ~10k points for plotting
thin = max(1, N_SWEEPS // 10_000)
t_axis = np.arange(0, N_SWEEPS, thin)
e_thin = energy[::thin]
ms_thin = m_sigma[::thin]
mt_thin = m_tau[::thin]
mb_thin = m_baxter[::thin]
ams_thin = abs_m_sigma[::thin]
amb_thin = abs_m_baxter[::thin]

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"Ashkin-Teller  L={L}  T={T}  U={U}  {point_label}",
    fontsize=13,
    fontweight="bold",
)

sweep_label = "sweep"

# (0,0) Energy — red
ax = axes[0, 0]
ax.plot(t_axis, e_thin, linewidth=0.3, color="tab:red", alpha=0.7)
e_mean = energy.mean()
ax.axhline(e_mean, color="black", ls="--", lw=0.8, label=f"⟨E⟩={e_mean:+.4f}")
ax.set_ylabel("Energy  E / N")
ax.legend(fontsize=8)
ax.set_title("Energy density")

# (0,1) m_sigma — blue
ax = axes[0, 1]
ax.plot(t_axis, ms_thin, linewidth=0.3, color="tab:blue", alpha=0.7)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_ylabel("m_σ")
ax.set_title("Sigma magnetization")

# (1,0) m_tau — orange
ax = axes[1, 0]
ax.plot(t_axis, mt_thin, linewidth=0.3, color="tab:orange", alpha=0.7)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_ylabel("m_τ")
ax.set_title("Tau magnetization")

# (1,1) m_baxter — purple
ax = axes[1, 1]
ax.plot(t_axis, mb_thin, linewidth=0.3, color="tab:purple", alpha=0.7)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_ylabel("m_B")
ax.set_title("Baxter order parameter")

# (2,0) |m_sigma| — green
ax = axes[2, 0]
ax.plot(t_axis, ams_thin, linewidth=0.3, color="tab:green", alpha=0.7)
ams_mean = abs_m_sigma.mean()
ax.axhline(ams_mean, color="black", ls="--", lw=0.8, label=f"⟨|m_σ|⟩={ams_mean:.4f}")
ax.set_ylabel("|m_σ|")
ax.set_xlabel(sweep_label)
ax.legend(fontsize=8)
ax.set_title("Absolute sigma magnetization")

# (2,1) |m_baxter| — teal
ax = axes[2, 1]
ax.plot(t_axis, amb_thin, linewidth=0.3, color="teal", alpha=0.7)
amb_mean = abs_m_baxter.mean()
ax.axhline(amb_mean, color="black", ls="--", lw=0.8, label=f"⟨|m_B|⟩={amb_mean:.4f}")
ax.set_ylabel("|m_B|")
ax.set_xlabel(sweep_label)
ax.legend(fontsize=8)
ax.set_title("Absolute Baxter order parameter")

fig.tight_layout()
suffix = "_random" if args.random else ""
out_path = f"ashkin_teller_L{L}_T{T}_U{U}{suffix}.png"
fig.savefig(out_path, dpi=150)
console.print(f"[green]Saved → {out_path}[/]")
plt.show()
