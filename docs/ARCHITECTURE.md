# Architecture — pbc_datagen

## Overview

Paper-quality dataset generator for independent spatial snapshots of three 2D lattice models:
**Pure Ising**, **Blume-Capel (BC)**, and **Ashkin-Teller (AT)**.

## Hybrid Architecture: Python Manager / C++ Worker

```
┌──────────────────────────────────────────────────────┐
│  Python Orchestrator                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐     │
│  │ PTEngine     │  │ Autocorr τ   │  │ HDF5 I/O │     │
│  │ KTH tuning + │  │ FFT-based    │  │ snapshots│     │
│  │ equilibrate  │  │ acf + τ_int  │  │  (stub)  │     │
│  └──────┬──────┘  └──────┬───────┘  └─────┬────┘     │
│         │                │                │           │
│         └────────────┬───┘────────────────┘           │
│                      │ pybind11                       │
├──────────────────────┼───────────────────────────────┤
│  C++ Backend         │                               │
│  ┌───────────────────▼─────────────────────────────┐ │
│  │  _core module                                   │ │
│  │  ┌────────────┐ ┌──────────┐ ┌────────────┐     │ │
│  │  │ IsingModel │ │ BCModel  │ │  ATModel   │     │ │
│  │  │ Wolff +    │ │ Geom.    │ │ Wiseman-   │     │ │
│  │  │ Metropolis │ │ Cluster +│ │ Domany +   │     │ │
│  │  │            │ │ Metropol.│ │ Metropolis │     │ │
│  │  └────────────┘ └──────────┘ └────────────┘     │ │
│  │  ┌──────────────────────────────────────────┐   │ │
│  │  │ PT Engine (header-only, templated)       │   │ │
│  │  │ pt_exchange · pt_rounds · label tracking │   │ │
│  │  └──────────────────────────────────────────┘   │ │
│  │  ┌──────────────┐ ┌────────────────────┐        │ │
│  │  │ PRNG         │ │ Flat 1D Lattice +  │        │ │
│  │  │ (imported)   │ │ PBC Neighbor Table  │        │ │
│  │  └──────────────┘ └────────────────────┘        │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## C++ Layer — Design Principles

- **Flat 1D array** for lattice memory: site `(r, c)` → index `r * L + c`. Cache-friendly.
- **Precomputed neighbor table**: `make_neighbor_table(L)` stores 4 neighbors per site in a flat `vector<int32_t>` to avoid modular arithmetic in inner loops.
- **PRNG**: Xoshiro256++ (Blackman & Vigna, 2018) vendored as header-only. `Rng` wrapper exposes `uniform()`, `rand_below()`, `jump()` for independent parallel streams.
- **No abstraction over physics**: three bespoke model structs, each with its own `sweep()`.
- **O(1) observable caching**: each model maintains cached sums (energy, magnetization, etc.) updated incrementally by `set_spin()`, Metropolis, and Wolff. `observables()` returns a dict of all cached values without recomputation.
- **Branchless inner loops** wherever possible (lookup tables for Metropolis acceptance).
- **PT engine** (header-only, templated): replica-exchange Metropolis criterion, label-based round-trip tracking, observable streaming. Templated over model type so a single `pt_rounds<Model>()` drives all three models.

## Hybrid Update Scheme (per sweep)

Each model's `sweep()` performs:

1. **Global cluster update** — suppresses critical slowing down.
2. **Full local Metropolis sweep** — ensures ergodicity and correct ensemble.

| Model | Cluster Algorithm | Local Update |
|---|---|---|
| Ising | Wolff (spin-flip) | Metropolis with precomputed exp table |
| Blume-Capel | Wolff (adapted for 3-state; vacancies block cluster growth) | Metropolis over {-1, 0, +1} |
| Ashkin-Teller | Embedded Wolff (Wiseman-Domany 1995): pick σ or τ, J_eff = J + U·fixed; auto-remap to (σ, s=στ) basis when U > 1 | Metropolis for σ and τ independently (2N proposals) |

## Parallel Tempering Pipeline

Each model is sampled via replica-exchange Monte Carlo (parallel tempering). The C++ `pt_rounds()` template runs the inner loop: sweep all replicas → attempt M random neighbor exchanges → track UP/DOWN labels for round-trip counting → optionally record per-slot observable streams. The Python `PTEngine` class drives the three-phase workflow:

1. **Phase A — Ladder Tuning** (`tune_ladder()`): KTH feedback (Katzgraber, Trebst, Huse 2006) iteratively redistributes temperatures to equalise diffusion current. Doubles sweep count each iteration until convergence (max relative ΔT < tol, f(T) linear with R² > 0.99).
2. **Phase B — Equilibration** (`equilibrate()`): Run on the locked ladder with observable tracking. Welch's t-test (first 20% vs last 20%, Bonferroni-corrected) detects drift. Doubling scheme up to `n_max`. Once equilibrated, measure τ_int from the converged batch (excluding 20% burn-in).
3. **Phase C — Production** (planned): Harvest decorrelated snapshots thinned by 3×τ_max, stream to HDF5.

## Python Layer

- **PTEngine** (`parallel_tempering.py`): Three-phase parallel tempering orchestrator. Pure-math helpers (`kth_redistribute`, `kth_check_convergence`, `welch_equilibration_check`) are stateless for testability. Model factory creates replicas with independent PRNG streams.
- **Autocorrelation** (`autocorrelation.py`): FFT-based `acf_fft()` (Wiener-Khinchin), `tau_int()` via first zero crossing, `tau_int_multi()` for sweep-dict bottleneck detection.
- **Validation** (`validation.py`): Equilibration trace plots (E, M), cluster scaling checks (⟨n⟩ ~ L^{y_h}). *(stub)*
- **I/O** (`io.py`): HDF5 snapshots with metadata (T, L, model params, τ_int, seed). *(stub)*

## Directory Layout

```
pbc_datagen/
├── CMakeLists.txt              # Build system (scikit-build-core)
├── pyproject.toml              # uv / Python project config
├── docs/
│   ├── ARCHITECTURE.md         # This file
│   ├── PLAN.md                 # Implementation plan
│   └── LESSONS.md              # Hard-won physics/build/testing insights
├── src/cpp/
│   ├── include/
│   │   ├── xoshiro256pp.hpp    # Vendored Xoshiro256++ PRNG engine
│   │   ├── prng.hpp            # Rng wrapper (uniform, rand_below, jump)
│   │   ├── lattice.hpp         # Flat lattice + PBC neighbor table
│   │   ├── ising.hpp           # IsingModel struct + SweepResult + observables()
│   │   ├── blume_capel.hpp     # BlumeCapelModel struct + SweepResult + observables()
│   │   ├── ashkin_teller.hpp   # AshkinTellerModel struct + SweepResult + observables()
│   │   └── pt_engine.hpp       # PT exchange, label tracking, pt_rounds<Model> template
│   ├── ising.cpp               # Ising: Wolff, Metropolis, sweep, cached observables
│   ├── blume_capel.cpp         # Blume-Capel: Wolff, Metropolis, sweep, cached observables
│   ├── ashkin_teller.cpp       # Ashkin-Teller: embedded Wolff, Metropolis, sweep, cached observables
│   └── bindings.cpp            # pybind11 _core module (all 3 models + PT engine)
├── python/pbc_datagen/
│   ├── __init__.py
│   ├── _core.pyi               # Type stubs for C++ extension (mypy)
│   ├── orchestrator.py         # Param-level parallelism + generate_dataset (stub)
│   ├── autocorrelation.py      # FFT-based acf_fft, tau_int, tau_int_multi
│   ├── parallel_tempering.py   # PTEngine: KTH ladder tuning, equilibration, production
│   ├── validation.py           # Equilibration & cluster scaling (stub)
│   └── io.py                   # HDF5/numpy disk I/O (stub)
├── tests/
│   ├── test_foundation.py      # PRNG + neighbor table tests
│   ├── exact_2x2.py            # Shared exact partition functions (Ising/BC/AT)
│   ├── test_observable_cache.py # O(1) cache consistency for all 3 models
│   ├── test_autocorrelation.py # FFT acf, τ_int for white noise & AR(1)
│   ├── test_pt_exchange.py     # pt_exchange single-gap acceptance
│   ├── test_pt_exchange_round.py # pt_exchange_round coverage & map consistency
│   ├── test_pt_labels.py       # pt_update_labels, histograms, round-trip counting
│   ├── test_pt_rounds.py       # pt_collect_obs + pt_rounds integration
│   ├── test_pt_detailed_balance.py # 2×2 chi-squared for PT (Ising/BC/AT)
│   ├── test_parallel_tempering.py  # KTH redistrib, convergence, PTEngine phases A+B
│   ├── ising/                  # Ising model tests
│   │   ├── test_model.py       # Construction, energy, magnetization
│   │   ├── test_wolff.py       # Wolff cluster kernel + detailed balance
│   │   ├── test_metropolis.py  # Metropolis sweep + detailed balance
│   │   └── test_sweep.py       # Combined sweep + ergodicity
│   ├── blume_capel/            # Blume-Capel model tests
│   │   ├── test_model.py       # Construction, energy, magnetization, quadrupole
│   │   ├── test_wolff.py       # Wolff with vacancy barriers + detailed balance
│   │   ├── test_metropolis.py  # Metropolis 3-state + 81-state chi-squared
│   │   └── test_sweep.py       # Combined sweep + ergodicity (Welch's t-test)
│   └── ashkin_teller/          # Ashkin-Teller model tests
│       ├── test_model.py       # Construction, energy, σ/τ/Baxter magnetizations
│       ├── test_wolff.py       # Embedded Wolff + remapping + detailed balance
│       ├── test_metropolis.py  # ΔE formulas + 256-state chi-squared
│       └── test_sweep.py       # Combined sweep + ergodicity + σ-τ symmetry
└── scripts/
    ├── bench_ising.py          # Ising sweep benchmark (rich progress)
    ├── demo_blume_capel.py     # BC observable time series + 2×2 panel plot
    ├── demo_ashkin_teller.py   # AT observable time series + 3×2 panel plot
    └── generate_dataset.py     # Main entry point (stub)
```
