# Architecture — pbc_datagen

## Overview

Paper-quality dataset generator for independent spatial snapshots of three 2D lattice models:
**Pure Ising**, **Blume-Capel (BC)**, and **Ashkin-Teller (AT)**.

## Hybrid Architecture: Python Manager / C++ Worker

```
┌─────────────────────────────────────────────────┐
│  Python Orchestrator (multiprocessing)           │
│  ┌───────────┐  ┌──────────────┐  ┌──────────┐  │
│  │ Temp array │  │ Autocorr τ   │  │ HDF5 I/O │  │
│  │ scheduling │  │ analysis     │  │ snapshots│  │
│  └─────┬─────┘  └──────┬───────┘  └─────┬────┘  │
│        │               │                │        │
│        └───────────┬───┘────────────────┘        │
│                    │ pybind11                     │
├────────────────────┼─────────────────────────────┤
│  C++ Backend       │                             │
│  ┌─────────────────▼───────────────────────────┐ │
│  │  _core module                               │ │
│  │  ┌────────────┐ ┌──────────┐ ┌────────────┐ │ │
│  │  │ IsingModel │ │ BCModel  │ │  ATModel   │ │ │
│  │  │ Wolff +    │ │ Geom.    │ │ Wiseman-   │ │ │
│  │  │ Metropolis │ │ Cluster +│ │ Domany +   │ │ │
│  │  │            │ │ Metropol.│ │ Metropolis │ │ │
│  │  └────────────┘ └──────────┘ └────────────┘ │ │
│  │  ┌──────────────┐ ┌────────────────────┐    │ │
│  │  │ PRNG         │ │ Flat 1D Lattice +  │    │ │
│  │  │ (imported)   │ │ PBC Neighbor Table  │    │ │
│  │  └──────────────┘ └────────────────────┘    │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## C++ Layer — Design Principles

- **Flat 1D array** for lattice memory: site `(r, c)` → index `r * L + c`. Cache-friendly.
- **Precomputed neighbor table**: `make_neighbor_table(L)` stores 4 neighbors per site in a flat `vector<int32_t>` to avoid modular arithmetic in inner loops.
- **PRNG**: Xoshiro256++ (Blackman & Vigna, 2018) vendored as header-only. `Rng` wrapper exposes `uniform()`, `rand_below()`, `jump()` for independent parallel streams.
- **No abstraction over physics**: three bespoke model structs, each with its own `sweep()`.
- **Branchless inner loops** wherever possible (lookup tables for Metropolis acceptance).

## Hybrid Update Scheme (per sweep)

Each model's `sweep()` performs:

1. **Global cluster update** — suppresses critical slowing down.
2. **Full local Metropolis sweep** — ensures ergodicity and correct ensemble.

| Model | Cluster Algorithm | Local Update |
|---|---|---|
| Ising | Wolff (spin-flip) | Metropolis with precomputed exp table |
| Blume-Capel | Wolff (adapted for 3-state; vacancies block cluster growth) | Metropolis over {-1, 0, +1} |
| Ashkin-Teller | Embedded Wolff (Wiseman-Domany 1995): pick σ or τ, J_eff = J + U·fixed; auto-remap to (σ, s=στ) basis when U > 1 | Metropolis for σ and τ independently (2N proposals) |

## Python Layer

- **Orchestrator**: `multiprocessing` pool over independent temperatures.
- **Autocorrelation**: Integrated autocorrelation time τ_int via FFT-based estimator.
- **Validation**: Equilibration trace plots (E, M), cluster scaling checks (⟨n⟩ ~ L^{y_h}).
- **I/O**: HDF5 snapshots with metadata (T, L, model params, τ_int, seed).

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
│   │   ├── ising.hpp           # IsingModel struct + SweepResult
│   │   ├── blume_capel.hpp     # BlumeCapelModel struct + SweepResult
│   │   └── ashkin_teller.hpp   # AshkinTellerModel struct + SweepResult
│   ├── ising.cpp               # Ising: Wolff, Metropolis, sweep
│   ├── blume_capel.cpp         # Blume-Capel: Wolff, Metropolis, sweep
│   ├── ashkin_teller.cpp       # Ashkin-Teller: embedded Wolff, Metropolis, sweep
│   └── bindings.cpp            # pybind11 _core module (all 3 models)
├── python/pbc_datagen/
│   ├── __init__.py
│   ├── _core.pyi               # Type stubs for C++ extension (mypy)
│   ├── orchestrator.py         # Parallel temperature manager (stub)
│   ├── autocorrelation.py      # τ_int calculation (stub)
│   ├── validation.py           # Equilibration & cluster scaling (stub)
│   └── io.py                   # HDF5/numpy disk I/O (stub)
├── tests/
│   ├── test_foundation.py      # PRNG + neighbor table tests
│   ├── exact_2x2.py            # Shared exact partition functions (Ising/BC/AT)
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
    ├── demo_blume_capel.py     # BC model demo
    └── generate_dataset.py     # Main entry point (stub)
```
