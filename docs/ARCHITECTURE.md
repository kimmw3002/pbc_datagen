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
- **Precomputed neighbor table**: `NeighborTable` stores 4 neighbors per site to avoid modular arithmetic in inner loops.
- **PRNG**: Import a header-only library (e.g. PCG or Xoshiro256++) rather than hand-rolling. Must support `jump()` for independent parallel streams.
- **No abstraction over physics**: three bespoke model structs, each with its own `sweep()`.
- **Branchless inner loops** wherever possible (lookup tables for Metropolis acceptance).

## Hybrid Update Scheme (per sweep)

Each model's `sweep()` performs:

1. **Global cluster update** — suppresses critical slowing down.
2. **Full local Metropolis sweep** — ensures ergodicity and correct ensemble.

| Model | Cluster Algorithm | Local Update |
|---|---|---|
| Ising | Wolff (spin-flip) | Metropolis with precomputed exp table |
| Blume-Capel | Geometric (Heringa & Blöte point reflection) | Metropolis over {-1, 0, +1} |
| Ashkin-Teller | Wiseman-Domany 3-step embedded Wolff (σ, τ, στ) | Metropolis for σ and τ independently |

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
│   └── PLAN.md                 # Implementation plan
├── src/cpp/
│   ├── include/
│   │   ├── prng.hpp            # PRNG (imported header-only lib)
│   │   ├── lattice.hpp         # Flat lattice + neighbor table
│   │   ├── ising.hpp           # Ising model header
│   │   ├── blume_capel.hpp     # Blume-Capel header
│   │   └── ashkin_teller.hpp   # Ashkin-Teller header
│   ├── ising.cpp               # Ising implementation
│   ├── blume_capel.cpp         # Blume-Capel implementation
│   ├── ashkin_teller.cpp       # Ashkin-Teller implementation
│   └── bindings.cpp            # pybind11 module
├── python/pbc_datagen/
│   ├── __init__.py
│   ├── orchestrator.py         # Parallel temperature manager
│   ├── autocorrelation.py      # τ_int calculation
│   ├── validation.py           # Equilibration & cluster scaling
│   └── io.py                   # HDF5/numpy disk I/O
├── tests/
│   └── ...                     # pytest test suite
└── scripts/
    └── generate_dataset.py     # Main entry point
```
