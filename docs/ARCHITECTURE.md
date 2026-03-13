# Architecture — pbc_datagen

## Overview

Paper-quality dataset generator for independent spatial snapshots of three 2D lattice models:
**Pure Ising**, **Blume-Capel (BC)**, and **Ashkin-Teller (AT)**.

## Hybrid Architecture: Python Manager / C++ Worker

```
┌──────────────────────────────────────────────────────────────┐
│  Python Orchestrator                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────┐  │
│  │ PTEngine    │ │ PTEngine2D  │ │ Single   │ │ Orchestr.│  │
│  │ 1D KTH +   │ │ 2D spectral │ │ Chain    │ │ campaign │  │
│  │ equil+prod  │ │ +convergence│ │ Engine   │ │ manager  │  │
│  └──────┬──────┘ └──────┬──────┘ └────┬─────┘ └────┬─────┘  │
│  ┌──────┴───┐ ┌─────────┴──┐ ┌───────┴──────┐ ┌───┴──────┐ │
│  │Autocorr τ│ │ Spectral   │ │ Convergence  │ │ HDF5 I/O │ │
│  │FFT-based │ │ connectiv. │ │ Welch 2-init │ │SnapshotWr│ │
│  └──────┬───┘ └─────┬─────┘ └──────┬────────┘ └────┬─────┘ │
│         │     ┌──────┴─────────────┐│               │       │
│         │     │ Registry (ModelInfo)││               │       │
│         │     └──────┬─────────────┘│               │       │
│         └──────┬─────┘──────────────┘───────────────┘       │
│                │ pybind11                                     │
├────────────────┼────────────────────────────────────────────┤
│  C++ Backend   │                                            │
│  ┌─────────────▼──────────────────────────────────────────┐ │
│  │  _core module                                          │ │
│  │  ┌────────────┐ ┌──────────┐ ┌────────────┐            │ │
│  │  │ IsingModel │ │ BCModel  │ │  ATModel   │            │ │
│  │  │ Wolff +    │ │ Geom.    │ │ Wiseman-   │            │ │
│  │  │ Metropolis │ │ Cluster +│ │ Domany +   │            │ │
│  │  │+snapshot() │ │ Metropol.│ │ Metropolis │            │ │
│  │  │+randomize()│ │+set_param│ │+set_param  │            │ │
│  │  │            │ │+dE_dparam│ │+dE_dparam  │            │ │
│  │  │            │ │+snapshot │ │+snapshot   │            │ │
│  │  │            │ │+randomize│ │+randomize  │            │ │
│  │  └────────────┘ └──────────┘ └────────────┘            │ │
│  │  ┌────────────────────────────────────────────────┐    │ │
│  │  │ PT Engine (header-only, templated)             │    │ │
│  │  │ 1D: pt_exchange · pt_rounds · label tracking   │    │ │
│  │  │ 2D: pt_exchange_param · pt_rounds_2d           │    │ │
│  │  └────────────────────────────────────────────────┘    │ │
│  │  ┌──────────────┐ ┌────────────────────┐               │ │
│  │  │ PRNG         │ │ Flat 1D Lattice +  │               │ │
│  │  │ (imported)   │ │ PBC Neighbor Table  │               │ │
│  │  └──────────────┘ └────────────────────┘               │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## C++ Layer — Design Principles

- **Flat 1D array** for lattice memory: site `(r, c)` → index `r * L + c`. Cache-friendly.
- **Precomputed neighbor table**: `make_neighbor_table(L)` stores 4 neighbors per site in a flat `vector<int32_t>` to avoid modular arithmetic in inner loops.
- **PRNG**: Xoshiro256++ (Blackman & Vigna, 2018) vendored as header-only. `Rng` wrapper exposes `uniform()`, `rand_below()`, `jump()` for independent parallel streams.
- **No abstraction over physics**: three bespoke model structs, each with its own `sweep()`.
- **Uniform snapshot interface**: every model exposes `snapshot()` → `(C, L, L)` numpy array (owning copy, correct dtype) and `randomize(rng)` for model-appropriate random initialization. Eliminates Python-side if-else for spin collection and randomization.
- **O(1) observable caching**: each model maintains cached sums (energy, magnetization, etc.) updated incrementally by `set_spin()`, Metropolis, and Wolff. `observables()` returns a dict of all cached values without recomputation.
- **Branchless inner loops** wherever possible (lookup tables for Metropolis acceptance).
- **PT engine** (header-only, templated): replica-exchange Metropolis criterion, label-based round-trip tracking, observable streaming. Templated over model type so a single `pt_rounds<Model>()` drives all three models. Extended with `pt_rounds_2d<Model>()` for 2D parameter-space PT (T × D or T × U grids) with separate T-direction and param-direction exchanges via `pt_exchange_param()`.
- **2D PT interface**: BC and AT models expose `set_param(double)` and `dE_dparam()` methods for uniform parameter-direction exchanges. Ising has no second parameter and does not participate in 2D PT.

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

### 1D Parallel Tempering (PTEngine)

Each model is sampled via replica-exchange Monte Carlo (parallel tempering). The C++ `pt_rounds()` template runs the inner loop: sweep all replicas → attempt M random neighbor exchanges → track UP/DOWN labels for round-trip counting → optionally record per-slot observable streams. The Python `PTEngine` class drives the three-phase workflow:

1. **Phase A — Ladder Tuning** (`tune_ladder()`): KTH feedback (Katzgraber, Trebst, Huse 2006) iteratively redistributes temperatures to equalise diffusion current. Doubles sweep count each iteration until convergence (max relative ΔT < tol, f(T) linear with R² > 0.7).
2. **Phase B — Equilibration** (`equilibrate()`): Run on the locked ladder with observable tracking. Welch's t-test (first 20% vs last 20%, Bonferroni-corrected) detects drift. Doubling scheme up to `n_max`. Once equilibrated, measure τ_int from the converged batch (excluding 20% burn-in).
3. **Phase C — Production** (`produce()`): Harvest decorrelated snapshots thinned by 3×τ_max, stream to HDF5 via `SnapshotWriter`. Tracks phase-crossing seeds for reproducibility.

### 2D Parameter-Space Parallel Tempering (PTEngine2D)

For BC and AT models near first-order transitions, 1D PT fails because energy gaps block replica diffusion (see `docs/LESSONS.md`). `PTEngine2D` extends PT to a rectangular (n_T × n_P) grid of replicas, exchanging along both T and parameter (D or U) directions. The C++ `pt_rounds_2d<Model>()` drives the inner loop; Python orchestrates:

1. **Phase A — Connectivity Check** (`check_connectivity()`): Build lazy random-walk transition matrix from edge acceptance rates, check spectral gap ≥ threshold via `spectral.check_connectivity()`. Ensures the grid is connected.
2. **Phase B — Convergence** (`equilibrate()`): Two independent initializations (cold + hot), run both, compare via `convergence.convergence_check()` (Welch t-test with Bonferroni). Doubling scheme. Measure τ_int on the converged run.
3. **Phase C — Production** (`produce()`): Harvest decorrelated snapshots, stream to HDF5 with 2D slot naming (T × param).

### Single-Chain MCMC (SingleChainEngine)

A simpler alternative for well-behaved parameter points that don't need replica exchange. Single chain at one (T, param) with Welch t-test equilibration and τ_int-thinned production.

## Python Layer

- **PTEngine** (`parallel_tempering.py`): Three-phase 1D parallel tempering orchestrator (A: KTH tuning, B: equilibration, C: production). Pure-math helpers (`kth_redistribute`, `kth_check_convergence`, `welch_equilibration_check`) are stateless for testability. Model factory creates replicas with independent PRNG streams.
- **PTEngine2D** (`pt_engine_2d.py`): Three-phase 2D parameter-space PT orchestrator for BC and AT models. Uses spectral connectivity (Phase A) and two-initialization convergence (Phase B) instead of KTH tuning and single-run Welch tests.
- **SingleChainEngine** (`single_chain.py`): Single-chain MCMC runner for one (T, param) point. Welch t-test equilibration, τ_int-thinned production. Helper `run_single_campaign()` manages fresh/resume workflow.
- **Autocorrelation** (`autocorrelation.py`): FFT-based `acf_fft()` (Wiener-Khinchin), `tau_int()` via first zero crossing, `tau_int_multi()` for sweep-dict bottleneck detection.
- **Spectral** (`spectral.py`): Connectivity check for 2D PT grids. Builds lazy random-walk transition matrix from edge acceptance rates, checks spectral gap (1 − λ₂) via dense `np.linalg.eigh` (grids are small, typically 10–200 nodes). Returns `ConnectivityResult` with Fiedler vector on failure for diagnostics.
- **Convergence** (`convergence.py`): Two-initialization convergence check. Compares observable streams from independent cold-start and hot-start PT runs using block-averaged Welch t-test with Bonferroni correction. Returns `ConvergenceResult` with per-slot disagreement map.
- **Registry** (`registry.py`): Single source of truth for model metadata. `ModelInfo` dataclass holds constructor, `n_channels`, `snapshot_dtype`, `param_label`, `set_param`, and PT round functions. All engines, I/O, orchestrator, and scripts look up model info via `get_model_info(name)` instead of maintaining their own dispatch dicts. Adding a new model requires one registry entry.
- **I/O** (`io.py`): HDF5 snapshot streaming via `SnapshotWriter` context manager. Flat schema (only supported format): root-level `(M, N, C, L, L)` snapshot dataset + `(M, N)` observable datasets, indexed by slot via `slot_keys` JSON attr. Snapshot dtype is parameterized (stored as `snapshot_dtype` HDF5 attr) — not hardcoded to `int8`. ~4 h5py calls per production round. Auto-extends datasets on resume. Metadata persisted every round for crash safety. Resume state via `read_resume_state()` / `read_resume_state_2d()`. See `docs/HDF5_SCHEMAS.md` for full schema reference.
- **Orchestrator** (`orchestrator.py`): Campaign manager for sequential 1D and 2D PT runs. Handles fresh/resume workflow, file discovery, OMP thread configuration, and seed derivation. Uses registry for model validation and param labels. Entry points: `generate_dataset()` (1D) and `generate_dataset_2d()` (2D).

## Directory Layout

```
pbc_datagen/
├── CMakeLists.txt              # Build system (scikit-build-core)
├── pyproject.toml              # uv / Python project config
├── PT_FORMAT.md                # .pt dataset schema reference
├── docs/
│   ├── ARCHITECTURE.md         # This file
│   ├── PLAN.md                 # Implementation plan
│   ├── LESSONS.md              # Hard-won physics/build/testing insights
│   └── HDF5_SCHEMAS.md         # HDF5 flat schema reference
├── src/cpp/
│   ├── include/
│   │   ├── xoshiro256pp.hpp    # Vendored Xoshiro256++ PRNG engine
│   │   ├── prng.hpp            # Rng wrapper (uniform, rand_below, jump)
│   │   ├── lattice.hpp         # Flat lattice + PBC neighbor table
│   │   ├── ising.hpp           # IsingModel struct + SweepResult + observables()
│   │   ├── blume_capel.hpp     # BCModel + set_param/dE_dparam for 2D PT
│   │   ├── ashkin_teller.hpp   # ATModel + set_param/dE_dparam for 2D PT
│   │   └── pt_engine.hpp       # 1D PT (pt_rounds) + 2D PT (pt_rounds_2d) + label tracking
│   ├── ising.cpp               # Ising: Wolff, Metropolis, sweep, cached observables
│   ├── blume_capel.cpp         # Blume-Capel: Wolff, Metropolis, sweep, cached observables
│   ├── ashkin_teller.cpp       # Ashkin-Teller: embedded Wolff, Metropolis, sweep, cached observables
│   └── bindings.cpp            # pybind11 _core module (all 3 models + 1D/2D PT engine)
├── python/pbc_datagen/
│   ├── __init__.py
│   ├── _core.pyi               # Type stubs for C++ extension (PTResult, PT2DResult, models)
│   ├── orchestrator.py         # Campaign manager: generate_dataset (1D) + generate_dataset_2d
│   ├── autocorrelation.py      # FFT-based acf_fft, tau_int, tau_int_multi
│   ├── parallel_tempering.py   # PTEngine: 1D KTH ladder tuning, equilibration, production
│   ├── pt_engine_2d.py         # PTEngine2D: 2D parameter-space PT (spectral + convergence)
│   ├── single_chain.py         # SingleChainEngine: single-chain MCMC (no replica exchange)
│   ├── spectral.py             # Spectral connectivity check for 2D PT grids
│   ├── convergence.py          # Two-initialization convergence check (Welch t-test)
│   ├── registry.py             # Model registry — single source of truth for model metadata
│   └── io.py                   # HDF5 streaming I/O (SnapshotWriter, resume state)
├── tests/
│   ├── conftest.py             # Pytest config (OMP_NUM_THREADS cap)
│   ├── test_foundation.py      # PRNG + neighbor table tests
│   ├── exact_2x2.py            # Shared exact partition functions (Ising/BC/AT)
│   ├── test_snapshot_method.py  # snapshot() shape/dtype/values + randomize() tests
│   ├── test_observable_cache.py # O(1) cache consistency for all 3 models
│   ├── test_autocorrelation.py # FFT acf, τ_int for white noise & AR(1)
│   ├── test_spectral.py        # Spectral connectivity: mixing, isolated clusters
│   ├── test_convergence.py     # Two-init convergence detection
│   ├── test_pt_exchange.py     # pt_exchange single-gap acceptance
│   ├── test_pt_exchange_round.py # pt_exchange_round coverage & map consistency
│   ├── test_pt_2d_exchange.py  # 2D PT exchange (T-direction + param-direction)
│   ├── test_pt_engine_2d.py    # PTEngine2D Phase B soft failure + disagreement slots
│   ├── test_pt_labels.py       # pt_update_labels, histograms, round-trip counting
│   ├── test_pt_rounds.py       # pt_collect_obs + pt_rounds integration
│   ├── test_pt_detailed_balance.py # 2×2 chi-squared for PT (Ising/BC/AT)
│   ├── test_parallel_tempering.py  # KTH redistrib, convergence, PTEngine phases A+B
│   ├── test_produce.py         # Phase C snapshot harvesting + HDF5 layout
│   ├── test_io.py              # HDF5 I/O: groups, datasets, resume logic
│   ├── test_orchestrator.py    # Campaign file discovery, execution, resume
│   ├── test_single_chain.py    # SingleChainEngine construction + campaign
│   ├── test_e2e.py             # End-to-end: fresh + resume for all 3 engine types
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
    ├── generate_dataset.py     # Main entry point for 1D PT dataset generation
    ├── generate_single.py      # CLI entry point for single-chain MCMC generation
    ├── generate_single_parallel.py # Parallel sweep of (T, param) grid via single-chain
    ├── convert_to_pt.py        # HDF5 → .pt converter (flat schema)
    ├── plot_obs_vs_T.py        # Observable vs temperature curves from HDF5 or .pt
    └── plot_snapshots.py       # Random snapshot samples from HDF5 or .pt
```
