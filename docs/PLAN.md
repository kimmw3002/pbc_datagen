# Implementation Plan — pbc_datagen

## Phase 1: C++ Backend & Hybrid Update Kernels

### 1.0 Foundation
- [x] Step 1.0.1: Project scaffold — directory structure, CMakeLists, pyproject.toml
- [x] Step 1.0.2: `prng.hpp` — import Xoshiro256++ (Blackman & Vigna), wrap in `Rng` class
- [x] Step 1.0.3: `lattice.hpp` — flat 1D lattice + precomputed PBC neighbor table

### 1.1 Ising Model
- [x] Step 1.1.1: `ising.hpp` + `ising.cpp` — IsingModel struct, constructor, set_temperature
- [x] Step 1.1.2: Wolff cluster update kernel
- [x] Step 1.1.3: Metropolis sweep with precomputed exp lookup table
- [ ] Step 1.1.4: `sweep()` = Wolff + Metropolis, observable tracking (E, M, cluster size)

### 1.2 Blume-Capel Model
- [ ] Step 1.2.1: `blume_capel.hpp` + `blume_capel.cpp` — BlumeCapelModel struct, constructor
- [ ] Step 1.2.2: Geometric Cluster Algorithm (Heringa & Blöte point reflection)
- [ ] Step 1.2.3: Local Metropolis sweep over {-1, 0, +1} with crystal field D
- [ ] Step 1.2.4: `sweep()` = GCA + Metropolis, observable tracking (E, M, ρ_vac, cluster size)

### 1.3 Ashkin-Teller Model
- [ ] Step 1.3.1: `ashkin_teller.hpp` + `ashkin_teller.cpp` — AshkinTellerModel struct
- [ ] Step 1.3.2: Wiseman-Domany 3-step embedded Wolff (σ, τ, στ clusters)
- [ ] Step 1.3.3: Local Metropolis sweep for σ and τ
- [ ] Step 1.3.4: `sweep()` = 3-step Wolff + Metropolis, observable tracking

### 1.4 pybind11 Bindings
- [ ] Step 1.4.1: `bindings.cpp` — expose all three models, sweep(), observables, lattice data

## Phase 2: Python Orchestrator & Parallel Execution

- [ ] Step 2.1: `orchestrator.py` — multiprocessing pool over temperature array
- [ ] Step 2.2: `io.py` — HDF5 snapshot writer with metadata
- [ ] Step 2.3: `generate_dataset.py` — CLI entry point

## Phase 3: Rigorous Statistical Validation

- [ ] Step 3.1: `autocorrelation.py` — FFT-based integrated autocorrelation time τ_int
- [ ] Step 3.2: `validation.py` — equilibration trace plots (E, M vs sweep)
- [ ] Step 3.3: `validation.py` — cluster scaling check ⟨n⟩ ~ L^{y_h}
- [ ] Step 3.4: Thinning logic — sample snapshots at intervals ≥ 3τ_int

## Test Plan

- [x] Unit: PRNG smoke test (determinism, range, uniformity, autocorrelation)
- [x] Unit: Neighbor table correctness for various L (shape, PBC, symmetry)
- [x] Unit: IsingModel construction, energy, magnetization, |m| on cold start & 2×2 checkerboard
- [ ] Unit: Ising exact results (2×2 partition function, known T_c ≈ 2.269)
- [ ] Unit: BC energy/magnetization consistency after sweep
- [ ] Unit: AT energy/magnetization consistency after sweep
- [ ] Integration: Full pipeline — equilibrate, measure τ_int, generate snapshots

## Test Strategy for Steps 1.1.2–1.1.4

Expose internal kernels via pybind11 (underscore-prefixed) so every building
block is individually testable from pytest.  The public API stays clean —
users only see `sweep()`.

### Step 1.1.2 — Wolff cluster update (`wolff_step`)

Bind as `_wolff_step() -> int` (returns cluster size).

| Test | What it checks |
|---|---|
| **Cluster flips spins** | After `_wolff_step()` on all-+1, at least one spin is -1 |
| **Energy change is consistent** | E_before - E_after matches recomputed energy |
| **Cluster size in [1, N]** | Returned size is within valid bounds |
| **Magnetization changes by 2×cluster_size/N** | |m_before - m_after| = 2 × cluster_size / N (Wolff flips a connected cluster) |
| **High-T large clusters** | At T >> T_c, mean cluster size → O(1) (disordered, bonds rarely activate) |
| **Low-T large clusters** | At T << T_c, mean cluster size → O(N) (ordered, most bonds activate) |

### Step 1.1.3 — Metropolis sweep (`_delta_energy`, `_metropolis_sweep`)

Bind `_delta_energy(site) -> int` and `_metropolis_sweep() -> int` (returns
number of accepted flips).

| Test | What it checks |
|---|---|
| **ΔE exact values on cold start** | `_delta_energy(any_site)` on all-+1 = +8 (flipping one +1 among four +1 neighbors: ΔE = 2×1×4 = 8) |
| **ΔE on checkerboard** | `_delta_energy(site)` on 2×2 checkerboard = -8 (all neighbors are antiparallel) |
| **ΔE is O(1)** | Only depends on 4 neighbors, not system size — verify same result for L=4 and L=64 |
| **Metropolis respects detailed balance** | Run many sweeps on 2×2 lattice, histogram all 16 states, compare to exact Boltzmann weights Z⁻¹exp(-E/T). Chi-squared test against exact probabilities |
| **Acceptance rate vs temperature** | At T→∞ acceptance ≈ 50% (random), at T→0 acceptance ≈ 0% for cold start (all flips cost +8) |
| **Energy is conserved mod ΔE** | E_after = E_before + ΔE for each accepted flip (when testing single flips) |

### Step 1.1.4 — Combined sweep + observables

Bind `sweep()` (1 Wolff step + 1 full Metropolis sweep).

| Test | What it checks |
|---|---|
| **2×2 Boltzmann distribution** | Run many sweeps, verify state histogram matches exact partition function P(E) = g(E)exp(-E/T)/Z |
| **Energy in valid range** | After sweep, -2L² ≤ E ≤ +2L² |
| **Observables track state** | energy() and magnetization() are consistent with the spin array |
| **Ergodicity** | Starting from all-+1 and all-−1 at T > T_c, both reach same ⟨E⟩ within tolerance |

### The 2×2 exact partition function (key validation tool)

The 2×2 Ising model has 2⁴ = 16 states and only 3 distinct energy levels:

| E | Degeneracy g(E) | States |
|---|---|---|
| -8 | 2 | all +1, all -1 |
| 0 | 12 | states with exactly 2 spins up (not checkerboard) |
| +8 | 2 | checkerboard patterns |

Z(T) = 2exp(8/T) + 12 + 2exp(-8/T)

This gives exact P(E) at any T, which we can compare against a histogram
from many sweeps.  This is the gold standard for verifying detailed balance.
