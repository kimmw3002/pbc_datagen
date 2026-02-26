# Implementation Plan — pbc_datagen

## Phase 1: C++ Backend & Hybrid Update Kernels

### 1.0 Foundation ✅

- [x] Step 1.0.1: Project scaffold — `src/cpp/`, `python/`, `tests/`, CMakeLists, pyproject.toml
- [x] Step 1.0.2: `prng.hpp` — Xoshiro256++ wrapped in `Rng` class
- [x] Step 1.0.3: `lattice.hpp` — flat 1D lattice + precomputed PBC neighbor table

### 1.1 Ising Model ✅

- [x] Step 1.1.1: `ising.hpp` + `ising.cpp` — IsingModel struct, constructor, set_temperature, energy, magnetization, abs_magnetization, set_spin
- [x] Step 1.1.2: `_wolff_step()` — Wolff single-cluster update (DFS with explicit stack, `std::vector<bool>` for in_cluster)
- [x] Step 1.1.3: `_metropolis_sweep()` — N random-site proposals with precomputed exp table (ΔE ∈ {-8,-4,0,+4,+8})
- [x] Step 1.1.4: `sweep(n_sweeps)` — Metropolis + Wolff repeated n times, returns dict of observable arrays (energy, m, abs_m). Bound via lambda in bindings.cpp that copies vectors into numpy arrays.

Tests: `tests/ising/` — test_model.py, test_wolff.py, test_metropolis.py, test_sweep.py. All include 2×2 exact partition function chi-squared checks for detailed balance.

### 1.2 Blume-Capel Model
- [x] Step 1.2.1: `blume_capel.hpp` + `blume_capel.cpp` — BlumeCapelModel struct, constructor, set_temperature, set_crystal_field, energy, magnetization, abs_magnetization, quadrupole, set_spin
- [ ] Step 1.2.2: Geometric Cluster Algorithm (Heringa & Blöte point reflection)
- [ ] Step 1.2.3: Local Metropolis sweep over {-1, 0, +1} with crystal field D
- [ ] Step 1.2.4: `sweep()` = GCA + Metropolis, observable tracking (E, M, Q, cluster size)

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

- [x] Unit: PRNG smoke test (determinism, range, uniformity, autocorrelation) — `tests/test_foundation.py`
- [x] Unit: Neighbor table correctness for various L (shape, PBC, symmetry) — `tests/test_foundation.py`
- [x] Unit: Ising model construction, energy, magnetization — `tests/ising/test_model.py`
- [x] Unit: Ising Wolff detailed balance (2×2 chi-squared at 10 temperatures) — `tests/ising/test_wolff.py`
- [x] Unit: Ising Metropolis detailed balance (2×2 chi-squared at 10 temperatures) — `tests/ising/test_metropolis.py`
- [x] Unit: Ising sweep detailed balance + ergodicity — `tests/ising/test_sweep.py`
- [ ] Unit: BC energy/magnetization consistency after sweep
- [ ] Unit: AT energy/magnetization consistency after sweep
- [ ] Integration: Full pipeline — equilibrate, measure τ_int, generate snapshots

## 2×2 Exact Partition Function (key validation tool)

The 2×2 Ising model has 2⁴ = 16 states and only 3 distinct energy levels:

| E | Degeneracy g(E) | States |
|---|---|---|
| -8 | 2 | all +1, all -1 |
| 0 | 12 | states with exactly 2 spins up (not checkerboard) |
| +8 | 2 | checkerboard patterns |

Z(T) = 2exp(8/T) + 12 + 2exp(-8/T)

This gives exact P(E) at any T, which we can compare against a histogram
from many sweeps.  This is the gold standard for verifying detailed balance.
