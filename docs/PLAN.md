# Implementation Plan — pbc_datagen

## Phase 1: C++ Backend & Hybrid Update Kernels

### 1.0 Foundation
- [x] Step 1.0.1: Project scaffold — directory structure, CMakeLists, pyproject.toml
- [ ] Step 1.0.2: `prng.hpp` — import a header-only PRNG library (e.g. PCG or Xoshiro256++)
- [ ] Step 1.0.3: `lattice.hpp` — flat 1D lattice + precomputed PBC neighbor table

### 1.1 Ising Model
- [ ] Step 1.1.1: `ising.hpp` + `ising.cpp` — IsingModel struct, constructor, set_temperature
- [ ] Step 1.1.2: Wolff cluster update kernel
- [ ] Step 1.1.3: Metropolis sweep with precomputed exp lookup table
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

- [ ] Unit: PRNG smoke test (verify imported library links and produces output)
- [ ] Unit: Neighbor table correctness for various L
- [ ] Unit: Ising exact results (2×2 partition function, known T_c ≈ 2.269)
- [ ] Unit: BC energy/magnetization consistency after sweep
- [ ] Unit: AT energy/magnetization consistency after sweep
- [ ] Integration: Full pipeline — equilibrate, measure τ_int, generate snapshots
