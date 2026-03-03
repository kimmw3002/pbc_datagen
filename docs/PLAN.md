# Implementation Plan — pbc_datagen

## Phase 1: C++ Backend & Hybrid Update Kernels ✅

### 1.0 Foundation ✅

`src/cpp/include/prng.hpp`, `lattice.hpp` — Xoshiro256++ PRNG, flat 1D lattice with precomputed PBC neighbor table.

### 1.1 Ising Model ✅

`src/cpp/ising.hpp` + `ising.cpp` — Wolff cluster + Metropolis hybrid sweep. 3 observables: energy, m, |m|.
Tests: `tests/ising/` — 2×2 exact partition function chi-squared checks.

### 1.2 Blume-Capel Model ✅

`src/cpp/blume_capel.hpp` + `blume_capel.cpp` — Wolff (vacancies block growth) + Metropolis over {-1,0,+1}. 4 observables: energy, m, |m|, q.
Tests: `tests/blume_capel/` — 2×2 exact (81 states) chi-squared + ergodicity via Welch's t-test.

### 1.3 Ashkin-Teller Model ✅

`src/cpp/ashkin_teller.hpp` + `ashkin_teller.cpp` — Embedded Wolff (Wiseman & Domany 1995) with auto σ,τ→σ,s remapping when U>1. 7 observables.
Tests: `tests/ashkin_teller/` — 2×2 exact (256 states) chi-squared + σ-τ symmetry.

### 1.4 pybind11 Bindings ✅

`src/cpp/bindings.cpp` — All three models bound. Type stubs in `_core.pyi`.

### 1.4.1 O(1) Observable Caching ✅

Incremental cache updates on all mutation paths. Tests: `tests/test_observable_cache.py` (14 tests).

### 1.5 C++ PT Inner Loop ✅

`src/cpp/include/pt_engine.hpp` — 7 composable functions: `pt_exchange`, `pt_exchange_round`, `pt_update_labels`, `pt_accumulate_histograms`, `pt_count_round_trips`, `pt_collect_obs`, `pt_rounds`.

Key decisions: `r2t`/`t2r`/`labels` mutated in-place via reference (not in PTResult dict). `PTResult` contains only output-only fields: `n_accepts`, `n_attempts`, `n_up`, `n_down`, `round_trip_count`, `obs_streams`.

Tests: `tests/test_pt_exchange.py` (3), `test_pt_exchange_round.py` (4), `test_pt_labels.py` (5), `test_pt_rounds.py` (6), `test_pt_detailed_balance.py` (4 integration) — 22 tests total.

## Model Interface

Every model exposes: `L`, `spins` (writable numpy view), `set_temperature()`, `energy()`, `sweep()`, `observables()` → `dict[str, float]`.

`observables()` is the unified API for reading all observable values from a model.
Bound via pybind11 lambda that converts C++ `vector<pair<string, double>>` → Python dict.

`sweep()` is for standalone single-chain MCMC. PT uses `pt_rounds()` (C++ composition loop), which calls `sweep(1)` + exchange + histograms internally, and reads observables via `observables()`.

| Model | observables() keys |
|-------|-------------------|
| Ising | `energy`, `m`, `abs_m` |
| Blume-Capel | `energy`, `m`, `abs_m`, `q` |
| Ashkin-Teller | `energy`, `m_sigma`, `abs_m_sigma`, `m_tau`, `abs_m_tau`, `m_baxter`, `abs_m_baxter` |

PT manager and τ_int iterate over ALL keys — never hardcode observable names.

## Phase 2: Orchestration Pipeline

### Architecture Overview

Three-phase pipeline per Hamiltonian parameter value. Parallelism across param values (embarrassingly parallel).

```
 Phase A (tune) → Phase B (equil.) → Phase C (produce)
   geometric T      locked ladder       3×τ_max thinning
   → KTH feedback   → Welch t-test      → HDF5 streaming
   → lock ladder    → lock τ_max
```

### 2.0 Autocorrelation Utility ✅

`python/pbc_datagen/autocorrelation.py` — `acf_fft()`, `tau_int()`, `tau_int_multi()`.
Tests: `tests/test_autocorrelation.py` (7 tests).

### 2.1 Parallel Tempering Orchestration ✅

`python/pbc_datagen/parallel_tempering.py`

- [x] Step 2.1.1: `PTEngine.__init__` — geometric T ladder, model factory, address maps, C++ function dispatch
- [x] Step 2.1.2: `tune_ladder()` — Phase A: KTH feedback (smoothed df/dT → η → CDF inversion), doubling N_sw, damped update, convergence check (T stability + f(T) linearity R²>0.99), post-tuning acceptance rate safety
- [x] Step 2.1.3: `equilibrate()` — Phase B: locked ladder, doubling Welch t-test (Bonferroni-corrected), τ_int measurement via `tau_int_multi` on last 80% of converged batch
- [x] Step 2.1.4: `produce()` — Phase C: snapshot harvesting loop, call `pt_rounds()` for `3×τ_max` rounds between snapshots, read spins + observables from `replicas[t2r[t]]`, stream to HDF5, resume-safe

Pure helper functions: `kth_redistribute()`, `kth_check_convergence()`, `welch_equilibration_check()`.

Tests: `tests/test_parallel_tempering.py` (15 tests: 4 KTH math, 3 convergence, 2 tune_ladder integration, 3 Welch check, 3 equilibrate), `tests/test_produce.py` (14 tests: 1 precondition, 4 layout, 2 integrity, 3 metadata, 4 resume).

#### Phase C — Production

- **Stopping condition:** collect `n_snapshots` per temperature slot (default 100).
- **Thinning rule:** save one snapshot every `max(1, 3 × τ_max)` sweeps.
- **Stream to HDF5:** append immediately, flush after each write.
- **Resume = read the HDF5.** Last snapshot per T slot = replica state.
- Track round-trip times. Warn if < 10 round trips completed.

---

### 2.2 Orchestrator & Param-Level Parallelism ✅

File: `python/pbc_datagen/orchestrator.py`

Top-level coordinator. Each Hamiltonian parameter value (D, U, etc.) runs an
independent PT campaign — embarrassingly parallel across `max_workers` cores.

```python
def find_existing_hdf5(output_dir, model_type, L, param_value, T_range, n_replicas):
    """Find the most recent HDF5 file for this exact config."""
    # Glob encodes model, L, param, T-range, and replica count
    label = _param_label(model_type)  # "D", "U", or None
    T_seg = f"T={T_range[0]:.4f}-{T_range[1]:.4f}_R{n_replicas}"
    if label is not None:
        pattern = f"{model_type}_L{L}_{label}={param_value:.4f}_{T_seg}_*.h5"
    else:
        pattern = f"{model_type}_L{L}_{T_seg}_*.h5"
    ...

def run_campaign(model_type, L, param_value, T_range, n_replicas,
                 n_snapshots, output_dir, force_new):
    """Run one PT campaign for a single param value. Called by worker."""
    existing = None if force_new else find_existing_hdf5(...)
    if existing:
        # Resume: derive fresh seed, load state
        seed, state = read_resume_state(existing)
        seed = hash(seed, state.n_snapshots_completed)  # deterministic new seed
        path = existing
    else:
        # Clean start: new timestamp, new file
        ts = int(time.time() * 1000)
        seed = ts % 2**63
        label = _param_label(model_type)  # None for Ising
        if label is not None:
            path = f"{output_dir}/{model_type}_L{L}_{label}={param_value:.4f}_{ts}.h5"
        else:
            path = f"{output_dir}/{model_type}_L{L}_{ts}.h5"
    engine = PTEngine(model_type, L, param_value, T_range, n_replicas, seed)
    engine.tune_ladder()                 # Phase A
    engine.equilibrate()                 # Phase B → locks τ_max
    engine.produce(path, n_snapshots)    # Phase C → streams to HDF5

def generate_dataset(
    model_type, L, param_values, T_range, n_replicas=20, n_snapshots=100,
    max_workers=4, output_dir="output/", force_new=False,
):
    """Distribute param values across workers."""
    with Pool(max_workers) as pool:
        pool.starmap(run_campaign, [
            (model_type, L, p, T_range, n_replicas, n_snapshots,
             output_dir, force_new)
            for p in param_values
        ])
```

Since pybind11 model objects are NOT picklable, each worker constructs its own
models internally. Workers receive scalar arguments only.

**Seeding:** on clean start, seed = ms timestamp (same as filename). On resume, `new_seed = hash(old_seed, n_snapshots_completed)`. Deterministic: same resume point → same continuation.

**Seed history:** HDF5 attr `seed_history: list[(int, int)]` — each entry is `(n_snapshots_at_start, seed)`. First entry is `(0, initial_seed)`. Each resume appends `(n_existing_snapshots, derived_seed)`. Enables full replay of which seed produced which snapshots.

**File naming:** `{model_type}_L{L}_{label}={param:.4f}_T={Tmin:.4f}-{Tmax:.4f}_R{n_replicas}_{timestamp_ms}.h5` for BC/AT. Ising has no tunable parameter: `ising_L{L}_T={Tmin:.4f}-{Tmax:.4f}_R{n_replicas}_{timestamp_ms}.h5`. T-range and replica count are encoded in the filename so `find_existing_hdf5` only matches files with the exact same config.

#### Resume

Default behaviour is resume. Each param's HDF5 file IS its checkpoint.
`snapshots in HDF5 < n_snapshots requested?` → resume. `>= n_snapshots` → skip. `--new` → fresh file.

#### Steps

- [x] Step 2.2.1: `generate_dataset()` — distribute param values across workers via `multiprocessing.Pool`
- [x] Step 2.2.2: `run_campaign()` — single-param entry point: construct PTEngine, run A→B→C
- [x] Step 2.2.3: Resume logic — scan HDF5 for completed/in-progress params, skip or restore

---

### 2.3 I/O ✅

File: `python/pbc_datagen/io.py`

#### HDF5 Streaming Writes

**HDF5 layout (one file per param value):**
```
blume_capel_L64_D=1.5000_1709312456789.h5
├── .attrs                              # seed, seed_history, model_type, L,
│                                       # param_value, locked T ladder, τ_max,
│                                       # address_map, round_trip_stats, "complete"
├── T=2.269/
│   ├── snapshots                       # (N, C, L, L) int8, resizable axis 0
│   ├── energy                          # (N,) float64, resizable
│   ├── m                               # (N,) float64, resizable
│   ├── abs_m                           # (N,) float64, resizable
│   └── .attrs                          # per-slot τ_int
├── T=2.300/
│   └── ...
└── ...
```

**Snapshot shape:** Ising/BC: `C=1` → `(N, 1, L, L)` int8. AT: `C=2` → `(N, 2, L, L)` int8.

**Observables:** each snapshot is paired with its observable values, stored as
separate `(N,)` float64 datasets keyed by observable name. The model's
`observables()` keys become HDF5 dataset names (e.g. `energy`, `abs_m`, `q`).
No need to recompute from saved configurations.

**Streaming protocol:**
1. `create_temperature_slot(path, T, L, C, obs_names)` → resizable snapshot dataset + one `(N,)` dataset per observable
2. `append_snapshot(path, T, spins, obs_dict)` → resize all datasets, write snapshot + each observable value
3. `file.flush()` after each append
4. Update attrs after each snapshot batch
5. Set `"complete": True` when done

#### Steps

- [x] Step 2.3.1: `SnapshotWriter` class — open/create HDF5, create groups, streaming append with flush
- [x] Step 2.3.2: `write_param_attrs()` — save T ladder, τ_max, address map as HDF5 attrs (updated each snapshot batch)
- [x] Step 2.3.3: `read_resume_state()` — load last snapshot per T slot + attrs for resume

---

### 2.4 CLI Entry Point ✅

File: `scripts/generate_dataset.py`

Simple argparse wrapper around `generate_dataset()`. `--new` flag for fresh start.

```
python scripts/generate_dataset.py \
    --model blume_capel --L 64 \
    --params 0.0 0.5 1.0 1.5 1.965 \
    --T-range 1.5 4.0 \
    --n-replicas 20 --n-snapshots 100 \
    --workers 4 --output-dir output/ \
    --new   # optional: ignore existing files, start fresh with new timestamp
```

- [x] Step 2.4.1: argparse CLI wrapping `generate_dataset()` with all parameters + `--new` flag

## Phase 3: 2D Parameter-Space Parallel Tempering

### 3.0 — Problem Statement

1D PT fails at first-order transitions. The energy gap is O(L²), exchanges are
exponentially suppressed, and the replica ladder splits into disconnected
ordered/disordered shelves that never mix.

The first-order line is a barrier in BOTH T and D directions — a naive 2D grid
doesn't automatically fix this. The grid must include D values below the
tricritical point (D_tcp) so replicas can cross the transition through the
second-order region, where no barrier exists.

2D PT converts an exponential barrier into a polynomial mixing time by routing
replicas around the first-order line endpoint.

- [x] Step 3.0.1: Document the failure mode with a 1D PT run at D near the first-order line

### 3.1 — Architecture Changes

- Remove Python `multiprocessing.Pool` (param-level parallelism)
- Add OpenMP for replica-level parallelism in C++ sweep loops
- Ising: keep 1D PT with KTH (Phases A→B→C unchanged), gains OpenMP for free
- BC/AT: new 2D PT with three new phases (A→B→C redefined below)

- [x] Step 3.1.1: Remove `multiprocessing.Pool` and `_worker_init()` from orchestrator
- [x] Step 3.1.2: Add OpenMP flags to CMakeLists.txt
- [x] Step 3.1.3: Add `py::gil_scoped_release` to pybind11 bindings for sweep/PT functions

### 3.2 — 2D Grid & Exchange

True 2D grid: `n_T × n_D` replicas, `slot(i,j) = i*n_D + j`.

- T spacing: geometric
- D spacing: linear
- Alternating T-direction and D-direction exchange rounds

2D exchange criterion:

```
Δ = (β_j − β_i)(E_a − E_b) + (D_j − D_i)(β_j·SqSum_a − β_i·SqSum_b)
```

Reduces to standard 1D criterion when `D_i = D_j`.
For AT: analogous with U and four-spin sum.

- [ ] Step 3.2.1: `PTEngine2D.__init__` — 2D grid construction, replica allocation
- [ ] Step 3.2.2: `pt_exchange_2d()` — C++ exchange with 2D criterion
- [ ] Step 3.2.3: `pt_exchange_round_2d()` — alternating T/D direction rounds
- [ ] Step 3.2.4: `pt_rounds_2d()` — composition loop for 2D grid
- [ ] Step 3.2.5: Tests: detailed balance for 2D exchange, reduces to 1D when D constant

### 3.3 — Phase A: Spectral Connectivity Check

Run a batch of exchanges, measure acceptance rate per edge (T-gaps and D-gaps).
Build M×M Markov transition matrix from measured acceptance rates. Compute
spectral gap (1 − λ₂) via scipy sparse eigensolver.

If spectral gap too small → grid has islands → FAIL with Fiedler vector
diagnostic showing which (T,D) slots are isolated.

This replaces naive "all gaps > 10%" which misses collective barriers and
false-fails well-connected grids with one weak link.

- [ ] Step 3.3.1: Build acceptance-rate adjacency matrix from exchange statistics
- [ ] Step 3.3.2: Construct Markov transition matrix from adjacency
- [ ] Step 3.3.3: Compute spectral gap via `scipy.sparse.linalg.eigsh`
- [ ] Step 3.3.4: Fiedler vector diagnostic for disconnected clusters
- [ ] Step 3.3.5: Tests: fully connected grid passes, grid with dead edge fails, Fiedler identifies island

### 3.4 — Phase B: Gelman-Rubin Two-Initialization Convergence

The fundamental problem: stationarity within a metastable basin looks like
equilibration — Welch t-test passes even when stuck in one phase.

Solution: run the full 2D PT twice from independent initializations:
- Run 1: all replicas cold-start (ordered phase)
- Run 2: all replicas random-start (vacancy phase)

Compare observable distributions at each slot:
- If they agree → genuinely equilibrated
- If they disagree → those slots haven't mixed, likely near the first-order line

The disagreement map reveals the first-order transition line as a byproduct.

This is the only robust check — it distinguishes "correctly in one phase" from
"stuck in one phase".

- [ ] Step 3.4.1: Cold-start and random-start initialization routines
- [ ] Step 3.4.2: Run two independent 2D PT campaigns
- [ ] Step 3.4.3: Gelman-Rubin comparison across slots
- [ ] Step 3.4.4: Disagreement map construction (transition line detection)
- [ ] Step 3.4.5: Manual validation: two converged runs agree, stuck runs disagree at expected slots

### 3.5 — Phase C: Production

Harvest snapshots at all (T, D) grid points.

- Track phase crossings (Q transitions across 0.5) instead of geometric round
  trips. Phase crossings are the physically meaningful mixing diagnostic: has a
  replica visited both the ordered and vacancy basins?
- Warn if phase crossings are insufficient at slots near the detected transition
- Thinning: measure τ_int from Phase B data, use 3×τ_max

- [ ] Step 3.5.1: Phase crossing tracker (Q-based)
- [ ] Step 3.5.2: Production loop with 2D PT and HDF5 streaming
- [ ] Step 3.5.3: Insufficient phase crossing warnings
- [ ] Step 3.5.4: Tests: phase crossing counting, thinning rule applied

### 3.6 — Pipeline Overview (2D PT)

```
Phase A (connectivity)  →  Phase B (convergence)    →  Phase C (produce)
  geometric T/D grid        two independent runs        3×τ_max thinning
  → run exchanges           → cold-start vs hot-start   → HDF5 streaming
  → spectral gap check      → Gelman-Rubin compare      → phase crossing tracking
  → Fiedler diagnostic      → locate transition line
```

### 3.7 — OpenMP

`#pragma omp parallel for` on the replica sweep loop (both 1D and 2D). Each
replica has independent state (lattice, RNG, cache) — no shared mutable state.

Add `-fopenmp` to CMakeLists.txt, `py::gil_scoped_release` in bindings. Ising
benefits too (1D PT sweep loop parallelized).

- [x] Step 3.7.1: Add OpenMP pragmas to `pt_rounds` and `pt_rounds_2d` sweep loops
- [ ] Step 3.7.2: Thread-local RNG seeding (deterministic from base seed + thread ID)
- [ ] Step 3.7.3: Manual validation: results correct with >1 thread

### 3.8 — Orchestrator Changes

- Remove `multiprocessing.Pool` and `_worker_init()`
- BC/AT: single `PTEngine2D` covering full (T, param) grid
- Ising: single `PTEngine` (1D), no multiprocessing needed
- User specifies: `T_range`, `param_range` (D_min, D_max), `n_T`, `n_D`
- D_min MUST be below D_tcp for the second-order bridge to work
- Rename CLI `--workers` → `--threads`: same user intent ("use N cores"), but now
  sets `OMP_NUM_THREADS` instead of `Pool(max_workers)`

- [ ] Step 3.8.1: Refactor `generate_dataset()` to dispatch 1D (Ising) vs 2D (BC/AT)
- [ ] Step 3.8.2: `run_campaign_2d()` — single-param-grid entry point
- [ ] Step 3.8.3: Validate D_min < D_tcp constraint
- [x] Step 3.8.4: Rename `--workers` to `--threads`, wire to `OMP_NUM_THREADS`
- [ ] Step 3.8.5: Manual validation: orchestrator dispatches correctly by model type

### 3.9 — HDF5 Output Changes

One HDF5 file per 2D PT campaign (covers all grid points).

- Group key includes both T and D: `T={T:.4f}_D={D:.4f}/`
- Filename encodes T_range, D_range, n_T, n_D

- [ ] Step 3.9.1: Update `SnapshotWriter` for 2D group keys
- [ ] Step 3.9.2: Update filename encoding for 2D campaigns
- [ ] Step 3.9.3: Update `read_resume_state()` for 2D layout
- [ ] Step 3.9.4: Tests: HDF5 round-trip with 2D group keys, resume from 2D file

---

## Validation & Diagnostics (manual)

Validation and diagnostics will be done by hand, not via automated tests.

## Test Plan

### Phase 1 Tests ✅

55 tests across `tests/ising/`, `tests/blume_capel/`, `tests/ashkin_teller/`, `tests/test_foundation.py`, `tests/test_observable_cache.py`. All pass.

### Phase 1.5 Tests — C++ PT Inner Loop ✅

22 tests: `test_pt_exchange.py` (3), `test_pt_exchange_round.py` (4), `test_pt_labels.py` (5), `test_pt_rounds.py` (6), `test_pt_detailed_balance.py` (4 integration).

### Phase 2 Tests — Autocorrelation ✅

7 tests: `tests/test_autocorrelation.py`.

### Phase 2 Tests — PT Orchestration ✅

15 tests in `tests/test_parallel_tempering.py`:
- KTH redistribution: linear-f no-change, bottleneck concentrates, endpoints fixed, sorted output
- Convergence: converged, temps unstable, f nonlinear
- tune_ladder: converges on 4×4 Ising, aborts on absurd range
- Welch check: stationary passes, drifting fails, Bonferroni correction
- equilibrate: requires locked ladder, sets positive τ_max, temps unchanged

### Phase 2 Tests — I/O (`tests/test_io.py`) ✅

12 tests: `TestSnapshotWriterCreation` (2), `TestSnapshotWriterAppend` (4), `TestWriteParamAttrs` (2), `TestReadResumeState` (4).

- [x] Unit: `SnapshotWriter` creates correct HDF5 group hierarchy
- [x] Unit: `append_snapshot` grows dataset by 1 along axis 0, data matches
- [x] Unit: `write_param_attrs` round-trips T ladder, τ_max, address map through HDF5 attrs
- [x] Unit: `read_resume_state` loads last snapshot per T slot and restores attrs

### Phase 2 Tests — Orchestrator (`tests/test_orchestrator.py`) ✅

15 tests: `TestFindExistingHdf5` (6), `TestDeriveSeed` (2), `TestRunCampaign` (3), `TestRunCampaignResume` (4).

- [x] Unit: `find_existing_hdf5` returns newest match, ignores wrong model/L/T_range/n_replicas
- [x] Unit: `_derive_seed` is deterministic, differs with offset
- [x] Integration: fresh campaign creates HDF5 with correct layout and filename
- [x] Integration: resume reuses file, appends to target, extends seed history
