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

### 1.4 pybind11 Bindings + Observable Caching ✅

`src/cpp/bindings.cpp` — All three models bound. Type stubs in `_core.pyi`. Incremental O(1) cache updates on all mutation paths (14 tests).

### 1.5 C++ PT Inner Loop ✅

`src/cpp/include/pt_engine.hpp` — 7 composable functions: `pt_exchange`, `pt_exchange_round`, `pt_update_labels`, `pt_accumulate_histograms`, `pt_count_round_trips`, `pt_collect_obs`, `pt_rounds`. 22 tests.

## Model Interface

Every model exposes: `L`, `spins` (writable numpy view), `set_temperature()`, `energy()`, `sweep()`, `observables()` → `dict[str, float]`, `snapshot()` → `(C, L, L)` numpy array, `randomize(rng)`.

| Model | observables() keys | snapshot shape |
|-------|-------------------|---------------|
| Ising | `energy`, `m`, `abs_m` | `(1, L, L)` int8 |
| Blume-Capel | `energy`, `m`, `abs_m`, `q` | `(1, L, L)` int8 |
| Ashkin-Teller | `energy`, `m_sigma`, `abs_m_sigma`, `m_tau`, `abs_m_tau`, `m_baxter`, `abs_m_baxter` | `(2, L, L)` int8 |

PT manager and τ_int iterate over ALL keys — never hardcode observable names.

## Phase 2: Orchestration Pipeline ✅

### 2.0 Autocorrelation Utility ✅

`python/pbc_datagen/autocorrelation.py` — `acf_fft()`, `tau_int()`, `tau_int_multi()`. 9 tests.

### 2.1 Parallel Tempering Orchestration ✅

`python/pbc_datagen/parallel_tempering.py` — PTEngine with KTH ladder tuning (Phase A), Welch t-test equilibration (Phase B), thinned production (Phase C). 29 tests.

### 2.2 Orchestrator ✅

`python/pbc_datagen/orchestrator.py` — Campaign manager, file discovery, resume logic. 15 tests.

### 2.3 I/O ✅

`python/pbc_datagen/io.py` — `SnapshotWriter` (flat schema), `read_resume_state()`, crash-safe metadata. 17 tests.

### 2.4 CLI + Single-Chain Runner ✅

`scripts/generate_dataset.py`, `python/pbc_datagen/single_chain.py`, `scripts/generate_single.py`. 23 tests.

## Phase 3: 2D Parameter-Space Parallel Tempering ✅ (mostly)

### 3.1–3.2 Architecture + 2D Grid Exchange ✅

OpenMP, `pt_rounds_2d()`, `pt_exchange_param()`, `dE_dparam()`/`set_param()` on BC/AT. 10 tests.

### 3.3 Phase A: Spectral Connectivity ✅

`python/pbc_datagen/spectral.py` — lazy random walk transition matrix, spectral gap, Fiedler diagnostic. 8 tests.

### 3.4 Phase B: Two-Initialization Convergence ✅

`python/pbc_datagen/convergence.py` — cold vs hot start comparison, disagreement map. 6 tests + 4 engine tests.

### 3.5 Phase C: Production ✅

Phase crossing tracker, 2D PT HDF5 streaming, insufficient crossing warnings.

### 3.8–3.9 Orchestrator + HDF5 ✅

1D/2D dispatch, `run_campaign_2d()`, flat `SnapshotWriter` for 2D slot keys, `read_resume_state_2d()`.

## Phase 4: Post-Processing ✅

`scripts/convert_to_pt.py`, `scripts/plot_obs_vs_T.py`, `scripts/plot_snapshots.py`. 3 E2E tests.

### 4.1 — Full Pipeline Integration Tests ✅

`tests/test_pipeline_e2e.py` — 48 integration tests covering the full pipeline for all 8 valid (model, mode) combinations.

## Phase 5: Great Migration — Model-Agnostic Python Stack ✅

### 5.0 — C++ `snapshot()` + `randomize()` Methods ✅

`src/cpp/{ising,blume_capel,ashkin_teller}.{hpp,cpp}`, `bindings.cpp` — uniform `snapshot()` → `(C, L, L)` numpy array and `randomize(rng)` on all models. Tests: `tests/test_snapshot_method.py`.

### 5.1 — Model Registry ✅

`python/pbc_datagen/registry.py` — `ModelInfo` dataclass + `MODEL_REGISTRY` dict + `get_model_info()` / `valid_model_names()`. Single source of truth replacing duplicated dispatch dicts across 4 files.

### 5.2 — Generalize I/O Dtype ✅

`io.py` — `create_datasets()` takes `snapshot_dtype` param, stores it as HDF5 attr. `write_round()` accepts any dtype. Float64 round-trip tested.

### 5.3 — Rewire Engines to Registry + `snapshot()` + `randomize()` ✅

All engines (`parallel_tempering.py`, `pt_engine_2d.py`, `single_chain.py`) and `orchestrator.py` now use `get_model_info()` for construction, PT dispatch, snapshot collection, and param labels. No more dispatch dicts.

### 5.4 — Remove Per-Group HDF5 Schema ✅

Flat schema is the only format. Removed per-group fallback from `io.py`, all 3 scripts, and tests.

### 5.5 — Fix Dtype in Scripts + Tests ✅

`convert_to_pt.py` reads `snapshot_dtype` attr for torch dtype mapping. All tests use registry for dtype assertions. CLI scripts use `registry.valid_model_names()`.

### 5.6 — Documentation ✅

- [x] Step 5.6.1: Update `docs/ARCHITECTURE.md` — registry pattern, `snapshot()`/`randomize()` interface, dtype-parameterized I/O
- [x] Step 5.6.2: Update `docs/HDF5_SCHEMAS.md` — `snapshot_dtype` attr; remove per-group schema
- [x] Step 5.6.3: Update `docs/LESSONS.md` — registry pattern lesson

### Post-Migration: Adding a New Model

```python
# 1. Write C++ model (xy.hpp, xy.cpp) with: set_temperature(), sweep(),
#    energy(), observables(), snapshot() → (1,L,L) float64, randomize()
# 2. Bind in bindings.cpp, add pt_rounds_xy instantiation
# 3. Add stub to _core.pyi
# 4. One registry entry:

MODEL_REGISTRY["xy"] = ModelInfo(
    name="xy",
    constructor=_core.XYModel,
    n_channels=1,
    snapshot_dtype=np.dtype(np.float64),
    param_label=None,
    set_param=None,
    pt_rounds_fn=_core.pt_rounds_xy,
    pt_rounds_2d_fn=None,
)
# Done. All engines, I/O, orchestrator, scripts work automatically.
```

**Total: 333 tests (276 default, 57 integration-only). All pass.**

---

## Future Work

- [ ] Manual validation: two converged 2D PT runs agree, stuck runs disagree (3.4.6)
- [ ] Tests: phase crossing counting, thinning rule (3.5.4)
- [ ] Thread-local RNG seeding — deterministic from base seed + thread ID (3.7.2)
- [ ] Manual validation: results correct with >1 thread (3.7.3)
- [ ] Validate D_min < D_tcp constraint in orchestrator (3.8.3)
- [ ] Manual validation: orchestrator dispatches correctly by model type (3.8.5)
- [ ] Tests: HDF5 round-trip with 2D group keys, resume from 2D file (3.9.4)
- [ ] 2D XY model (C++ O(2) Wolff cluster + float64 snapshots + registry entry)
- [ ] Classical Heisenberg model (3-channel float64, O(3) Wolff cluster)
