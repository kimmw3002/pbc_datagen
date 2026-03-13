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

Every model exposes: `L`, `spins` (writable numpy view), `set_temperature()`, `energy()`, `sweep()`, `observables()` → `dict[str, float]`.

| Model | observables() keys |
|-------|-------------------|
| Ising | `energy`, `m`, `abs_m` |
| Blume-Capel | `energy`, `m`, `abs_m`, `q` |
| Ashkin-Teller | `energy`, `m_sigma`, `abs_m_sigma`, `m_tau`, `abs_m_tau`, `m_baxter`, `abs_m_baxter` |

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

**Total: 331 tests (322 default, 9 integration-only). All pass.**

---

## Phase 5: Great Migration — Model-Agnostic Python Stack

### Problem

Adding a new C++ model (e.g., 2D XY with `float64` angles, classical Heisenberg with 3D vectors) requires touching ~16 files due to:

1. **Hardcoded `int8` dtype** — `SnapshotWriter`, production loops, tests, scripts all assume `np.int8`
2. **If-else model dispatch** — `C = 2 if ashkin_teller else 1`, spin collection via `.spins` vs `.sigma`/`.tau`, parameter init via `set_crystal_field` vs `set_four_spin_coupling` — repeated in 3 engine files
3. **Duplicated registries** — `_MODEL_CONSTRUCTORS` and `_PT_ROUNDS_FN` dicts duplicated across `parallel_tempering.py`, `pt_engine_2d.py`, `single_chain.py`, `orchestrator.py`
4. **Legacy HDF5 format** — per-group schema fallback code in `io.py` and all 3 scripts

### Goal

After migration, adding a new model requires only:
1. Write C++ model + pybind11 bindings (with `snapshot()` + `randomize()`)
2. Add one entry to a Python model registry

Everything else flows automatically.

### 5.0 — C++ `snapshot()` + `randomize()` Methods

Additive — zero breakage. Two new uniform interface methods on every model.

`snapshot()` returns `(C, L, L)` numpy array (correct dtype, owning copy). Eliminates Python-side if-else for spin collection.

`randomize(Rng&)` randomizes all spins using model-appropriate logic. Eliminates Python-side `_randomize_all()` if-else.

- [x] Step 5.0.1: `ising.hpp` + `ising.cpp` — `snapshot()` returns spins copy; `randomize()` sets each spin ±1
- [x] Step 5.0.2: `bindings.cpp` — bind `IsingModel.snapshot()` → `py::array_t<int8_t>` shape `(1, L, L)`; bind `randomize()`
- [x] Step 5.0.3: Same for `BlumeCapelModel` — `snapshot()` → `(1, L, L)` int8; `randomize()` draws from {−1, 0, +1}
- [x] Step 5.0.4: Same for `AshkinTellerModel` — `snapshot()` → `(2, L, L)` int8 (stacks σ+τ in C++); `randomize()` both layers ±1
- [x] Step 5.0.5: `_core.pyi` — add `snapshot()` and `randomize()` stubs to all 3 classes
- [x] Step 5.0.6: Test: `tests/test_snapshot_method.py` — shape, dtype, value-match for `snapshot()`; `randomize()` produces non-constant configs; observable cache correctness after `randomize()` (3 tests in `test_observable_cache.py`)
- [x] Step 5.0.7: Rebuild + full test suite (334 tests pass, 9 deselected)

### 5.1 — Model Registry

Single source of truth. New file `python/pbc_datagen/registry.py`.

```python
@dataclasses.dataclass(frozen=True)
class ModelInfo:
    name: str                    # "ising", "blume_capel", etc.
    constructor: type            # _core.IsingModel, etc.
    n_channels: int              # 1 or 2 (or 3 for Heisenberg)
    snapshot_dtype: np.dtype     # np.dtype(np.int8), np.dtype(np.float64)
    param_label: str | None      # None for Ising, "D" for BC, "U" for AT
    set_param: Callable | None   # e.g. lambda m, v: m.set_crystal_field(v)
    pt_rounds_fn: Callable       # _core.pt_rounds_ising, etc.
    pt_rounds_2d_fn: Callable | None  # None for 1D-only models
```

- [x] Step 5.1.1: Create `registry.py` with `ModelInfo`, `MODEL_REGISTRY`, `get_model_info()`, `valid_model_names()`
- [x] Step 5.1.2: Populate with ising, blume_capel, ashkin_teller entries

### 5.2 — Generalize I/O Dtype

- [ ] Step 5.2.1: `io.py` — add `snapshot_dtype: np.dtype` param to `create_datasets()` (no default — caller must specify)
- [ ] Step 5.2.2: `io.py` — widen `write_round()` annotation from `NDArray[np.int8]` to `NDArray[Any]`
- [ ] Step 5.2.3: `io.py` — store `snapshot_dtype` as HDF5 attr (string: `"int8"`, `"float64"`) for readers
- [ ] Step 5.2.4: Test: `tests/test_io.py` — add float64 round-trip test; update creation test for explicit dtype

### 5.3 — Rewire Engines to Registry + `snapshot()` + `randomize()`

- [x] Step 5.3.1: `parallel_tempering.py` — import `get_model_info`; delete `_MODEL_CONSTRUCTORS`, `_PT_ROUNDS_FN`
- [x] Step 5.3.2: `parallel_tempering.py` — rewrite `_make_replicas()` via `info.constructor` + `info.set_param`
- [x] Step 5.3.3: `parallel_tempering.py` — `PTEngine.__init__` uses `info.pt_rounds_fn`
- [x] Step 5.3.4: `parallel_tempering.py` — `produce()`: `info.n_channels`, `info.snapshot_dtype`, `replica.snapshot()`
- [x] Step 5.3.5: `pt_engine_2d.py` — delete dicts; use registry; `_randomize_all()` → `model.randomize()`
- [x] Step 5.3.6: `pt_engine_2d.py` — `produce()`: same n_channels/dtype/snapshot() changes
- [x] Step 5.3.7: `single_chain.py` — delete `_MODEL_CONSTRUCTORS`; rewrite `_make_model()` via registry
- [x] Step 5.3.8: `single_chain.py` — `produce()`: same changes
- [x] Step 5.3.9: `orchestrator.py` — delete `_VALID_MODELS`, `_PARAM_LABELS`; use `get_model_info().param_label`
- [x] Step 5.3.10: Run full test suite — all 334 tests pass

### 5.4 — Remove Per-Group HDF5 Schema Everywhere

Flat schema is the only format. Old per-group `.h5` files become unsupported.

- [ ] Step 5.4.1: `io.py` — `read_resume_state()`: remove per-group fallback; raise if `slot_keys` missing
- [ ] Step 5.4.2: `io.py` — `read_resume_state_2d()`: same
- [ ] Step 5.4.3: `io.py` — `_snapshot_count()`: remove per-group fallback
- [ ] Step 5.4.4: `scripts/convert_to_pt.py` — remove `_read_per_group_schema()` and schema-detection dispatch
- [ ] Step 5.4.5: `scripts/plot_obs_vs_T.py` — remove per-group reading branch
- [ ] Step 5.4.6: `scripts/plot_snapshots.py` — remove per-group reading branch
- [ ] Step 5.4.7: `tests/test_io.py` — remove old-format tests (`test_old_format_group_*`)
- [ ] Step 5.4.8: Run full test suite

### 5.5 — Fix Dtype in Scripts + Tests

- [ ] Step 5.5.1: `scripts/convert_to_pt.py` — read `snapshot_dtype` attr; map to torch dtype dynamically (fallback `"int8"`)
- [ ] Step 5.5.2: `tests/test_produce.py` — dtype assertion via registry instead of hardcoded `np.int8`
- [ ] Step 5.5.3: `tests/test_e2e.py` — same
- [ ] Step 5.5.4: `tests/test_single_chain.py` — same
- [ ] Step 5.5.5: `tests/test_io.py` — `_random_spins` helper accepts dtype param
- [ ] Step 5.5.6: CLI scripts (`generate_dataset.py`, `generate_single.py`, `generate_single_parallel.py`) — `VALID_MODELS` → `registry.valid_model_names()`
- [ ] Step 5.5.7: Full verify: `mypy . && ruff check . && ruff format --check . && pytest`

### 5.6 — Documentation

- [ ] Step 5.6.1: Update `docs/ARCHITECTURE.md` — registry pattern, `snapshot()`/`randomize()` interface, dtype-parameterized I/O
- [ ] Step 5.6.2: Update `docs/HDF5_SCHEMAS.md` — `snapshot_dtype` attr; remove per-group schema
- [ ] Step 5.6.3: Update `docs/LESSONS.md` — registry pattern lesson

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
