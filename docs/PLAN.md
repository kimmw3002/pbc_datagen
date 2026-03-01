# Implementation Plan — pbc_datagen

## Phase 1: C++ Backend & Hybrid Update Kernels ✅

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

### 1.2 Blume-Capel Model ✅

- [x] Step 1.2.1: `blume_capel.hpp` + `blume_capel.cpp` — struct, constructor, observables (E, m, |m|, Q)
- [x] Step 1.2.2: `_wolff_step()` — Wolff cluster flip; vacancies block growth, D doesn't affect bonds
- [x] Step 1.2.3: `_metropolis_sweep()` — symmetric proposal over {-1,0,+1}\\{current}, `_delta_energy(site, new_spin)` used by sweep
- [x] Step 1.2.4: `sweep()` — Metropolis + Wolff, returns dict with keys `energy`, `m`, `abs_m`, `q`

Tests: `tests/blume_capel/` — test_model.py, test_wolff.py, test_metropolis.py, test_sweep.py. All include 2×2 exact partition function (81 states) chi-squared checks. Ergodicity verified via Welch's t-test (all-magnetic vs all-vacancy starts).

### 1.3 Ashkin-Teller Model ✅

`src/cpp/ashkin_teller.hpp` + `ashkin_teller.cpp` — Two coupled Ising layers with four-spin coupling U.
Embedded Wolff cluster (Wiseman & Domany, 1995) with automatic σ,τ → σ,s=στ remapping when U > 1.
Metropolis sweep in physical (σ,τ) basis. 7 observables: energy, m_σ, |m_σ|, m_τ, |m_τ|, m_B, |m_B|.

- [x] Step 1.3.1: Struct, constructor, observables, `set_four_spin_coupling` (auto-remapping)
- [x] Step 1.3.2: `_wolff_step()` — Embedded Wolff with 4 modes (σ/τ non-remapped, σ/s remapped)
- [x] Step 1.3.3: `_metropolis_sweep()` — 2N proposals (N for σ, N for τ)
- [x] Step 1.3.4: `sweep()` — Metropolis + embedded Wolff, returns dict with 7 observable arrays

Tests: `tests/ashkin_teller/` — test_model.py, test_wolff.py, test_metropolis.py, test_sweep.py. 2×2 exact partition function (256 states) chi-squared checks. σ-τ symmetry verified under remapping.

### 1.4 pybind11 Bindings ✅

`src/cpp/bindings.cpp` — All three models fully bound: constructors, properties (L, T, spins/sigma/tau), observables, internal update methods, and `sweep()` returning numpy dict. Type stubs in `_core.pyi`.

### 1.4.1 O(1) Observable Caching ✅

All three models cache observables incrementally — `energy()`, `magnetization()`, etc. return
in O(1) instead of O(N). Caches update on every mutation path: `set_spin`/`set_sigma`/`set_tau`,
`_metropolis_sweep()`, and `_wolff_step()`. Hamiltonian parameter changes (`set_crystal_field`,
`set_four_spin_coupling`) also keep caches consistent. AT stores three bond coupling sums
separately so `energy()` recomputes from `−σ_c − τ_c − U·four` when U changes.

Tests: `tests/test_observable_cache.py` — 14 tests across all 3 models × 4–5 mutation paths,
Python O(N) recomputation vs C++ cached values. Includes AT remapped (U > 1) regime.

### 1.5 C++ PT Inner Loop

`src/cpp/pt_engine.hpp` — Small, independently testable functions that each
do one thing. Composed into `pt_rounds()` which is a thin loop calling them
in order. Every function is bound via pybind11 and tested in isolation.

temperatures sorted ascending: `temps[0] = T_min` (coldest),
`temps[M-1] = T_max` (hottest). Slot 0 = cold end, slot M-1 = hot end.

Directional label convention (ints):
- `LABEL_NONE = 0` — no label yet
- `LABEL_UP = 1` — last visited T_min (cold end, slot 0)
- `LABEL_DOWN = -1` — last visited T_max (hot end, slot M-1)

#### 1.5.1 `pt_exchange` — single-gap exchange decision

```cpp
bool pt_exchange(double E_i, double E_j, double T_i, double T_j, Rng& rng);
```

Pure function. Returns true (accept) or false (reject). Metropolis criterion:
```
A = min(1, exp((1/T_i - 1/T_j) × (E_i - E_j)))
```
When `T_i == T_j`, always returns true (Δ(1/T) = 0 → exp(0) = 1).
No side effects — easy to test with exact values.

#### 1.5.2 `pt_exchange_round` — M random swap proposals

```cpp
template<typename Model>
void pt_exchange_round(
    std::vector<Model*>& replicas,       // M replicas (read energy only)
    const std::vector<double>& temps,    // M temperatures, sorted ascending
    std::vector<int>& r2t,               // replica_to_temp (mutated)
    std::vector<int>& t2r,               // temp_to_replica (mutated)
    std::vector<int>& n_accepts,         // per-gap accept count (mutated)
    std::vector<int>& n_attempts,        // per-gap attempt count (mutated)
    Rng& rng
);
```

Makes M random swap proposals (like Metropolis picks N random sites):
1. Pick a random gap `g = rng.rand_below(M - 1)`.
2. Look up replicas at slots `g` and `g+1` via `t2r`, read their energies.
3. Call `pt_exchange(E_g, E_{g+1}, temps[g], temps[g+1], rng)`.
4. On accept: swap `r2t` and `t2r` entries. Increment `n_accepts[g]`.
5. Always increment `n_attempts[g]`.
6. Repeat M times.

No parity tracking — every gap is reachable on every round.
Testable: pass known energies + temps, check map changes and counts.

#### 1.5.3 `pt_update_labels` — assign directional labels at extremes

```cpp
void pt_update_labels(
    std::vector<int>& labels,            // per-replica labels (mutated)
    const std::vector<int>& t2r,         // temp_to_replica
    int M                                // number of replicas
);
```

- `labels[t2r[0]]` = `LABEL_UP` (replica at coldest slot → released upward)
- `labels[t2r[M-1]]` = `LABEL_DOWN` (replica at hottest slot → released downward)
- All other labels unchanged.

Pure index arithmetic — trivially testable.

#### 1.5.4 `pt_accumulate_histograms` — increment diffusion counters

```cpp
void pt_accumulate_histograms(
    std::vector<int>& n_up,              // per-T-slot up counter (mutated)
    std::vector<int>& n_down,            // per-T-slot down counter (mutated)
    const std::vector<int>& labels,      // per-replica labels
    const std::vector<int>& t2r,         // temp_to_replica
    int M
);
```

For each T slot `t`: look up `labels[t2r[t]]`. If `LABEL_UP` → `n_up[t]++`.
If `LABEL_DOWN` → `n_down[t]++`. If `LABEL_NONE` → skip.

Pure index arithmetic — trivially testable with synthetic labels.

#### 1.5.5 `pt_count_round_trips` — detect completed round trips

```cpp
int pt_count_round_trips(
    const std::vector<int>& labels,      // current labels (after update)
    const std::vector<int>& prev_labels, // labels before this round's update
    const std::vector<int>& t2r,         // temp_to_replica
    int M
);
```

A round trip completes when an `UP`-labeled replica reaches T_max (slot M-1)
and gets relabeled `DOWN`. Specifically: check the replica at slot M-1 —
if `prev_labels[r] == LABEL_UP` and `labels[r] == LABEL_DOWN`, that's one
completed trip. Returns the count (0 or 1 per call, since only one replica
sits at slot M-1).

Testable with synthetic label arrays — no models needed.

#### 1.5.6 `pt_collect_obs` — record per-T-slot observables

```cpp
template<typename Model>
void pt_collect_obs(
    std::vector<std::vector<double>>& obs_streams, // [M][n_rounds_so_far]
    const std::vector<Model*>& replicas,
    const std::vector<int>& t2r,
    int M
);
```

For each T slot `t`: read `replicas[t2r[t]]->energy()`, append to
`obs_streams[t]`. Only called when `track_observables` is true.

#### 1.5.7 `pt_rounds` — thin composition loop

```cpp
template<typename Model>
PTResult pt_rounds(
    std::vector<Model*>& replicas,       // M replicas
    const std::vector<double>& temps,    // M temperatures, sorted ascending
    std::vector<int>& r2t,               // address map (mutated)
    std::vector<int>& t2r,               // inverse map (mutated)
    std::vector<int>& labels,            // directional labels (mutated)
    int n_rounds,
    Rng& rng,
    bool track_observables = false
);
```

The loop body is just calls to the functions above:
```
for round in 0..n_rounds:
    for each replica r:
        replicas[r]->set_temperature(temps[r2t[r]])
        replicas[r]->sweep(1)
    prev_labels = copy(labels)
    pt_exchange_round(replicas, temps, r2t, t2r, n_accepts, n_attempts, rng)
    pt_update_labels(labels, t2r, M)
    pt_accumulate_histograms(n_up, n_down, labels, t2r, M)
    round_trip_count += pt_count_round_trips(labels, prev_labels, t2r, M)
    if track_observables:
        pt_collect_obs(obs_streams, replicas, t2r, M)
```

Returns `PTResult` dict with all accumulated state.

Bound in `bindings.cpp` three times via template instantiation (only
`pt_exchange_round`, `pt_collect_obs`, and `pt_rounds` are templated;
the rest are plain functions):
```cpp
// Non-templated (bound once)
m.def("pt_exchange", &pt_exchange);
m.def("pt_update_labels", &pt_update_labels);
m.def("pt_accumulate_histograms", &pt_accumulate_histograms);
m.def("pt_count_round_trips", &pt_count_round_trips);

// Templated (bound per model)
m.def("pt_exchange_round_ising", ...);
m.def("pt_collect_obs_ising", ...);
m.def("pt_rounds_ising", ...);
// ... _bc and _at variants
```

#### Steps

- [x] Step 1.5.1: `pt_exchange()` — pure accept/reject function + pybind11 binding + tests
- [x] Step 1.5.2: `pt_exchange_round()` — M random gap proposals + map updates + binding + tests
- [x] Step 1.5.3: `pt_update_labels()` — label assignment at extremes + binding + tests
- [x] Step 1.5.4: `pt_accumulate_histograms()` — diffusion counter increments + binding + tests
- [x] Step 1.5.5: `pt_count_round_trips()` — round-trip detection + binding + tests
- [ ] Step 1.5.6: `pt_collect_obs()` — per-T-slot energy recording + binding + tests
- [ ] Step 1.5.7: `pt_rounds()` — thin composition loop + binding + integration test
- [ ] Step 1.5.8: Update `_core.pyi` type stubs for all new functions

## Model Interface (contract between C++ models and Python orchestration)

Every model must expose a consistent interface so that the PT manager,
autocorrelation analysis, and I/O code work identically for Ising, BC, and AT.

```python
class Model:
    L: int                                  # lattice linear size
    spins: npt.NDArray[np.int8]             # (L, L) writable view into spin array

    def set_temperature(self, T: float) -> None: ...
    def set_spin(self, site: int, value: int) -> None: ...  # single-site setter
    def energy(self) -> int | float: ...    # int for Ising, float for BC/AT
    def sweep(self, n_sweeps: int) -> dict[str, npt.NDArray]: ...
```

`spins` is a writable numpy view — `model.spins[:] = snapshot` loads a full
configuration for resume. For AT, use `model.sigma[:] = ...` and
`model.tau[:] = ...` separately (two (L, L) views).

**The `sweep()` return dict is the key unifying contract.**
Each model returns ALL its observables as named arrays:

| Model | Keys returned by sweep() |
|-------|--------------------------|
| Ising | `energy`, `m`, `abs_m` |
| Blume-Capel | `energy`, `m`, `abs_m`, `q` |
| Ashkin-Teller | `energy`, `m_sigma`, `abs_m_sigma`, `m_tau`, `abs_m_tau`, `m_baxter`, `abs_m_baxter` |

The PT manager and τ_int calculation iterate over ALL keys in the dict —
they never hardcode observable names. This means adding a new observable to a
model automatically gets picked up by thinning (bottleneck rule) without
changing any Python orchestration code.

## Phase 2: Orchestration Pipeline

### Architecture Overview

Three-phase pipeline per Hamiltonian parameter value. No pilot run — start from
geometric T spacing and let KTH tuning handle it. Parallelism is across param
values (embarrassingly parallel), not across replicas within a PT run.

```
 For each param value (e.g. D) — distributed across workers:

 ┌───────────┐    ┌───────────┐    ┌───────────┐
 │  PT       │───▶│  PT       │───▶│  PT       │
 │  Phase A  │    │  Phase B  │    │  Phase C  │
 │  (tune)   │    │  (equil.) │    │ (produce) │
 └─────┬─────┘    └──┬────┬──┘    └─────┬─────┘
       │              │    │              │
       │  locked      │    │ true τ_max   │
       │  T ladder    │    │ (for thin)   │
       ▼              │    ▼              ▼
 ┌───────────┐        │  ┌──────────┐  ┌────────┐
 │ Geometric │        └─▶│ ACF/τ_int│  │ HDF5   │
 │ T start   │           └──────────┘  │ I/O    │
 └───────────┘                         └────────┘

 Data flow summary:
 ───────────────────────────────────────────────────────────────────────────────
 Stage        Produces                    Consumed by
 ───────────────────────────────────────────────────────────────────────────────
 Phase A      tuned T array → LOCK        Phase B
 Phase B      equilibrated streams,       Phase C
              true PT τ_max → LOCK
 Phase C      thinned snapshots           HDF5 I/O
 ───────────────────────────────────────────────────────────────────────────────

 τ_int is measured ONCE — in Phase B on the locked PT ladder.
 PT injects decorrelated configs from hot replicas, so PT τ_int is the
 true production value. This τ_max is locked and used for thinning in Phase C.
```

**Orchestrator inputs:**
```python
generate_dataset(
    model_type: str,                        # "ising" | "blume_capel" | "ashkin_teller"
    L: int,                                 # lattice linear size
    param_values: list[float],              # explicit list of Hamiltonian param values
    T_range: tuple[float, float],           # (T_min, T_max)
    n_replicas: int = 20,                   # replicas per PT ladder
    n_snapshots: int = 100,                 # snapshots per temperature slot
    max_workers: int = 4,                   # CPU cores for param-level parallelism
    output_dir: str = "output/",            # directory for per-param HDF5 files
    force_new: bool = False,                # if True, ignore existing files, create new
)
```

For Ising (no Hamiltonian parameter beyond T), `param_values` is `[0.0]`.
For Blume-Capel, it lists D values. For Ashkin-Teller, U values.

**File naming:** each param value gets its own HDF5 file, timestamped:
`{output_dir}/{model_type}_L{L}_{param_label}={param:.4f}_{timestamp_ms}.h5`
Examples:
- `output/ising_L64_1709312456789.h5` (Ising has no param label)
- `output/blume_capel_L64_D=1.5000_1709312456789.h5`
- `output/ashkin_teller_L64_U=0.8000_1709312456789.h5`

The timestamp suffix makes each run unique. `--new` doesn't delete old files —
it simply ignores them and creates a new file with a fresh timestamp.
No concurrent writes — each worker owns its file exclusively.

**Seeding:** on clean start, seed = current millisecond timestamp (same one used
in the filename), stored in HDF5 root attrs as `seed`. On resume, derive a fresh
seed from the stored one: `new_seed = hash(old_seed, n_snapshots_completed)`.
This avoids replaying the same RNG sequence while staying deterministic
(same resume point → same continuation). Full RNG state save is unnecessary
because snapshots are already decorrelated by 3×τ_max thinning.

**Seed history:** HDF5 attr `seed_history: list[(int, int)]` — each entry is
`(n_snapshots_at_start, seed)`. First entry is `(0, initial_seed)`. Each resume
appends `(n_existing_snapshots, derived_seed)`. This tells you exactly which
seed produced which segment of snapshots, enabling full replay.

### Why Not On-The-Fly ACF

Computing ACF continuously during severe critical slowing down (CSD) is a statistical trap:

1. **Biased mean:** The ACF formula depends on the sample mean Ō. If the running
   window N < 10×τ_int, the simulation hasn't explored phase space — Ō is wrong.
2. **Artificial decorrelation:** A biased Ō makes ρ(t) cross zero prematurely,
   producing a falsely small τ_int. Snapshots get harvested too frequently →
   correlated garbage.

**Solution:** Always compute ACF on a frozen block of completed samples — never
on a growing window. Phase B computes ACF on the equilibrated fixed-temperature
streams after the ladder is locked. This captures the true PT dynamics (replica
exchanges shorten τ_int). The resulting τ_max is locked for Phase C thinning.

### Module Map

```
src/cpp/
└── pt_engine.hpp           # Templated PT hot loop (sweep + exchange + histograms)

python/pbc_datagen/
├── autocorrelation.py      # FFT-based ACF + τ_int
├── parallel_tempering.py   # PT orchestration (A/B/C phases, KTH feedback)
├── orchestrator.py         # Top-level coordinator + param-level parallelism
└── io.py                   # HDF5 snapshot writer
```

---

### 2.0 Autocorrelation Utility

File: `python/pbc_datagen/autocorrelation.py`

- [ ] Step 2.0.1: `acf_fft(x)` — FFT-based normalized autocorrelation function
- [ ] Step 2.0.2: `tau_int(x)` — integrated autocorrelation time via first zero crossing
- [ ] Step 2.0.3: `tau_int_multi(sweep_dict)` — τ_int for every key in a `sweep()` result dict; returns per-observable τ_int values and the bottleneck τ_max

#### ACF Math

**Autocorrelation function** for observable time series O of length N:
```
ρ(t) = Σ_{i=1}^{N-t} (O_i - Ō)(O_{i+t} - Ō)  /  Σ_{i=1}^{N} (O_i - Ō)²
```
Compute via FFT (O(N log N) instead of O(N²)):
1. Center the series: x = O - mean(O)
2. FFT: X = fft(x, n=2N)  (zero-padded to avoid circular correlation)
3. Power spectrum: S = |X|²
4. ACF = ifft(S)[:N].real / ifft(S)[0].real  (normalize so ρ(0) = 1)

**Integrated autocorrelation time:**
```
τ_int = 1/2 + Σ_{t=1}^{t_cut} ρ(t)
```
where `t_cut` = first lag where ρ(t) ≤ 0 (first zero crossing). Simple, robust, no tuning parameters.

**Bottleneck rule:** compute τ_int for EVERY observable the model reports,
then use `τ_max = max(τ_E, τ_{|M|}, τ_Q, ...)`. Thin at intervals ≥ 3×τ_max.
Near the tricritical point in BC, τ_Q (quadrupole/vacancy density) will be the bottleneck.

---

### 2.1 Parallel Tempering Orchestration

File: `python/pbc_datagen/parallel_tempering.py`

Python-side PT orchestration — Phase A/B/C logic, KTH feedback, convergence
checks. The hot inner loop (sweep + exchange + histograms) lives in C++ (§1.5).
Works with any model conforming to the **Model Interface** (see above).

#### Exchange Logic

Adjacent-only replica swaps maximize energy-distribution overlap.
Each round makes M random swap proposals: pick a random gap `g` from
`{0, ..., M-2}`, attempt exchange between slots `g` and `g+1`. Repeat M times.
Same logic as Metropolis (N random site proposals per sweep).

Metropolis acceptance criterion (for adjacent slots i and i+1, T_i < T_{i+1}):
```
ΔE = E_i - E_{i+1}
A = min(1, exp((1/T_i - 1/T_{i+1}) × ΔE))
```
Since T_i < T_{i+1}, the factor `(1/T_i - 1/T_{i+1})` is always positive.
On acceptance: swap temperature labels in the address map, NOT spin configurations.
Copying large lattice arrays is wasteful; swapping a pair of integers is O(1).

#### Address Map (Replica ↔ Temperature Permutation)

Maintain two integer arrays initialised to identity at startup:

- `replica_to_temp[r]` → which temperature index replica `r` is currently at.
- `temp_to_replica[t]` → which replica index currently sits at temperature `t`.

These are inverse permutations of each other (keep both for O(1) lookup).

On accepted swap between temperature slots `t` and `t+1`:
1. Look up replicas: `r_a = temp_to_replica[t]`, `r_b = temp_to_replica[t+1]`.
2. Swap temperature assignments: `replica_to_temp[r_a] ↔ replica_to_temp[r_b]`.
3. Update inverse map: `temp_to_replica[t] = r_b`, `temp_to_replica[t+1] = r_a`.

Each replica's spin array stays in place — only the mapping changes.

#### Fixed-Temperature Stream Tracking (not Fixed-Replica)

**Critical distinction:** observables and snapshots must be tracked at each
**temperature slot** (e.g., "whatever replica currently sits at T = 2.269"),
NOT at each physical replica.

- Use `temp_to_replica[t]` to find the current replica at slot `t`,
  then read that replica's energy/magnetisation.
- The fixed-temperature stream belongs to a single canonical ensemble.
- PT teleports decorrelated configurations from the hot end, reducing τ_int
  compared to single-chain MCMC. This is the whole point.
- **Do NOT track a specific replica across temperatures** — its observables
  change with T and don't belong to any single ensemble.

#### Round-Trip Time Tracking & Directional Labeling

Even if the fixed-temperature ACF looks small, PT may be broken: replicas
might swap locally (bouncing between T₀ and T₁) but never traverse the full
ladder. Bottleneck configurations never reach hot temperatures to melt.

**Directional labels (shared by KTH tuning and round-trip counting):**
Each replica carries a persistent directional label:
- `up`: assigned when the replica most recently visited T_min (cold end)
- `down`: assigned when the replica most recently visited T_max (hot end)
- Unlabeled at start (no label until first extreme visit)

A round trip is completed when an `up`-labeled replica reaches T_max (label
flips to `down`) and then returns to T_min (label flips back to `up`).
Phase A uses these same labels to build diffusion histograms for KTH feedback.

**Track round-trip time (t_RT):**
1. Tag each replica with a persistent ID and directional label.
2. When a replica reaches T_min, mark it `up` ("released upward").
3. Track sweep count until it reaches T_max and returns to T_min.
4. Store in HDF5 root attrs: `round_trip_count` (total completed across all
   replicas), `mean_round_trip_time`, `min_round_trip_time`, `max_round_trip_time`.

**Post-production round-trip check:** after Phase C completes all `n_snapshots`,
check total `round_trip_count`. If < 10, print a warning:

```
WARNING: Only {n} round-trips completed — snapshots may not be ergodic.
Re-run with higher --n-snapshots to collect more (resume is automatic).
```

This is a warning, not a hard error — the data is already on disk and may still
be usable. The user can simply re-run the same command with a larger
`--n-snapshots` value; resume logic sees existing snapshots < requested count
and continues producing more.

#### Phase A — Ladder Tuning (Katzgraber-Trebst-Huse 2006)

Start from geometric T spacing and optimise temperature placement to **minimise
round-trip time**, not to flatten acceptance rates. The key insight: concentrate
temperatures at thermodynamic bottlenecks (e.g. phase transitions) where replica
diffusion slows down. Acceptance rates should be *non-uniform* — smaller gaps
(more temperatures) where diffusion is hard, larger gaps where it's easy.

**Diffusion histograms:**
At each temperature slot T_i, maintain two counters:
- `n_up[i]`: times the replica at T_i carried the `up` label
- `n_down[i]`: times the replica at T_i carried the `down` label

After each round of swap moves, increment the appropriate counter at each
temperature slot based on the directional label of the replica sitting there.

**Algorithm:**
1. Start from geometric T spacing between T_min and T_max (M = `n_replicas`).
2. Set initial round count N_sw = 500.
3. Run N_sw rounds of (`sweep(1)` on all replicas + M random exchange
   proposals). `sweep(1)` already performs Metropolis + Wolff internally.
   After each round, update diffusion histograms using the directional
   labels from `RoundTripTracker`.
4. Compute the steady-state up-fraction at each temperature:
   ```
   f(T_i) = n_up[i] / (n_up[i] + n_down[i])
   ```
   Skip slots where `n_up[i] + n_down[i] = 0` (unlabeled).
   f should be monotonically decreasing: ≈1 near T_min, ≈0 near T_max.
   When converged, f(T) is approximately **linear** (constant df/dT) —
   this means the diffusion current is uniform across the ladder.

5. Estimate df/dT at each gap midpoint. Raw finite differences are noisy,
   so smooth with a 3-point windowed linear regression (fit a line through
   `(T_{i-1}, f_{i-1}), (T_i, f_i), (T_{i+1}, f_{i+1})` and take the slope;
   use one-sided at boundaries). Clamp to a positive floor to prevent zero
   or negative values from statistical fluctuations:
   ```
   ε = max(median(raw_slopes) × 0.01, 1e-6)
   df_dT[i] = max(smoothed_slope[i], ε)
   ```

6. Compute the optimised temperature density at each gap:
   ```
   η[i] = sqrt(df_dT[i] / ΔT[i])     where ΔT[i] = T[i+1] - T[i]
   ```
   This places more temperatures where df/dT is large (bottleneck regions
   where replicas struggle to diffuse) and fewer where diffusion is easy.

7. Redistribute temperatures by integrating η:
   ```
   # Cumulative distribution:
   S[0] = 0
   S[i] = S[i-1] + η[i-1] × ΔT[i-1]     for i = 1, ..., M-1
   S[i] /= S[M-1]                          # normalise to [0, 1]

   # New temperatures: invert the CDF.
   # For each k = 1, ..., M-2, find T'_k such that S(T'_k) = k/(M-1)
   # via linear interpolation on the S(T) curve.
   # T'_0 = T_min and T'_{M-1} = T_max are fixed.
   ```

8. Apply damped update to prevent oscillation:
   ```
   T_i^new = (1 - γ) × T_i^old + γ × T_i^target     (γ ≈ 0.5)
   ```

9. Reset diffusion histograms (n_up, n_down) to zero.
   Double N_sw for the next iteration — but cap doubling at iteration 10
   (N_sw maxes out at 500 × 2^10 ≈ 512k). After that, N_sw stays fixed.

10. Repeat from step 3 until convergence or max iterations.

**Defaults:** N_sw_initial = 500, max_iterations = 100, γ = 0.5.

**Convergence criterion (both must hold):**
1. **Temperatures fixed:** `max_i |T_i^new - T_i^old| / T_i^old < tol` (tol ≈ 0.01)
2. **f(T) linear:** R² of a linear fit to f(T_i) vs T_i exceeds 0.99

Or `max_iterations` reached (hard failure).

**Hard failure (either triggers abort):**
1. **Failed to converge:** `max_iterations` reached without meeting both
   convergence conditions. The ladder cannot stabilise.
2. **Under-resolved ladder:** after tuning completes, check swap acceptance
   rates at each gap. If ANY gap has acceptance rate < 10%, abort.
   Too few replicas spanning too large a T range.

Both are hard stops — the user must re-run with more replicas or a narrower
T range. Continuing would produce correlated garbage with broken diffusion.

#### Phase B — Equilibration & τ_int Measurement

- **Ladder is LOCKED.** No more temperature adjustments. Adjusting temperatures
  during production violates detailed balance and invalidates the dataset.

**Doubling equilibration scheme:**
1. Set `N = 100_000` sweeps.
2. Call `pt_rounds(..., n_rounds=N, track_observables=True)`. The C++ loop
   records per-T-slot observable streams and returns them in `PTResult`.
3. For every observable key, at every temperature slot: Welch's t-test
   comparing the first 20% vs last 20% of the batch.
   Bonferroni-corrected threshold: `α = 0.05 / n_tests` where
   `n_tests = n_replicas × n_observables` (e.g., 20 × 3 = 60 → α ≈ 0.0008).
   This prevents false failures from multiple testing.
4. If ALL tests pass (p > α for every observable × every T slot) →
   equilibrated. Proceed to τ_int measurement.
5. If ANY test fails → double `N`, run a new batch of `2N` sweeps
   (simulation state carries forward, old batch discarded for testing),
   go to step 3.
6. Cap at a maximum (e.g., `N = 6_400_000`). If still not converged,
   raise an error and abort — the system cannot equilibrate, likely
   needs more replicas or a narrower T range.

**Measure true PT τ_int.** Once equilibrated, compute `tau_int_multi` on
the last converged batch (excluding first 20% as burn-in margin). This
captures the real PT autocorrelation (shortened by replica exchanges).
The resulting `τ_max` is locked and passed to Phase C for thinning.

#### Phase C — Production

- **Stopping condition:** collect `n_snapshots` per temperature slot (default 100).
- Harvest spin snapshots at each temperature slot.
- **Thinning rule:** save one snapshot every `max(1, 3 × τ_max)` sweeps.
  τ_max comes from **Phase B** measurement on the locked PT ladder.
  Correlation between successive snapshots: e^{-3} ≈ 0.05 (safely independent).
  Total sweeps = `n_snapshots × max(1, 3 × τ_max)`.
- **Stream to HDF5:** each snapshot is appended to the HDF5 file immediately
  (resizable dataset, flushed after each write). No in-memory buffering.
  If the process crashes, all snapshots collected so far are safely on disk.
- **Resume = read the HDF5.** The snapshots ARE the replica states. On resume:
  1. Open HDF5, count existing snapshots per `(param, T)` slot.
  2. Load the last snapshot at each T slot → replica spin state.
  3. Restore address map + locked T ladder + τ_max from HDF5 attributes.
  4. Continue sweeping and appending until `n_snapshots` reached.
  State is only persisted after `>3τ_int` sweeps (i.e., at each snapshot save),
  so the worst-case data loss on crash is one thinning interval — acceptable.
- Track round-trip times continuously. Log warnings if round-trips stall.

#### Steps

- [ ] Step 2.1.1: `PTEngine.__init__` — replica creation from geometric T ladder, model factory, address map initialisation, select correct `pt_rounds_*` C++ function based on model type
- [ ] Step 2.1.2: `tune_ladder()` — Phase A: outer KTH feedback loop in Python (call `pt_rounds()` for N_sw rounds → read f(T) from returned histograms → compute df/dT, η, redistribute T → damped update → check convergence). Doubling N_sw schedule, post-tuning acceptance rate safety check
- [ ] Step 2.1.3: `equilibrate()` — Phase B: locked ladder, call `pt_rounds(..., track_observables=True)`, Welch t-test on returned obs streams, doubling scheme, measure true PT τ_int → lock τ_max
- [ ] Step 2.1.4: `produce()` — Phase C: snapshot harvesting loop, call `pt_rounds()` for `3×τ_max` rounds between snapshots, read spins via `temp_to_replica`, stream to HDF5

---

### 2.2 Orchestrator & Param-Level Parallelism

File: `python/pbc_datagen/orchestrator.py`

Top-level coordinator. Each Hamiltonian parameter value (D, U, etc.) runs an
independent PT campaign — embarrassingly parallel across `max_workers` cores.

```python
def find_existing_hdf5(output_dir, model_type, L, param_value):
    """Find the most recent HDF5 file for this (model, L, param) combo."""
    # Glob for matching files, sort by timestamp suffix, return newest or None
    pattern = f"{model_type}_L{L}_{param_label}={param_value:.4f}_*.h5"
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
        path = f"{output_dir}/{model_type}_L{L}_{param_label}={param_value:.4f}_{ts}.h5"
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

**One HDF5 per param value.** No concurrent writes — each worker owns its file.

#### Resume

Default behaviour is resume. Each param's HDF5 file IS its checkpoint.
The resume check is simple: **snapshots in HDF5 < `n_snapshots` requested?**

1. If `force_new=False` and a matching HDF5 exists with snapshots < `n_snapshots`:
   resume from it — derive fresh seed, load replica states, continue Phase C.
   Re-running with a higher `--n-snapshots` just extends production.
2. If matching HDF5 exists and snapshots >= `n_snapshots`: skip (already complete).
3. If `force_new=True` (`--new`): ignore existing files, create a new timestamped
   HDF5. Old files are preserved (user can delete manually if desired).
4. If no matching HDF5 exists: clean start with ms-timestamp seed.

#### Steps

- [ ] Step 2.2.1: `generate_dataset()` — distribute param values across workers via `multiprocessing.Pool`
- [ ] Step 2.2.2: `run_campaign()` — single-param entry point: construct PTEngine, run A→B→C
- [ ] Step 2.2.3: Resume logic — scan HDF5 for completed/in-progress params, skip or restore

---

### 2.3 I/O

File: `python/pbc_datagen/io.py`

#### HDF5 Streaming Writes

Snapshots are streamed to HDF5 during production — **not** buffered in memory.
For large L and long runs, holding all snapshots in RAM is infeasible.

**HDF5 layout (one file per param value):**
```
blume_capel_L64_D=1.5000_1709312456789.h5
├── .attrs                              # seed, model_type, L, param_value,
│                                       # locked T ladder, τ_max, address_map,
│                                       # round_trip_stats, "complete" flag
├── T=2.269/
│   ├── snapshots                       # (N, C, L, L) int8, resizable axis 0
│   └── .attrs                          # per-slot τ_int
├── T=2.300/
│   └── ...
└── ...
```

**Snapshot shape depends on model:**
- Ising / Blume-Capel: `C=1` → shape `(N, 1, L, L)` int8
- Ashkin-Teller: `C=2` → shape `(N, 2, L, L)` int8 (channel 0 = σ, channel 1 = τ)

Uniform (N, C, L, L) shape across all models simplifies I/O code.

The last snapshot in each `T` slot doubles as the resume state for that replica.
Root `.attrs` stores seed, address map, and locked τ_max — everything needed
to resume Phase C without a separate checkpoint file.

**Streaming protocol:**
1. `create_temperature_slot(path, T, L, C)` → creates group + resizable dataset
   with `maxshape=(None, C, L, L)`, `dtype=int8`, `chunks=(1, C, L, L)`.
2. `append_snapshot(path, T, spins)` → `dset.resize(n+1, axis=0); dset[n] = spins`.
3. `file.flush()` after each append for crash safety.
4. Update param-level attrs (address map, sweep count) after each snapshot batch.
5. Set `"complete": True` attr when param production finishes.

#### Steps

- [ ] Step 2.3.1: `SnapshotWriter` class — open/create HDF5, create groups, streaming append with flush
- [ ] Step 2.3.2: `write_param_attrs()` — save T ladder, τ_max, address map as HDF5 attrs (updated each snapshot batch)
- [ ] Step 2.3.3: `read_resume_state()` — load last snapshot per T slot + attrs for resume

---

### 2.4 CLI Entry Point

File: `scripts/generate_dataset.py`

Simple argparse wrapper around `generate_dataset()`. No TDD — just a thin script.
Default behaviour is resume (pick up where you left off). `--new` ignores existing
files and creates fresh ones (old files are NOT deleted).

```
python scripts/generate_dataset.py \
    --model blume_capel --L 64 \
    --params 0.0 0.5 1.0 1.5 1.965 \
    --T-range 1.5 4.0 \
    --n-replicas 20 --n-snapshots 100 \
    --workers 4 --output-dir output/ \
    --new   # optional: ignore existing files, start fresh with new timestamp
```

- [ ] Step 2.4.1: argparse CLI wrapping `generate_dataset()` with all parameters + `--new` flag

## Phase 3: Validation & Diagnostics

- [ ] Step 3.1: `validation.py` — equilibration trace plots (E, M vs sweep)
- [ ] Step 3.2: `validation.py` — cluster scaling check ⟨n⟩ ~ L^{y_h}
- [ ] Step 3.3: Round-trip diagnostics — t_RT histogram, replica diffusion plot
- [ ] Step 3.4: Per-gap acceptance rate plot + f(T) fraction plot (temperatures should cluster at f ≈ 0.5 bottleneck after KTH tuning)

## Test Plan

### Phase 1 Tests (completed) ✅

- [x] Unit: PRNG smoke test (determinism, range, uniformity, autocorrelation) — `tests/test_foundation.py`
- [x] Unit: Neighbor table correctness for various L (shape, PBC, symmetry) — `tests/test_foundation.py`
- [x] Unit: Ising model construction, energy, magnetization — `tests/ising/test_model.py`
- [x] Unit: Ising Wolff detailed balance (2×2 chi-squared at 10 temperatures) — `tests/ising/test_wolff.py`
- [x] Unit: Ising Metropolis detailed balance (2×2 chi-squared at 10 temperatures) — `tests/ising/test_metropolis.py`
- [x] Unit: Ising sweep detailed balance + ergodicity — `tests/ising/test_sweep.py`
- [x] Unit: BC Wolff — vacancy barrier, cluster on pure ±1 matches Ising, seed-on-vacancy returns 0 — `tests/blume_capel/test_wolff.py`
- [x] Unit: BC Metropolis — `_delta_energy` correctness, 2×2 detailed balance (81 states) — `tests/blume_capel/test_metropolis.py`
- [x] Unit: BC sweep — detailed balance + ergodicity, observables dict includes Q — `tests/blume_capel/test_sweep.py`
### Phase 1 Tests — Ashkin-Teller (`tests/ashkin_teller/`)

- [x] Unit: AT construction, cold-start energy, σ/τ/Baxter magnetizations — `test_model.py`
- [x] Unit: AT Wolff — effective coupling J_eff = J + U·fixed, cluster growth, flip correctness — `test_wolff.py`
- [x] Unit: AT Wolff remapping — U > 1 activates s = στ basis, bond probabilities stay in [0,1] — `test_wolff.py`
- [x] Unit: AT Metropolis — ΔE formulas, 2×2 detailed balance (256 states, chi-squared) — `test_metropolis.py`
- [x] Unit: AT sweep — detailed balance + ergodicity, observables dict has 7 keys — `test_sweep.py`
- [x] Unit: AT sweep with U > 1 — remapped cluster produces correct equilibrium statistics — `test_sweep.py`

### Phase 2 Tests — Autocorrelation (`tests/test_autocorrelation.py`)

- [ ] Unit: `acf_fft` on synthetic AR(1) signal matches known analytical ACF
- [ ] Unit: `tau_int` on AR(1) with known τ recovers correct value within tolerance
- [ ] Unit: `tau_int` on white noise returns ≈ 0.5
- [ ] Unit: `tau_int_multi` returns per-observable dict and correct bottleneck τ_max

### Phase 1.5 Tests — C++ PT Inner Loop

#### `tests/test_pt_exchange.py` — exchange decision (Step 1.5.1) ✅
- [x] Unit: Same T → always returns true (1/T_i - 1/T_j = 0)
- [x] Unit: Known E and T → exact accept probability over many calls
- [x] Unit: Extreme T gap with typical energies → always returns false

#### `tests/test_pt_exchange_round.py` — M random swap proposals (Step 1.5.2) ✅
- [x] Unit: After one round, total attempts across all gaps == M
- [x] Unit: After many rounds, every gap has attempts > 0 (uniform coverage)
- [x] Unit: On accepted swap, r2t and t2r stay inverse permutations
- [x] Unit: Same T → n_accepts == n_attempts at every gap

#### `tests/test_pt_labels.py` — labels, histograms, round trips (Steps 1.5.3–1.5.5) ✅
- [x] Unit: `pt_update_labels` sets UP at slot 0, DOWN at slot M-1, others unchanged
- [x] Unit: `pt_accumulate_histograms` increments n_up/n_down matching replica labels
- [x] Unit: `pt_accumulate_histograms` skips LABEL_NONE slots
- [x] Unit: `pt_count_round_trips` detects UP→DOWN transition at slot M-1
- [x] Unit: `pt_count_round_trips` returns 0 when no transition occurs

#### `tests/test_pt_rounds.py` — composition loop integration (Steps 1.5.6–1.5.7)
- [ ] Unit: `pt_collect_obs` records valid energies per T slot per round
- [ ] Unit: `pt_rounds` with track_observables=False returns empty obs_streams
- [ ] Integration: `pt_rounds` with close T over many rounds produces round trips > 0
- [ ] Integration: f(T) monotonicity — UP fraction higher at cold end than hot end

### Phase 2 Tests — PT Orchestration (`tests/test_parallel_tempering.py`)

- [ ] Unit: KTH tuning concentrates temperatures at bottleneck (f(T) ≈ 0.5 region) on a known test case
- [ ] Unit: KTH f(T) is approximately linear (R² > 0.99) after convergence
- [ ] Unit: KTH convergence requires both f(T) linearity and temperature stability
- [ ] Unit: Ladder is immutable during Phase B and Phase C (no T changes)

### Phase 2 Tests — I/O (`tests/test_io.py`)

- [ ] Unit: `SnapshotWriter` creates correct HDF5 group hierarchy
- [ ] Unit: `append_snapshot` grows dataset by 1 along axis 0, data matches
- [ ] Unit: `write_param_attrs` round-trips T ladder, τ_max, address map through HDF5 attrs
- [ ] Unit: `read_resume_state` loads last snapshot per T slot and restores attrs

### Phase 2 Tests — Integration (`tests/test_integration.py`)

- [ ] Integration: Full pipeline on 4×4 Ising — PT(A→B→C) → HDF5
- [ ] Integration: Verify HDF5 layout matches spec (groups, resizable datasets, attrs)
- [ ] Integration: Round-trip times are finite (replicas actually diffuse)
- [ ] Integration: Resume from existing HDF5 continues without re-running Phase A/B

