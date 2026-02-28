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

### 1.2 Blume-Capel Model ✅

- [x] Step 1.2.1: `blume_capel.hpp` + `blume_capel.cpp` — struct, constructor, observables (E, m, |m|, Q)
- [x] Step 1.2.2: `_wolff_step()` — Wolff cluster flip; vacancies block growth, D doesn't affect bonds
- [x] Step 1.2.3: `_metropolis_sweep()` — symmetric proposal over {-1,0,+1}\\{current}, `_delta_energy(site, new_spin)` used by sweep
- [x] Step 1.2.4: `sweep()` — Metropolis + Wolff, returns dict with keys `energy`, `m`, `abs_m`, `q`

Tests: `tests/blume_capel/` — test_model.py, test_wolff.py, test_metropolis.py, test_sweep.py. All include 2×2 exact partition function (81 states) chi-squared checks. Ergodicity verified via Welch's t-test (all-magnetic vs all-vacancy starts).

### 1.3 Ashkin-Teller Model

**Hamiltonian:**
```
H = -J Σ_{<ij>} σ_i σ_j  -  J Σ_{<ij>} τ_i τ_j  -  U Σ_{<ij>} σ_i σ_j τ_i τ_j
```
where J = 1 (fixed ferromagnetic unit), U is the four-spin coupling, and σ_i, τ_i ∈ {-1, +1}.

Public API: `set_four_spin(U)` sets U. J = 1 is fixed and not user-settable.
For the Z(4) subspace, both two-spin couplings are equal (J_σ = J_τ = 1);
anisotropy arises only internally when the remapping (below) is needed.

#### Cluster Algorithm — Embedded Wolff (Wiseman & Domany, 1995)

**Case 1: U ≤ 1 (no remapping)**

1. Randomly pick target variable: σ or τ (50/50).
2. Hold the other variable fixed. This gives an effective Ising model where
   the bond coupling between sites j, k is:
   `J_eff(j,k) = J + U · (fixed_j)(fixed_k)`
3. Pick a random seed site. Grow Wolff cluster: for each neighbor k of a
   cluster site j with aligned target spin, add to cluster with probability
   `1 - exp(-2 · J_eff(j,k) / T)`.
4. Flip all target spins in the cluster. The fixed variable is unchanged.

**Case 2: U > 1 (remapping required)**

When U > 1, the effective coupling J_eff = J - U < 0 for anti-aligned fixed
spins, making the bond probability invalid (> 1). Fix by introducing
s_j = σ_j τ_j and working in (σ, s) basis:

| Physical coupling | Value | Working coupling | Value |
|---|---|---|---|
| J_σ | 1 | J_σ | 1 |
| J_τ | 1 | J_s | U |
| U   | U | U'  | 1 |

Since U' = 1 ≤ min(J_σ, J_s), bond probabilities are valid. The cluster
algorithm is the same as Case 1 but operating on (σ, s) instead of (σ, τ):

1. Randomly pick target variable: σ or s (50/50).
2. Hold the other variable fixed.
   - If clustering σ (holding s fixed): `J_eff(j,k) = 1 + 1 · s_j s_k`
   - If clustering s (holding σ fixed): `J_eff(j,k) = U + 1 · σ_j σ_k`
3. Grow Wolff cluster (alignment checked on the target variable; for s,
   check `σ_j τ_j == σ_seed τ_seed`).
4. Flip target variable in cluster.

Translation back to physical (σ, τ) after flipping:
- Flipping σ with s held fixed → both σ AND τ flip (since τ = s·σ)
- Flipping s with σ held fixed → only τ flips (since τ = s·σ and s changed)

**Internal representation:** The struct always stores physical σ and τ arrays.
Computes s = σ·τ on the fly when remapping is active. The `remapped_` flag is
set automatically by `set_four_spin(U)`. Anisotropic internal couplings
(J_a, J_b, U_eff) support both cases without branching in the cluster loop.

#### Metropolis Sweep

Operates in physical (σ, τ) basis regardless of remapping. For each of 2N
random proposals (N for σ, N for τ):

ΔE for σ_i → -σ_i:
```
ΔE = 2σ_i Σ_{j∈nbr(i)} σ_j (J + U τ_i τ_j)
```
ΔE for τ_i → -τ_i:
```
ΔE = 2τ_i Σ_{j∈nbr(i)} τ_j (J + U σ_i σ_j)
```

Accept with probability min(1, exp(-ΔE / T)).

#### Observables

| Key | Definition |
|-----|-----------|
| `energy` | Full Hamiltonian |
| `m_sigma` | (1/N) Σ σ_i |
| `abs_m_sigma` | (1/N) \|Σ σ_i\| |
| `m_tau` | (1/N) Σ τ_i |
| `abs_m_tau` | (1/N) \|Σ τ_i\| |
| `m_baxter` | (1/N) Σ σ_i τ_i  (Baxter order parameter) |
| `abs_m_baxter` | (1/N) \|Σ σ_i τ_i\| |

#### Steps

- [x] Step 1.3.1: `ashkin_teller.hpp` + `ashkin_teller.cpp` — AshkinTellerModel struct, constructor, `set_temperature`, `set_four_spin_coupling` (auto-remapping, U ≥ 0), energy, magnetizations (σ, τ, Baxter), `set_sigma`/`set_tau`
- [x] Step 1.3.2: `_wolff_step()` — Embedded Wolff: pick σ or τ (or σ/s when remapped), compute J_eff per bond, grow cluster, flip with physical-basis translation
- [x] Step 1.3.3: `_metropolis_sweep()` — 2N proposals (N for σ, N for τ), ΔE formulas above
- [ ] Step 1.3.4: `sweep()` — Metropolis + embedded Wolff, returns dict with 7 observable arrays

Tests: `tests/ashkin_teller/` — test_model.py, test_wolff.py, test_metropolis.py, test_sweep.py. 2×2 exact partition function (256 states) chi-squared checks. Verify remapping produces consistent observables for U > 1.

### 1.4 pybind11 Bindings
- [ ] Step 1.4.1: `bindings.cpp` — expose all three models, sweep(), observables, lattice data

## Model Interface (contract between C++ models and Python orchestration)

Every model must expose a consistent interface so that the PT manager,
autocorrelation analysis, and I/O code work identically for Ising, BC, and AT.

```python
class Model:
    L: int                                  # lattice linear size
    spins: npt.NDArray[np.int8]             # (L, L) view into spin array

    def set_temperature(self, T: float) -> None: ...
    def energy(self) -> int | float: ...    # int for Ising, float for BC/AT
    def sweep(self, n_sweeps: int) -> dict[str, npt.NDArray]: ...
```

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

Five-stage pipeline. Each stage is a separate module with a clean public API for TDD.

```
 For each param value (e.g. D) in linspace:

 ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
 │  1. Pilot  │───▶│ 2. Ladder │───▶│  3. PT    │───▶│  4. PT    │───▶│  5. PT    │
 │  (single-  │    │  Design   │    │  Phase A  │    │  Phase B  │    │  Phase C  │
 │   chain)   │    │           │    │  (tune)   │    │  (equil.) │    │ (produce) │
 └─────┬──┬──┘    └─────┬─────┘    └─────┬─────┘    └──┬────┬──┘    └─────┬─────┘
       │  │             │                │              │    │              │
       │  │  naive τ_int│  warm-start    │  locked      │    │ true τ_max   │
       │  │  + E hists  │  β ladder      │  β ladder    │    │ (for thin)   │
       │  ▼             ▼                ▼              │    ▼              ▼
       │  ┌──────────────────────────────────────────┐  │  ┌──────────┐  ┌────────┐
       └─▶│           Autocorrelation (acf_fft)      │◀─┘  │ τ_max    │  │ HDF5   │
          └──────────────────────────────────────────┘     │ locked   │─▶│ I/O    │
                                                           └──────────┘  └────────┘

 Data flow summary:
 ───────────────────────────────────────────────────────────────────────────────
 Stage        Produces                    Consumed by
 ───────────────────────────────────────────────────────────────────────────────
 1. Pilot     naive τ_int, E histograms   2. Ladder Design
 2. Ladder    warm-start β array          3. Phase A
 3. Phase A   tuned β array → LOCK        4. Phase B
 4. Phase B   equilibrated streams,       5. Phase C
              true PT τ_max → LOCK
 5. Phase C   thinned snapshots           HDF5 I/O
 ───────────────────────────────────────────────────────────────────────────────

 KEY INSIGHT: τ_int is measured TWICE.
   • Pilot τ_int (single-chain, naive) — only used for ladder design.
   • Phase B τ_int (PT, true) — used for production thinning.
   PT injects decorrelated configs from hot replicas, so PT τ_int ≪ pilot τ_int.
   Using the pilot value for thinning would discard good snapshots.
```

**Orchestrator inputs:**
```python
generate_dataset(
    model_type: str,                        # "ising" | "blume_capel" | "ashkin_teller"
    L: int,                                 # lattice linear size
    param_range: tuple[float, float, int],  # (start, end, num) → np.linspace
    T_range: tuple[float, float],           # (T_min, T_max); ΔT auto-selected
    max_workers: int = 4,                   # CPU cores available
    pilot_sweeps: int = 1_000_000,          # sweeps per pilot point
    output_path: str = "dataset.h5",
)
```

For Ising (no Hamiltonian parameter beyond T), `param_range` is a single dummy value
`(0.0, 0.0, 1)`. For Blume-Capel, it sweeps D. For Ashkin-Teller, it sweeps U (the four-spin coupling).

### Why Not On-The-Fly ACF

Computing ACF continuously during severe critical slowing down (CSD) is a statistical trap:

1. **Biased mean:** The ACF formula depends on the sample mean Ō. If the running
   window N < 10×τ_int, the simulation hasn't explored phase space — Ō is wrong.
2. **Artificial decorrelation:** A biased Ō makes ρ(t) cross zero prematurely,
   producing a falsely small τ_int. Snapshots get harvested too frequently →
   correlated garbage.

**Solution:** Always compute ACF on a frozen block of completed samples — never
on a growing window. This happens at two points in the pipeline:

1. **Pilot** (single-chain): ACF on ~1M sweeps per temperature. Produces a
   naive τ_int that overestimates the true production value (no PT mixing).
   Used only for ladder design.
2. **Phase B** (PT, locked ladder): ACF on the equilibrated fixed-temperature
   streams. Captures the true PT dynamics (replica exchanges shorten τ_int).
   This τ_max is locked and used for production thinning in Phase C.

### Module Map

```
python/pbc_datagen/
├── autocorrelation.py      # FFT-based ACF + τ_int
├── pilot.py                # Single-chain τ_int profiling         ← NEW
├── ladder.py               # PT ladder design from pilot data     ← NEW
├── parallel_tempering.py   # PT engine + round-trip tracking      ← NEW
├── orchestrator.py         # Top-level coordinator (redesigned)
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

### 2.1 Pilot Run

File: `python/pbc_datagen/pilot.py`

The pilot run serves two purposes:
1. **τ_int profiling:** determine worst-case τ_int across all observables and
   temperatures → locked thinning interval for production.
2. **Energy distributions:** E(T) histograms at each temperature → input for
   PT ladder design (§2.2).

#### Method

For each Hamiltonian parameter value (e.g., each D):
- Run single-chain MCMC at a grid of temperatures (e.g., 50 points from T_min to T_max).
- At each T: run `pilot_sweeps` (default 1M) sweeps using the model's `sweep()`.
  Store full observable arrays in memory.
- Compute FFT-based ACF (§2.0) on the entire frozen block for each observable.
- Extract τ_int for each observable at each T.
- Record `τ_max(T) = max(τ_int across observables)` at each T.
- Record energy histograms for ladder design.

#### Parallelization

Each `(param_value, T)` pair is a fully independent single-chain run.
Distribute across `max_workers` via `multiprocessing.Pool.map`.

Since pybind11 model objects are NOT picklable, each worker constructs its own
model instance. Workers receive `(model_type, L, param_value, T, n_sweeps, seed)`
and return `(T, observable_arrays, energy_histogram)`.

#### Steps

- [ ] Step 2.1.1: `pilot_single(model_type, L, params, T, n_sweeps, seed)` — run one point, return observables + energy histogram
- [ ] Step 2.1.2: `pilot_sweep(model_type, L, params, T_grid, n_sweeps, max_workers)` — parallel pilot across all T for one param set
- [ ] Step 2.1.3: `PilotResult` dataclass — τ_int(T) per observable, energy histograms, worst-case τ_max

---

### 2.2 PT Ladder Design

File: `python/pbc_datagen/ladder.py`

Given pilot energy distributions and a replica budget, compute the optimal β ladder.

**Inputs:**
- Pilot energy histograms E(T) at each pilot temperature
- `max_replicas`: upper bound on ladder size
- `T_min`, `T_max`: temperature endpoints (fixed)

**Method:**
1. Compute energy-distribution overlap between each pair of adjacent pilot temperatures.
2. Inverse-overlap weighting (same principle as KTH):
   ```
   w_i = 1 / max(overlap(T_i, T_{i+1}), ε)    (poor overlap → high weight)
   C_i = Σ_{k<i} w_k / Σ_all w_k              (cumulative resistance)
   β_i = β_min + C_i × (β_max - β_min)         (new β positions)
   ```
3. Concentrate replicas near phase transitions (poor overlap → more replicas).
4. Constrain to `max_replicas` points.

**Output:** `np.ndarray` of β values (length ≤ max_replicas), sorted.

This gives the PT engine (§2.3) a *warm start* — the initial ladder is already
near-optimal, so Phase A (KTH tuning) converges quickly rather than starting
from a blind geometric spacing.

#### Steps

- [ ] Step 2.2.1: `energy_overlap(hist_i, hist_j)` — histogram overlap metric between two temperature points
- [ ] Step 2.2.2: `design_ladder(pilot_result, max_replicas, T_min, T_max)` — compute optimal β array
- [ ] Step 2.2.3: Overlap quality check — warn if any adjacent pair has overlap below threshold

---

### 2.3 Parallel Tempering Engine

File: `python/pbc_datagen/parallel_tempering.py`

Model-agnostic PT engine. Works with any model conforming to the **Model Interface** (see above).

#### Exchange Logic

Adjacent-only replica swaps maximize energy-distribution overlap.
Even/odd alternating pattern: on even rounds propose swaps for pairs (0,1), (2,3), ...;
on odd rounds propose (1,2), (3,4), .... This ensures every gap is attempted every 2 rounds.

Metropolis acceptance criterion:
```
Δβ = β_{i+1} - β_i
ΔE = E_{i+1} - E_i
A = min(1, exp(Δβ × ΔE))
```
On acceptance: swap spin configurations (numpy array copy), NOT temperature labels.
This preserves the identity of each temperature slot.

#### Fixed-Temperature Stream Tracking (not Fixed-Replica)

**Critical distinction:** observables and snapshots must be tracked at each
**temperature slot** (e.g., "whatever replica currently sits at T = 2.269"),
NOT at each physical replica.

- When replicas swap, a new lattice configuration enters the temperature slot.
- The fixed-temperature stream belongs to a single canonical ensemble.
- PT teleports decorrelated configurations from the hot end, reducing τ_int
  compared to single-chain MCMC. This is the whole point.
- **Do NOT track a specific replica across temperatures** — its observables
  change with T and don't belong to any single ensemble.

#### Round-Trip Time Tracking

Even if the fixed-temperature ACF looks small, PT may be broken: replicas
might swap locally (bouncing between T₀ and T₁) but never traverse the full
ladder. Bottleneck configurations never reach hot temperatures to melt.

**Track round-trip time (t_RT):**
1. Tag each replica with a persistent ID.
2. When a replica reaches T_min, mark it "released upward."
3. Track sweep count until it reaches T_max and returns to T_min.
4. Production run must allow dozens of complete round-trips per replica.
5. If mean t_RT is comparable to total production length → dataset is suspect.

#### Phase A — Ladder Tuning (Katzgraber-Trebst-Huse 2006)

The pilot (§2.1) provides a warm-start ladder. Phase A fine-tunes it in situ.

**Why this matters:** Even the pilot-informed ladder may not be perfect — the
pilot ran single-chain, but PT energy distributions shift due to replica mixing.
KTH feedback closes the gap.

**Algorithm:**
1. Start from the pilot-designed ladder (§2.2), NOT a blind geometric spacing.
2. Run `n_tune` rounds of (sweeps + swap attempts).
3. Track empirical swap acceptance rates via exponential moving average:
   `a_i ← (1 - α) × a_i + α × measured_i`    (α ≈ 0.1)
4. Every `update_interval` rounds, redistribute β values:

**Feedback formula:**
```
ε = 0.05                          # floor to prevent division by zero
w_i = 1 / max(a_i, ε)             # "resistance" of gap i

# Cumulative resistance determines new β positions:
C_0 = 0
C_i = Σ_{k=0}^{i-1} w_k / Σ_{k=0}^{M-2} w_k     for i = 1, ..., M-1

# Target β values:
β_i^target = β_min + C_i × (β_max - β_min)

# Damped update (prevents oscillation):
β_i^new = (1 - γ) × β_i^old + γ × β_i^target     (γ ≈ 0.3)
```

**Convergence criterion:** max_i |a_i - a_target| < tolerance, or max tuning rounds reached.
Target acceptance rate: ~23% (optimal for adjacent-only swaps in 2D lattice models).

#### Phase B — Equilibration & τ_int Measurement

- **Ladder is LOCKED.** No more temperature adjustments. Adjusting temperatures
  during production violates detailed balance and invalidates the dataset.
- Run sweeps + swaps until macroscopic observables stabilize.
- Convergence check: sliding window (e.g., last 20% vs first 20% of window) — if means
  agree within tolerance for E and |M| at every replica, equilibration is complete.
- **Measure true PT τ_int.** Once equilibrated, compute `tau_int_multi` on
  the fixed-temperature observable streams collected during Phase B. This
  captures the real PT autocorrelation (shortened by replica exchanges).
  The resulting `τ_max` is locked and passed to Phase C for thinning.
  - The pilot τ_int (§2.1) overestimates because it was single-chain.
    It is only used for ladder design — never for production thinning.

#### Phase C — Production

- Harvest spin snapshots at each temperature slot.
- **Thinning rule:** save one snapshot every `max(1, 3 × τ_max)` sweeps.
  τ_max comes from **Phase B** measurement on the locked PT ladder.
  Correlation between successive snapshots: e^{-3} ≈ 0.05 (safely independent).
- **Stream to HDF5:** each snapshot is appended to the HDF5 file immediately
  (resizable dataset, flushed after each write). No in-memory buffering.
  If the process crashes, all snapshots collected so far are safely on disk.
- **Checkpoint after each snapshot batch:** update a JSON sidecar file with
  `sweep_index`, per-slot snapshot counts, and save replica spin states into
  an HDF5 `_checkpoint/` group. On resume, restore replicas and continue
  from the last checkpoint — no need to re-run pilot or ladder design.
- Track round-trip times continuously. Log warnings if round-trips stall.

#### Steps

- [ ] Step 2.3.1: `PTEngine.__init__` — replica creation from β ladder, model factory
- [ ] Step 2.3.2: `_sweep_all()` — sweep all replicas (parallelized across workers, see §2.4)
- [ ] Step 2.3.3: `_attempt_exchanges()` — even/odd adjacent Metropolis swaps
- [ ] Step 2.3.4: `_track_observables()` — record fixed-temperature stream data
- [ ] Step 2.3.5: `RoundTripTracker` — replica tagging, T_min/T_max arrival detection, statistics
- [ ] Step 2.3.6: `tune_ladder()` — Phase A: KTH feedback loop
- [ ] Step 2.3.7: `equilibrate()` — Phase B: locked ladder, sliding-window convergence, measure true PT τ_int on equilibrated streams → lock τ_max
- [ ] Step 2.3.8: `produce()` — Phase C: snapshot harvesting, thinned by Phase B τ_max

---

### 2.4 Worker Allocation & Scheduling

**Key insight:** PT requires all replicas to *exist* simultaneously for exchanges,
but does NOT require them to sweep in parallel. Parallelism is a performance
optimization, not a correctness requirement.

#### Intra-PT (replicas within one PT run)

Each sweep round distributes R replicas across W workers:

- **R ≤ W:** One replica per worker, all sweep in parallel. Simple.
- **R > W:** Sweep in `ceil(R/W)` batches of W. After *all* batches complete,
  do exchanges. The algorithm is identical — just `ceil(R/W)` times slower
  per round than having R workers.

Workers own their model instances (pybind11 objects aren't picklable).
Each worker holds `ceil(R/W)` replicas and sweeps them sequentially within
its process. After all workers finish, the main process collects
`(replica_id, energy, spins)`, makes exchange decisions, and sends back
`(new_T, new_spins)` or no-op.

#### Inter-PT (across Hamiltonian parameter values)

Each param value (e.g., D) runs an independent PT campaign. The outer loop is
sequential by default. If `R_per_campaign < W`, the system could run
`floor(W / R_per_campaign)` campaigns simultaneously, but this adds complexity
and is deferred to a future optimization.

#### Pilot Parallelism

Each `(param_value, T)` pair is fully independent → distribute across W workers
via `Pool.map`. Trivially parallel — no batching needed.

#### Steps

- [ ] Step 2.4.1: `WorkerPool` class — manages replica-to-worker assignment, batched sweep dispatch, result collection
- [ ] Step 2.4.2: Integration with `PTEngine._sweep_all()` — plug WorkerPool in

---

### 2.5 Orchestrator

File: `python/pbc_datagen/orchestrator.py`

Top-level coordinator: pilot → ladder → PT → I/O.

```python
def generate_dataset(
    model_type: str,                        # "ising" | "blume_capel" | "ashkin_teller"
    L: int,                                 # lattice linear size
    param_range: tuple[float, float, int],  # (start, end, num) → np.linspace
    T_range: tuple[float, float],           # (T_min, T_max)
    max_workers: int = 4,
    pilot_sweeps: int = 1_000_000,
    output_path: str = "dataset.h5",
) -> None: ...
```

For each param value in `linspace(start, end, num)`:
1. **Pilot** (§2.1) → `PilotResult` (τ_int profiles, energy histograms)
2. **Ladder** (§2.2) → β array (length ≤ max_workers)
3. **PT** (§2.3) → Phase A (tune) → Phase B (equilibrate) → Phase C (produce)

Steps 1–2 are cheap relative to production and are always re-run on resume.
Only Phase C (production) is resumable.

#### Checkpoint / Resume Protocol

Checkpoint file: `{output_path}.checkpoint.json` (sibling of the HDF5 file).
Written after every snapshot batch during Phase C.

```json
{
  "model_type": "blume_capel",
  "L": 64,
  "completed_params": [0.0, 0.5, 1.0],
  "current_param": {
    "value": 1.5,
    "beta_ladder": [0.30, 0.35, 0.40, "..."],
    "tau_max": 42.7,
    "sweep_index": 15000,
    "n_snapshots": {"2.269": 84, "2.300": 84, "...": "..."}
  }
}
```

Replica spin states are binary data — stored in the HDF5 file under a
`_checkpoint/{param_value}/` group, NOT in JSON. Deleted once the param
value's production completes.

**On resume:**
1. Load checkpoint JSON.
2. Skip all `completed_params` — their snapshots are already in the HDF5.
3. For `current_param`: reload β ladder + τ_max, restore replica spins from
   HDF5 `_checkpoint/`, continue Phase C from `sweep_index`.
4. For remaining params: run full pilot → ladder → production.
5. On clean completion of all params: delete checkpoint JSON and
   HDF5 `_checkpoint/` groups.

#### Steps

- [ ] Step 2.5.1: `generate_dataset()` — main entry point implementing the above loop
- [ ] Step 2.5.2: Progress reporting and logging (per-D, per-phase)
- [ ] Step 2.5.3: `save_checkpoint()` / `load_checkpoint()` — JSON sidecar + HDF5 replica state
- [ ] Step 2.5.4: Resume logic — detect existing checkpoint, skip completed params, restore in-progress

---

### 2.6 I/O

File: `python/pbc_datagen/io.py`

#### HDF5 Streaming Writes

Snapshots are streamed to HDF5 during production — **not** buffered in memory.
For large L and long runs, holding all snapshots in RAM is infeasible.

**HDF5 layout:**
```
dataset.h5
├── blume_capel/                        # model group
│   ├── D=0.000/                        # param value group
│   │   ├── T=2.269/
│   │   │   ├── snapshots               # (N, L, L) int8, resizable axis 0
│   │   │   └── .attrs                  # τ_int, τ_max, seed, round_trip_stats
│   │   ├── T=2.300/
│   │   │   └── ...
│   │   └── .attrs                      # locked β ladder, pilot τ_max
│   └── D=1.965/
│       └── ...
└── _checkpoint/                        # replica spins for resume
    └── D=1.500/                        # (deleted on param completion)
        └── replica_spins               # (R, L, L) int8
```

**Streaming protocol:**
1. `create_temperature_slot(path, T, L)` → creates group + resizable dataset
   with `maxshape=(None, L, L)`, `dtype=int8`, `chunks=(1, L, L)`.
2. `append_snapshot(path, T, spins)` → `dset.resize(n+1, axis=0); dset[n] = spins`.
3. `file.flush()` after each append for crash safety.
4. `save_replica_state(path, param, spins_list)` → write `_checkpoint/` group.
5. `finalize_param(path, param)` → write final metadata attrs, delete `_checkpoint/` group.

#### Steps

- [ ] Step 2.6.1: `SnapshotWriter` class — open/create HDF5, create groups, streaming append with flush
- [ ] Step 2.6.2: `save_replica_state()` / `load_replica_state()` — checkpoint binary data in HDF5
- [ ] Step 2.6.3: `finalize_param()` — write metadata attrs, clean up `_checkpoint/`
- [ ] Step 2.6.4: `SnapshotReader` class — load snapshots by `(model, param, T)` query
- [ ] Step 2.6.5: `dataset_summary()` — list available `(model, param, T, n_snapshots)`

---

### 2.7 CLI Entry Point

File: `scripts/generate_dataset.py`

- [ ] Step 2.7.1: argparse CLI wrapping `generate_dataset()` with all parameters

## Phase 3: Validation & Diagnostics

- [ ] Step 3.1: `validation.py` — equilibration trace plots (E, M vs sweep)
- [ ] Step 3.2: `validation.py` — cluster scaling check ⟨n⟩ ~ L^{y_h}
- [ ] Step 3.3: Round-trip diagnostics — t_RT histogram, replica diffusion plot
- [ ] Step 3.4: Per-gap acceptance rate plot (should be uniform after KTH tuning)
- [ ] Step 3.5: Pilot τ_int profile plot — τ_int(T) for each observable, highlighting bottleneck

## Test Plan

### Phase 1 Tests (completed)

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

- [ ] Unit: AT construction, cold-start energy, σ/τ/Baxter magnetizations — `test_model.py`
- [ ] Unit: AT Wolff — effective coupling J_eff = J + U·fixed, cluster growth, flip correctness — `test_wolff.py`
- [ ] Unit: AT Wolff remapping — U > 1 activates s = στ basis, bond probabilities stay in [0,1] — `test_wolff.py`
- [ ] Unit: AT Metropolis — ΔE formulas, 2×2 detailed balance (256 states, chi-squared) — `test_metropolis.py`
- [ ] Unit: AT sweep — detailed balance + ergodicity, observables dict has 7 keys — `test_sweep.py`
- [ ] Unit: AT sweep with U > 1 — remapped cluster produces correct equilibrium statistics — `test_sweep.py`

### Phase 2 Tests — Autocorrelation (`tests/test_autocorrelation.py`)

- [ ] Unit: `acf_fft` on synthetic AR(1) signal matches known analytical ACF
- [ ] Unit: `tau_int` on AR(1) with known τ recovers correct value within tolerance
- [ ] Unit: `tau_int` on white noise returns ≈ 0.5
- [ ] Unit: `tau_int_multi` returns per-observable dict and correct bottleneck τ_max

### Phase 2 Tests — Pilot (`tests/test_pilot.py`)

- [ ] Unit: `pilot_single` runs correct number of sweeps and returns expected observable keys
- [ ] Unit: `pilot_single` energy histogram has correct bin range for model
- [ ] Unit: `pilot_sweep` distributes work across workers (mock Pool to verify)
- [ ] Unit: `PilotResult` τ_int profile shape matches T_grid length

### Phase 2 Tests — Ladder (`tests/test_ladder.py`)

- [ ] Unit: `energy_overlap` — identical histograms → 1.0, disjoint histograms → 0.0
- [ ] Unit: `design_ladder` — more replicas near poor-overlap region than easy-overlap region
- [ ] Unit: `design_ladder` — respects `max_replicas` constraint exactly
- [ ] Unit: `design_ladder` — endpoints match T_min and T_max

### Phase 2 Tests — PT Engine (`tests/test_parallel_tempering.py`)

- [ ] Unit: Exchange acceptance matches exact Boltzmann formula (deterministic test)
- [ ] Unit: Even/odd alternation covers all gaps over 2 rounds
- [ ] Unit: Fixed-temperature stream records the configuration currently *at* that T slot
- [ ] Unit: `RoundTripTracker` detects a completed min→max→min trip
- [ ] Unit: `RoundTripTracker` does NOT count incomplete trips
- [ ] Unit: KTH tuning converges toward uniform acceptance on a known test case
- [ ] Unit: Ladder is immutable during Phase B and Phase C (no β changes)

### Phase 2 Tests — Worker Pool (`tests/test_worker_pool.py`)

- [ ] Unit: R ≤ W → one batch, all replicas in parallel
- [ ] Unit: R > W → ceil(R/W) batches, each batch ≤ W replicas
- [ ] Unit: Worker results collected correctly after batched dispatch

### Phase 2 Tests — I/O (`tests/test_io.py`)

- [ ] Unit: `SnapshotWriter` creates correct HDF5 group hierarchy
- [ ] Unit: `append_snapshot` grows dataset by 1 along axis 0, data matches
- [ ] Unit: `save_replica_state` / `load_replica_state` round-trips spin arrays exactly
- [ ] Unit: `finalize_param` writes attrs and deletes `_checkpoint/` group
- [ ] Unit: `SnapshotReader` loads correct slice by `(model, param, T)`

### Phase 2 Tests — Checkpoint / Resume (`tests/test_checkpoint.py`)

- [ ] Unit: `save_checkpoint` writes valid JSON with expected keys
- [ ] Unit: `load_checkpoint` restores completed_params and current_param state
- [ ] Unit: Resume skips completed params and continues from sweep_index
- [ ] Unit: Clean completion deletes checkpoint JSON and `_checkpoint/` HDF5 groups

### Phase 2 Tests — Integration (`tests/test_integration.py`)

- [ ] Integration: Full pipeline on 4×4 Ising — pilot → ladder → PT(A→B→C) → HDF5
- [ ] Integration: Verify HDF5 layout matches spec (groups, resizable datasets, attrs)
- [ ] Integration: Round-trip times are finite (replicas actually diffuse)
- [ ] Integration: Kill-and-resume produces same snapshot count as uninterrupted run

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
