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
- [x] Step 1.2.2: `_wolff_step()` — Wolff single-cluster flip (spin-0 sites are "dead" barriers)
- [x] Step 1.2.3: `_metropolis_sweep()` — N random-site proposals over {-1, 0, +1} with crystal field D
- [x] Step 1.2.4: `sweep()` = Wolff + Metropolis, observable tracking (E, M, |M|, Q)

#### BC Wolff Design Notes

Standard Wolff cluster algorithm, adapted for 3-state spins:
- If `spin[seed] == 0` → return immediately (cluster_size = 0). Vacancies can't seed clusters.
- DFS grows through neighbors with `spin[j] == seed_spin` (±1). Spin-0 sites naturally fail this
  check and act as barriers that fragment the lattice.
- Bond probability: `p_add = 1 - exp(-2/T)` — identical to Ising.
  The crystal field D does NOT affect Wolff bond probabilities because (-s)² = s²; the D·s² term
  cancels exactly under cluster flip.
- Cluster flip: s → -s for all cluster members. Vacancies are never touched.
- Wolff alone is NOT ergodic for BC — it cannot create/destroy vacancies. Metropolis handles 0 ↔ ±1 transitions.

#### BC Metropolis Design Notes

- Proposal: pick random site, propose `new_spin` uniformly from `{-1, 0, +1} \ {current}` (symmetric, no Hastings correction needed).
- `ΔE = -(s_new - s_old) × Σ_neighbors s_j  +  D × (s_new² - s_old²)`
- Accept if ΔE ≤ 0 or `uniform() < exp(-ΔE/T)`.
- Possible transitions and their crystal field part D·(s_new² - s_old²):
  - ±1 → ∓1: crystal part = 0 (magnetic flip, no vacancy change)
  - ±1 → 0:  crystal part = -D (creating a vacancy)
  -  0 → ±1: crystal part = +D (filling a vacancy)

#### 2×2 BC Exact Partition Function

3⁴ = 81 states. Enumerate all in Python for chi-squared tests.
H = -Σ_{⟨ij⟩} s_i s_j + D Σ_i s_i² over 4 nearest-neighbor bonds (PBC).
Tests at multiple (T, D) pairs validate detailed balance for both Wolff and Metropolis.

### 1.3 Ashkin-Teller Model
- [ ] Step 1.3.1: `ashkin_teller.hpp` + `ashkin_teller.cpp` — AshkinTellerModel struct
- [ ] Step 1.3.2: Wiseman-Domany 3-step embedded Wolff (σ, τ, στ clusters)
- [ ] Step 1.3.3: Local Metropolis sweep for σ and τ
- [ ] Step 1.3.4: `sweep()` = 3-step Wolff + Metropolis, observable tracking

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

## Phase 2: Parallel Tempering & Python Orchestration

### 2.0 Autocorrelation Utility (prerequisite for Phase C)
- [ ] Step 2.0.1: `autocorrelation.py` — FFT-based autocorrelation function + τ_int via first zero crossing

#### τ_int Calculation Method

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

### 2.1 Parallel Tempering Manager

File: `python/pbc_datagen/parallel_tempering.py`

Model-agnostic: works with any model conforming to the **Model Interface** (see above).

- [ ] Step 2.1.1: Core `ParallelTemperingManager` class — replica creation, geometric β ladder init
- [ ] Step 2.1.2: Exchange logic — adjacent-only swaps with even/odd alternation
- [ ] Step 2.1.3: Phase A (Ladder Tuning) — dynamic temperature adjustment with KTH feedback
- [ ] Step 2.1.4: Phase B (Equilibration) — locked ladder, convergence monitoring
- [ ] Step 2.1.5: Phase C (Production) — snapshot harvesting with τ_int-based thinning
- [ ] Step 2.1.6: Multiprocessing support — optional `multiprocessing.Pool` for large systems

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

#### Phase A — Dynamic Temperature Ladder Optimization (Katzgraber-Trebst-Huse 2006)

**Why this matters:** A naive linearly/geometrically spaced ladder fails at phase transitions
where the heat capacity spikes. The energy distributions of adjacent replicas stop overlapping
and the swap acceptance rate drops to zero, breaking replica flow through temperature space.

**Algorithm:**
1. Initialize M replicas with geometric inverse-temperature spacing:
   `β_i = β_min × (β_max/β_min)^(i/(M-1))`
2. Run `n_tune` rounds of (sweeps + swap attempts).
3. Track empirical swap acceptance rates via exponential moving average:
   `a_i ← (1 - α) × a_i + α × measured_i`    (α ≈ 0.1)
4. Every `update_interval` rounds, redistribute β values:

**Feedback formula (the key math):**
```
ε = 0.05                          # floor to prevent division by zero
w_i = 1 / max(a_i, ε)             # "resistance" of gap i (low acceptance = high resistance)

# Cumulative resistance determines new β positions:
C_0 = 0
C_i = Σ_{k=0}^{i-1} w_k / Σ_{k=0}^{M-2} w_k     for i = 1, ..., M-1

# Target β values:
β_i^target = β_min + C_i × (β_max - β_min)

# Damped update (prevents oscillation):
β_i^new = (1 - γ) × β_i^old + γ × β_i^target     (γ ≈ 0.3)
```

This concentrates replicas near phase transitions where energy distributions barely overlap
(high resistance → more replicas inserted). Regions with easy swaps get sparser coverage.

**Convergence criterion:** max_i |a_i - a_target| < tolerance, or max tuning rounds reached.
Target acceptance rate: ~23% (optimal for adjacent-only swaps in 2D lattice models).

#### Phase B — Equilibration

- **Ladder is LOCKED.** No more temperature adjustments. This is critical: adjusting
  temperatures during production violates detailed balance and invalidates the dataset.
- Run sweeps + swaps until macroscopic observables stabilize.
- Convergence check: sliding window (e.g., last 20% vs first 20% of window) — if means
  agree within tolerance for E and |M| at every replica, equilibration is complete.

#### Phase C — Production

- Harvest spin snapshots at each temperature point.
- Compute τ_int for EVERY observable in the `sweep()` dict (from autocorrelation.py).
- **Bottleneck rule:** `τ_max = max(τ_obs for obs in sweep_dict.keys())`.
  Near T_c the bottleneck is typically |M|; near the BC tricritical point it's Q.
- **Thinning rule:** save one snapshot every `max(1, 3 × τ_max)` sweeps.
  Correlation between successive snapshots: e^{-3} ≈ 0.05 (safely independent).
- Store snapshots with metadata: (T, D, L, seed, sweep_index, τ_max, per-observable τ_int).

#### Multiprocessing Design

pybind11 model objects are NOT picklable. Two approaches:

**Option A — Process-per-replica (preferred for large L):**
Each worker process owns its own model instance. After sweeps, workers send
`(energy, spins_as_numpy)` back to the main process via `multiprocessing.Queue`.
Main process makes swap decisions and sends back `(new_temperature, new_spins)` or no-op.

**Option B — Sequential (simpler, fine for small L):**
Single process, loop over replicas. For L ≤ 64, sweep() takes ~μs and the overhead
of process synchronization likely exceeds the computation itself.

The manager should auto-select based on `L` and `n_replicas`, or accept a `parallel=True/False` flag.

### 2.2 I/O & CLI
- [ ] Step 2.2.1: `io.py` — HDF5 snapshot writer with metadata (T, D, L, seed, τ_int)
- [ ] Step 2.2.2: `generate_dataset.py` — CLI entry point orchestrating PT runs

## Phase 3: Validation & Diagnostics

- [ ] Step 3.1: `validation.py` — equilibration trace plots (E, M vs sweep)
- [ ] Step 3.2: `validation.py` — cluster scaling check ⟨n⟩ ~ L^{y_h}
- [ ] Step 3.3: Replica flow diagnostics — round-trip time histogram, per-gap acceptance plot

## Test Plan

- [x] Unit: PRNG smoke test (determinism, range, uniformity, autocorrelation) — `tests/test_foundation.py`
- [x] Unit: Neighbor table correctness for various L (shape, PBC, symmetry) — `tests/test_foundation.py`
- [x] Unit: Ising model construction, energy, magnetization — `tests/ising/test_model.py`
- [x] Unit: Ising Wolff detailed balance (2×2 chi-squared at 10 temperatures) — `tests/ising/test_wolff.py`
- [x] Unit: Ising Metropolis detailed balance (2×2 chi-squared at 10 temperatures) — `tests/ising/test_metropolis.py`
- [x] Unit: Ising sweep detailed balance + ergodicity — `tests/ising/test_sweep.py`
- [x] Unit: BC Wolff — vacancy barrier, cluster on pure ±1 matches Ising, seed-on-vacancy returns 0 — `tests/blume_capel/test_wolff.py`
- [x] Unit: BC Metropolis — `_delta_energy` correctness, 2×2 detailed balance (81 states) — `tests/blume_capel/test_metropolis.py`
- [x] Unit: BC sweep — detailed balance + ergodicity, observables dict includes Q — `tests/blume_capel/test_sweep.py`
- [ ] Unit: AT energy/magnetization consistency after sweep
- [ ] Unit: ACF + τ_int on synthetic AR(1) signal (known τ), first-zero-crossing cutoff — `tests/test_autocorrelation.py`
- [ ] Unit: PT swap acceptance matches exact Boltzmann formula — `tests/test_parallel_tempering.py`
- [ ] Unit: PT ladder tuning converges to uniform acceptance — `tests/test_parallel_tempering.py`
- [ ] Unit: PT phase separation — no ladder changes during Phase B/C — `tests/test_parallel_tempering.py`
- [ ] Integration: Full pipeline — PT equilibrate, measure τ_int, generate thinned snapshots

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
