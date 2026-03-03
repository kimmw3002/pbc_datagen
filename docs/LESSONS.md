# Lessons Learned

Hard-won insights from building this codebase. Read this before writing new code.

## Physics / Algorithm

- **Metropolis site selection must be random**, not sequential. A sequential sweep
  (site 0, 1, 2, ..., N-1) creates correlations between proposals. At high T with
  ~100% acceptance it degenerates into deterministic oscillation (all+1 → all-1 → ...).
  Always pick N random sites per sweep.

- **Both Wolff and Metropolis independently satisfy detailed balance** for the Ising model.
  Either alone is a valid sampler. The hybrid (Wolff + Metropolis) is used because Wolff
  kills critical slowing down while Metropolis handles local decorrelation.

- **The 2×2 exact partition function is the gold standard test.** Z(T) = 2exp(8/T) + 12 + 2exp(-8/T).
  Only 3 energy levels (E = -8, 0, +8). Chi-squared test against exact P(E) at multiple
  temperatures is the definitive check for detailed balance.

## Statistics / Testing

- **Chi-squared requires expected counts >= 5 per bin.** At low T the rare states (e.g. E=+8)
  have tiny probability. With too few samples the expected count drops below 5 and the test
  becomes unreliable. Fix: increase samples (we use 500k for the 2×2 tests).

- **After dropping low-expected bins, rescale expected to match the observed sum.** Dropping
  bins with expected < 5 removes a tiny fraction of probability mass, making `sum(obs) !=
  sum(exp)`. scipy's `chisquare` rejects this mismatch. Fix: `exp *= obs.sum() / exp.sum()`
  after filtering. The rescaling is negligible (< 0.001%) and keeps the test valid.

- **Welch t-test on raw MCMC samples = inflated false-positive rate.** `ttest_ind`
  assumes independence, but MCMC chains have autocorrelation time τ_int. The standard
  error is underestimated by √(2τ_int), inflating the t-statistic. With Bonferroni
  over 150 tests (3 obs × 50 T-slots), the per-test false-positive rate exceeds 50%
  for τ_int ≈ 10, making overall rejection near-certain. Fix: estimate τ_int from
  the series, average into blocks of `ceil(3×τ_int)`, then run `ttest_ind` on the
  approximately-independent block means. IID data gets block_size ≈ 2 (thousands of
  blocks, no power loss); drifting data gets huge τ → 0 blocks → correct rejection.

- **Welch t-test + near-constant observables = catastrophic cancellation.** At low T,
  observables like `abs_m_baxter` saturate to ~1.0 with only float-noise variance (~1e-30).
  If the first/last 20% segments have a tiny systematic offset (e.g., 1e-15 from warmup),
  `ttest_ind` produces t≈180, p=0.0 — false rejection. Exactly constant data gives NaN.
  Fix: check `std < atol + rtol × |mean|` on both segments before calling `ttest_ind`;
  also treat NaN p-values as passing. Same spirit as the `tau_int` constant-observable
  guard (commit 300237b).

## C++ / Build

- **Rebuild after C++ changes:** `uv sync --all-extras --reinstall-package pbc-datagen`.
  Plain `uv sync` only tracks Python metadata and won't recompile .cpp/.hpp changes.

- **Update `_core.pyi` when adding new bindings.** mypy uses the stub file, not the compiled
  .so. Forgetting to update it causes spurious mypy errors.

- **`std::vector<bool>` packs 8 bools per byte** — good for large visited/in_cluster arrays
  but slightly slower per-access than `std::vector<char>`.

- **Use explicit stack, not recursion, for DFS** on lattice clusters. Recursive DFS can
  blow the call stack on large lattices (256×256 = 65k deep). `std::vector<int>` as a
  stack lives on the heap and grows as needed.
