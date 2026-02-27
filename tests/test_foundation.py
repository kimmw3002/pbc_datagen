"""Red-phase tests for Steps 1.0.2 (PRNG) and 1.0.3 (Lattice neighbor table).

These tests define the expected pybind11 API for the foundation layer.
All imports are lazy (inside test functions) so pytest can *collect*
the tests even though the C++ module doesn't exist yet.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# PRNG tests  (Step 1.0.2)
# ---------------------------------------------------------------------------


def test_rng_produces_floats_in_unit_interval() -> None:
    """Rng.uniform() must return floats in [0, 1)."""
    from pbc_datagen._core import Rng

    rng = Rng(seed=42)
    samples = [rng.uniform() for _ in range(1000)]
    assert all(0.0 <= x < 1.0 for x in samples)


def test_rng_deterministic_with_same_seed() -> None:
    """Same seed must produce the identical sequence."""
    from pbc_datagen._core import Rng

    rng_a = Rng(seed=123)
    rng_b = Rng(seed=123)
    seq_a = [rng_a.uniform() for _ in range(100)]
    seq_b = [rng_b.uniform() for _ in range(100)]
    assert seq_a == seq_b


def test_rng_different_seeds_produce_different_sequences() -> None:
    """Different seeds must produce different sequences (with overwhelming probability)."""
    from pbc_datagen._core import Rng

    rng_a = Rng(seed=1)
    rng_b = Rng(seed=2)
    seq_a = [rng_a.uniform() for _ in range(50)]
    seq_b = [rng_b.uniform() for _ in range(50)]
    assert seq_a != seq_b


def test_rng_autocorrelation_negligible() -> None:
    """Consecutive PRNG outputs must be uncorrelated at short lags.

    For N samples from a good PRNG, the sample autocorrelation at any
    non-zero lag should be ~0 with std ≈ 1/sqrt(N).  We check lags 1–20
    against a 4-sigma threshold.
    """
    from pbc_datagen._core import Rng

    N = 100_000
    rng = Rng(seed=7)
    samples = np.array([rng.uniform() for _ in range(N)])

    # Center the sequence (subtract mean)
    x = samples - samples.mean()
    var = np.dot(x, x)

    # Check autocorrelation at lags 1 through 20
    threshold = 4.0 / np.sqrt(N)  # ~4 sigma ≈ 0.013
    for lag in range(1, 21):
        acf = np.dot(x[:-lag], x[lag:]) / var
        assert abs(acf) < threshold, (
            f"Autocorrelation at lag {lag} = {acf:.5f}, exceeds ±{threshold:.5f}"
        )


def test_rng_uniformity() -> None:
    """PRNG output must be uniformly distributed across [0, 1).

    Split [0, 1) into bins and use a chi-squared test: if the PRNG
    is uniform, each bin should get roughly N/num_bins samples.
    """
    from pbc_datagen._core import Rng
    from scipy.stats import chisquare

    N = 100_000
    num_bins = 20
    rng = Rng(seed=314)
    samples = np.array([rng.uniform() for _ in range(N)])

    # Count how many samples land in each bin
    observed, _ = np.histogram(samples, bins=num_bins, range=(0.0, 1.0))
    expected = np.full(num_bins, N / num_bins)

    result = chisquare(observed, expected)
    assert result.pvalue > 0.001, (
        f"Chi-squared uniformity test failed: p={result.pvalue:.6f}, stat={result.statistic:.2f}"
    )


def test_rng_rand_below_stays_in_range() -> None:
    """Rng.rand_below(n) must return integers in [0, n)."""
    from pbc_datagen._core import Rng

    rng = Rng(seed=99)
    for n in [2, 5, 10, 100, 1000]:
        values = [rng.rand_below(n) for _ in range(500)]
        assert all(0 <= v < n for v in values), f"rand_below({n}) produced out-of-range value"


def test_rng_rand_below_uniformity() -> None:
    """rand_below(n) must produce a uniform distribution over [0, n).

    We use n = 128*128 = 16384 (a typical lattice size) and draw enough
    samples so each bin has ~100 expected counts, then chi-squared test.
    """
    from pbc_datagen._core import Rng
    from scipy.stats import chisquare

    n = 128 * 128
    samples_per_bin = 100
    N = n * samples_per_bin  # 1_638_400 total samples

    rng = Rng(seed=2025)
    counts = np.zeros(n, dtype=np.int64)
    for _ in range(N):
        counts[rng.rand_below(n)] += 1

    expected = np.full(n, samples_per_bin, dtype=np.float64)
    result = chisquare(counts, expected)
    assert result.pvalue > 0.001, (
        f"rand_below({n}) uniformity failed: chi2={result.statistic:.1f}, p={result.pvalue:.6f}"
    )


def test_rng_rand_below_autocorrelation() -> None:
    """Consecutive rand_below() outputs must be uncorrelated at short lags.

    We draw 500k samples from rand_below(128*128), center the sequence,
    and check that the autocorrelation at lags 1–20 is within 4 sigma of zero.
    """
    from pbc_datagen._core import Rng

    n = 128 * 128
    N = 500_000
    rng = Rng(seed=555)
    samples = np.array([rng.rand_below(n) for _ in range(N)], dtype=np.float64)

    x = samples - samples.mean()
    var = np.dot(x, x)

    threshold = 4.0 / np.sqrt(N)  # ~4 sigma
    for lag in range(1, 21):
        acf = np.dot(x[:-lag], x[lag:]) / var
        assert abs(acf) < threshold, (
            f"rand_below autocorrelation at lag {lag} = {acf:.5f}, exceeds ±{threshold:.5f}"
        )


# ---------------------------------------------------------------------------
# Neighbor-table tests  (Step 1.0.3)
# ---------------------------------------------------------------------------


def _expected_neighbors(L: int, site: int) -> set[int]:
    """Pure-Python reference: the 4 PBC neighbors of *site* on an L×L grid."""
    r, c = divmod(site, L)
    north = ((r - 1) % L) * L + c
    south = ((r + 1) % L) * L + c
    east = r * L + (c + 1) % L
    west = r * L + (c - 1) % L
    return {north, south, east, west}


def test_neighbor_table_shape() -> None:
    """make_neighbor_table(L) returns an (L*L, 4) integer array."""
    from pbc_datagen._core import make_neighbor_table

    L = 8
    table = make_neighbor_table(L)
    assert isinstance(table, np.ndarray)
    assert table.shape == (L * L, 4)
    assert np.issubdtype(table.dtype, np.integer)


def test_neighbor_table_values_in_range() -> None:
    """Every neighbor index must lie in [0, L*L)."""
    from pbc_datagen._core import make_neighbor_table

    L = 16
    table = make_neighbor_table(L)
    assert np.all(table >= 0)
    assert np.all(table < L * L)


def test_neighbor_table_pbc_corners() -> None:
    """Corner sites must wrap around in both directions via PBC."""
    from pbc_datagen._core import make_neighbor_table

    L = 4
    table = make_neighbor_table(L)
    # Top-left corner: site 0 = (0,0)
    assert set(table[0]) == _expected_neighbors(L, 0)
    # Bottom-right corner: site L*L-1 = (L-1, L-1)
    assert set(table[L * L - 1]) == _expected_neighbors(L, L * L - 1)


def test_neighbor_table_pbc_edges() -> None:
    """Edge (non-corner) sites must wrap in one direction."""
    from pbc_datagen._core import make_neighbor_table

    L = 6
    table = make_neighbor_table(L)
    # Top edge, middle column: site (0, 3)
    site = 3
    assert set(table[site]) == _expected_neighbors(L, site)
    # Left edge, middle row: site (3, 0)
    site = 3 * L
    assert set(table[site]) == _expected_neighbors(L, site)


def test_neighbor_table_bulk_site() -> None:
    """A bulk site (away from edges) has straightforward ±1 and ±L neighbors."""
    from pbc_datagen._core import make_neighbor_table

    L = 8
    table = make_neighbor_table(L)
    # Site (3, 4) = index 28.  None of its neighbors should wrap.
    site = 3 * L + 4
    expected = {site - L, site + L, site - 1, site + 1}
    assert set(table[site]) == expected


def test_neighbor_table_symmetry() -> None:
    """Neighbor relation is symmetric: if j in neighbors(i), then i in neighbors(j)."""
    from pbc_datagen._core import make_neighbor_table

    L = 10
    table = make_neighbor_table(L)
    for i in range(L * L):
        for j in table[i]:
            assert i in table[j], f"Site {j} is a neighbor of {i}, but {i} is not a neighbor of {j}"


def test_neighbor_table_various_sizes() -> None:
    """Neighbor table must be correct for several lattice sizes."""
    from pbc_datagen._core import make_neighbor_table

    for L in [2, 3, 5, 16, 32]:
        table = make_neighbor_table(L)
        assert table.shape == (L * L, 4), f"Wrong shape for L={L}"
        for site in range(L * L):
            assert set(table[site]) == _expected_neighbors(L, site), (
                f"Wrong neighbors for site {site} at L={L}"
            )
