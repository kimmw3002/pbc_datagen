"""Tests for pt_exchange_round() — M random swap proposals (Step 1.5.2).

Each round picks M random gaps, attempts Metropolis exchange, and updates
the replica↔temperature address maps on accept.
"""

from __future__ import annotations


def _make_rng(seed: int = 42):
    from pbc_datagen._core import Rng

    return Rng(seed)


def _make_ising_replicas(M: int, L: int = 4, seed_base: int = 100):
    """Create M Ising replicas with distinct seeds."""
    from pbc_datagen._core import IsingModel

    replicas = []
    for i in range(M):
        model = IsingModel(L, seed_base + i)
        replicas.append(model)
    return replicas


def _identity_maps(M: int) -> tuple[list[int], list[int]]:
    """Return identity r2t and t2r maps."""
    return list(range(M)), list(range(M))


def _is_inverse_permutation(r2t: list[int], t2r: list[int]) -> bool:
    """Check that r2t and t2r are inverse permutations of each other."""
    M = len(r2t)
    if len(t2r) != M:
        return False
    for r in range(M):
        if t2r[r2t[r]] != r:
            return False
    for t in range(M):
        if r2t[t2r[t]] != t:
            return False
    return True


class TestAttemptCounting:
    """After one round of M replicas, total attempts across all gaps == M."""

    def test_total_attempts_equals_M(self):
        from pbc_datagen._core import pt_exchange_round_ising

        M = 5
        replicas = _make_ising_replicas(M)
        temps = [1.0, 1.5, 2.0, 2.5, 3.0]
        for i, r in enumerate(replicas):
            r.set_temperature(temps[i])
        r2t, t2r = _identity_maps(M)
        n_accepts = [0] * (M - 1)
        n_attempts = [0] * (M - 1)

        pt_exchange_round_ising(replicas, temps, r2t, t2r, n_accepts, n_attempts, _make_rng())

        assert sum(n_attempts) == M


class TestUniformCoverage:
    """After many rounds, every gap has attempts > 0 (uniform random coverage)."""

    def test_all_gaps_attempted(self):
        from pbc_datagen._core import pt_exchange_round_ising

        M = 5
        replicas = _make_ising_replicas(M)
        temps = [1.0, 1.5, 2.0, 2.5, 3.0]
        for i, r in enumerate(replicas):
            r.set_temperature(temps[i])
        r2t, t2r = _identity_maps(M)
        n_accepts = [0] * (M - 1)
        n_attempts = [0] * (M - 1)
        rng = _make_rng()

        # 100 rounds of M=5 proposals each = 500 proposals across 4 gaps
        for _ in range(100):
            pt_exchange_round_ising(replicas, temps, r2t, t2r, n_accepts, n_attempts, rng)

        for g in range(M - 1):
            assert n_attempts[g] > 0, f"Gap {g} was never attempted"


class TestMapConsistency:
    """On accepted swaps, r2t and t2r must stay inverse permutations."""

    def test_maps_stay_inverse(self):
        from pbc_datagen._core import pt_exchange_round_ising

        M = 5
        replicas = _make_ising_replicas(M)
        temps = [1.0, 1.5, 2.0, 2.5, 3.0]
        for i, r in enumerate(replicas):
            r.set_temperature(temps[i])
        r2t, t2r = _identity_maps(M)
        n_accepts = [0] * (M - 1)
        n_attempts = [0] * (M - 1)
        rng = _make_rng()

        for _ in range(200):
            pt_exchange_round_ising(replicas, temps, r2t, t2r, n_accepts, n_attempts, rng)
            assert _is_inverse_permutation(r2t, t2r), (
                f"Maps are not inverse permutations: r2t={r2t}, t2r={t2r}"
            )


class TestSameTemperature:
    """When all temps are identical, every exchange accepts (Δ(1/T) = 0)."""

    def test_same_temp_all_accept(self):
        from pbc_datagen._core import pt_exchange_round_ising

        M = 4
        replicas = _make_ising_replicas(M)
        T = 2.0
        temps = [T] * M
        for r in replicas:
            r.set_temperature(T)
        r2t, t2r = _identity_maps(M)
        n_accepts = [0] * (M - 1)
        n_attempts = [0] * (M - 1)
        rng = _make_rng()

        n_rounds = 100
        for _ in range(n_rounds):
            pt_exchange_round_ising(replicas, temps, r2t, t2r, n_accepts, n_attempts, rng)

        for g in range(M - 1):
            assert n_accepts[g] == n_attempts[g], (
                f"Gap {g}: accepts={n_accepts[g]} != attempts={n_attempts[g]}"
            )
