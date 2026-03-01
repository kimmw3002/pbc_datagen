"""Tests for pt_collect_obs and pt_rounds (Steps 1.5.6–1.5.7).

pt_collect_obs: records ALL per-T-slot observables each round.
pt_rounds: thin composition loop calling sweep + exchange + labels + histograms.
"""

from __future__ import annotations


def _make_rng(seed: int = 42):
    from pbc_datagen._core import Rng

    return Rng(seed)


def _make_ising_replicas(M: int, L: int = 4, seed_base: int = 100):
    from pbc_datagen._core import IsingModel

    return [IsingModel(L, seed_base + i) for i in range(M)]


# Ising observables() returns: energy, m, abs_m
ISING_OBS_KEYS = {"energy", "m", "abs_m"}


# ── 1.5.6  pt_collect_obs ────────────────────────────────────────────


class TestCollectObs:
    def test_records_all_observables_per_slot(self):
        """After one call, each obs key has one value per T slot."""
        from pbc_datagen._core import pt_collect_obs_ising

        M = 3
        replicas = _make_ising_replicas(M)
        t2r = list(range(M))
        for i, r in enumerate(replicas):
            r.set_temperature(1.0 + i)

        obs_streams: dict[str, list[list[float]]] = {
            k: [[] for _ in range(M)] for k in ISING_OBS_KEYS
        }

        pt_collect_obs_ising(obs_streams, replicas, t2r, M)

        for key in ISING_OBS_KEYS:
            for t in range(M):
                assert len(obs_streams[key][t]) == 1

        # Energy should match the replica at that slot
        for t in range(M):
            assert obs_streams["energy"][t][0] == replicas[t2r[t]].energy()

    def test_accumulates_over_calls(self):
        """Multiple calls append to the streams."""
        from pbc_datagen._core import pt_collect_obs_ising

        M = 2
        replicas = _make_ising_replicas(M)
        t2r = list(range(M))
        for r in replicas:
            r.set_temperature(2.0)

        obs_streams: dict[str, list[list[float]]] = {
            k: [[] for _ in range(M)] for k in ISING_OBS_KEYS
        }

        for _ in range(5):
            pt_collect_obs_ising(obs_streams, replicas, t2r, M)

        for key in ISING_OBS_KEYS:
            for t in range(M):
                assert len(obs_streams[key][t]) == 5


# ── 1.5.7  pt_rounds ────────────────────────────────────────────────


class TestPtRoundsNoObs:
    def test_returns_empty_obs_when_not_tracking(self):
        """track_observables=False → obs_streams is empty dict."""
        from pbc_datagen._core import pt_rounds_ising

        M = 3
        replicas = _make_ising_replicas(M)
        temps = [1.5, 2.0, 2.5]
        for i, r in enumerate(replicas):
            r.set_temperature(temps[i])
        r2t = list(range(M))
        t2r = list(range(M))
        labels = [0] * M

        result = pt_rounds_ising(replicas, temps, r2t, t2r, labels, 10, _make_rng(), False)

        assert result["obs_streams"] == {}


class TestPtRoundsIntegration:
    def test_round_trips_with_close_temps(self):
        """Close temperatures → high acceptance → replicas diffuse → round trips > 0."""
        from pbc_datagen._core import pt_rounds_ising

        M = 4
        replicas = _make_ising_replicas(M, L=4)
        temps = [2.0, 2.01, 2.02, 2.03]
        for i, r in enumerate(replicas):
            r.set_temperature(temps[i])
        r2t = list(range(M))
        t2r = list(range(M))
        labels = [0] * M

        result = pt_rounds_ising(replicas, temps, r2t, t2r, labels, 500, _make_rng(), False)

        assert result["round_trip_count"] > 0

    def test_f_monotonicity(self):
        """UP fraction should be higher at cold end than hot end."""
        from pbc_datagen._core import pt_rounds_ising

        M = 5
        replicas = _make_ising_replicas(M, L=4)
        temps = [1.5, 1.8, 2.1, 2.4, 2.7]
        for i, r in enumerate(replicas):
            r.set_temperature(temps[i])
        r2t = list(range(M))
        t2r = list(range(M))
        labels = [0] * M

        result = pt_rounds_ising(replicas, temps, r2t, t2r, labels, 1000, _make_rng(), False)

        n_up = result["n_up"]
        n_down = result["n_down"]

        f = []
        for t in range(M):
            total = n_up[t] + n_down[t]
            if total > 0:
                f.append(n_up[t] / total)
            else:
                f.append(None)

        assert f[0] is not None and f[-1] is not None
        assert f[0] > f[-1], f"f[0]={f[0]:.3f} should be > f[-1]={f[-1]:.3f}"

    def test_obs_streams_has_all_keys_when_tracking(self):
        """track_observables=True → obs_streams has all Ising keys, n_rounds entries."""
        from pbc_datagen._core import pt_rounds_ising

        M = 3
        n_rounds = 20
        replicas = _make_ising_replicas(M, L=4)
        temps = [1.5, 2.0, 2.5]
        for i, r in enumerate(replicas):
            r.set_temperature(temps[i])
        r2t = list(range(M))
        t2r = list(range(M))
        labels = [0] * M

        result = pt_rounds_ising(replicas, temps, r2t, t2r, labels, n_rounds, _make_rng(), True)

        obs = result["obs_streams"]
        assert set(obs.keys()) == ISING_OBS_KEYS
        for key in ISING_OBS_KEYS:
            assert len(obs[key]) == M
            for t in range(M):
                assert len(obs[key][t]) == n_rounds
