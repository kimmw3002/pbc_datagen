"""2D parameter-space parallel tempering — Phase A/B/C pipeline.

Phase A: Spectral connectivity check.
Phase B: Two-initialization convergence (Gelman-Rubin).
Phase C: Production snapshot harvesting with phase-crossing tracking.

Used for Blume-Capel (D) and Ashkin-Teller (U) where 1D PT fails
near first-order transitions.  The 2D grid routes replicas around
the transition endpoint through the second-order region.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Union

import numpy as np
import numpy.typing as npt
from loguru import logger

import pbc_datagen._core as _core
from pbc_datagen.autocorrelation import tau_int_multi
from pbc_datagen.convergence import convergence_check
from pbc_datagen.io import SnapshotWriter, _slot_group_name_2d
from pbc_datagen.spectral import check_connectivity

Model = Union[_core.BlumeCapelModel, _core.AshkinTellerModel]

_PARAM_LABELS: dict[str, str] = {
    "blume_capel": "D",
    "ashkin_teller": "U",
}

_MODEL_CONSTRUCTORS: dict[str, type] = {
    "blume_capel": _core.BlumeCapelModel,
    "ashkin_teller": _core.AshkinTellerModel,
}

_PT_ROUNDS_2D_FN: dict[str, Callable[..., _core.PT2DResult]] = {
    "blume_capel": _core.pt_rounds_2d_bc,
    "ashkin_teller": _core.pt_rounds_2d_at,
}


# ---------------------------------------------------------------------------
# Replica creation helpers
# ---------------------------------------------------------------------------


def _make_replicas_2d(
    model_type: str,
    L: int,
    M: int,
    seed: int,
    *,
    init: str = "cold",
) -> tuple[list[Model], _core.Rng]:
    """Create M replicas for a 2D PT grid.

    Parameters
    ----------
    init : str
        ``"cold"`` — all spins +1 (constructor default).
        ``"random"`` — randomize spins after construction.
    """
    rng = _core.Rng(seed)
    ctor = _MODEL_CONSTRUCTORS[model_type]
    replicas: list[Model] = []

    for i in range(M):
        replica_seed = (seed + i + 1) % (2**63)
        m: Model = ctor(L, replica_seed)
        replicas.append(m)

    if init == "random":
        np_rng = np.random.default_rng(seed ^ 0xDEAD_BEEF)
        _randomize_all(replicas, model_type, L, np_rng)

    return replicas, rng


def _randomize_all(
    replicas: list[Model],
    model_type: str,
    L: int,
    np_rng: np.random.Generator,
) -> None:
    """Randomize spins for all replicas (hot-start initialization)."""
    N = L * L
    for m in replicas:
        if model_type == "blume_capel":
            vals = np_rng.choice([-1, 0, 1], size=N)
            for site in range(N):
                m.set_spin(site, int(vals[site]))  # type: ignore[union-attr]
        elif model_type == "ashkin_teller":
            svals = np_rng.choice([-1, 1], size=N)
            tvals = np_rng.choice([-1, 1], size=N)
            for site in range(N):
                m.set_sigma(site, int(svals[site]))  # type: ignore[union-attr]
                m.set_tau(site, int(tvals[site]))  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_acceptance(result: _core.PT2DResult, label: str) -> None:
    """Log mean T- and P-direction exchange acceptance rates at DEBUG level."""
    t_acc = np.array(result["t_accepts"], dtype=np.float64)
    t_att = np.array(result["t_attempts"], dtype=np.float64)
    p_acc = np.array(result["p_accepts"], dtype=np.float64)
    p_att = np.array(result["p_attempts"], dtype=np.float64)
    t_rate = float(t_acc.sum() / t_att.sum()) if t_att.sum() > 0 else 0.0
    p_rate = float(p_acc.sum() / p_att.sum()) if p_att.sum() > 0 else 0.0
    logger.debug(
        "  {} accept: T={:.3f}  P={:.3f}",
        label,
        t_rate,
        p_rate,
    )


# ---------------------------------------------------------------------------
# PTEngine2D
# ---------------------------------------------------------------------------


class PTEngine2D:
    """2D parameter-space parallel tempering for BC/AT models.

    Grid: ``n_T × n_P`` replicas.  ``slot(i, j) = i * n_P + j``.
    Row i = temperature index (0 = cold), column j = parameter index.

    Phases:
      A) check_connectivity()  — spectral gap on exchange graph
      B) equilibrate()         — two-init convergence + τ_int
      C) produce()             — snapshot harvesting + phase crossings
    """

    def __init__(
        self,
        model_type: str,
        L: int,
        T_range: tuple[float, float],
        param_range: tuple[float, float],
        n_T: int,
        n_P: int,
        seed: int,
    ) -> None:
        if model_type not in _MODEL_CONSTRUCTORS:
            msg = f"2D PT only supports blume_capel and ashkin_teller, got {model_type!r}"
            raise ValueError(msg)

        self.model_type = model_type
        self.L = L
        self.n_T = n_T
        self.n_P = n_P
        self.M = n_T * n_P
        self.seed = seed
        self.param_label = _PARAM_LABELS[model_type]

        self.temps = np.geomspace(T_range[0], T_range[1], n_T)
        self.params = np.linspace(param_range[0], param_range[1], n_P)

        # Create replicas (cold-start by default)
        self.replicas: list[Model]
        self.replicas, self.rng = _make_replicas_2d(model_type, L, self.M, seed, init="cold")

        # Address maps — identity at start
        self.r2s = list(range(self.M))
        self.s2r = list(range(self.M))

        # C++ dispatch
        self._pt_rounds_2d = _PT_ROUNDS_2D_FN[model_type]

        # State flags
        self.connectivity_checked = False
        self.tau_max: float | None = None

        logger.info(
            "PTEngine2D created: model={} L={} grid={}×{} ({}R) "
            "T=[{:.4f},{:.4f}] {}=[{:.4f},{:.4f}]",
            model_type,
            L,
            n_T,
            n_P,
            self.M,
            T_range[0],
            T_range[1],
            self.param_label,
            param_range[0],
            param_range[1],
        )

    def _run_pt(
        self,
        replicas: list[Model],
        r2s: list[int],
        s2r: list[int],
        n_rounds: int,
        rng: _core.Rng,
        track_obs: bool,
    ) -> _core.PT2DResult:
        """Run pt_rounds_2d on the given replica set."""
        return self._pt_rounds_2d(
            replicas,
            self.temps.tolist(),
            self.params.tolist(),
            r2s,
            s2r,
            n_rounds,
            rng,
            track_obs,
        )

    # ---- Phase A ----

    def check_connectivity(
        self,
        n_rounds: int = 100,
        min_gap: float | None = None,
    ) -> None:
        """Phase A: run exchanges and verify spectral connectivity.

        The default threshold scales as ``2/M`` (inverse grid size) so
        that large grids aren't penalised by their naturally smaller
        spectral gaps.

        Raises RuntimeError with Fiedler diagnostic if the grid is
        disconnected or poorly connected.
        """
        if min_gap is None:
            min_gap = 2.0 / self.M
        logger.info(
            "Phase A: checking connectivity (n_rounds={}, min_gap={:.6f})",
            n_rounds,
            min_gap,
        )

        result = self._run_pt(self.replicas, self.r2s, self.s2r, n_rounds, self.rng, False)

        # Compute per-edge acceptance rates
        t_acc = np.array(result["t_accepts"], dtype=np.float64)
        t_att = np.array(result["t_attempts"], dtype=np.float64)
        p_acc = np.array(result["p_accepts"], dtype=np.float64)
        p_att = np.array(result["p_attempts"], dtype=np.float64)

        t_rates = np.where(t_att > 0, t_acc / t_att, 0.0)
        p_rates = np.where(p_att > 0, p_acc / p_att, 0.0)

        conn = check_connectivity(self.n_T, self.n_P, t_rates, p_rates, min_gap)

        if not conn.passed:
            diag = ""
            if conn.fiedler is not None:
                signs = conn.fiedler.reshape(self.n_T, self.n_P)
                diag = f"\nFiedler vector sign pattern (T rows × param cols):\n{np.sign(signs)}"
            msg = (
                f"Grid is poorly connected: spectral gap = {conn.gap:.6f} "
                f"< {min_gap}.{diag}\n"
                f"Try adding more T or {self.param_label} points, or widen the ranges."
            )
            raise RuntimeError(msg)

        logger.info("Phase A: passed, spectral gap = {:.4f}", conn.gap)
        self.connectivity_checked = True

    # ---- Phase B ----

    def equilibrate(
        self,
        n_initial: int = 100,
        n_max: int = 51200,
        alpha: float = 0.05,
    ) -> None:
        """Phase B: two-initialization convergence check + τ_int measurement.

        Runs cold-start and random-start 2D PT campaigns, compares via
        convergence_check.  Uses doubling schedule.  On success, measures
        τ_int and keeps the cold-start replicas for production.

        Raises RuntimeError if connectivity not checked or convergence fails.
        """
        if not self.connectivity_checked:
            msg = "Must call check_connectivity() before equilibrate()"
            raise RuntimeError(msg)

        logger.info("Phase B: two-initialization convergence check")
        n_rounds = n_initial

        # Create replicas once — they are mutated in-place across doubling iterations
        replicas_cold, rng_cold = _make_replicas_2d(
            self.model_type, self.L, self.M, self.seed, init="cold"
        )
        r2s_cold = list(range(self.M))
        s2r_cold = list(range(self.M))

        replicas_hot, rng_hot = _make_replicas_2d(
            self.model_type, self.L, self.M, self.seed + 1, init="random"
        )
        r2s_hot = list(range(self.M))
        s2r_hot = list(range(self.M))

        while n_rounds <= n_max:
            logger.debug("Phase B: cold run — n_rounds={}", n_rounds)
            result_cold = self._run_pt(replicas_cold, r2s_cold, s2r_cold, n_rounds, rng_cold, True)
            _log_acceptance(result_cold, label="cold")

            logger.debug("Phase B: hot run — n_rounds={}", n_rounds)
            result_hot = self._run_pt(replicas_hot, r2s_hot, s2r_hot, n_rounds, rng_hot, True)
            _log_acceptance(result_hot, label="hot")

            # Compare
            obs_cold: dict[str, npt.NDArray[np.float64]] = result_cold["obs_streams"]
            obs_hot: dict[str, npt.NDArray[np.float64]] = result_hot["obs_streams"]
            del result_hot  # free hot obs_streams early
            n_tests = len(obs_cold) * len(obs_cold[next(iter(obs_cold))])
            logger.debug("Phase B: convergence_check — {} (obs × slots) tests", n_tests)
            conv = convergence_check(obs_cold, obs_hot, alpha)
            del obs_hot
            n_disagree_total = sum(sum(flags) for flags in conv.disagreement_map.values())
            logger.debug(
                "Phase B: convergence_check done — {}/{} slots disagree",
                n_disagree_total,
                n_tests,
            )

            if conv.converged:
                logger.info("Phase B: converged at n_rounds={}", n_rounds)
                del replicas_hot, rng_hot, r2s_hot, s2r_hot

                # Measure τ_int from cold-start data (last 80%)
                burn_in = n_rounds // 5
                trimmed: dict[str, npt.NDArray[np.float64]] = {}
                for name, slots in obs_cold.items():
                    for s in range(self.M):
                        key = f"{name}_s{s}"
                        trimmed[key] = slots[s, burn_in:]

                _, self.tau_max = tau_int_multi(trimmed)
                logger.info("Phase B: tau_max={:.1f}", self.tau_max)

                # Hand the warm cold-start replicas to Phase C
                self.replicas = replicas_cold
                self.r2s = r2s_cold
                self.s2r = s2r_cold
                self.rng = rng_cold
                return

            # Log disagreement map
            n_disagree = 0
            for obs_name, flags in conv.disagreement_map.items():
                for s, disagree in enumerate(flags):
                    if disagree:
                        i_t = s // self.n_P
                        j_p = s % self.n_P
                        n_disagree += 1
                        if n_disagree <= 10:
                            logger.debug(
                                "Disagree: {} at slot ({},{}) T={:.4f} {}={:.4f}",
                                obs_name,
                                i_t,
                                j_p,
                                self.temps[i_t],
                                self.param_label,
                                self.params[j_p],
                            )

            logger.debug(
                "Phase B: {} disagreements at n_rounds={}, doubling",
                n_disagree,
                n_rounds,
            )
            n_rounds *= 2

        msg = (
            f"Two-initialization convergence failed after {n_max} rounds. "
            f"The cold-start and random-start runs still disagree — the grid "
            f"may be insufficient near the first-order line."
        )
        raise RuntimeError(msg)

    # ---- Phase C ----

    def produce(
        self,
        path: str | Path,
        n_snapshots: int = 100,
        seed_history: list[tuple[int, int]] | None = None,
    ) -> None:
        """Phase C: harvest snapshots at all (T, param) grid points.

        Thinning: ``max(1, ceil(3 × τ_max))`` rounds between snapshots.
        Tracks phase crossings (Q transitions across 0.5) for BC.

        Resume-safe: if *path* already has snapshots, only the remaining
        are collected.
        """
        if self.tau_max is None:
            msg = "tau_max not set — call equilibrate() before produce()"
            raise RuntimeError(msg)

        if seed_history is None:
            seed_history = [(0, self.seed)]

        L = self.L
        C = 2 if self.model_type == "ashkin_teller" else 1
        obs_names = list(self.replicas[0].observables().keys())
        thinning = max(1, math.ceil(3 * self.tau_max))
        pl = self.param_label

        logger.info("Phase C: producing snapshots (target={}, thinning={})", n_snapshots, thinning)

        # Phase-crossing tracker: per-replica last-Q and crossing count (BC only)
        track_crossings = self.model_type == "blume_capel"
        last_q: list[float | None] = [None] * self.M
        phase_crossings = 0

        slot_keys = [
            _slot_group_name_2d(self.temps[s // self.n_P], self.params[s % self.n_P], pl)
            for s in range(self.M)
        ]

        # Metadata dict — written alongside every flush so crash-resume works
        metadata: dict[str, object] = {
            "model_type": self.model_type,
            "L": L,
            "param_label": pl,
            "temps": np.asarray(self.temps, dtype=np.float64),
            "params": np.asarray(self.params, dtype=np.float64),
            "tau_max": self.tau_max,
            "seed": self.seed,
            "seed_history": json.dumps(seed_history),
            "pt_mode": "2d",
        }

        with SnapshotWriter(path) as writer:
            # Create flat datasets or open existing ones for resume
            if "snapshots" not in writer._file:
                writer.create_datasets(slot_keys, n_snapshots, C, L, obs_names)
            else:
                writer.open_datasets()

            # Count existing snapshots (all slots in lockstep)
            n_existing = writer.snapshot_count
            n_remaining = n_snapshots - n_existing
            if n_remaining <= 0:
                logger.info("Phase C: already have {}/{}, done", n_existing, n_snapshots)
                return

            if n_existing > 0:
                logger.info("Phase C: resuming from {}/{}", n_existing, n_snapshots)

            for snap_i in range(n_remaining):
                # Thinning rounds
                self._run_pt(self.replicas, self.r2s, self.s2r, thinning, self.rng, False)

                # Batch-collect one snapshot from every slot
                all_spins = np.empty((self.M, C, L, L), dtype=np.int8)
                all_obs: dict[str, npt.NDArray[np.float64]] = {
                    name: np.empty(self.M, dtype=np.float64) for name in obs_names
                }
                for s in range(self.M):
                    replica = self.replicas[self.s2r[s]]

                    if self.model_type == "ashkin_teller":
                        all_spins[s] = np.stack(
                            [replica.sigma, replica.tau]  # type: ignore[union-attr]
                        ).astype(np.int8)
                    else:
                        all_spins[s] = replica.spins[np.newaxis].copy()  # type: ignore[union-attr]

                    obs = replica.observables()
                    for name in obs_names:
                        all_obs[name][s] = obs[name]

                    # Phase crossing tracking (BC: Q crosses 0.5)
                    if track_crossings:
                        r = self.s2r[s]
                        q = obs.get("q", 0.0)
                        if last_q[r] is not None:
                            if (last_q[r] > 0.5) != (q > 0.5):  # type: ignore[operator]
                                phase_crossings += 1
                        last_q[r] = q

                writer.write_round(all_spins, all_obs)

                # Persist metadata + address maps alongside every flush
                metadata["r2s"] = np.asarray(self.r2s, dtype=np.int64)
                metadata["s2r"] = np.asarray(self.s2r, dtype=np.int64)
                writer.write_metadata(metadata)
                writer.flush()

                done = n_existing + snap_i + 1
                if done % 10 == 0 or snap_i == n_remaining - 1:
                    logger.info("Phase C: {}/{} snapshots", done, n_snapshots)

        if track_crossings:
            if phase_crossings < 10:
                logger.warning(
                    "Phase C: only {} phase crossings — snapshots near "
                    "the transition may not be ergodic",
                    phase_crossings,
                )
            else:
                logger.info("Phase C: {} phase crossings", phase_crossings)
