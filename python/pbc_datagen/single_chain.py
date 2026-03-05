"""Single-chain MCMC runner — no replica exchange, single (param, T) point.

Simpler alternative to PTEngine for cases that don't need parallel tempering.
Uses the same C++ sweep() API: one Metropolis+Wolff hybrid sweep per call.

Reuses existing infrastructure:
- welch_equilibration_check() for equilibration detection
- tau_int_multi() for autocorrelation measurement
- SnapshotWriter / write_param_attrs / read_resume_state for HDF5 I/O
- _param_label / _derive_seed from orchestrator
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
from loguru import logger

import pbc_datagen._core as _core
from pbc_datagen.autocorrelation import tau_int_multi
from pbc_datagen.io import SnapshotWriter, _t_group_name, read_resume_state
from pbc_datagen.orchestrator import _derive_seed, _param_label
from pbc_datagen.parallel_tempering import welch_equilibration_check

Model = Union[_core.IsingModel, _core.BlumeCapelModel, _core.AshkinTellerModel]

_MODEL_CONSTRUCTORS = {
    "ising": _core.IsingModel,
    "blume_capel": _core.BlumeCapelModel,
    "ashkin_teller": _core.AshkinTellerModel,
}


def _make_model(model_type: str, L: int, param_value: float, T: float, seed: int) -> Model:
    """Create a single model instance and set its temperature + param."""
    m: Model
    if model_type == "blume_capel":
        bc = _core.BlumeCapelModel(L, seed)
        bc.set_crystal_field(param_value)
        bc.set_temperature(T)
        m = bc
    elif model_type == "ashkin_teller":
        at = _core.AshkinTellerModel(L, seed)
        at.set_four_spin_coupling(param_value)
        at.set_temperature(T)
        m = at
    else:
        m = _core.IsingModel(L, seed)
        m.set_temperature(T)
    return m


class SingleChainEngine:
    """Single-chain MCMC engine for one (param, T) point.

    Two-phase pipeline:
      1) equilibrate() — doubling Welch t-test, measure tau_int
      2) produce()     — harvest decorrelated snapshots to HDF5
    """

    def __init__(
        self,
        model_type: str,
        L: int,
        param_value: float,
        T: float,
        seed: int,
    ) -> None:
        if model_type not in _MODEL_CONSTRUCTORS:
            msg = f"Unknown model type: {model_type!r}"
            raise ValueError(msg)

        self.model_type = model_type
        self.L = L
        self.param_value = param_value
        self.T = T
        self.seed = seed

        self.model: Model = _make_model(model_type, L, param_value, T, seed)
        self.tau_max: float | None = None

        logger.info(
            "SingleChainEngine created: model={} L={} T={:.4f} seed={}",
            model_type,
            L,
            T,
            seed,
        )

    def equilibrate(
        self,
        n_initial: int = 10_000,
        n_max: int = 5_120_000,
        alpha: float = 0.05,
    ) -> None:
        """Equilibrate via doubling Welch t-test, then measure tau_int.

        Runs n_initial sweeps, checks equilibration.  If the test fails,
        doubles and retries up to n_max.

        Once equilibrated, trims 20% burn-in, measures tau_int_multi,
        and locks tau_max.

        Raises:
            RuntimeError: If equilibration fails after reaching n_max.
        """
        logger.info("Equilibrating (Welch t-test, n_initial={})", n_initial)
        n_sweeps = n_initial

        while n_sweeps <= n_max:
            # Run sweeps and collect per-sweep observables
            result = self.model.sweep(n_sweeps)

            # Convert sweep result to welch_equilibration_check format:
            # {obs_name: [series_as_list]} — 1 "T slot"
            obs_streams: dict[str, list[list[float]]] = {}
            for name, values in result.items():
                obs_streams[name] = [values.tolist()]

            if welch_equilibration_check(obs_streams, alpha=alpha):
                logger.debug("Welch test passed at n_sweeps={}", n_sweeps)
                # Trim 20% burn-in, measure tau_int
                burn_in = n_sweeps // 5
                trimmed: dict[str, npt.NDArray[np.float64]] = {}
                for name, values in result.items():
                    trimmed[name] = np.array(values[burn_in:], dtype=np.float64)

                _, self.tau_max = tau_int_multi(trimmed)
                logger.info("Equilibrated, tau_max={:.1f}", self.tau_max)
                return

            logger.debug("Welch test failed at n_sweeps={}, doubling", n_sweeps)
            n_sweeps *= 2

        msg = f"Equilibration failed: Welch t-test still failing after {n_max} sweeps."
        raise RuntimeError(msg)

    def produce(
        self,
        path: str | Path,
        n_snapshots: int = 100,
        seed_history: list[tuple[int, int]] | None = None,
    ) -> None:
        """Harvest decorrelated snapshots and stream to HDF5.

        Runs ``max(1, ceil(3 * tau_max))`` sweeps between each snapshot.

        Resume-safe: if *path* already has snapshots, only the remaining
        ``n_snapshots - n_existing`` are collected.

        Args:
            path: Output HDF5 file path.
            n_snapshots: Target total snapshots.
            seed_history: PRNG audit trail.

        Raises:
            RuntimeError: If tau_max is not set (equilibrate() not called).
        """
        if self.tau_max is None:
            msg = "tau_max not set — call equilibrate() before produce()"
            raise RuntimeError(msg)

        if seed_history is None:
            seed_history = [(0, self.seed)]

        L = self.L
        C = 2 if self.model_type == "ashkin_teller" else 1
        obs_names = list(self.model.observables().keys())
        thinning = max(1, math.ceil(3 * self.tau_max))

        logger.info(
            "Producing snapshots (target={}, thinning={} sweeps)",
            n_snapshots,
            thinning,
        )

        slot_keys = [_t_group_name(self.T)]

        # Metadata dict — written alongside every flush so crash-resume works
        metadata: dict[str, object] = {
            "model_type": self.model_type,
            "L": L,
            "param_value": self.param_value,
            "T_ladder": np.array([self.T], dtype=np.float64),
            "tau_max": self.tau_max,
            "r2t": np.array([0], dtype=np.int64),
            "t2r": np.array([0], dtype=np.int64),
            "seed": self.seed,
            "seed_history": json.dumps(seed_history),
        }

        with SnapshotWriter(path) as writer:
            # Create flat datasets or open existing ones for resume
            if "snapshots" not in writer._file:
                writer.create_datasets(slot_keys, n_snapshots, C, L, obs_names)
            else:
                writer.open_datasets()

            # Count existing snapshots
            n_existing = writer.snapshot_count
            n_remaining = n_snapshots - n_existing
            if n_remaining <= 0:
                logger.info(
                    "Already have {}/{} snapshots, nothing to do",
                    n_existing,
                    n_snapshots,
                )
                return

            if n_existing > 0:
                logger.info("Resuming from {}/{} snapshots", n_existing, n_snapshots)

            # Main production loop
            for snap_i in range(n_remaining):
                # Decorrelation sweeps (discard the observable time series)
                self.model.sweep(thinning)

                # Harvest snapshot — M=1 batch
                if self.model_type == "ashkin_teller":
                    spins = np.stack(
                        [self.model.sigma, self.model.tau]  # type: ignore[union-attr]
                    ).astype(np.int8)
                else:
                    spins = self.model.spins[np.newaxis].copy()  # type: ignore[union-attr]

                all_spins = spins[np.newaxis]  # (1, C, L, L)
                obs = self.model.observables()
                all_obs = {name: np.array([obs[name]]) for name in obs_names}

                writer.write_round(all_spins, all_obs)
                writer.write_metadata(metadata)
                writer.flush()

                done = n_existing + snap_i + 1
                if done % 10 == 0 or snap_i == n_remaining - 1:
                    logger.info("{}/{} snapshots collected", done, n_snapshots)


# ---------------------------------------------------------------------------
# File discovery + campaign runner (mirrors orchestrator.py)
# ---------------------------------------------------------------------------


def find_existing_single_hdf5(
    output_dir: str | Path,
    model_type: str,
    L: int,
    param_value: float,
    T: float,
) -> Path | None:
    """Find the most recent single-chain HDF5 file for this config.

    Filename pattern (no _R{n} since there are no replicas):
    - Ising: ``ising_L{L}_T={T:.4f}_{ts}.h5``
    - BC:    ``blume_capel_L{L}_D={D:.4f}_T={T:.4f}_{ts}.h5``
    - AT:    ``ashkin_teller_L{L}_U={U:.4f}_T={T:.4f}_{ts}.h5``

    Returns the newest match by timestamp suffix, or None.
    """
    directory = Path(output_dir)
    label = _param_label(model_type)
    if label is not None:
        pattern = f"{model_type}_L{L}_{label}={param_value:.4f}_T={T:.4f}_*.h5"
    else:
        pattern = f"{model_type}_L{L}_T={T:.4f}_*.h5"

    matches = sorted(directory.glob(pattern))
    if not matches:
        logger.debug("No existing single-chain HDF5 for pattern {}", pattern)
        return None

    def _timestamp(p: Path) -> int:
        return int(p.stem.rsplit("_", 1)[-1])

    result = max(matches, key=_timestamp)
    logger.debug("Found existing single-chain HDF5: {}", result.name)
    return result


def run_single_campaign(
    model_type: str,
    L: int,
    param_value: float,
    T: float,
    n_snapshots: int,
    output_dir: str | Path,
    force_new: bool = False,
) -> Path:
    """Run one single-chain campaign for a given (model, L, param, T).

    Fresh start: create new file, equilibrate, produce.
    Resume: find existing file, derive new seed, extend seed_history,
    equilibrate, produce (appends remaining snapshots).

    Returns the path to the output HDF5 file.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    label = _param_label(model_type)

    existing = (
        None if force_new else find_existing_single_hdf5(directory, model_type, L, param_value, T)
    )

    if existing is not None:
        # --- Resume ---
        logger.info("Resuming single-chain campaign from {}", existing.name)
        old_seed, state = read_resume_state(existing)
        seed_history: list[tuple[int, int]] = state["seed_history"]

        counts = state["snapshot_counts"]
        n_existing = min(counts.values()) if counts else 0

        if n_existing >= n_snapshots:
            logger.info(
                "Already complete ({}/{} snapshots), skipping",
                n_existing,
                n_snapshots,
            )
            return existing

        new_seed = _derive_seed(old_seed, n_existing)
        seed_history.append((n_existing, new_seed))

        engine = SingleChainEngine(model_type, L, param_value, T, new_seed)
        # Restore tau_max from saved state — no need to re-equilibrate
        engine.tau_max = state["tau_max"]
        engine.produce(existing, n_snapshots, seed_history=seed_history)
        return existing
    else:
        # --- Fresh start ---
        ts = int(time.time() * 1000)
        seed = ts % (2**63)
        if label is not None:
            filename = f"{model_type}_L{L}_{label}={param_value:.4f}_T={T:.4f}_{ts}.h5"
        else:
            filename = f"{model_type}_L{L}_T={T:.4f}_{ts}.h5"
        path = directory / filename
        logger.info("Fresh single-chain campaign: {}", filename)

        engine = SingleChainEngine(model_type, L, param_value, T, seed)
        engine.equilibrate()
        engine.produce(path, n_snapshots)
        logger.info("Single-chain campaign complete: {}", filename)
        return path
