"""Parallel tempering orchestration — Phase A/B/C pipeline.

Phase A: KTH ladder tuning (Katzgraber-Trebst-Huse, PRE 2006).
Phase B: Equilibration + τ_int measurement.
Phase C: Production snapshot harvesting.

The hot inner loop (sweep + exchange + histograms) lives in C++
(pt_engine.hpp).  This module handles the outer feedback logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
from scipy import stats

import pbc_datagen._core as _core
from pbc_datagen.autocorrelation import tau_int_multi
from pbc_datagen.io import SnapshotWriter, write_param_attrs

# Union of all model types — mypy needs this to see set_temperature etc.
Model = Union[_core.IsingModel, _core.BlumeCapelModel, _core.AshkinTellerModel]


# ---------------------------------------------------------------------------
# Pure-math helpers (no model objects, fully testable with synthetic data)
# ---------------------------------------------------------------------------


def kth_redistribute(
    temps: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute target temperatures from KTH feedback.

    Given current temperatures and the measured up-fraction f(T) at each
    slot, redistribute temperatures to equalise diffusion current.

    The algorithm (KTH 2006):
    1. Estimate df/dT via 3-point smoothed slopes, clamped to a positive floor.
    2. Compute density η[i] = sqrt(df_dT[i] / ΔT[i]) at each gap.
    3. Integrate η to build a CDF, then invert it to place new temperatures.

    Endpoints T_min and T_max are always preserved.

    Args:
        temps: Current temperature array, sorted ascending.  Shape (M,).
        f: Up-fraction at each temperature slot.  Shape (M,).
           Should be ≈1 at cold end, ≈0 at hot end.

    Returns:
        Target temperature array, shape (M,).  Endpoints unchanged.
    """
    M = len(temps)
    if M < 3:
        return temps.copy()

    # --- Step 1: smoothed df/dT via 3-point windowed linear regression ---
    # Raw finite differences at gap midpoints
    raw_slopes = np.abs(np.diff(f) / np.diff(temps))

    # Smooth with 3-point window (average neighboring slopes)
    smoothed = np.empty_like(raw_slopes)
    smoothed[0] = np.mean(raw_slopes[:2])  # one-sided at left boundary
    smoothed[-1] = np.mean(raw_slopes[-2:])  # one-sided at right boundary
    for i in range(1, len(raw_slopes) - 1):
        smoothed[i] = np.mean(raw_slopes[i - 1 : i + 2])

    # Clamp to positive floor to avoid zero/negative from noise
    eps = max(float(np.median(raw_slopes)) * 0.01, 1e-6)
    df_dT = np.maximum(smoothed, eps)

    # --- Step 2: temperature density η ---
    dT = np.diff(temps)
    eta = np.sqrt(df_dT / dT)

    # --- Step 3: integrate η to build CDF, invert to place new temps ---
    # Cumulative distribution S(T)
    S = np.zeros(M)
    for i in range(1, M):
        S[i] = S[i - 1] + eta[i - 1] * dT[i - 1]
    S /= S[-1]  # normalise to [0, 1]

    # Invert: for each k = 1..M-2, find T such that S(T) = k/(M-1)
    target = np.empty(M)
    target[0] = temps[0]
    target[-1] = temps[-1]
    for k in range(1, M - 1):
        s_target = k / (M - 1)
        # Linear interpolation on the (temps, S) curve
        idx = int(np.searchsorted(S, s_target, side="right") - 1)
        idx = np.clip(idx, 0, M - 2)
        # Fraction within the interval [S[idx], S[idx+1]]
        frac = (s_target - S[idx]) / (S[idx + 1] - S[idx])
        target[k] = temps[idx] + frac * (temps[idx + 1] - temps[idx])

    return target


def kth_check_convergence(
    old_temps: npt.NDArray[np.float64],
    new_temps: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    tol: float = 0.01,
    r2_threshold: float = 0.99,
) -> bool:
    """Check KTH convergence: temperatures stable AND f(T) linear.

    Args:
        old_temps: Previous iteration temperatures.
        new_temps: Current iteration temperatures.
        f: Up-fraction at each temperature slot.
        tol: Maximum allowed relative temperature change.
        r2_threshold: Minimum R² for f(T) linearity.

    Returns:
        True if both conditions are met.
    """
    # Condition 1: temperatures stable
    # Skip endpoints (they never change) — check interior only
    interior = slice(1, -1)
    rel_change = np.abs(new_temps[interior] - old_temps[interior]) / old_temps[interior]
    if np.max(rel_change) >= tol:
        return False

    # Condition 2: f(T) is linear (R² > threshold)
    # Linear regression: f = a*T + b
    coeffs = np.polyfit(old_temps, f, 1)
    f_pred = np.polyval(coeffs, old_temps)
    ss_res = np.sum((f - f_pred) ** 2)
    ss_tot = np.sum((f - np.mean(f)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return bool(r2 >= r2_threshold)


def welch_equilibration_check(
    obs_streams: dict[str, list[list[float]]],
    alpha: float = 0.05,
) -> bool:
    """Check equilibration via Welch's t-test on observable streams.

    For every observable at every temperature slot, compare the first 20%
    of samples against the last 20%.  If the means differ significantly,
    the system hasn't equilibrated yet.

    Uses Bonferroni correction: the per-test threshold is
    α / (n_observables × n_T_slots) to control family-wise error rate.

    Args:
        obs_streams: PTResult obs_streams — obs_streams[name][T_slot]
            is a list of float values (one per PT round).
        alpha: Family-wise significance level (default 0.05).

    Returns:
        True if ALL tests pass (equilibrated), False if any fails.
    """
    if not obs_streams:
        return True

    # Count total number of tests for Bonferroni correction
    obs_names = list(obs_streams.keys())
    M = len(obs_streams[obs_names[0]])  # number of T slots
    n_tests = len(obs_names) * M
    threshold = alpha / n_tests

    for name in obs_names:
        for t in range(M):
            series = obs_streams[name][t]
            n = len(series)
            if n < 10:
                # Too few samples to test meaningfully
                return False
            cut = n // 5  # 20%
            first = series[:cut]
            last = series[-cut:]
            _, p_value = stats.ttest_ind(first, last, equal_var=False)
            if p_value < threshold:
                return False

    return True


# ---------------------------------------------------------------------------
# Model factory — creates M replica objects from a model type string
# ---------------------------------------------------------------------------

_MODEL_CONSTRUCTORS = {
    "ising": _core.IsingModel,
    "blume_capel": _core.BlumeCapelModel,
    "ashkin_teller": _core.AshkinTellerModel,
}

_PT_ROUNDS_FN = {
    "ising": _core.pt_rounds_ising,
    "blume_capel": _core.pt_rounds_bc,
    "ashkin_teller": _core.pt_rounds_at,
}


def _make_replicas(
    model_type: str, L: int, param_value: float, n_replicas: int, seed: int
) -> tuple[list[Model], _core.Rng]:
    """Create n_replicas model objects and a shared RNG.

    Each replica gets a unique seed derived from the master seed.
    The shared RNG (for exchange decisions) uses the master seed directly.
    """
    rng = _core.Rng(seed)

    replicas: list[Model] = []
    for i in range(n_replicas):
        # Each replica needs its own RNG — derive a unique seed
        replica_seed = (seed + i + 1) % (2**63)
        m: Model
        if model_type == "blume_capel":
            bc = _core.BlumeCapelModel(L, replica_seed)
            bc.set_crystal_field(param_value)
            m = bc
        elif model_type == "ashkin_teller":
            at = _core.AshkinTellerModel(L, replica_seed)
            at.set_four_spin_coupling(param_value)
            m = at
        else:
            m = _core.IsingModel(L, replica_seed)
        replicas.append(m)

    return replicas, rng


# ---------------------------------------------------------------------------
# PTEngine — the main orchestrator class
# ---------------------------------------------------------------------------


class PTEngine:
    """Parallel tempering engine for a single Hamiltonian parameter value.

    Manages M replicas across a temperature ladder.  Phases:
      A) tune_ladder()  — KTH feedback to optimise T placement
      B) equilibrate()  — locked ladder, measure τ_int
      C) produce()      — harvest decorrelated snapshots
    """

    def __init__(
        self,
        model_type: str,
        L: int,
        param_value: float,
        T_range: tuple[float, float],
        n_replicas: int,
        seed: int,
    ) -> None:
        if model_type not in _MODEL_CONSTRUCTORS:
            msg = f"Unknown model type: {model_type!r}"
            raise ValueError(msg)

        self.model_type = model_type
        self.L = L
        self.param_value = param_value
        self.n_replicas = n_replicas
        self.seed = seed

        # Geometric T ladder (ascending: slot 0 = cold, slot M-1 = hot)
        T_min, T_max = T_range
        self.temps = np.geomspace(T_min, T_max, n_replicas)

        # Create replicas and RNG
        self.replicas: list[Model]
        self.replicas, self.rng = _make_replicas(model_type, L, param_value, n_replicas, seed)

        # Address maps — identity permutation at start
        self.r2t = list(range(n_replicas))
        self.t2r = list(range(n_replicas))

        # Directional labels — unlabeled at start
        self.labels = [_core.LABEL_NONE] * n_replicas

        # Select the right C++ pt_rounds function
        self._pt_rounds = _PT_ROUNDS_FN[model_type]  # type: ignore[assignment]

        # State flags
        self.ladder_locked = False
        self.tau_max: float | None = None

    def tune_ladder(
        self,
        n_sw_initial: int = 500,
        max_iterations: int = 100,
        gamma: float = 0.5,
        tol: float = 0.01,
        min_acceptance: float = 0.10,
    ) -> None:
        """Phase A: KTH feedback loop to optimise temperature placement.

        Runs rounds of PT, measures diffusion histograms, redistributes
        temperatures.  Doubles N_sw each iteration (capped at iter 10).

        Raises:
            RuntimeError: If max_iterations reached without convergence,
                or if any gap acceptance rate < min_acceptance after tuning.
        """
        M = self.n_replicas
        n_sw = n_sw_initial

        for iteration in range(max_iterations):
            # Set replica temperatures from current ladder
            for r in range(M):
                self.replicas[r].set_temperature(self.temps[self.r2t[r]])

            # Run PT rounds — C++ handles sweep + exchange + histograms
            result: _core.PTResult = self._pt_rounds(  # type: ignore[operator]
                self.replicas,
                self.temps.tolist(),
                self.r2t,
                self.t2r,
                self.labels,
                n_sw,
                self.rng,
                False,  # no observable tracking in Phase A
            )

            # r2t, t2r, labels are mutated in-place by the C++ binding
            # (ivec_to_list writes back into the original Python lists).
            # We read histograms from the returned dict.

            # Compute f(T) = n_up / (n_up + n_down) at each slot
            n_up = np.array(result["n_up"], dtype=np.float64)
            n_down = np.array(result["n_down"], dtype=np.float64)
            total = n_up + n_down
            # Mask slots with no labeled visits
            valid = total > 0
            f = np.zeros(M)
            f[valid] = n_up[valid] / total[valid]

            # Skip redistribution if too few slots have data
            n_valid = int(np.sum(valid))
            if n_valid < 3:
                # Not enough data yet — double and retry
                if iteration < 10:
                    n_sw = min(n_sw * 2, n_sw_initial * (2**10))
                continue

            # Compute target temperatures via KTH redistribution
            old_temps = self.temps.copy()
            target = kth_redistribute(self.temps, f)

            # Damped update
            self.temps = (1 - gamma) * self.temps + gamma * target

            # Check convergence
            if kth_check_convergence(old_temps, self.temps, f, tol=tol):
                break

            # Double N_sw (cap at iteration 10)
            if iteration < 10:
                n_sw = min(n_sw * 2, n_sw_initial * (2**10))

            # Reset histograms by resetting labels
            self.labels = [_core.LABEL_NONE] * M
        else:
            msg = f"KTH tuning failed to converge after {max_iterations} iterations"
            raise RuntimeError(msg)

        # Post-tuning safety: check acceptance rates at each gap
        # Run one final batch to measure acceptance rates on the tuned ladder
        for r in range(M):
            self.replicas[r].set_temperature(self.temps[self.r2t[r]])

        check_result: _core.PTResult = self._pt_rounds(  # type: ignore[operator]
            self.replicas,
            self.temps.tolist(),
            self.r2t,
            self.t2r,
            self.labels,
            max(1000, n_sw),
            self.rng,
            False,
        )

        # r2t, t2r, labels already mutated in-place by C++ binding

        n_accepts = np.array(check_result["n_accepts"], dtype=np.float64)
        n_attempts = np.array(check_result["n_attempts"], dtype=np.float64)
        rates = np.where(n_attempts > 0, n_accepts / n_attempts, 0.0)

        bad_gaps = np.where(rates < min_acceptance)[0]
        if len(bad_gaps) > 0:
            worst = bad_gaps[0]
            msg = (
                f"Gap {worst} (T={self.temps[worst]:.4f}–"
                f"{self.temps[worst + 1]:.4f}) has acceptance rate "
                f"{rates[worst]:.3f} < {min_acceptance}. "
                f"Add more replicas or narrow the T range."
            )
            raise RuntimeError(msg)

        self.ladder_locked = True

    def equilibrate(
        self,
        n_initial: int = 100_000,
        n_max: int = 6_400_000,
        alpha: float = 0.05,
    ) -> None:
        """Phase B: equilibrate on the locked ladder, then measure τ_int.

        Doubling scheme: run n_initial rounds with observable tracking,
        check equilibration via Welch t-test.  If any test fails, double n
        and retry.  Cap at n_max.

        Once equilibrated, measure τ_int on the last converged batch
        (excluding first 20% as burn-in) and lock τ_max.

        Raises:
            RuntimeError: If ladder is not locked (must call tune_ladder first).
            RuntimeError: If equilibration fails after reaching n_max.
        """
        if not self.ladder_locked:
            msg = "Ladder must be locked before equilibration — call tune_ladder() first"
            raise RuntimeError(msg)

        M = self.n_replicas
        n_rounds = n_initial

        while n_rounds <= n_max:
            # Set replica temperatures
            for r in range(M):
                self.replicas[r].set_temperature(self.temps[self.r2t[r]])

            # Run PT rounds with observable tracking
            result: _core.PTResult = self._pt_rounds(  # type: ignore[operator]
                self.replicas,
                self.temps.tolist(),
                self.r2t,
                self.t2r,
                self.labels,
                n_rounds,
                self.rng,
                True,  # track observables
            )

            obs_streams: dict[str, list[list[float]]] = result["obs_streams"]

            # Welch t-test equilibration check
            if welch_equilibration_check(obs_streams, alpha=alpha):
                # Equilibrated — measure τ_int on last 80% (discard first 20% burn-in)
                burn_in = n_rounds // 5
                trimmed: dict[str, npt.NDArray[np.float64]] = {}
                for name, slots in obs_streams.items():
                    # Use all T slots — tau_int_multi takes per-observable arrays,
                    # so we pick the bottleneck across all slots
                    for t in range(M):
                        key = f"{name}_T{t}"
                        trimmed[key] = np.array(slots[t][burn_in:], dtype=np.float64)

                _, self.tau_max = tau_int_multi(trimmed)
                return

            # Not equilibrated — double and retry
            n_rounds *= 2

        msg = (
            f"Equilibration failed: Welch t-test still failing after "
            f"{n_max} rounds. System cannot equilibrate — try more "
            f"replicas or a narrower T range."
        )
        raise RuntimeError(msg)

    def produce(
        self,
        path: str | Path,
        n_snapshots: int = 100,
        seed_history: list[tuple[int, int]] | None = None,
    ) -> None:
        """Phase C: harvest decorrelated snapshots and stream to HDF5.

        Runs ``max(1, round(3 × τ_max))`` PT rounds between each snapshot
        harvest.  Each harvest reads spins + observables from every T slot
        and appends them to the HDF5 file.

        Resume-safe: if *path* already contains snapshots, only the
        remaining ``n_snapshots - n_existing`` are collected.

        Args:
            path: Output HDF5 file path.
            n_snapshots: Target total snapshots per T slot.
            seed_history: PRNG audit trail.  Defaults to ``[(0, self.seed)]``
                for a fresh run.  On resume the orchestrator passes the
                extended history from ``read_resume_state()``.

        Raises:
            RuntimeError: If tau_max is not set (equilibrate() not called).
        """
        if self.tau_max is None:
            msg = "tau_max not set — call equilibrate() before produce()"
            raise RuntimeError(msg)

        if seed_history is None:
            seed_history = [(0, self.seed)]

        M = self.n_replicas
        L = self.L

        # Channel count: AT has 2 spin layers (σ, τ), others have 1
        C = 2 if self.model_type == "ashkin_teller" else 1
        obs_names = list(self.replicas[0].observables().keys())

        # Thinning: at least 1 round between snapshots
        thinning = max(1, round(3 * self.tau_max))

        with SnapshotWriter(path) as writer:
            # Create T slots only if they don't already exist
            first_key = f"T={self.temps[0]}"
            if first_key not in writer._file:
                for T in self.temps:
                    writer.create_temperature_slot(T=T, L=L, C=C, obs_names=obs_names)

            # Count existing snapshots (all T slots are in lockstep)
            n_existing = writer._file[first_key]["snapshots"].shape[0]
            n_remaining = n_snapshots - n_existing
            if n_remaining <= 0:
                return

            # Main production loop — collect only the remaining snapshots
            for _ in range(n_remaining):
                # Set replica temperatures from current address map
                for r in range(M):
                    self.replicas[r].set_temperature(self.temps[self.r2t[r]])

                # Run thinning rounds (sweep + exchange, no obs tracking)
                self._pt_rounds(  # type: ignore[operator]
                    self.replicas,
                    self.temps.tolist(),
                    self.r2t,
                    self.t2r,
                    self.labels,
                    thinning,
                    self.rng,
                    False,
                )

                # Harvest one snapshot from every T slot
                for t in range(M):
                    replica = self.replicas[self.t2r[t]]
                    T = self.temps[t]

                    # Build (C, L, L) spin array
                    if self.model_type == "ashkin_teller":
                        spins = np.stack(
                            [replica.sigma, replica.tau]  # type: ignore[union-attr]
                        ).astype(np.int8)
                    else:
                        spins = replica.spins[np.newaxis].copy()  # type: ignore[union-attr]

                    obs = replica.observables()
                    writer.append_snapshot(T=T, spins=spins, obs_dict=obs)

        # Write campaign metadata
        write_param_attrs(
            path,
            model_type=self.model_type,
            L=L,
            param_value=self.param_value,
            T_ladder=self.temps,
            tau_max=self.tau_max,
            r2t=self.r2t,
            t2r=self.t2r,
            seed=self.seed,
            seed_history=seed_history,
        )
