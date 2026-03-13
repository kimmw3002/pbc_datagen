# Orchestrator — sequential PT campaigns with OpenMP parallelism.

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from loguru import logger

from pbc_datagen.io import read_resume_state, read_resume_state_2d
from pbc_datagen.parallel_tempering import PTEngine
from pbc_datagen.pt_engine_2d import PTEngine2D
from pbc_datagen.registry import get_model_info


def _param_label(model_type: str) -> str | None:
    """Return the Hamiltonian parameter label for file naming, or None.

    Ising has no tunable Hamiltonian parameter (J=1 is fixed in C++),
    so it returns None.  Blume-Capel → "D", Ashkin-Teller → "U".
    """
    info = get_model_info(model_type)  # raises ValueError if unknown
    return info.param_label


def _derive_seed(old_seed: int, n_existing: int) -> int:
    """Deterministic seed derivation for resume.

    Same (old_seed, n_existing) always produces the same new seed.
    Uses SHA-256 to mix old_seed and n_existing into a new 63-bit seed.
    """
    data = f"{old_seed}:{n_existing}".encode()
    digest = hashlib.sha256(data).digest()
    # Take first 8 bytes as a 63-bit unsigned integer (fits in Python int)
    return int.from_bytes(digest[:8], "little") % (2**63)


def find_existing_hdf5(
    output_dir: str | Path,
    model_type: str,
    L: int,
    param_value: float,
    T_range: tuple[float, float],
    n_replicas: int,
) -> Path | None:
    """Find the most recent HDF5 file for this exact config.

    The glob encodes model, L, param (if any), T-range, and replica
    count so that a re-run with different T-range or n_replicas does
    not accidentally resume an incompatible file.

    Returns the newest match (by timestamp suffix), or None if no match.
    """
    directory = Path(output_dir)
    label = _param_label(model_type)
    T_seg = f"T={T_range[0]:.4f}-{T_range[1]:.4f}_R{n_replicas}"
    if label is not None:
        pattern = f"{model_type}_L{L}_{label}={param_value:.4f}_{T_seg}_*.h5"
    else:
        pattern = f"{model_type}_L{L}_{T_seg}_*.h5"

    matches = sorted(directory.glob(pattern))
    if not matches:
        logger.debug("No existing HDF5 for pattern {}", pattern)
        return None

    # Sort by timestamp suffix (the integer before .h5) — return newest
    def _timestamp(p: Path) -> int:
        stem = p.stem  # e.g. "blume_capel_L4_D=1.5000_2000000000000"
        return int(stem.rsplit("_", 1)[-1])

    result = max(matches, key=_timestamp)
    logger.debug("Found existing HDF5: {}", result.name)
    return result


def run_campaign(
    model_type: str,
    L: int,
    param_value: float,
    T_range: tuple[float, float],
    n_replicas: int,
    n_snapshots: int,
    output_dir: str | Path,
    force_new: bool = False,
) -> Path:
    """Run one PT campaign for a single parameter value.

    Fresh start: create new file, run A→B→C.
    Resume: find existing file, derive new seed, extend seed_history,
    run A→B→C (produce appends remaining snapshots).

    Returns the path to the output HDF5 file.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    label = _param_label(model_type)

    existing = (
        None
        if force_new
        else find_existing_hdf5(directory, model_type, L, param_value, T_range, n_replicas)
    )

    if existing is not None:
        # --- Resume path ---
        logger.info("Resuming campaign from {}", existing.name)
        old_seed, state = read_resume_state(existing)
        seed_history: list[tuple[int, int]] = state["seed_history"]

        # Count existing snapshots (all T slots in lockstep — use first)
        counts = state["snapshot_counts"]
        n_existing = min(counts.values()) if counts else 0

        if n_existing >= n_snapshots:
            logger.info("Already complete ({}/{} snapshots), skipping", n_existing, n_snapshots)
            return existing  # already done

        # Derive a fresh seed so resumed snapshots use an independent PRNG stream
        new_seed = _derive_seed(old_seed, n_existing)
        seed_history.append((n_existing, new_seed))
        logger.debug(
            "Resume: {}/{} snapshots exist, derived seed={}",
            n_existing,
            n_snapshots,
            new_seed,
        )

        # Restore engine with the locked ladder and τ_max from the saved state
        # — no need to re-tune or re-equilibrate.
        engine = PTEngine(model_type, L, param_value, T_range, n_replicas, new_seed)
        engine.temps = state["T_ladder"]
        engine.ladder_locked = True
        engine.tau_max = state["tau_max"]
        engine.produce(existing, n_snapshots, seed_history=seed_history)
        return existing
    else:
        # --- Fresh start ---
        ts = int(time.time() * 1000)
        seed = ts % (2**63)
        T_seg = f"T={T_range[0]:.4f}-{T_range[1]:.4f}_R{n_replicas}"
        if label is not None:
            filename = f"{model_type}_L{L}_{label}={param_value:.4f}_{T_seg}_{ts}.h5"
        else:
            filename = f"{model_type}_L{L}_{T_seg}_{ts}.h5"
        path = directory / filename
        logger.info("Fresh campaign: {}", filename)

        engine = PTEngine(model_type, L, param_value, T_range, n_replicas, seed)
        engine.tune_ladder()
        engine.equilibrate()
        engine.produce(path, n_snapshots)
        logger.info("Campaign complete: {}", filename)
        return path


def set_omp_threads(n_threads: int) -> None:
    """Set OMP_NUM_THREADS before any C++ code runs.

    Must be called before the first pybind11 call that triggers an
    OpenMP parallel region — the thread pool is created once and reused.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    logger.debug("OMP_NUM_THREADS={}", n_threads)


def find_existing_hdf5_2d(
    output_dir: str | Path,
    model_type: str,
    L: int,
    T_range: tuple[float, float],
    param_range: tuple[float, float],
    n_T: int,
    n_P: int,
) -> Path | None:
    """Find the most recent 2D HDF5 file for this exact config."""
    directory = Path(output_dir)
    label = _param_label(model_type)
    if label is None:
        return None  # Ising doesn't use 2D PT
    T_seg = f"T={T_range[0]:.4f}-{T_range[1]:.4f}"
    P_seg = f"{label}={param_range[0]:.4f}-{param_range[1]:.4f}"
    pattern = f"{model_type}_L{L}_{T_seg}_{P_seg}_{n_T}x{n_P}_*.h5"

    matches = sorted(directory.glob(pattern))
    if not matches:
        logger.debug("No existing 2D HDF5 for pattern {}", pattern)
        return None

    def _timestamp(p: Path) -> int:
        return int(p.stem.rsplit("_", 1)[-1])

    result = max(matches, key=_timestamp)
    logger.debug("Found existing 2D HDF5: {}", result.name)
    return result


def run_campaign_2d(
    model_type: str,
    L: int,
    T_range: tuple[float, float],
    param_range: tuple[float, float],
    n_T: int,
    n_P: int,
    n_snapshots: int,
    output_dir: str | Path,
    force_new: bool = False,
    connectivity_rounds: int = 100,
) -> Path:
    """Run one 2D PT campaign covering the full (T, param) grid.

    Fresh start: A→B→C.  Resume: skip to C with saved state.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    label = _param_label(model_type)

    existing = (
        None
        if force_new
        else find_existing_hdf5_2d(directory, model_type, L, T_range, param_range, n_T, n_P)
    )

    if existing is not None:
        logger.info("Resuming 2D campaign from {}", existing.name)
        old_seed, state = read_resume_state_2d(existing)
        seed_history: list[tuple[int, int]] = state["seed_history"]

        counts = state["snapshot_counts"]
        n_existing = min(counts.values()) if counts else 0
        if n_existing >= n_snapshots:
            logger.info("Already complete ({}/{}), skipping", n_existing, n_snapshots)
            return existing

        new_seed = _derive_seed(old_seed, n_existing)
        seed_history.append((n_existing, new_seed))

        engine = PTEngine2D(model_type, L, T_range, param_range, n_T, n_P, new_seed)
        engine.temps = state["temps"]
        engine.params = state["params"]
        engine.connectivity_checked = True
        engine.tau_max = state["tau_max"]
        engine.produce(existing, n_snapshots, seed_history=seed_history)
        return existing
    else:
        ts = int(time.time() * 1000)
        seed = ts % (2**63)
        T_seg = f"T={T_range[0]:.4f}-{T_range[1]:.4f}"
        P_seg = f"{label}={param_range[0]:.4f}-{param_range[1]:.4f}"
        filename = f"{model_type}_L{L}_{T_seg}_{P_seg}_{n_T}x{n_P}_{ts}.h5"
        path = directory / filename
        logger.info("Fresh 2D campaign: {}", filename)

        engine = PTEngine2D(model_type, L, T_range, param_range, n_T, n_P, seed)
        engine.check_connectivity(n_rounds=connectivity_rounds)
        engine.equilibrate()
        engine.produce(path, n_snapshots)
        logger.info("2D campaign complete: {}", filename)
        return path


def generate_dataset(
    model_type: str,
    L: int,
    param_values: list[float],
    T_range: tuple[float, float],
    n_replicas: int = 20,
    n_snapshots: int = 100,
    output_dir: str | Path = "output/",
    force_new: bool = False,
) -> None:
    """Run 1D PT campaigns sequentially for each param value.

    For Ising (and legacy single-param BC/AT runs).
    Call ``set_omp_threads()`` before this function.
    """
    logger.info(
        "Generating dataset: {} param values (sequential, OpenMP sweeps)",
        len(param_values),
    )
    for p in param_values:
        run_campaign(model_type, L, p, T_range, n_replicas, n_snapshots, output_dir, force_new)


def generate_dataset_2d(
    model_type: str,
    L: int,
    T_range: tuple[float, float],
    param_range: tuple[float, float],
    n_T: int = 10,
    n_P: int = 10,
    n_snapshots: int = 100,
    output_dir: str | Path = "output/",
    force_new: bool = False,
    connectivity_rounds: int = 100,
) -> None:
    """Run a single 2D PT campaign covering the full (T, param) grid.

    For Blume-Capel and Ashkin-Teller near first-order transitions.
    Call ``set_omp_threads()`` before this function.
    """
    run_campaign_2d(
        model_type,
        L,
        T_range,
        param_range,
        n_T,
        n_P,
        n_snapshots,
        output_dir,
        force_new,
        connectivity_rounds=connectivity_rounds,
    )
