# Orchestrator — param-level parallelism for PT campaigns.

from __future__ import annotations

import hashlib
import time
from multiprocessing import Pool
from pathlib import Path

from loguru import logger

from pbc_datagen.io import read_resume_state
from pbc_datagen.parallel_tempering import PTEngine


def _worker_init(log_prefix: str | None) -> None:
    """Pool initializer: set up loguru in worker processes.

    Forked workers inherit handler objects but the enqueue background
    thread dies on fork.  Set up fresh handlers in each worker.

    Each worker writes to its own log file (``{log_prefix}_worker_{pid}.log``)
    so output from different workers never interleaves.
    """
    import os
    import sys

    logger.remove()
    logger.enable("pbc_datagen")

    fmt_plain = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}"
    fmt_color = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
    logger.add(sys.stdout, format=fmt_color, level="INFO", colorize=True)
    if log_prefix is not None:
        log_file = f"{log_prefix}_worker_{os.getpid()}.log"
        logger.add(log_file, format=fmt_plain, level="DEBUG")


_VALID_MODELS: set[str] = {"ising", "blume_capel", "ashkin_teller"}

_PARAM_LABELS: dict[str, str] = {
    "blume_capel": "D",
    "ashkin_teller": "U",
}


def _param_label(model_type: str) -> str | None:
    """Return the Hamiltonian parameter label for file naming, or None.

    Ising has no tunable Hamiltonian parameter (J=1 is fixed in C++),
    so it returns None.  Blume-Capel → "D", Ashkin-Teller → "U".
    """
    if model_type not in _VALID_MODELS:
        msg = f"Unknown model type: {model_type!r}"
        raise ValueError(msg)
    return _PARAM_LABELS.get(model_type)


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


def generate_dataset(
    model_type: str,
    L: int,
    param_values: list[float],
    T_range: tuple[float, float],
    n_replicas: int = 20,
    n_snapshots: int = 100,
    max_workers: int = 4,
    output_dir: str | Path = "output/",
    force_new: bool = False,
    log_prefix: str | None = None,
) -> None:
    """Distribute param values across workers via multiprocessing.Pool."""
    logger.info(
        "Generating dataset: {} param values across {} workers",
        len(param_values),
        max_workers,
    )
    with Pool(max_workers, initializer=_worker_init, initargs=(log_prefix,)) as pool:
        pool.starmap(
            run_campaign,
            [
                (model_type, L, p, T_range, n_replicas, n_snapshots, output_dir, force_new)
                for p in param_values
            ],
        )
