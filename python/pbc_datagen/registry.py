# Model registry — single source of truth for model metadata.
#
# Replaces duplicated _MODEL_CONSTRUCTORS, _PT_ROUNDS_FN, _PARAM_LABELS
# dicts across parallel_tempering.py, pt_engine_2d.py, single_chain.py,
# and orchestrator.py.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import pbc_datagen._core as _core


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Everything the engines need to know about a model type."""

    name: str
    constructor: type
    n_channels: int
    snapshot_dtype: np.dtype[Any]
    param_label: str | None
    set_param: Callable[..., None] | None
    pt_rounds_fn: Callable[..., _core.PTResult]
    pt_rounds_2d_fn: Callable[..., _core.PT2DResult] | None


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "ising": ModelInfo(
        name="ising",
        constructor=_core.IsingModel,
        n_channels=1,
        snapshot_dtype=np.dtype(np.int8),
        param_label=None,
        set_param=None,
        pt_rounds_fn=_core.pt_rounds_ising,
        pt_rounds_2d_fn=None,
    ),
    "blume_capel": ModelInfo(
        name="blume_capel",
        constructor=_core.BlumeCapelModel,
        n_channels=1,
        snapshot_dtype=np.dtype(np.int8),
        param_label="D",
        set_param=lambda m, v: m.set_crystal_field(v),
        pt_rounds_fn=_core.pt_rounds_bc,
        pt_rounds_2d_fn=_core.pt_rounds_2d_bc,
    ),
    "ashkin_teller": ModelInfo(
        name="ashkin_teller",
        constructor=_core.AshkinTellerModel,
        n_channels=2,
        snapshot_dtype=np.dtype(np.int8),
        param_label="U",
        set_param=lambda m, v: m.set_four_spin_coupling(v),
        pt_rounds_fn=_core.pt_rounds_at,
        pt_rounds_2d_fn=_core.pt_rounds_2d_at,
    ),
}


def get_model_info(name: str) -> ModelInfo:
    """Look up model info by name, raising ValueError if unknown."""
    if name not in MODEL_REGISTRY:
        msg = f"Unknown model type: {name!r}. Valid: {valid_model_names()}"
        raise ValueError(msg)
    return MODEL_REGISTRY[name]


def valid_model_names() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(MODEL_REGISTRY.keys())
