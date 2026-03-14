# Model registry — single source of truth for model metadata.
#
# Replaces duplicated _MODEL_CONSTRUCTORS, _PT_ROUNDS_FN, _PARAM_LABELS
# dicts across parallel_tempering.py, pt_engine_2d.py, single_chain.py,
# and orchestrator.py.

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

import pbc_datagen._core as _core

if TYPE_CHECKING:
    from matplotlib.cm import ScalarMappable  # noqa: F401
    from matplotlib.colors import Colormap, Normalize


@dataclass(frozen=True, slots=True)
class VizInfo:
    """Visualization metadata for a model (pure data, no matplotlib dependency)."""

    viz_type: str  # "discrete" or "continuous"
    cmap_colors: tuple[str, ...] | None  # hex colors for discrete ListedColormap
    cmap_name: str | None  # matplotlib colormap name for continuous
    boundaries: tuple[float, ...] | None  # BoundaryNorm edges for discrete
    vmin: float | None  # Normalize range for continuous
    vmax: float | None
    tick_values: tuple[float, ...] | None  # colorbar tick positions
    tick_labels: tuple[str, ...] | None  # colorbar tick labels


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
    viz: VizInfo


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
        viz=VizInfo(
            viz_type="discrete",
            cmap_colors=("#2166ac", "#b2182b"),
            cmap_name=None,
            boundaries=(-1.5, 0, 1.5),
            vmin=None,
            vmax=None,
            tick_values=(-1, 1),
            tick_labels=("-1", "+1"),
        ),
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
        viz=VizInfo(
            viz_type="discrete",
            cmap_colors=("#2166ac", "#f7f7f7", "#b2182b"),
            cmap_name=None,
            boundaries=(-1.5, -0.5, 0.5, 1.5),
            vmin=None,
            vmax=None,
            tick_values=(-1, 0, 1),
            tick_labels=("-1", "0", "+1"),
        ),
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
        viz=VizInfo(
            viz_type="discrete",
            cmap_colors=("#2166ac", "#b2182b"),
            cmap_name=None,
            boundaries=(-1.5, 0, 1.5),
            vmin=None,
            vmax=None,
            tick_values=(-1, 1),
            tick_labels=("-1", "+1"),
        ),
    ),
    "xy": ModelInfo(
        name="xy",
        constructor=_core.XYModel,
        n_channels=1,
        snapshot_dtype=np.dtype(np.float64),
        param_label=None,
        set_param=None,
        pt_rounds_fn=_core.pt_rounds_xy,
        pt_rounds_2d_fn=None,
        viz=VizInfo(
            viz_type="continuous",
            cmap_colors=None,
            cmap_name="twilight",
            boundaries=None,
            vmin=0.0,
            vmax=2 * math.pi,
            tick_values=(0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi),
            tick_labels=("0", "\u03c0/2", "\u03c0", "3\u03c0/2", "2\u03c0"),
        ),
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


def make_cmap_norm(viz: VizInfo) -> tuple[Colormap, Normalize]:
    """Build (colormap, norm) from VizInfo. This is the only function that imports matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

    cmap: Colormap
    norm_out: Normalize
    if viz.viz_type == "discrete":
        assert viz.cmap_colors is not None
        assert viz.boundaries is not None
        cmap = ListedColormap(list(viz.cmap_colors))
        norm_out = BoundaryNorm(list(viz.boundaries), cmap.N)
    else:
        assert viz.cmap_name is not None
        cmap = plt.get_cmap(viz.cmap_name)
        norm_out = Normalize(vmin=viz.vmin, vmax=viz.vmax)
    return cmap, norm_out
