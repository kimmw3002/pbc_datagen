# HDF5 Schemas

Two HDF5 schemas exist in this codebase. All scripts (`convert_to_pt.py`,
`plot_obs_vs_T.py`, `plot_snapshots.py`) handle both transparently.

## How to tell them apart

| Check | Flat schema | Per-group schema |
|-------|-------------|------------------|
| `"slot_keys" in f.attrs` | `True` | `False` |
| `"count" in f.attrs` | `True` | `False` |
| Root-level `"snapshots"` dataset | Yes `(M, N, C, L, L)` | No |
| Groups named `T=...` | No | Yes |

## 1. Flat Schema (current)

Produced by: `SnapshotWriter` (all current engines: `PTEngine`, `PTEngine2D`,
`SingleChainEngine`).

All data lives in root-level datasets indexed by slot number. Fast batch
writes (~4 h5py calls per production round).

```
file.h5
├── .attrs
│   ├── model_type      str     "ising" | "blume_capel" | "ashkin_teller"
│   ├── L               int     lattice side length
│   ├── slot_keys       str     JSON list: ["T=1.0000", "T=1.5000", ...]
│   │                            or 2D: ["T=1.0000_D=1.9000", ...]
│   ├── obs_names       str     JSON list: ["energy", "m", "abs_m", ...]
│   ├── count           int     number of snapshots actually written per slot
│   ├── seed            int     initial PRNG seed
│   ├── seed_history    str     JSON: [[n_at_start, seed], ...]
│   ├── tau_max         float   max autocorrelation time from Phase B
│   ├── pt_mode         str     "2d" if 2D PT (absent for 1D PT / single-chain)
│   ├── param_value     float   Hamiltonian parameter (1D PT / single-chain only)
│   ├── param_label     str     "D" or "U" (2D PT only)
│   ├── T_ladder        array   temperature ladder (1D PT / single-chain)
│   ├── temps           array   temperature grid (2D PT)
│   ├── params          array   parameter grid (2D PT)
│   ├── r2t / t2r       array   replica <-> temperature maps (1D PT)
│   └── r2s / s2r       array   replica <-> slot maps (2D PT)
├── snapshots   (M, N, C, L, L)  int8     M=slots, N=max snapshots, C=channels
├── energy      (M, N)           float64
├── m           (M, N)           float64  (Ising/BC)
├── abs_m       (M, N)           float64  (Ising/BC)
├── q           (M, N)           float64  (BC only)
├── m_sigma     (M, N)           float64  (AT only)
└── ...         (M, N)           float64  (other AT observables)
```

**Key points:**
- `slot_keys[i]` names what slot index `i` represents.
- Only `count` snapshots per slot are valid; the rest is pre-allocated zeros.
- Observable dataset names match `obs_names` exactly.

## 2. Per-Group Schema (legacy)

Produced by: early `PTEngine2D` runs (before flat schema was added).

Each (T, param) combination is an HDF5 group. Reads are ~7500x slower than
flat schema for large grids.

```
file.h5
├── .attrs
│   ├── model_type      str
│   ├── L               int
│   ├── pt_mode         str     "2d"
│   ├── temps           array   temperature grid
│   ├── params          array   parameter grid
│   ├── seed            int
│   ├── seed_history    str
│   ├── tau_max         float
│   ├── r2s / s2r       array
│   └── (no slot_keys, no count, no obs_names)
├── T=0.3000_D=1.9000/
│   ├── snapshots       (N, C, L, L)  int8
│   ├── energy          (N,)          float64
│   ├── m               (N,)          float64
│   ├── abs_m           (N,)          float64
│   └── q               (N,)          float64
├── T=0.3000_D=1.9041/
│   ├── snapshots       ...
│   └── ...
└── ...
```

**Key points:**
- No `slot_keys` or `count` attrs. Presence of `T=...` groups is the indicator.
- Each group is independent; snapshot count may vary across groups.
- Observable names are discovered from group keys (excluding `"snapshots"`).
- Group names encode T (and param for 2D) with full float precision (not %.4f).

## Detection logic in scripts

All reader code (converter, plot scripts) uses this pattern:

```python
with h5py.File(path, "r") as f:
    if "slot_keys" in f.attrs:
        # Flat schema
    else:
        # Per-group schema (look for T=... groups)
```
