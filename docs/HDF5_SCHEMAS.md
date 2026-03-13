# HDF5 Schema

## Flat Schema (only supported format)

Produced by: `SnapshotWriter` (all engines: `PTEngine`, `PTEngine2D`,
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
│   ├── snapshot_dtype  str     numpy dtype name: "int8", "float64", etc.
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
├── snapshots   (M, N, C, L, L)  dtype     M=slots, N=max snapshots, C=channels
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
- `snapshot_dtype` records the numpy dtype of the `snapshots` dataset (e.g.
  `"int8"` for discrete spin models, `"float64"` for continuous models like XY).
  Readers should use this attr to reconstruct the correct dtype; scripts fall
  back to `"int8"` for files written before this attr was introduced.
- The snapshot dataset dtype matches `snapshot_dtype` — it is **not** always
  `int8`. Future models (e.g. XY, Heisenberg) will use `float64`.

## Legacy per-group schema (unsupported)

An earlier per-group schema stored each (T, param) combination as a separate
HDF5 group (e.g. `T=0.3000_D=1.9000/snapshots`). This format is **no longer
supported** by any reader or writer in the codebase. Files in this format must
be regenerated.
