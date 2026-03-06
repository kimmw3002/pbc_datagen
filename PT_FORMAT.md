# .pt Dataset Format

Each `.pt` file is a flat list of dicts saved with `torch.save()`:

```python
records = torch.load("dataset.pt", weights_only=False)
# records[i] is a dict with "state", "T", observables, and optionally a Hamiltonian parameter
```

## Ising

```python
{"state": Tensor(1,L,L), "T": float, "energy": float, "m": float, "abs_m": float}
```

- `state`: int8 tensor, spins in {-1, +1}, shape `(1, L, L)`
- `T`: temperature
- Observables: `energy`, `m`, `abs_m`

## Blume-Capel

```python
{"state": Tensor(1,L,L), "T": float, "D": float, "energy": float, "m": float, "abs_m": float, "q": float}
```

- `state`: int8 tensor, spins in {-1, 0, +1}, shape `(1, L, L)`
- `T`: temperature, `D`: crystal field
- Observables: `energy`, `m`, `abs_m`, `q`

## Ashkin-Teller

```python
{"state": Tensor(2,L,L), "T": float, "U": float, "energy": float, "m_sigma": float, "abs_m_sigma": float, "m_tau": float, "abs_m_tau": float, "m_baxter": float, "abs_m_baxter": float}
```

- `state`: int8 tensor, 2 layers (sigma, tau), spins in {-1, +1}, shape `(2, L, L)`
- `T`: temperature, `U`: 4-spin coupling
- Observables: `energy`, `m_sigma`, `abs_m_sigma`, `m_tau`, `abs_m_tau`, `m_baxter`, `abs_m_baxter`

## Notes

- All `state` tensors are `torch.int8`
- No compression on `torch.save()`
- Model type can be inferred from keys: `"D"` present = blume_capel, `"U"` present = ashkin_teller, neither = ising
