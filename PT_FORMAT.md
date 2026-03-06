# Ising

## ising_128_renew.pt

The ising_128_renew.pt contains:


`[{"state": S_0, "beta": b_0}, {"state": S_1, "beta": b_1}, ...]`

## Configuration
Where:
- state: Final spin configurations as L×L tensors, L = 128. Each spin has value +1 or -1.
- beta: Beta values, where beta = 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60

The critical beta is $\beta_c = 0.4407$ so beta until 0.44 is disordered, from 0.46 is ordered.

Each beta value has 1,000 snapshots. So 16 * 1,000 = 16,000 snapshots in total.

# Blume-Capel

The blumecapel.pt contains:

`[{"state": S_0, "beta": b_0, "delta": D_0}, {"state": S_1, "beta": b_1, "delta": D_1}, ...]`

## Configuration
Where:
- state: Final spin configurations as L×L tensors, L = 64. Each spin has value +1 or 0 or -1.
- beta: beta values, where beta = 0.50 to 3.00.
- delta: delta values, where delta = 1.70 to 2.20 (step 0.01)

Each beta, delta value pair has 100 snapshots. 108,200 snapshots in total.
