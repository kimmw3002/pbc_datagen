# pbc_datagen

Paper-quality dataset generator for independent spatial snapshots of three 2D lattice models:
**Pure Ising**, **Blume-Capel**, and **Ashkin-Teller**.

## Architecture

Python orchestrates (temperature scheduling, autocorrelation analysis, disk I/O).
C++ does the heavy lifting (lattice memory, PRNG, MCMC update kernels).
They talk via [pybind11](https://pybind11.readthedocs.io/).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Prerequisites

- Python 3.10+
- A C++17 compiler (g++ or clang++)
- CMake 3.20+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --all-extras
```

This single command:
1. Creates/updates the virtual environment
2. Installs Python dependencies (numpy, scipy, h5py, etc.)
3. Invokes CMake, which compiles all C++ source into a shared library (`_core.so`)
4. Places the compiled binary into the package so Python can import it

After this, you can do:
```python
from pbc_datagen._core import IsingModel
```

## Development workflow

**Python changes** take effect immediately (editable install).

**C++ changes** require a forced rebuild — `uv sync` alone won't recompile
if only `.cpp`/`.hpp` files changed (it only tracks Python metadata):
```bash
uv sync --all-extras --reinstall-package pbc-datagen
```

There is no runtime compilation — all C++ is compiled at install time, not at import time.

## How pybind11 works (quick primer)

If you're new to C++/Python interop:

- CPython can load any `.so` shared library as a module, as long as it exports a `PyInit_<name>` function following CPython's C API.
- Writing that by hand is tedious. pybind11 is a header-only C++ library that generates all the boilerplate at compile time using C++ templates.
- The file `src/cpp/bindings.cpp` is where C++ classes/functions are exposed to Python. Each `.def(...)` call tells pybind11 "make this C++ method callable from Python."
- The compiled `.so` is a standard CPython extension module — `import` just loads it, no magic.

## Running tests

```bash
pytest
```

## Project structure

```
pbc_datagen/
├── CMakeLists.txt              # C++ build config
├── pyproject.toml              # Python project config (uv + scikit-build-core)
├── docs/
│   ├── ARCHITECTURE.md         # System design
│   └── PLAN.md                 # Implementation plan
├── src/cpp/
│   ├── include/                # C++ headers (PRNG, lattice, model definitions)
│   └── bindings.cpp            # pybind11 glue — exposes C++ to Python
├── python/pbc_datagen/         # Python package (orchestrator, validation, I/O)
├── tests/                      # pytest test suite
└── scripts/
    └── generate_dataset.py     # Main entry point
```
