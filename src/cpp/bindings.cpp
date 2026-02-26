// pybind11 module definition — binds C++ models to Python.
// TODO: Phase 1 implementation

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "pbc_datagen C++ backend: Ising, Blume-Capel, Ashkin-Teller lattice models";
    // TODO: bind IsingModel, BlumeCapelModel, AshkinTellerModel
}
