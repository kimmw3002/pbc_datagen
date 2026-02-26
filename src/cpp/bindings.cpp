// pybind11 module definition — binds C++ models to Python.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "lattice.hpp"
#include "prng.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "pbc_datagen C++ backend: Ising, Blume-Capel, Ashkin-Teller lattice models";

    // --- PRNG ---------------------------------------------------------
    py::class_<pbc::Rng>(m, "Rng")
        .def(py::init<uint64_t>(), py::arg("seed"))
        .def("uniform", &pbc::Rng::uniform)
        .def("rand_below", &pbc::Rng::rand_below, py::arg("n"));

    // --- Lattice ------------------------------------------------------
    m.def(
        "make_neighbor_table",
        [](int L) {
            auto flat = pbc::make_neighbor_table(L);
            int N = L * L;
            // Copy into a numpy array with shape (N, 4).
            py::array_t<int32_t> result({N, 4});
            auto buf = result.mutable_unchecked<2>();
            for (int i = 0; i < N; ++i) {
                for (int d = 0; d < 4; ++d) {
                    buf(i, d) = flat[i * 4 + d];
                }
            }
            return result;
        },
        py::arg("L"),
        "Build the PBC neighbor table for an L*L square lattice.\n"
        "Returns an (L*L, 4) int32 array.  Columns: north, south, east, west.");
}
