// pybind11 module definition — binds C++ models to Python.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "blume_capel.hpp"
#include "ising.hpp"
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

    // --- Ising model --------------------------------------------------
    py::class_<pbc::IsingModel>(m, "IsingModel")
        .def(py::init<int, uint64_t>(), py::arg("L"), py::arg("seed"))
        .def_readonly("L", &pbc::IsingModel::L)
        .def_property_readonly("T", [](const pbc::IsingModel& self) {
            return self.T_;
        })
        .def_property_readonly("spins", [](const pbc::IsingModel& self) {
            // Return a numpy view into the spin vector (no copy).
            // Shape: (L, L), dtype: int8.  Row r, column c = site r*L+c.
            return py::array_t<int8_t>(
                {self.L, self.L},                               // shape (L, L)
                {self.L * (int)sizeof(int8_t), (int)sizeof(int8_t)},  // strides: row-major
                self.spin.data(),                               // data pointer
                py::cast(self)                                  // parent object keeps memory alive
            );
        })
        .def("set_temperature", &pbc::IsingModel::set_temperature, py::arg("T"))
        .def("set_spin", &pbc::IsingModel::set_spin, py::arg("site"), py::arg("value"))
        .def("energy", &pbc::IsingModel::energy)
        .def("magnetization", &pbc::IsingModel::magnetization)
        .def("abs_magnetization", &pbc::IsingModel::abs_magnetization)
        .def("_wolff_step", &pbc::IsingModel::_wolff_step)
        .def("_delta_energy", &pbc::IsingModel::_delta_energy, py::arg("site"))
        .def("_metropolis_sweep", &pbc::IsingModel::_metropolis_sweep)
        .def("sweep", [](pbc::IsingModel& self, int n_sweeps) {
            auto result = self.sweep(n_sweeps);
            auto n = static_cast<py::ssize_t>(n_sweeps);

            // Copy each vector into a numpy array
            py::array_t<int>    energy(n, result.energy.data());
            py::array_t<double> m(n, result.m.data());
            py::array_t<double> abs_m(n, result.abs_m.data());

            py::dict out;
            out["energy"] = std::move(energy);
            out["m"]      = std::move(m);
            out["abs_m"]  = std::move(abs_m);
            return out;
        }, py::arg("n_sweeps"));

    // --- Blume-Capel model -----------------------------------------------
    py::class_<pbc::BlumeCapelModel>(m, "BlumeCapelModel")
        .def(py::init<int, uint64_t>(), py::arg("L"), py::arg("seed"))
        .def_readonly("L", &pbc::BlumeCapelModel::L)
        .def_property_readonly("T", [](const pbc::BlumeCapelModel& self) {
            return self.T_;
        })
        .def_property_readonly("D", [](const pbc::BlumeCapelModel& self) {
            return self.D_;
        })
        .def_property_readonly("spins", [](const pbc::BlumeCapelModel& self) {
            return py::array_t<int8_t>(
                {self.L, self.L},
                {self.L * (int)sizeof(int8_t), (int)sizeof(int8_t)},
                self.spin.data(),
                py::cast(self)
            );
        })
        .def("set_temperature", &pbc::BlumeCapelModel::set_temperature, py::arg("T"))
        .def("set_crystal_field", &pbc::BlumeCapelModel::set_crystal_field, py::arg("D"))
        .def("set_spin", &pbc::BlumeCapelModel::set_spin, py::arg("site"), py::arg("value"))
        .def("energy", &pbc::BlumeCapelModel::energy)
        .def("magnetization", &pbc::BlumeCapelModel::magnetization)
        .def("abs_magnetization", &pbc::BlumeCapelModel::abs_magnetization)
        .def("quadrupole", &pbc::BlumeCapelModel::quadrupole)
        .def("_wolff_step", &pbc::BlumeCapelModel::_wolff_step)
        .def("_delta_energy", &pbc::BlumeCapelModel::_delta_energy,
             py::arg("site"), py::arg("new_spin"))
        .def("_metropolis_sweep", &pbc::BlumeCapelModel::_metropolis_sweep)
        .def("sweep", [](pbc::BlumeCapelModel& self, int n_sweeps) {
            auto result = self.sweep(n_sweeps);
            auto n = static_cast<py::ssize_t>(n_sweeps);

            py::array_t<double> energy(n, result.energy.data());
            py::array_t<double> m(n, result.m.data());
            py::array_t<double> abs_m(n, result.abs_m.data());
            py::array_t<double> q(n, result.q.data());

            py::dict out;
            out["energy"] = std::move(energy);
            out["m"]      = std::move(m);
            out["abs_m"]  = std::move(abs_m);
            out["q"]      = std::move(q);
            return out;
        }, py::arg("n_sweeps"));
}
