// pybind11 module definition — binds C++ models to Python.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ashkin_teller.hpp"
#include "blume_capel.hpp"
#include "ising.hpp"
#include "lattice.hpp"
#include "prng.hpp"
#include "pt_engine.hpp"

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
        .def("observables", [](const pbc::IsingModel& self) {
            py::dict out;
            for (auto& [name, val] : self.observables())
                out[py::cast(name)] = val;
            return out;
        })
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
        .def("observables", [](const pbc::BlumeCapelModel& self) {
            py::dict out;
            for (auto& [name, val] : self.observables())
                out[py::cast(name)] = val;
            return out;
        })
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

    // --- Ashkin-Teller model -------------------------------------------------
    py::class_<pbc::AshkinTellerModel>(m, "AshkinTellerModel")
        .def(py::init<int, uint64_t>(), py::arg("L"), py::arg("seed"))
        .def_readonly("L", &pbc::AshkinTellerModel::L)
        .def_property_readonly("T", [](const pbc::AshkinTellerModel& self) {
            return self.T_;
        })
        .def_property_readonly("U", [](const pbc::AshkinTellerModel& self) {
            return self.U_;
        })
        .def_property_readonly("sigma", [](const pbc::AshkinTellerModel& self) {
            return py::array_t<int8_t>(
                {self.L, self.L},
                {self.L * (int)sizeof(int8_t), (int)sizeof(int8_t)},
                self.sigma.data(),
                py::cast(self)
            );
        })
        .def_property_readonly("tau", [](const pbc::AshkinTellerModel& self) {
            return py::array_t<int8_t>(
                {self.L, self.L},
                {self.L * (int)sizeof(int8_t), (int)sizeof(int8_t)},
                self.tau.data(),
                py::cast(self)
            );
        })
        .def("set_temperature", &pbc::AshkinTellerModel::set_temperature, py::arg("T"))
        .def("set_four_spin_coupling", &pbc::AshkinTellerModel::set_four_spin_coupling,
             py::arg("U"))
        .def("set_sigma", &pbc::AshkinTellerModel::set_sigma,
             py::arg("site"), py::arg("value"))
        .def("set_tau", &pbc::AshkinTellerModel::set_tau,
             py::arg("site"), py::arg("value"))
        .def("energy", &pbc::AshkinTellerModel::energy)
        .def("m_sigma", &pbc::AshkinTellerModel::m_sigma)
        .def("abs_m_sigma", &pbc::AshkinTellerModel::abs_m_sigma)
        .def("m_tau", &pbc::AshkinTellerModel::m_tau)
        .def("abs_m_tau", &pbc::AshkinTellerModel::abs_m_tau)
        .def("m_baxter", &pbc::AshkinTellerModel::m_baxter)
        .def("abs_m_baxter", &pbc::AshkinTellerModel::abs_m_baxter)
        .def("_delta_energy_sigma", &pbc::AshkinTellerModel::_delta_energy_sigma,
             py::arg("site"))
        .def("_delta_energy_tau", &pbc::AshkinTellerModel::_delta_energy_tau,
             py::arg("site"))
        .def("_metropolis_sweep", &pbc::AshkinTellerModel::_metropolis_sweep)
        .def("_wolff_step", &pbc::AshkinTellerModel::_wolff_step)
        .def("observables", [](const pbc::AshkinTellerModel& self) {
            py::dict out;
            for (auto& [name, val] : self.observables())
                out[py::cast(name)] = val;
            return out;
        })
        .def("sweep", [](pbc::AshkinTellerModel& self, int n_sweeps) {
            auto result = self.sweep(n_sweeps);
            auto n = static_cast<py::ssize_t>(n_sweeps);

            py::array_t<double> energy(n, result.energy.data());
            py::array_t<double> m_sigma(n, result.m_sigma.data());
            py::array_t<double> abs_m_sigma(n, result.abs_m_sigma.data());
            py::array_t<double> m_tau(n, result.m_tau.data());
            py::array_t<double> abs_m_tau(n, result.abs_m_tau.data());
            py::array_t<double> m_baxter(n, result.m_baxter.data());
            py::array_t<double> abs_m_baxter(n, result.abs_m_baxter.data());

            py::dict out;
            out["energy"]       = std::move(energy);
            out["m_sigma"]      = std::move(m_sigma);
            out["abs_m_sigma"]  = std::move(abs_m_sigma);
            out["m_tau"]        = std::move(m_tau);
            out["abs_m_tau"]    = std::move(abs_m_tau);
            out["m_baxter"]     = std::move(m_baxter);
            out["abs_m_baxter"] = std::move(abs_m_baxter);
            return out;
        }, py::arg("n_sweeps"));

    // --- PT engine: non-templated functions --------------------------------
    m.def("pt_exchange", &pbc::pt_exchange,
          py::arg("E_i"), py::arg("E_j"),
          py::arg("T_i"), py::arg("T_j"),
          py::arg("rng"),
          "Single-gap Metropolis exchange decision.\n"
          "Returns True (accept) or False (reject).");

    // pybind11's automatic STL conversion copies Python lists into temporary
    // C++ vectors — mutations inside C++ would be lost.  These lambdas
    // manually copy int lists → vectors, call the C++ function, then write
    // the mutated values back into the original Python lists.
    // The vectors are tiny (M ≈ 20 ints) so the overhead is negligible.
    // In production, pt_rounds() keeps everything in C++ with no copying.
    auto list_to_ivec = [](py::list& lst) {
        std::vector<int> v(lst.size());
        for (size_t i = 0; i < lst.size(); ++i)
            v[i] = lst[i].cast<int>();
        return v;
    };
    auto ivec_to_list = [](const std::vector<int>& v, py::list& lst) {
        for (size_t i = 0; i < v.size(); ++i)
            lst[i] = v[i];
    };

    m.attr("LABEL_NONE") = pbc::LABEL_NONE;
    m.attr("LABEL_UP")   = pbc::LABEL_UP;
    m.attr("LABEL_DOWN") = pbc::LABEL_DOWN;

    m.def("pt_update_labels",
        [list_to_ivec, ivec_to_list](py::list py_labels, py::list py_t2r, int M) {
            auto labels = list_to_ivec(py_labels);
            auto t2r    = list_to_ivec(py_t2r);
            pbc::pt_update_labels(labels, t2r, M);
            ivec_to_list(labels, py_labels);
        },
        py::arg("labels"), py::arg("t2r"), py::arg("M"));

    m.def("pt_accumulate_histograms",
        [list_to_ivec, ivec_to_list](
            py::list py_n_up, py::list py_n_down,
            py::list py_labels, py::list py_t2r, int M) {
            auto n_up   = list_to_ivec(py_n_up);
            auto n_down = list_to_ivec(py_n_down);
            auto labels = list_to_ivec(py_labels);
            auto t2r    = list_to_ivec(py_t2r);
            pbc::pt_accumulate_histograms(n_up, n_down, labels, t2r, M);
            ivec_to_list(n_up, py_n_up);
            ivec_to_list(n_down, py_n_down);
        },
        py::arg("n_up"), py::arg("n_down"),
        py::arg("labels"), py::arg("t2r"), py::arg("M"));

    m.def("pt_count_round_trips",
        [list_to_ivec](py::list py_labels, py::list py_prev, py::list py_t2r, int M) {
            auto labels = list_to_ivec(py_labels);
            auto prev   = list_to_ivec(py_prev);
            auto t2r    = list_to_ivec(py_t2r);
            return pbc::pt_count_round_trips(labels, prev, t2r, M);
        },
        py::arg("labels"), py::arg("prev_labels"),
        py::arg("t2r"), py::arg("M"));

    // Helper: convert PTResult to a Python dict.
    // r2t, t2r, labels are NOT included — they're mutated in-place
    // via ivec_to_list in each binding lambda.
    auto pt_result_to_dict = [](const pbc::PTResult& res) {
        py::dict out;
        out["n_accepts"]        = res.n_accepts;
        out["n_attempts"]       = res.n_attempts;
        out["n_up"]             = res.n_up;
        out["n_down"]           = res.n_down;
        out["round_trip_count"] = res.round_trip_count;
        // obs_streams: map<string, vector<vector<double>>>
        //           → dict[str, list[list[float]]]
        py::dict streams;
        for (auto& [name, slots] : res.obs_streams) {
            py::list slot_list;
            for (auto& s : slots)
                slot_list.append(py::cast(s));
            streams[py::cast(name)] = slot_list;
        }
        out["obs_streams"] = streams;
        return out;
    };

    // --- PT engine: templated per-model ------------------------------------
    m.def("pt_exchange_round_ising",
        [list_to_ivec, ivec_to_list](
           py::list py_replicas,
           std::vector<double> temps,
           py::list py_r2t, py::list py_t2r,
           py::list py_acc, py::list py_att,
           pbc::Rng& rng) {
            // Extract model pointers
            std::vector<pbc::IsingModel*> reps;
            for (auto& obj : py_replicas)
                reps.push_back(obj.cast<pbc::IsingModel*>());
            // Copy mutable int lists
            auto r2t = list_to_ivec(py_r2t);
            auto t2r = list_to_ivec(py_t2r);
            auto acc = list_to_ivec(py_acc);
            auto att = list_to_ivec(py_att);

            pbc::pt_exchange_round(reps, temps, r2t, t2r, acc, att, rng);

            // Write mutations back
            ivec_to_list(r2t, py_r2t);
            ivec_to_list(t2r, py_t2r);
            ivec_to_list(acc, py_acc);
            ivec_to_list(att, py_att);
        },
        py::arg("replicas"), py::arg("temps"),
        py::arg("r2t"), py::arg("t2r"),
        py::arg("n_accepts"), py::arg("n_attempts"),
        py::arg("rng"));

    m.def("pt_exchange_round_bc",
        [list_to_ivec, ivec_to_list](
           py::list py_replicas,
           std::vector<double> temps,
           py::list py_r2t, py::list py_t2r,
           py::list py_acc, py::list py_att,
           pbc::Rng& rng) {
            std::vector<pbc::BlumeCapelModel*> reps;
            for (auto& obj : py_replicas)
                reps.push_back(obj.cast<pbc::BlumeCapelModel*>());
            auto r2t = list_to_ivec(py_r2t);
            auto t2r = list_to_ivec(py_t2r);
            auto acc = list_to_ivec(py_acc);
            auto att = list_to_ivec(py_att);

            pbc::pt_exchange_round(reps, temps, r2t, t2r, acc, att, rng);

            ivec_to_list(r2t, py_r2t);
            ivec_to_list(t2r, py_t2r);
            ivec_to_list(acc, py_acc);
            ivec_to_list(att, py_att);
        },
        py::arg("replicas"), py::arg("temps"),
        py::arg("r2t"), py::arg("t2r"),
        py::arg("n_accepts"), py::arg("n_attempts"),
        py::arg("rng"));

    m.def("pt_exchange_round_at",
        [list_to_ivec, ivec_to_list](
           py::list py_replicas,
           std::vector<double> temps,
           py::list py_r2t, py::list py_t2r,
           py::list py_acc, py::list py_att,
           pbc::Rng& rng) {
            std::vector<pbc::AshkinTellerModel*> reps;
            for (auto& obj : py_replicas)
                reps.push_back(obj.cast<pbc::AshkinTellerModel*>());
            auto r2t = list_to_ivec(py_r2t);
            auto t2r = list_to_ivec(py_t2r);
            auto acc = list_to_ivec(py_acc);
            auto att = list_to_ivec(py_att);

            pbc::pt_exchange_round(reps, temps, r2t, t2r, acc, att, rng);

            ivec_to_list(r2t, py_r2t);
            ivec_to_list(t2r, py_t2r);
            ivec_to_list(acc, py_acc);
            ivec_to_list(att, py_att);
        },
        py::arg("replicas"), py::arg("temps"),
        py::arg("r2t"), py::arg("t2r"),
        py::arg("n_accepts"), py::arg("n_attempts"),
        py::arg("rng"));

    // --- pt_collect_obs (templated, test-only bindings) ---
    // Takes a dict[str, list[list[float]]] and appends one round of readings.
    m.def("pt_collect_obs_ising",
        [list_to_ivec](py::dict py_obs, py::list py_replicas,
                        py::list py_t2r, int M) {
            std::vector<pbc::IsingModel*> reps;
            for (auto& obj : py_replicas)
                reps.push_back(obj.cast<pbc::IsingModel*>());
            auto t2r = list_to_ivec(py_t2r);
            // Convert py_obs → ObsStreams
            pbc::ObsStreams obs;
            for (auto& item : py_obs) {
                auto name = item.first.cast<std::string>();
                py::list slots = item.second.cast<py::list>();
                obs[name].resize(M);
                for (int t = 0; t < M; ++t) {
                    py::list inner = slots[t].cast<py::list>();
                    for (auto& v : inner)
                        obs[name][t].push_back(v.cast<double>());
                }
            }
            pbc::pt_collect_obs(obs, reps, t2r, M);
            // Write back
            for (auto& [name, slots] : obs) {
                py::list slot_list;
                for (auto& s : slots)
                    slot_list.append(py::cast(s));
                py_obs[py::cast(name)] = slot_list;
            }
        },
        py::arg("obs_streams"), py::arg("replicas"),
        py::arg("t2r"), py::arg("M"));

    // --- pt_rounds (templated, one per model) ---
    m.def("pt_rounds_ising",
        [list_to_ivec, ivec_to_list, pt_result_to_dict](
            py::list py_replicas, std::vector<double> temps,
            py::list py_r2t, py::list py_t2r,
            py::list py_labels, int n_rounds,
            pbc::Rng& rng, bool track_obs) {
            std::vector<pbc::IsingModel*> reps;
            for (auto& obj : py_replicas)
                reps.push_back(obj.cast<pbc::IsingModel*>());
            auto r2t    = list_to_ivec(py_r2t);
            auto t2r    = list_to_ivec(py_t2r);
            auto labels = list_to_ivec(py_labels);

            pbc::PTResult res;
            {
                // Release the GIL while running pure C++ — allows OpenMP
                // threads to run and other Python threads to proceed.
                py::gil_scoped_release release;
                res = pbc::pt_rounds(reps, temps, r2t, t2r, labels,
                                     n_rounds, rng, track_obs);
            }

            // r2t, t2r, labels were mutated by pt_rounds via reference —
            // write the final values back into the original Python lists.
            ivec_to_list(r2t, py_r2t);
            ivec_to_list(t2r, py_t2r);
            ivec_to_list(labels, py_labels);
            return pt_result_to_dict(res);
        },
        py::arg("replicas"), py::arg("temps"),
        py::arg("r2t"), py::arg("t2r"), py::arg("labels"),
        py::arg("n_rounds"), py::arg("rng"),
        py::arg("track_observables"));

    m.def("pt_rounds_bc",
        [list_to_ivec, ivec_to_list, pt_result_to_dict](
            py::list py_replicas, std::vector<double> temps,
            py::list py_r2t, py::list py_t2r,
            py::list py_labels, int n_rounds,
            pbc::Rng& rng, bool track_obs) {
            std::vector<pbc::BlumeCapelModel*> reps;
            for (auto& obj : py_replicas)
                reps.push_back(obj.cast<pbc::BlumeCapelModel*>());
            auto r2t    = list_to_ivec(py_r2t);
            auto t2r    = list_to_ivec(py_t2r);
            auto labels = list_to_ivec(py_labels);

            pbc::PTResult res;
            {
                py::gil_scoped_release release;
                res = pbc::pt_rounds(reps, temps, r2t, t2r, labels,
                                     n_rounds, rng, track_obs);
            }

            ivec_to_list(r2t, py_r2t);
            ivec_to_list(t2r, py_t2r);
            ivec_to_list(labels, py_labels);
            return pt_result_to_dict(res);
        },
        py::arg("replicas"), py::arg("temps"),
        py::arg("r2t"), py::arg("t2r"), py::arg("labels"),
        py::arg("n_rounds"), py::arg("rng"),
        py::arg("track_observables"));

    m.def("pt_rounds_at",
        [list_to_ivec, ivec_to_list, pt_result_to_dict](
            py::list py_replicas, std::vector<double> temps,
            py::list py_r2t, py::list py_t2r,
            py::list py_labels, int n_rounds,
            pbc::Rng& rng, bool track_obs) {
            std::vector<pbc::AshkinTellerModel*> reps;
            for (auto& obj : py_replicas)
                reps.push_back(obj.cast<pbc::AshkinTellerModel*>());
            auto r2t    = list_to_ivec(py_r2t);
            auto t2r    = list_to_ivec(py_t2r);
            auto labels = list_to_ivec(py_labels);

            pbc::PTResult res;
            {
                py::gil_scoped_release release;
                res = pbc::pt_rounds(reps, temps, r2t, t2r, labels,
                                     n_rounds, rng, track_obs);
            }

            ivec_to_list(r2t, py_r2t);
            ivec_to_list(t2r, py_t2r);
            ivec_to_list(labels, py_labels);
            return pt_result_to_dict(res);
        },
        py::arg("replicas"), py::arg("temps"),
        py::arg("r2t"), py::arg("t2r"), py::arg("labels"),
        py::arg("n_rounds"), py::arg("rng"),
        py::arg("track_observables"));
}
