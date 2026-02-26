// Blume-Capel model implementation — Step 1.2.1: struct, constructor, observables.

#include "blume_capel.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace pbc {

BlumeCapelModel::BlumeCapelModel(int L, uint64_t seed)
    : L(L),
      N(L * L),
      T_(0.0),
      D_(0.0),
      spin(static_cast<size_t>(L * L), int8_t{1}),   // cold start: all +1
      nbr(make_neighbor_table(L)),
      rng(seed) {}

void BlumeCapelModel::set_temperature(double T) {
    if (T <= 0.0) {
        throw std::invalid_argument("Temperature must be positive");
    }
    T_ = T;
}

void BlumeCapelModel::set_crystal_field(double D) {
    D_ = D;
}

void BlumeCapelModel::set_spin(int site, int8_t value) {
    if (site < 0 || site >= N) {
        throw std::out_of_range("site index out of range");
    }
    if (value != 1 && value != -1 && value != 0) {
        throw std::invalid_argument("spin value must be -1, 0, or +1");
    }
    spin[static_cast<size_t>(site)] = value;
}

double BlumeCapelModel::energy() const {
    // Coupling term: -J Σ_{<ij>} s_i s_j.
    // Sum over all (site, neighbor) pairs — each bond counted twice — then halve.
    int coupling_sum = 0;
    for (int i = 0; i < N; ++i) {
        int si = spin[static_cast<size_t>(i)];
        for (int d = 0; d < 4; ++d) {
            int j = nbr[static_cast<size_t>(i * 4 + d)];
            coupling_sum += si * spin[static_cast<size_t>(j)];
        }
    }
    double coupling = -coupling_sum / 2.0;

    // Crystal-field term: D Σ_i s_i².
    int sq_sum = 0;
    for (int i = 0; i < N; ++i) {
        int si = spin[static_cast<size_t>(i)];
        sq_sum += si * si;
    }
    double crystal = D_ * sq_sum;

    return coupling + crystal;
}

double BlumeCapelModel::magnetization() const {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += spin[static_cast<size_t>(i)];
    }
    return static_cast<double>(sum) / N;
}

double BlumeCapelModel::abs_magnetization() const {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += spin[static_cast<size_t>(i)];
    }
    return std::abs(static_cast<double>(sum)) / N;
}

double BlumeCapelModel::quadrupole() const {
    int sq_sum = 0;
    for (int i = 0; i < N; ++i) {
        int si = spin[static_cast<size_t>(i)];
        sq_sum += si * si;
    }
    return static_cast<double>(sq_sum) / N;
}

}  // namespace pbc
