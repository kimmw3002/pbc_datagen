// Ising model implementation — Step 1.1.1: struct, constructor, observables.

#include "ising.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace pbc {

IsingModel::IsingModel(int L, uint64_t seed)
    : L(L),
      N(L * L),
      T_(0.0),
      spin(static_cast<size_t>(L * L), int8_t{1}),   // cold start: all +1
      nbr(make_neighbor_table(L)),
      rng(seed) {}

void IsingModel::set_temperature(double T) {
    if (T <= 0.0) {
        throw std::invalid_argument("Temperature must be positive");
    }
    T_ = T;
}

void IsingModel::set_spin(int site, int8_t value) {
    if (site < 0 || site >= N) {
        throw std::out_of_range("site index out of range");
    }
    if (value != 1 && value != -1) {
        throw std::invalid_argument("spin value must be +1 or -1");
    }
    spin[static_cast<size_t>(site)] = value;
}

int IsingModel::energy() const {
    // Sum s_i * s_j over all (site, neighbor) pairs.  Each bond is
    // counted twice (once from each end), so divide by 2.
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        int si = spin[static_cast<size_t>(i)];
        for (int d = 0; d < 4; ++d) {
            int j = nbr[static_cast<size_t>(i * 4 + d)];
            sum += si * spin[static_cast<size_t>(j)];
        }
    }
    return -sum / 2;   // H = -J Σ_{<ij>} s_i s_j
}

double IsingModel::magnetization() const {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += spin[static_cast<size_t>(i)];
    }
    return static_cast<double>(sum) / N;
}

double IsingModel::abs_magnetization() const {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += spin[static_cast<size_t>(i)];
    }
    return std::abs(static_cast<double>(sum)) / N;
}

}  // namespace pbc
