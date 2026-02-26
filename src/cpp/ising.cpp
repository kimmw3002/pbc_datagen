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

int IsingModel::_delta_energy(int site) const {
    // ΔE = 2 * s_i * (sum of neighbor spins).
    // This is the energy change if spin[site] were flipped.
    int si = spin[static_cast<size_t>(site)];
    int neighbor_sum = 0;
    for (int d = 0; d < 4; ++d) {
        int j = nbr[static_cast<size_t>(site * 4 + d)];
        neighbor_sum += spin[static_cast<size_t>(j)];
    }
    return 2 * si * neighbor_sum;
}

int IsingModel::_metropolis_sweep() {
    // Precomputed acceptance probabilities.
    // For the Ising model on a square lattice, ΔE ∈ {-8, -4, 0, +4, +8}.
    // ΔE ≤ 0 is always accepted.  For ΔE > 0 we need exp(-ΔE/T).
    // Index by ΔE/4: slot 1 → ΔE=+4, slot 2 → ΔE=+8.
    double exp_table[3];
    exp_table[0] = 1.0;                       // ΔE = 0: always accept
    exp_table[1] = std::exp(-4.0 / T_);       // ΔE = +4
    exp_table[2] = std::exp(-8.0 / T_);       // ΔE = +8

    int accepted = 0;

    for (int step = 0; step < N; ++step) {
        // Pick a random site
        int site = static_cast<int>(rng.rand_below(static_cast<uint64_t>(N)));

        int dE = _delta_energy(site);

        // Accept if ΔE ≤ 0, otherwise accept with probability exp(-ΔE/T)
        if (dE <= 0 || rng.uniform() < exp_table[dE / 4]) {
            spin[static_cast<size_t>(site)] =
                static_cast<int8_t>(-spin[static_cast<size_t>(site)]);
            ++accepted;
        }
    }

    return accepted;
}

int IsingModel::_wolff_step() {
    // Bond activation probability: aligned neighbors join the cluster
    // with this probability.  At low T it approaches 1 (large clusters),
    // at high T it approaches 0 (clusters of size ~1).
    double p_add = 1.0 - std::exp(-2.0 / T_);

    // Pick a random seed spin
    int seed = static_cast<int>(rng.rand_below(static_cast<uint64_t>(N)));
    int8_t seed_spin = spin[static_cast<size_t>(seed)];

    // Track which sites belong to the cluster
    std::vector<bool> in_cluster(static_cast<size_t>(N), false);

    // DFS stack for growing the cluster
    std::vector<int> stack;
    stack.push_back(seed);
    in_cluster[static_cast<size_t>(seed)] = true;
    int cluster_size = 0;

    while (!stack.empty()) {
        int site = stack.back();
        stack.pop_back();
        ++cluster_size;

        // Try to add each of the 4 neighbors
        for (int d = 0; d < 4; ++d) {
            int j = nbr[static_cast<size_t>(site * 4 + d)];
            if (!in_cluster[static_cast<size_t>(j)]
                && spin[static_cast<size_t>(j)] == seed_spin
                && rng.uniform() < p_add) {
                in_cluster[static_cast<size_t>(j)] = true;
                stack.push_back(j);
            }
        }
    }

    // Flip every spin in the cluster
    for (int i = 0; i < N; ++i) {
        if (in_cluster[static_cast<size_t>(i)]) {
            spin[static_cast<size_t>(i)] = static_cast<int8_t>(-spin[static_cast<size_t>(i)]);
        }
    }

    return cluster_size;
}

IsingModel::SweepResult IsingModel::sweep(int n_sweeps) {
    if (T_ <= 0.0) {
        throw std::invalid_argument(
            "Temperature not set — call set_temperature() before sweep()");
    }

    SweepResult result;
    result.energy.resize(static_cast<size_t>(n_sweeps));
    result.m.resize(static_cast<size_t>(n_sweeps));
    result.abs_m.resize(static_cast<size_t>(n_sweeps));

    for (int i = 0; i < n_sweeps; ++i) {
        _metropolis_sweep();
        _wolff_step();

        auto idx = static_cast<size_t>(i);
        result.energy[idx] = energy();
        result.m[idx]      = magnetization();
        result.abs_m[idx]  = abs_magnetization();
    }

    return result;
}

}  // namespace pbc
