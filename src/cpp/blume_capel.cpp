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

int BlumeCapelModel::_wolff_step() {
    // Pick a random seed site
    int seed = static_cast<int>(rng.rand_below(static_cast<uint64_t>(N)));
    int8_t seed_spin = spin[static_cast<size_t>(seed)];

    // Vacancies can't seed clusters — return immediately
    if (seed_spin == 0) {
        return 0;
    }

    // Bond activation probability — identical to Ising.
    // D doesn't enter here because (-s)² = s²: the crystal-field
    // contribution cancels exactly when a cluster is flipped.
    double p_add = 1.0 - std::exp(-2.0 / T_);

    // Track which sites belong to the cluster
    std::vector<bool> in_cluster(static_cast<size_t>(N), false);

    // Explicit DFS stack (not recursion — avoids stack overflow on large lattices)
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
            // Neighbor must: (1) not already be in cluster,
            //                (2) have the same spin as the seed (±1),
            //                (3) pass the bond activation coin flip.
            // Spin-0 neighbors naturally fail check (2) and act as barriers.
            if (!in_cluster[static_cast<size_t>(j)]
                && spin[static_cast<size_t>(j)] == seed_spin
                && rng.uniform() < p_add) {
                in_cluster[static_cast<size_t>(j)] = true;
                stack.push_back(j);
            }
        }
    }

    // Flip every spin in the cluster: s → -s
    for (int i = 0; i < N; ++i) {
        if (in_cluster[static_cast<size_t>(i)]) {
            spin[static_cast<size_t>(i)] =
                static_cast<int8_t>(-spin[static_cast<size_t>(i)]);
        }
    }

    return cluster_size;
}

}  // namespace pbc
