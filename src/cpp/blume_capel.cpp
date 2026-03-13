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
      rng(seed),
      cached_energy_(-2.0 * L * L),  // all aligned, D=0: -2N + 0*N
      cached_m_sum_(L * L),           // all +1: sum = N
      cached_sq_sum_(L * L),
      wl_in_cluster_(static_cast<size_t>(L * L), 0) {}        // all 1²: sum = N

void BlumeCapelModel::set_temperature(double T) {
    if (T <= 0.0) {
        throw std::invalid_argument("Temperature must be positive");
    }
    T_ = T;
}

void BlumeCapelModel::set_crystal_field(double D) {
    cached_energy_ += (D - D_) * cached_sq_sum_;
    D_ = D;
}

void BlumeCapelModel::set_spin(int site, int8_t value) {
    if (site < 0 || site >= N) {
        throw std::out_of_range("site index out of range");
    }
    if (value != 1 && value != -1 && value != 0) {
        throw std::invalid_argument("spin value must be -1, 0, or +1");
    }
    int8_t old = spin[static_cast<size_t>(site)];
    if (old == value) return;

    cached_energy_ += _delta_energy(site, value);
    cached_m_sum_ += (value - old);
    cached_sq_sum_ += (value * value - old * old);

    spin[static_cast<size_t>(site)] = value;
}

double BlumeCapelModel::energy() const {
    return cached_energy_;
}

double BlumeCapelModel::magnetization() const {
    return static_cast<double>(cached_m_sum_) / N;
}

double BlumeCapelModel::abs_magnetization() const {
    return std::abs(static_cast<double>(cached_m_sum_)) / N;
}

double BlumeCapelModel::quadrupole() const {
    return static_cast<double>(cached_sq_sum_) / N;
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

    // Reuse persistent workspace (no heap allocation).
    wl_stack_.clear();
    wl_cluster_sites_.clear();

    wl_stack_.push_back(seed);
    wl_in_cluster_[static_cast<size_t>(seed)] = 1;

    while (!wl_stack_.empty()) {
        int site = wl_stack_.back();
        wl_stack_.pop_back();
        wl_cluster_sites_.push_back(site);

        // Try to add each of the 4 neighbors
        for (int d = 0; d < 4; ++d) {
            int j = nbr[static_cast<size_t>(site * 4 + d)];
            // Neighbor must: (1) not already be in cluster,
            //                (2) have the same spin as the seed (±1),
            //                (3) pass the bond activation coin flip.
            // Spin-0 neighbors naturally fail check (2) and act as barriers.
            if (!wl_in_cluster_[static_cast<size_t>(j)]
                && spin[static_cast<size_t>(j)] == seed_spin
                && rng.uniform() < p_add) {
                wl_in_cluster_[static_cast<size_t>(j)] = 1;
                wl_stack_.push_back(j);
            }
        }
    }

    int cluster_size = static_cast<int>(wl_cluster_sites_.size());

    // Flip cluster and update cache — O(cluster_size), not O(N).
    // Energy: only boundary bonds change (same as Ising).
    // Crystal field: (-s)² = s², so D term is unchanged.
    // Quadrupole: unchanged for same reason.
    // Magnetization: all cluster spins were seed_spin, now -seed_spin.
    int delta_energy = 0;
    for (int site : wl_cluster_sites_) {
        auto idx = static_cast<size_t>(site);
        for (int d = 0; d < 4; ++d) {
            int j = nbr[static_cast<size_t>(site * 4 + d)];
            if (!wl_in_cluster_[static_cast<size_t>(j)]) {
                delta_energy += 2 * spin[idx] * spin[static_cast<size_t>(j)];
            }
        }
        spin[idx] = static_cast<int8_t>(-spin[idx]);
    }
    cached_energy_ += delta_energy;
    cached_m_sum_ -= 2 * seed_spin * cluster_size;
    // cached_sq_sum_ unchanged: (-s)² = s²

    // Clear in_cluster for only the sites we used — O(cluster_size).
    for (int site : wl_cluster_sites_) {
        wl_in_cluster_[static_cast<size_t>(site)] = 0;
    }

    return cluster_size;
}

double BlumeCapelModel::_delta_energy(int site, int8_t new_spin) const {
    int8_t old_spin = spin[static_cast<size_t>(site)];

    // Sum of neighbor spins
    int neighbor_sum = 0;
    for (int d = 0; d < 4; ++d) {
        int j = nbr[static_cast<size_t>(site * 4 + d)];
        neighbor_sum += spin[static_cast<size_t>(j)];
    }

    // ΔE = -(s_new - s_old) × Σ_neighbors  +  D × (s_new² - s_old²)
    int ds = new_spin - old_spin;
    int dsq = new_spin * new_spin - old_spin * old_spin;
    return -ds * neighbor_sum + D_ * dsq;
}

int BlumeCapelModel::_metropolis_sweep() {
    // Proposal lookup: for each current spin value, the two other options.
    // Index by (old_spin + 1) to map {-1, 0, +1} → {0, 1, 2}.
    static constexpr int8_t proposals[3][2] = {
        { 1,  0},  // old_spin = -1: propose +1 or 0
        { 1, -1},  // old_spin =  0: propose +1 or -1
        {-1,  0},  // old_spin = +1: propose -1 or 0
    };

    int accepted = 0;

    for (int step = 0; step < N; ++step) {
        // Pick a random site
        int site = static_cast<int>(rng.rand_below(static_cast<uint64_t>(N)));
        int8_t old_spin = spin[static_cast<size_t>(site)];

        // Pick new spin uniformly from the 2 alternatives
        int8_t new_spin = proposals[old_spin + 1][rng.rand_below(2)];

        double dE = _delta_energy(site, new_spin);

        // Accept if ΔE ≤ 0, otherwise with probability exp(-ΔE/T)
        if (dE <= 0.0 || rng.uniform() < std::exp(-dE / T_)) {
            cached_energy_ += dE;
            cached_m_sum_ += (new_spin - old_spin);
            cached_sq_sum_ += (new_spin * new_spin - old_spin * old_spin);
            spin[static_cast<size_t>(site)] = new_spin;
            ++accepted;
        }
    }

    return accepted;
}

BlumeCapelModel::SweepResult BlumeCapelModel::sweep(int n_sweeps) {
    if (T_ <= 0.0) {
        throw std::invalid_argument(
            "Temperature not set — call set_temperature() before sweep()");
    }

    SweepResult result;
    auto n = static_cast<size_t>(n_sweeps);
    result.energy.resize(n);
    result.m.resize(n);
    result.abs_m.resize(n);
    result.q.resize(n);

    for (int i = 0; i < n_sweeps; ++i) {
        _metropolis_sweep();
        _wolff_step();

        auto idx = static_cast<size_t>(i);
        result.energy[idx] = energy();
        result.m[idx]      = magnetization();
        result.abs_m[idx]  = abs_magnetization();
        result.q[idx]      = quadrupole();
    }

    return result;
}

BlumeCapelModel::ObsVec BlumeCapelModel::observables() const {
    return {
        {"energy",  energy()},
        {"m",       magnetization()},
        {"abs_m",   abs_magnetization()},
        {"q",       quadrupole()},
    };
}

std::vector<int8_t> BlumeCapelModel::snapshot() const {
    return spin;
}

void BlumeCapelModel::randomize() {
    // Each spin drawn uniformly from {-1, 0, +1}.
    cached_m_sum_ = 0;
    cached_sq_sum_ = 0;
    for (int i = 0; i < N; ++i) {
        int r = static_cast<int>(rng.rand_below(3));  // 0, 1, 2
        int8_t val = static_cast<int8_t>(r - 1);      // -1, 0, +1
        spin[static_cast<size_t>(i)] = val;
        cached_m_sum_ += val;
        cached_sq_sum_ += val * val;
    }

    // Recompute energy: H = -Σ_{<ij>} s_i s_j + D Σ s_i²
    int coupling = 0;
    for (int i = 0; i < N; ++i) {
        int si = spin[static_cast<size_t>(i)];
        int j_south = nbr[static_cast<size_t>(i * 4 + 1)];
        int j_east  = nbr[static_cast<size_t>(i * 4 + 2)];
        coupling += si * spin[static_cast<size_t>(j_south)];
        coupling += si * spin[static_cast<size_t>(j_east)];
    }
    cached_energy_ = -static_cast<double>(coupling) + D_ * cached_sq_sum_;
}

}  // namespace pbc
