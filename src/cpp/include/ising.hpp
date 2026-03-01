#pragma once
// Pure 2D Ising model — Wolff cluster + Metropolis hybrid update.
//
// H = -J Σ_{<ij>} s_i s_j,   J = 1,   s_i ∈ {+1, -1}.
//
// This file declares the struct and its methods.  The update kernels
// (Wolff, Metropolis) will be added in Steps 1.1.2–1.1.4.

#include "lattice.hpp"
#include "prng.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pbc {

struct IsingModel {
    int L;                          // lattice linear size
    int N;                          // total number of sites = L * L
    double T_;                      // temperature (0.0 = "not yet set")
    std::vector<int8_t> spin;       // spin array, length N, each +1 or -1
    std::vector<int32_t> nbr;       // PBC neighbor table, length N*4
    Rng rng;                        // PRNG instance

    // Cached observables — updated incrementally on every spin mutation.
    // energy() and magnetization() read these in O(1) instead of O(N).
    int cached_energy_;             // H = -J Σ_{<ij>} s_i s_j
    int cached_m_sum_;              // Σ_i s_i  (raw sum, NOT divided by N)

    // Persistent Wolff workspace — allocated once in constructor, reused
    // every _wolff_step() call to avoid per-step heap allocations.
    // char instead of bool: vector<bool> packs bits (slower per-access).
    std::vector<char> wl_in_cluster_;       // 1 if site is in cluster
    std::vector<int>  wl_stack_;            // DFS frontier
    std::vector<int>  wl_cluster_sites_;    // sites that were added

    // Construct an L×L Ising model.  Spins start in the cold state
    // (all +1).  Temperature must be set separately via set_temperature().
    IsingModel(int L, uint64_t seed);

    // Set the simulation temperature.  Must be > 0.
    void set_temperature(double T);

    // Set an individual spin (for testing / manual configuration).
    // value must be +1 or -1.
    void set_spin(int site, int8_t value);

    // Total energy:  H = -J Σ_{<ij>} s_i s_j  (integer, since J=1).
    int energy() const;

    // Intensive magnetization:  m = (1/N) Σ_i s_i.
    double magnetization() const;

    // Intensive absolute magnetization:  |m| = (1/N) |Σ_i s_i|.
    double abs_magnetization() const;

    // Wolff single-cluster update (Wolff, 1989).
    // Grows a cluster from a random seed spin by adding aligned neighbors
    // with probability p_add = 1 - exp(-2J/T), then flips the cluster.
    // Returns the cluster size.
    int _wolff_step();

    // Local energy change if spin[site] were flipped (does NOT flip it).
    // ΔE = 2 * s_i * Σ_{j ∈ neighbors(i)} s_j.
    int _delta_energy(int site) const;

    // Metropolis sweep: N random-site single-spin-flip proposals.
    // Each proposal is accepted with probability min(1, exp(-ΔE/T)).
    // Returns the number of accepted flips.
    int _metropolis_sweep();

    // Result of sweep(): per-iteration observables.
    struct SweepResult {
        std::vector<int>    energy;   // total energy after each iteration
        std::vector<double> m;        // intensive magnetization after each
        std::vector<double> abs_m;    // intensive |magnetization| after each
    };

    // Combined update: n_sweeps iterations of (Metropolis sweep + Wolff step).
    // After each iteration, records (E, m, |m|) into the returned arrays.
    // Requires set_temperature() to have been called (T_ > 0).
    SweepResult sweep(int n_sweeps);
};

}  // namespace pbc
