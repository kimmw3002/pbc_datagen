#pragma once
// 2D Blume-Capel model — Wolff cluster (adapted for 3-state) + Metropolis hybrid.
//
// H = -J Σ_{<ij>} s_i s_j  +  D Σ_i s_i²,   J = 1,   s_i ∈ {-1, 0, +1}.
//
// The s = 0 state is a "vacancy": it decouples from neighbors and
// contributes nothing to the crystal-field term (0² = 0).
//
// Step 1.2.1: struct, constructor, observables.
// Steps 1.2.2–1.2.4 will add GCA, Metropolis, and sweep().

#include "lattice.hpp"
#include "prng.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pbc {

struct BlumeCapelModel {
    int L;                          // lattice linear size
    int N;                          // total number of sites = L * L
    double T_;                      // temperature (0.0 = "not yet set")
    double D_;                      // crystal-field parameter (default 0.0)
    std::vector<int8_t> spin;       // spin array, length N, each -1, 0, or +1
    std::vector<int32_t> nbr;       // PBC neighbor table, length N*4
    Rng rng;                        // PRNG instance

    // Cached observables — updated incrementally on every spin mutation.
    double cached_energy_;          // H = -J Σ_{<ij>} s_i s_j + D Σ_i s_i²
    int cached_m_sum_;              // Σ_i s_i
    int cached_sq_sum_;             // Σ_i s_i²  (also needed by set_crystal_field)

    // Persistent Wolff workspace — allocated once, reused every step.
    std::vector<char> wl_in_cluster_;
    std::vector<int>  wl_stack_;
    std::vector<int>  wl_cluster_sites_;

    // Construct an L×L Blume-Capel model.  Spins start in the cold state
    // (all +1).  Temperature and crystal field must be set separately.
    BlumeCapelModel(int L, uint64_t seed);

    // Set the simulation temperature.  Must be > 0.
    void set_temperature(double T);

    // Set the crystal-field parameter D.  Any real value is valid:
    //   D > 0 → penalizes magnetic sites, favors vacancies
    //   D < 0 → penalizes vacancies, favors ordering
    //   D = 0 → pure Ising limit
    void set_crystal_field(double D);

    // Uniform parameter interface for 2D PT (delegates to set_crystal_field).
    void set_param(double p) { set_crystal_field(p); }

    // dE/dD = Σ s_i²  (coefficient of D in the energy).
    // Used by pt_exchange_param for param-direction exchanges.
    double dE_dparam() const { return static_cast<double>(cached_sq_sum_); }

    // Set an individual spin (for testing / manual configuration).
    // value must be -1, 0, or +1.
    void set_spin(int site, int8_t value);

    // Total energy:  H = -J Σ_{<ij>} s_i s_j  +  D Σ_i s_i².
    // Returns a double because D is continuous.
    double energy() const;

    // Intensive magnetization:  m = (1/N) Σ_i s_i.
    double magnetization() const;

    // Intensive absolute magnetization:  |m| = (1/N) |Σ_i s_i|.
    double abs_magnetization() const;

    // Quadrupole order parameter:  Q = (1/N) Σ_i s_i².
    // Q = 1 when all sites are magnetic (±1), Q = 0 when all vacant.
    double quadrupole() const;

    // All observables as name-value pairs (for generic PT collection).
    using ObsVec = std::vector<std::pair<std::string, double>>;
    ObsVec observables() const;

    // Wolff single-cluster update, adapted for 3-state spins.
    //
    // If the random seed lands on a vacancy (spin = 0), returns 0
    // immediately — vacancies can't seed clusters.
    //
    // Otherwise, grows a cluster via DFS through neighbors with the
    // same spin as the seed (±1).  Spin-0 sites fail the alignment
    // check and act as natural barriers that fragment the lattice.
    //
    // Bond probability: p_add = 1 - exp(-2J/T), identical to Ising.
    // The crystal field D does NOT affect Wolff bonds because
    // (-s)² = s²; the D·s² term cancels exactly under cluster flip.
    //
    // Returns the cluster size (0 if seed was vacant).
    int _wolff_step();

    // Local energy change if spin[site] were changed to new_spin.
    //
    // ΔE = -(s_new - s_old) × Σ_neighbors s_j  +  D × (s_new² - s_old²)
    //
    // Unlike Ising (where the only move is s → -s), BC has 3 spin values
    // so the energy change depends on WHICH new spin is proposed.
    // Returns a double because D is continuous.
    double _delta_energy(int site, int8_t new_spin) const;

    // One Metropolis sweep: N random-site proposals.
    //
    // For each proposal: pick a random site, choose new_spin uniformly
    // from {-1, 0, +1} \ {current} (symmetric proposal — no Hastings
    // correction needed).  Accept if ΔE ≤ 0 or uniform() < exp(-ΔE/T).
    //
    // Returns the number of accepted proposals.
    int _metropolis_sweep();

    // Result of sweep(): per-iteration observables.
    struct SweepResult {
        std::vector<double> energy;   // total energy (double because D is continuous)
        std::vector<double> m;        // intensive magnetization
        std::vector<double> abs_m;    // intensive |magnetization|
        std::vector<double> q;        // quadrupole order parameter
    };

    // Combined update: n_sweeps iterations of (Metropolis sweep + Wolff step).
    // After each iteration, records (E, m, |m|, Q) into the returned arrays.
    // Requires set_temperature() to have been called (T_ > 0).
    SweepResult sweep(int n_sweeps);
};

}  // namespace pbc
