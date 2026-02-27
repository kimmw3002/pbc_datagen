#pragma once
// 2D Blume-Capel model — Geometric Cluster Algorithm + Metropolis hybrid.
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
};

}  // namespace pbc
