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
};

}  // namespace pbc
