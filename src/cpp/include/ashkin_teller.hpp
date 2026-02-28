#pragma once
// 2D Ashkin-Teller model — two coupled Ising layers with four-spin coupling.
//
// H = -J Σ_{<ij>} σ_i σ_j  -  J Σ_{<ij>} τ_i τ_j  -  U Σ_{<ij>} σ_i σ_j τ_i τ_j
//
// J = 1 (fixed), U ≥ 0 is the four-spin coupling.
// σ_i, τ_i ∈ {-1, +1}.
//
// Step 1.3.1: struct, constructor, observables.
// Steps 1.3.2–1.3.4 will add embedded Wolff, Metropolis, and sweep().

#include "lattice.hpp"
#include "prng.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pbc {

struct AshkinTellerModel {
    int L;                          // lattice linear size
    int N;                          // total number of sites = L * L
    double T_;                      // temperature (0.0 = "not yet set")
    double U_;                      // four-spin coupling (default 0.0)
    bool remapped_;                 // true when U > 1 (work in σ, s=στ basis)
    std::vector<int8_t> sigma;      // σ spin array, length N, each ±1
    std::vector<int8_t> tau;        // τ spin array, length N, each ±1
    std::vector<int32_t> nbr;       // PBC neighbor table, length N*4
    Rng rng;                        // PRNG instance

    // Construct an L×L Ashkin-Teller model.  Both σ and τ start in the
    // cold state (all +1).  Temperature and U must be set separately.
    AshkinTellerModel(int L, uint64_t seed);

    // Set the simulation temperature.  Must be > 0.
    void set_temperature(double T);

    // Set the four-spin coupling U.  Must be ≥ 0.
    // Automatically sets remapped_ = (U > 1).
    void set_four_spin_coupling(double U);

    // Set an individual σ spin (for testing / manual configuration).
    // value must be +1 or -1.
    void set_sigma(int site, int8_t value);

    // Set an individual τ spin (for testing / manual configuration).
    // value must be +1 or -1.
    void set_tau(int site, int8_t value);

    // Total energy in the physical (σ, τ) basis:
    //   H = -Σ σ_i σ_j  -  Σ τ_i τ_j  -  U Σ σ_i σ_j τ_i τ_j
    // Returns double because U is continuous.
    double energy() const;

    // Intensive σ magnetization:  m_σ = (1/N) Σ σ_i.
    double m_sigma() const;

    // Intensive absolute σ magnetization:  |m_σ| = (1/N) |Σ σ_i|.
    double abs_m_sigma() const;

    // Intensive τ magnetization:  m_τ = (1/N) Σ τ_i.
    double m_tau() const;

    // Intensive absolute τ magnetization:  |m_τ| = (1/N) |Σ τ_i|.
    double abs_m_tau() const;

    // Baxter order parameter:  m_B = (1/N) Σ σ_i τ_i.
    double m_baxter() const;

    // Absolute Baxter order parameter:  |m_B| = (1/N) |Σ σ_i τ_i|.
    double abs_m_baxter() const;

    // Embedded Wolff single-cluster update (Wiseman & Domany, 1995).
    //
    // Projects the two-layer model onto a single Ising-like variable:
    //   U ≤ 1: cluster on σ or τ (50/50), hold the other fixed.
    //   U > 1: remap to (σ, s=στ) basis; cluster on σ or s (50/50).
    //
    // Bond coupling is per-neighbor-pair:
    //   J_eff(j,k) = J_base + U_eff × fixed_j × fixed_k
    // where J_base and U_eff depend on the chosen channel and remapping.
    //
    // Returns the cluster size (always ≥ 1).
    int _wolff_step();
};

}  // namespace pbc
