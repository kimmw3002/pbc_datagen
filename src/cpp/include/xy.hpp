#pragma once
// 2D XY model — O(2) Wolff cluster + Metropolis hybrid update.
//
// H = -J Σ_{<ij>} cos(θ_i - θ_j),   J = 1,   θ_i ∈ [0, 2π).
//
// This is the first continuous-spin model in the codebase.  Spins are
// angles (doubles) rather than discrete ±1 values.
//
// Step 6.0: struct, constructor, observables.
// Steps 6.1–6.3 will add Wolff, Metropolis, and sweep().

#include "lattice.hpp"
#include "prng.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pbc {

// Two-pi constant, used throughout for angle normalization.
constexpr double TWO_PI = 2.0 * 3.14159265358979323846;

struct XYModel {
    int L;                          // lattice linear size
    int N;                          // total number of sites = L * L
    double T_;                      // temperature (0.0 = "not yet set")
    std::vector<double> theta;      // spin angles, length N, each in [0, 2π)
    std::vector<int32_t> nbr;       // PBC neighbor table, length N*4
    Rng rng;                        // PRNG instance

    // Cached observables — updated incrementally on every spin mutation.
    // Unlike discrete models, these are floating-point sums and may
    // accumulate rounding error over many mutations.  Periodic full
    // recomputation is recommended (see sweep()).
    double cached_energy_;          // H = -J Σ cos(θ_i - θ_j)
    double cached_mx_sum_;          // Σ cos(θ_i)
    double cached_my_sum_;          // Σ sin(θ_i)

    // Persistent Wolff workspace — allocated once in constructor, reused
    // every _wolff_step() call to avoid per-step heap allocations.
    std::vector<char> wl_in_cluster_;
    std::vector<int>  wl_stack_;
    std::vector<int>  wl_cluster_sites_;

    // Construct an L×L XY model.  Spins start in the cold state
    // (all θ = 0, pointing along +x).  Temperature must be set
    // separately via set_temperature().
    XYModel(int L, uint64_t seed);

    // Set the simulation temperature.  Must be > 0.
    void set_temperature(double T);

    // Set an individual spin angle.  The angle is normalized to [0, 2π).
    // Updates cached observables incrementally.
    void set_spin(int site, double angle);

    // Total energy:  H = -J Σ_{<ij>} cos(θ_i - θ_j).
    // Returns the cached value (O(1)).
    double energy() const;

    // Intensive magnetization components:
    //   mx = (1/N) Σ cos(θ_i),   my = (1/N) Σ sin(θ_i)
    double mx() const;
    double my() const;

    // Intensive absolute magnetization:  |m| = (1/N) √(mx_sum² + my_sum²).
    double abs_magnetization() const;

    // All observables as name-value pairs (for generic PT collection).
    using ObsVec = std::vector<std::pair<std::string, double>>;
    ObsVec observables() const;

    // Normalize an angle to [0, 2π).
    static double normalize_angle(double a);
};

}  // namespace pbc
