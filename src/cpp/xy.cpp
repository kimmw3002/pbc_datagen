// XY model implementation — Step 6.0: construction + observables.

#include "xy.hpp"

#include <cmath>
#include <stdexcept>

namespace pbc {

// --- normalize_angle ---------------------------------------------------------
// Bring an arbitrary angle into [0, 2π) using fmod + correction for negatives.
double XYModel::normalize_angle(double a) {
    a = std::fmod(a, TWO_PI);
    if (a < 0.0) a += TWO_PI;
    return a;
}

// --- Constructor -------------------------------------------------------------
XYModel::XYModel(int L, uint64_t seed)
    : L(L),
      N(L * L),
      T_(0.0),
      theta(L * L, 0.0),            // cold start: all angles = 0
      nbr(make_neighbor_table(L)),
      rng(seed),
      cached_energy_(0.0),
      cached_mx_sum_(0.0),
      cached_my_sum_(0.0),
      wl_in_cluster_(L * L, 0),
      wl_stack_(),
      wl_cluster_sites_()
{
    // Cold start: all θ = 0.
    // cos(0 - 0) = 1 for every bond → E = -J × 2N = -2N.
    cached_energy_ = -2.0 * N;
    // cos(0) = 1, sin(0) = 0 for every site.
    cached_mx_sum_ = static_cast<double>(N);
    cached_my_sum_ = 0.0;
}

// --- set_temperature ---------------------------------------------------------
void XYModel::set_temperature(double T) {
    if (T <= 0.0) {
        throw std::invalid_argument("Temperature must be positive, got "
                                    + std::to_string(T));
    }
    T_ = T;
}

// --- set_spin ----------------------------------------------------------------
// Set the angle at a site and update cached observables incrementally.
void XYModel::set_spin(int site, double angle) {
    angle = normalize_angle(angle);
    double old_theta = theta[site];

    // Update magnetization sums: subtract old contribution, add new.
    cached_mx_sum_ += std::cos(angle) - std::cos(old_theta);
    cached_my_sum_ += std::sin(angle) - std::sin(old_theta);

    // Update energy: for each neighbor j, the bond contribution changes
    // from -cos(old - θ_j) to -cos(new - θ_j).
    for (int d = 0; d < 4; ++d) {
        int j = nbr[site * 4 + d];
        double theta_j = theta[j];
        // Old bond: -cos(old_theta - theta_j)
        // New bond: -cos(angle - theta_j)
        // ΔE for this bond = -cos(angle - theta_j) + cos(old_theta - theta_j)
        cached_energy_ += std::cos(old_theta - theta_j)
                        - std::cos(angle - theta_j);
    }

    theta[site] = angle;
}

// --- energy ------------------------------------------------------------------
double XYModel::energy() const {
    return cached_energy_;
}

// --- magnetization components ------------------------------------------------
double XYModel::mx() const {
    return cached_mx_sum_ / N;
}

double XYModel::my() const {
    return cached_my_sum_ / N;
}

double XYModel::abs_magnetization() const {
    return std::sqrt(cached_mx_sum_ * cached_mx_sum_
                   + cached_my_sum_ * cached_my_sum_) / N;
}

// --- observables -------------------------------------------------------------
XYModel::ObsVec XYModel::observables() const {
    return {
        {"energy",  energy()},
        {"mx",      mx()},
        {"my",      my()},
        {"abs_m",   abs_magnetization()},
    };
}

// --- _wolff_step -------------------------------------------------------------
// O(2) Wolff cluster algorithm (Wolff, 1989).
//
// The O(2) generalization of the Ising Wolff algorithm:
//
//   1. Pick a random reflection axis r̂ = (cos φ, sin φ).
//   2. Pick a random seed site.
//   3. DFS cluster growth: for neighbor j of cluster site i,
//      compute projections proj_i = cos(θ_i − φ), proj_j = cos(θ_j − φ).
//      If proj_i * proj_j > 0, add j with probability
//        p_add = 1 − exp(−2β × proj_i × proj_j).
//      Otherwise p_add = 0 (the min(0, ·) clamp).
//   4. Reflect cluster spins perpendicular to r̂:
//        s' = s − 2(s·r̂) r̂   →   θ' = 2φ + π − θ  (mod 2π).
//      This flips the projection: cos(θ'−φ) = −cos(θ−φ), which is
//      required for the boundary rejection ratio to equal exp(−βΔE).
//   5. Update cached energy and magnetization incrementally.
//
// Energy update:  only boundary bonds (cluster ↔ non-cluster) change.
// For a boundary bond (i∈C, j∉C):
//   cos(θ'_i − θ_j) − cos(θ_i − θ_j) = −2 cos(θ_i−φ) cos(θ_j−φ).
// Iterating cluster sites only, the factor-of-2 from both directed
// bond directions cancels with the 1/2 in E = −(1/2)Σ_directed, so:
//   ΔE = Σ_{i∈C, d, j=nbr∉C} [cos(θ_i−θ_j) + cos(2φ−θ_i−θ_j)]
//      = Σ_{i∈C, d, j=nbr∉C} 2 cos(θ_i−φ) cos(θ_j−φ).

int XYModel::_wolff_step() {
    // 1. Random reflection axis r̂ = (cos φ, sin φ).
    double phi = rng.uniform() * TWO_PI;

    // 2. Random seed site.
    int seed = static_cast<int>(rng.rand_below(static_cast<uint64_t>(N)));

    double beta = 1.0 / T_;

    // 3. DFS cluster growth.
    // Reuse persistent workspace — zeroed for exactly the sites used last call.
    wl_stack_.clear();
    wl_cluster_sites_.clear();

    wl_stack_.push_back(seed);
    wl_in_cluster_[static_cast<size_t>(seed)] = 1;

    while (!wl_stack_.empty()) {
        int site = wl_stack_.back();
        wl_stack_.pop_back();
        wl_cluster_sites_.push_back(site);

        // Projection of spin i onto the reflection axis r̂.
        double proj_i = std::cos(theta[static_cast<size_t>(site)] - phi);

        for (int d = 0; d < 4; ++d) {
            int j = nbr[static_cast<size_t>(site * 4 + d)];
            if (wl_in_cluster_[static_cast<size_t>(j)]) continue;

            double proj_j = std::cos(theta[static_cast<size_t>(j)] - phi);
            double coupling = proj_i * proj_j;

            // Only activate when both projections point the same way
            // (coupling > 0).  When coupling ≤ 0, the exponent argument
            // is ≥ 0 so p_add = 0 (via the min(0, ·) clamp).
            if (coupling > 0.0
                && rng.uniform() < 1.0 - std::exp(-2.0 * beta * coupling)) {
                wl_in_cluster_[static_cast<size_t>(j)] = 1;
                wl_stack_.push_back(j);
            }
        }
    }

    int cluster_size = static_cast<int>(wl_cluster_sites_.size());

    // 4–5. Reflect cluster spins and update cached observables.
    //
    // The reflection is PERPENDICULAR to r̂:
    //   s' = s − 2(s·r̂) r̂   →   θ' = 2φ + π − θ  (mod 2π).
    //
    // This flips the r̂-projection: cos(θ'−φ) = −cos(θ−φ), which is
    // essential for detailed balance (the rejection probability ratio
    // at boundary bonds equals exp(−βΔE) only when projections flip).
    //
    // Energy: only boundary bonds (cluster ↔ non-cluster) change.
    // For a directed bond (i∈C → j∉C):
    //   cos(θ'_i − θ_j) − cos(θ_i − θ_j) = −2 cos(θ_i−φ) cos(θ_j−φ)
    // Counting each boundary pair once from the cluster side, the
    // factor-of-2 from both directions cancels with the 1/2 in E,
    // giving: ΔE = Σ_{i∈C,d,j∉C} 2 cos(θ_i−φ) cos(θ_j−φ).
    double delta_E  = 0.0;
    double delta_mx = 0.0;
    double delta_my = 0.0;

    // Precompute the reflection offset: 2φ + π.
    double refl_offset = 2.0 * phi + M_PI;

    for (int site : wl_cluster_sites_) {
        auto idx = static_cast<size_t>(site);
        double ti = theta[idx];

        // Energy: accumulate boundary bond changes.
        for (int d = 0; d < 4; ++d) {
            int j = nbr[static_cast<size_t>(site * 4 + d)];
            if (!wl_in_cluster_[static_cast<size_t>(j)]) {
                double tj = theta[static_cast<size_t>(j)];
                // cos(θ_i−θ_j) − cos(θ'_i−θ_j)  where θ'_i = 2φ+π−θ_i
                // = cos(θ_i−θ_j) − cos(2φ+π−θ_i−θ_j)
                // = cos(θ_i−θ_j) + cos(2φ−θ_i−θ_j)   [cos(x+π) = −cos(x)]
                delta_E += std::cos(ti - tj)
                         + std::cos(2.0 * phi - ti - tj);
            }
        }

        // Reflect: θ → 2φ + π − θ  (mod 2π).
        double new_theta = normalize_angle(refl_offset - ti);

        // Magnetization: subtract old, add new.
        delta_mx += std::cos(new_theta) - std::cos(ti);
        delta_my += std::sin(new_theta) - std::sin(ti);

        theta[idx] = new_theta;
    }

    cached_energy_  += delta_E;
    cached_mx_sum_  += delta_mx;
    cached_my_sum_  += delta_my;

    // Clear in_cluster for only the sites we used — O(cluster_size).
    for (int site : wl_cluster_sites_) {
        wl_in_cluster_[static_cast<size_t>(site)] = 0;
    }

    return cluster_size;
}

}  // namespace pbc
