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

}  // namespace pbc
