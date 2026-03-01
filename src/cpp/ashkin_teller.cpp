// Ashkin-Teller model implementation — Step 1.3.1: struct, constructor, observables.

#include "ashkin_teller.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace pbc {

AshkinTellerModel::AshkinTellerModel(int L, uint64_t seed)
    : L(L),
      N(L * L),
      T_(0.0),
      U_(0.0),
      remapped_(false),
      sigma(static_cast<size_t>(L * L), int8_t{1}),   // cold start: all +1
      tau(static_cast<size_t>(L * L), int8_t{1}),      // cold start: all +1
      nbr(make_neighbor_table(L)),
      rng(seed),
      // Cold start: 2N bonds, each product = +1.
      cached_sigma_coupling_(2 * L * L),
      cached_tau_coupling_(2 * L * L),
      cached_four_spin_(2 * L * L),
      cached_sigma_sum_(L * L),
      cached_tau_sum_(L * L),
      cached_baxter_sum_(L * L),
      wl_in_cluster_(static_cast<size_t>(L * L), 0) {}

void AshkinTellerModel::set_temperature(double T) {
    if (T <= 0.0) {
        throw std::invalid_argument("Temperature must be positive");
    }
    T_ = T;
}

void AshkinTellerModel::set_four_spin_coupling(double U) {
    if (U < 0.0) {
        throw std::invalid_argument("Four-spin coupling U must be non-negative");
    }
    U_ = U;
    remapped_ = (U > 1.0);
}

void AshkinTellerModel::set_sigma(int site, int8_t value) {
    if (site < 0 || site >= N) {
        throw std::out_of_range("site index out of range");
    }
    if (value != 1 && value != -1) {
        throw std::invalid_argument("sigma spin value must be +1 or -1");
    }
    auto sdx = static_cast<size_t>(site);
    int8_t old_si = sigma[sdx];
    if (old_si == value) return;

    int8_t ti = tau[sdx];
    // Compute neighbor sums for bond cache updates
    int sigma_nbr = 0, sigma_tau_nbr = 0;
    for (int d = 0; d < 4; ++d) {
        auto j = static_cast<size_t>(nbr[static_cast<size_t>(site * 4 + d)]);
        sigma_nbr += sigma[j];
        sigma_tau_nbr += sigma[j] * tau[j];
    }
    cached_sigma_coupling_ -= 2 * old_si * sigma_nbr;
    cached_four_spin_ -= 2 * old_si * ti * sigma_tau_nbr;
    cached_sigma_sum_ -= 2 * old_si;
    cached_baxter_sum_ -= 2 * old_si * ti;

    sigma[sdx] = value;
}

void AshkinTellerModel::set_tau(int site, int8_t value) {
    if (site < 0 || site >= N) {
        throw std::out_of_range("site index out of range");
    }
    if (value != 1 && value != -1) {
        throw std::invalid_argument("tau spin value must be +1 or -1");
    }
    auto sdx = static_cast<size_t>(site);
    int8_t old_ti = tau[sdx];
    if (old_ti == value) return;

    int8_t si = sigma[sdx];
    int tau_nbr = 0, sigma_tau_nbr = 0;
    for (int d = 0; d < 4; ++d) {
        auto j = static_cast<size_t>(nbr[static_cast<size_t>(site * 4 + d)]);
        tau_nbr += tau[j];
        sigma_tau_nbr += sigma[j] * tau[j];
    }
    cached_tau_coupling_ -= 2 * old_ti * tau_nbr;
    cached_four_spin_ -= 2 * si * old_ti * sigma_tau_nbr;
    cached_tau_sum_ -= 2 * old_ti;
    cached_baxter_sum_ -= 2 * si * old_ti;

    tau[sdx] = value;
}

double AshkinTellerModel::energy() const {
    return -cached_sigma_coupling_ - cached_tau_coupling_
           - U_ * cached_four_spin_;
}

double AshkinTellerModel::m_sigma() const {
    return static_cast<double>(cached_sigma_sum_) / N;
}

double AshkinTellerModel::abs_m_sigma() const {
    return std::abs(static_cast<double>(cached_sigma_sum_)) / N;
}

double AshkinTellerModel::m_tau() const {
    return static_cast<double>(cached_tau_sum_) / N;
}

double AshkinTellerModel::abs_m_tau() const {
    return std::abs(static_cast<double>(cached_tau_sum_)) / N;
}

double AshkinTellerModel::m_baxter() const {
    return static_cast<double>(cached_baxter_sum_) / N;
}

double AshkinTellerModel::abs_m_baxter() const {
    return std::abs(static_cast<double>(cached_baxter_sum_)) / N;
}

double AshkinTellerModel::_delta_energy_sigma(int site) const {
    // ΔE for flipping σ_i → -σ_i, holding τ fixed.
    //
    // ΔE = 2σ_i Σ_{j∈nbr(i)} σ_j (1 + U τ_i τ_j)
    //
    // Derivation: the terms involving σ_i in H are:
    //   -σ_i Σ_j σ_j  (two-spin coupling)
    //   -U σ_i Σ_j σ_j τ_i τ_j  (four-spin coupling)
    // Flipping σ_i → -σ_i changes each of these by 2× its value.
    auto si = static_cast<size_t>(site);
    int8_t sigma_i = sigma[si];
    int8_t tau_i   = tau[si];

    double sum = 0.0;
    for (int d = 0; d < 4; ++d) {
        auto j = static_cast<size_t>(nbr[static_cast<size_t>(site * 4 + d)]);
        // Each neighbor contributes σ_j × (1 + U τ_i τ_j)
        sum += sigma[j] * (1.0 + U_ * tau_i * tau[j]);
    }

    return 2.0 * sigma_i * sum;
}

double AshkinTellerModel::_delta_energy_tau(int site) const {
    // ΔE for flipping τ_i → -τ_i, holding σ fixed.
    //
    // ΔE = 2τ_i Σ_{j∈nbr(i)} τ_j (1 + U σ_i σ_j)
    //
    // Identical structure to _delta_energy_sigma, with σ ↔ τ swapped.
    auto si = static_cast<size_t>(site);
    int8_t tau_i   = tau[si];
    int8_t sigma_i = sigma[si];

    double sum = 0.0;
    for (int d = 0; d < 4; ++d) {
        auto j = static_cast<size_t>(nbr[static_cast<size_t>(site * 4 + d)]);
        sum += tau[j] * (1.0 + U_ * sigma_i * sigma[j]);
    }

    return 2.0 * tau_i * sum;
}

int AshkinTellerModel::_metropolis_sweep() {
    // 2N proposals: N for σ flips, then N for τ flips.
    // Neighbor sums are computed inline so bond caches can be updated.
    int accepted = 0;

    // --- N σ-flip proposals ---
    for (int step = 0; step < N; ++step) {
        int site = static_cast<int>(rng.rand_below(static_cast<uint64_t>(N)));
        auto sdx = static_cast<size_t>(site);
        int8_t si = sigma[sdx], ti = tau[sdx];

        int sigma_nbr = 0, sigma_tau_nbr = 0;
        for (int d = 0; d < 4; ++d) {
            auto j = static_cast<size_t>(nbr[static_cast<size_t>(site * 4 + d)]);
            sigma_nbr += sigma[j];
            sigma_tau_nbr += sigma[j] * tau[j];
        }

        double dE = 2.0 * si * (sigma_nbr + U_ * ti * sigma_tau_nbr);
        if (dE <= 0.0 || rng.uniform() < std::exp(-dE / T_)) {
            cached_sigma_coupling_ -= 2 * si * sigma_nbr;
            cached_four_spin_ -= 2 * si * ti * sigma_tau_nbr;
            cached_sigma_sum_ -= 2 * si;
            cached_baxter_sum_ -= 2 * si * ti;
            sigma[sdx] = static_cast<int8_t>(-si);
            ++accepted;
        }
    }

    // --- N τ-flip proposals ---
    for (int step = 0; step < N; ++step) {
        int site = static_cast<int>(rng.rand_below(static_cast<uint64_t>(N)));
        auto sdx = static_cast<size_t>(site);
        int8_t si = sigma[sdx], ti = tau[sdx];

        int tau_nbr = 0, sigma_tau_nbr = 0;
        for (int d = 0; d < 4; ++d) {
            auto j = static_cast<size_t>(nbr[static_cast<size_t>(site * 4 + d)]);
            tau_nbr += tau[j];
            sigma_tau_nbr += sigma[j] * tau[j];
        }

        double dE = 2.0 * ti * (tau_nbr + U_ * si * sigma_tau_nbr);
        if (dE <= 0.0 || rng.uniform() < std::exp(-dE / T_)) {
            cached_tau_coupling_ -= 2 * ti * tau_nbr;
            cached_four_spin_ -= 2 * si * ti * sigma_tau_nbr;
            cached_tau_sum_ -= 2 * ti;
            cached_baxter_sum_ -= 2 * si * ti;
            tau[sdx] = static_cast<int8_t>(-ti);
            ++accepted;
        }
    }

    return accepted;
}

int AshkinTellerModel::_wolff_step() {
    // --- 1. Pick a channel (0 or 1) at random (50/50) ---
    int channel = (rng.uniform() < 0.5) ? 0 : 1;

    // --- 2. Determine clustering mode and coupling constants ---
    //
    // mode encodes which variable pair we cluster on:
    //   0: cluster σ, hold τ fixed       (U ≤ 1)  → flip σ only
    //   1: cluster τ, hold σ fixed       (U ≤ 1)  → flip τ only
    //   2: cluster σ, hold s=στ fixed    (U > 1)  → flip both σ and τ
    //   3: cluster s=στ, hold σ fixed    (U > 1)  → flip τ only
    //
    // Instead of copying N spins into target[] and fixed[] arrays,
    // we compute them on-the-fly from sigma[] and tau[] using lambdas.
    // The 'mode' variable is loop-invariant, so the branch predictor
    // handles the switch at zero cost.

    int mode;
    double J_base, U_eff;

    if (!remapped_) {
        J_base = 1.0;
        U_eff  = U_;
        mode   = channel;  // 0 or 1
    } else if (channel == 0) {
        // Cluster on σ, hold s fixed.  Flipping σ with s held → both flip.
        J_base = 1.0;
        U_eff  = 1.0;
        mode   = 2;
    } else {
        // Cluster on s, hold σ fixed.  Flipping s with σ held → only τ flips.
        J_base = U_;
        U_eff  = 1.0;
        mode   = 3;
    }

    // Compute the target spin (variable we cluster on) at site idx.
    // A lambda is a small unnamed function defined inline:
    //   [&] captures all local variables by reference (sigma, tau, mode).
    //   (size_t idx) is the parameter.
    //   -> int8_t is the return type.
    auto target_at = [&](size_t idx) -> int8_t {
        switch (mode) {
            case 0:  return sigma[idx];                                    // σ
            case 1:  return tau[idx];                                      // τ
            case 2:  return sigma[idx];                                    // σ
            case 3:  return static_cast<int8_t>(sigma[idx] * tau[idx]);    // s = στ
            default: return 0;  // unreachable
        }
    };

    // Compute the fixed spin (variable held constant) at site idx.
    auto fixed_at = [&](size_t idx) -> int8_t {
        switch (mode) {
            case 0:  return tau[idx];                                      // τ
            case 1:  return sigma[idx];                                    // σ
            case 2:  return static_cast<int8_t>(sigma[idx] * tau[idx]);    // s = στ
            case 3:  return sigma[idx];                                    // σ
            default: return 0;  // unreachable
        }
    };

    // --- 3. Precompute the two possible bond probabilities ---
    //
    // When the fixed-layer spins at j and k are aligned:
    //   J_eff = J_base + U_eff  →  p_aligned = 1 - exp(-2(J_base + U_eff)/T)
    // When anti-aligned:
    //   J_eff = J_base - U_eff  →  p_anti    = 1 - exp(-2(J_base - U_eff)/T)

    double p_aligned = 1.0 - std::exp(-2.0 * (J_base + U_eff) / T_);
    double p_anti    = 1.0 - std::exp(-2.0 * (J_base - U_eff) / T_);

    // --- 4. Pick a random seed site ---
    int seed = static_cast<int>(rng.rand_below(static_cast<uint64_t>(N)));
    int8_t seed_target = target_at(static_cast<size_t>(seed));

    // --- 5. Grow the cluster via DFS (persistent workspace) ---
    wl_stack_.clear();
    wl_cluster_sites_.clear();

    wl_stack_.push_back(seed);
    wl_in_cluster_[static_cast<size_t>(seed)] = 1;

    while (!wl_stack_.empty()) {
        int site = wl_stack_.back();
        wl_stack_.pop_back();
        wl_cluster_sites_.push_back(site);

        auto sdx = static_cast<size_t>(site);
        int8_t f_site = fixed_at(sdx);

        for (int d = 0; d < 4; ++d) {
            int j = nbr[static_cast<size_t>(site * 4 + d)];
            auto jdx = static_cast<size_t>(j);
            if (wl_in_cluster_[jdx]) continue;

            // Only aligned target-variable neighbors can join
            if (target_at(jdx) != seed_target) continue;

            // Bond probability depends on fixed-layer alignment
            double p_add = (f_site * fixed_at(jdx) > 0)
                               ? p_aligned
                               : p_anti;

            if (rng.uniform() < p_add) {
                wl_in_cluster_[jdx] = 1;
                wl_stack_.push_back(j);
            }
        }
    }

    int cluster_size = static_cast<int>(wl_cluster_sites_.size());

    // --- 6. Flip the cluster and update cached observables ---
    // Only iterate over cluster members — O(cluster_size), not O(N).
    //
    // Bond caches: only boundary bonds (cluster ↔ non-cluster) change.
    // Interior bonds (both endpoints flip) cancel out.
    //
    // Mode 0 (flip σ):        sigma_coupling + four_spin change
    // Mode 1/3 (flip τ):      tau_coupling + four_spin change
    // Mode 2 (flip both σ,τ): sigma_coupling + tau_coupling change,
    //                          four_spin UNCHANGED (both flip → cancels)
    int d_sigma_c = 0, d_tau_c = 0, d_four = 0;
    int d_sigma_sum = 0, d_tau_sum = 0, d_baxter_sum = 0;

    for (int site : wl_cluster_sites_) {
        auto idx = static_cast<size_t>(site);
        int8_t si = sigma[idx], ti = tau[idx];

        // Boundary bond contributions (pre-flip values)
        for (int d = 0; d < 4; ++d) {
            int j = nbr[static_cast<size_t>(site * 4 + d)];
            if (wl_in_cluster_[static_cast<size_t>(j)]) continue;
            auto jdx = static_cast<size_t>(j);
            int8_t sj = sigma[jdx], tj = tau[jdx];

            if (mode == 0) {
                d_sigma_c -= 2 * si * sj;
                d_four -= 2 * si * sj * ti * tj;
            } else if (mode == 2) {
                d_sigma_c -= 2 * si * sj;
                d_tau_c -= 2 * ti * tj;
                // four_spin unchanged
            } else {  // mode 1 or 3: flip τ only
                d_tau_c -= 2 * ti * tj;
                d_four -= 2 * si * sj * ti * tj;
            }
        }

        // Magnetization sums + flip
        if (mode == 0) {
            d_sigma_sum -= 2 * si;
            d_baxter_sum -= 2 * si * ti;
            sigma[idx] = static_cast<int8_t>(-si);
        } else if (mode == 2) {
            d_sigma_sum -= 2 * si;
            d_tau_sum -= 2 * ti;
            // baxter unchanged: (-σ)(-τ) = στ
            sigma[idx] = static_cast<int8_t>(-si);
            tau[idx]   = static_cast<int8_t>(-ti);
        } else {  // mode 1 or 3: flip τ only
            d_tau_sum -= 2 * ti;
            d_baxter_sum -= 2 * si * ti;
            tau[idx] = static_cast<int8_t>(-ti);
        }
    }

    cached_sigma_coupling_ += d_sigma_c;
    cached_tau_coupling_ += d_tau_c;
    cached_four_spin_ += d_four;
    cached_sigma_sum_ += d_sigma_sum;
    cached_tau_sum_ += d_tau_sum;
    cached_baxter_sum_ += d_baxter_sum;

    // Clear in_cluster for only the sites we used — O(cluster_size).
    for (int site : wl_cluster_sites_) {
        wl_in_cluster_[static_cast<size_t>(site)] = 0;
    }

    return cluster_size;
}

AshkinTellerModel::SweepResult AshkinTellerModel::sweep(int n_sweeps) {
    if (T_ <= 0.0) {
        throw std::invalid_argument(
            "Temperature not set — call set_temperature() before sweep()");
    }

    SweepResult result;
    auto n = static_cast<size_t>(n_sweeps);
    result.energy.resize(n);
    result.m_sigma.resize(n);
    result.abs_m_sigma.resize(n);
    result.m_tau.resize(n);
    result.abs_m_tau.resize(n);
    result.m_baxter.resize(n);
    result.abs_m_baxter.resize(n);

    for (int i = 0; i < n_sweeps; ++i) {
        _metropolis_sweep();
        _wolff_step();

        auto idx = static_cast<size_t>(i);
        result.energy[idx]       = energy();
        result.m_sigma[idx]      = m_sigma();
        result.abs_m_sigma[idx]  = abs_m_sigma();
        result.m_tau[idx]        = m_tau();
        result.abs_m_tau[idx]    = abs_m_tau();
        result.m_baxter[idx]     = m_baxter();
        result.abs_m_baxter[idx] = abs_m_baxter();
    }

    return result;
}

AshkinTellerModel::ObsVec AshkinTellerModel::observables() const {
    return {
        {"energy",       energy()},
        {"m_sigma",      m_sigma()},
        {"abs_m_sigma",  abs_m_sigma()},
        {"m_tau",        m_tau()},
        {"abs_m_tau",    abs_m_tau()},
        {"m_baxter",     m_baxter()},
        {"abs_m_baxter", abs_m_baxter()},
    };
}

}  // namespace pbc
