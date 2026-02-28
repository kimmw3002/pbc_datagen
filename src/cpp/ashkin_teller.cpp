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
      rng(seed) {}

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
    sigma[static_cast<size_t>(site)] = value;
}

void AshkinTellerModel::set_tau(int site, int8_t value) {
    if (site < 0 || site >= N) {
        throw std::out_of_range("site index out of range");
    }
    if (value != 1 && value != -1) {
        throw std::invalid_argument("tau spin value must be +1 or -1");
    }
    tau[static_cast<size_t>(site)] = value;
}

double AshkinTellerModel::energy() const {
    // H = -J Σ σ_i σ_j  -  J Σ τ_i τ_j  -  U Σ σ_i σ_j τ_i τ_j
    //
    // Sum over all (site, neighbor) pairs — each bond counted twice — then halve.
    int sigma_sum = 0;
    int tau_sum = 0;
    int four_spin_sum = 0;

    for (int i = 0; i < N; ++i) {
        int si = sigma[static_cast<size_t>(i)];
        int ti = tau[static_cast<size_t>(i)];
        for (int d = 0; d < 4; ++d) {
            auto j = static_cast<size_t>(nbr[static_cast<size_t>(i * 4 + d)]);
            int sj = sigma[j];
            int tj = tau[j];
            sigma_sum    += si * sj;
            tau_sum      += ti * tj;
            four_spin_sum += si * sj * ti * tj;
        }
    }

    // Each bond was counted twice (once from each endpoint)
    return -sigma_sum / 2.0 - tau_sum / 2.0 - U_ * four_spin_sum / 2.0;
}

double AshkinTellerModel::m_sigma() const {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += sigma[static_cast<size_t>(i)];
    }
    return static_cast<double>(sum) / N;
}

double AshkinTellerModel::abs_m_sigma() const {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += sigma[static_cast<size_t>(i)];
    }
    return std::abs(static_cast<double>(sum)) / N;
}

double AshkinTellerModel::m_tau() const {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += tau[static_cast<size_t>(i)];
    }
    return static_cast<double>(sum) / N;
}

double AshkinTellerModel::abs_m_tau() const {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += tau[static_cast<size_t>(i)];
    }
    return std::abs(static_cast<double>(sum)) / N;
}

double AshkinTellerModel::m_baxter() const {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += sigma[static_cast<size_t>(i)] * tau[static_cast<size_t>(i)];
    }
    return static_cast<double>(sum) / N;
}

double AshkinTellerModel::abs_m_baxter() const {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += sigma[static_cast<size_t>(i)] * tau[static_cast<size_t>(i)];
    }
    return std::abs(static_cast<double>(sum)) / N;
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
    // Operates in the physical (σ, τ) basis regardless of U.
    int accepted = 0;

    // --- N σ-flip proposals ---
    for (int step = 0; step < N; ++step) {
        int site = static_cast<int>(rng.rand_below(static_cast<uint64_t>(N)));
        double dE = _delta_energy_sigma(site);
        if (dE <= 0.0 || rng.uniform() < std::exp(-dE / T_)) {
            sigma[static_cast<size_t>(site)] =
                static_cast<int8_t>(-sigma[static_cast<size_t>(site)]);
            ++accepted;
        }
    }

    // --- N τ-flip proposals ---
    for (int step = 0; step < N; ++step) {
        int site = static_cast<int>(rng.rand_below(static_cast<uint64_t>(N)));
        double dE = _delta_energy_tau(site);
        if (dE <= 0.0 || rng.uniform() < std::exp(-dE / T_)) {
            tau[static_cast<size_t>(site)] =
                static_cast<int8_t>(-tau[static_cast<size_t>(site)]);
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

    // --- 5. Grow the cluster via DFS ---
    std::vector<bool> in_cluster(static_cast<size_t>(N), false);
    std::vector<int> stack;
    stack.push_back(seed);
    in_cluster[static_cast<size_t>(seed)] = true;
    int cluster_size = 0;

    while (!stack.empty()) {
        int site = stack.back();
        stack.pop_back();
        ++cluster_size;

        auto sdx = static_cast<size_t>(site);
        int8_t f_site = fixed_at(sdx);

        for (int d = 0; d < 4; ++d) {
            int j = nbr[static_cast<size_t>(site * 4 + d)];
            auto jdx = static_cast<size_t>(j);
            if (in_cluster[jdx]) continue;

            // Only aligned target-variable neighbors can join
            if (target_at(jdx) != seed_target) continue;

            // Bond probability depends on fixed-layer alignment
            double p_add = (f_site * fixed_at(jdx) > 0)
                               ? p_aligned
                               : p_anti;

            if (rng.uniform() < p_add) {
                in_cluster[jdx] = true;
                stack.push_back(j);
            }
        }
    }

    // --- 6. Flip the cluster ---
    for (int i = 0; i < N; ++i) {
        if (!in_cluster[static_cast<size_t>(i)]) continue;
        auto idx = static_cast<size_t>(i);
        if (mode == 0) {
            // σ-clustering (non-remapped): flip σ only
            sigma[idx] = static_cast<int8_t>(-sigma[idx]);
        } else if (mode == 2) {
            // σ-clustering (remapped): flip both σ and τ
            sigma[idx] = static_cast<int8_t>(-sigma[idx]);
            tau[idx]   = static_cast<int8_t>(-tau[idx]);
        } else {
            // mode 1 (τ-clustering) or mode 3 (s-clustering): flip τ only
            tau[idx] = static_cast<int8_t>(-tau[idx]);
        }
    }

    return cluster_size;
}

}  // namespace pbc
