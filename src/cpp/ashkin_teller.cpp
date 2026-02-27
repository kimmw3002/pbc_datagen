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

}  // namespace pbc
