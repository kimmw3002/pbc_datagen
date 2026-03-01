// pt_engine.hpp — Parallel tempering inner-loop functions.
//
// Small, independently testable functions composed into pt_rounds().
// Each is bound via pybind11 and tested in isolation from Python.
//
// Temperature convention: temps sorted ascending, temps[0] = T_min (coldest).

#pragma once

#include <cmath>
#include <vector>

#include "prng.hpp"

namespace pbc {

// ── 1.5.1  pt_exchange ─────────────────────────────────────────────
// Pure accept/reject for a single gap.
// Returns true (accept) when min(1, exp((1/T_i - 1/T_j)*(E_i - E_j))) > u.
inline bool pt_exchange(double E_i, double E_j,
                        double T_i, double T_j,
                        Rng& rng) {
    double delta = (1.0 / T_i - 1.0 / T_j) * (E_i - E_j);
    if (delta >= 0.0) return true;           // always accept
    return rng.uniform() < std::exp(delta);  // Metropolis test
}

// ── 1.5.2  pt_exchange_round ───────────────────────────────────────
// Make M random swap proposals (one "round").
//
// For each proposal: pick a random gap g ∈ [0, M-2], look up the two
// replicas sitting at temperature slots g and g+1 via t2r, read their
// energies, call pt_exchange.  On accept: swap the r2t / t2r entries.
//
// replicas  – M model pointers (read energy only, not mutated)
// temps     – M temperatures sorted ascending
// r2t       – replica_to_temp map (mutated on accept)
// t2r       – temp_to_replica map (mutated on accept)
// n_accepts – per-gap accept counter, length M-1 (mutated)
// n_attempts– per-gap attempt counter, length M-1 (mutated)
// rng       – random number generator

template <typename Model>
void pt_exchange_round(
    std::vector<Model*>& replicas,
    const std::vector<double>& temps,
    std::vector<int>& r2t,
    std::vector<int>& t2r,
    std::vector<int>& n_accepts,
    std::vector<int>& n_attempts,
    Rng& rng)
{
    int M = static_cast<int>(replicas.size());

    for (int proposal = 0; proposal < M; ++proposal) {
        // Pick a random gap g ∈ [0, M-2]
        int g = static_cast<int>(rng.rand_below(static_cast<uint64_t>(M - 1)));

        // Which replicas sit at slots g and g+1?
        int r_lo = t2r[g];
        int r_hi = t2r[g + 1];

        // Read their energies
        double E_lo = replicas[r_lo]->energy();
        double E_hi = replicas[r_hi]->energy();

        ++n_attempts[g];

        if (pt_exchange(E_lo, E_hi, temps[g], temps[g + 1], rng)) {
            // Accept: swap the maps
            r2t[r_lo] = g + 1;
            r2t[r_hi] = g;
            t2r[g]     = r_hi;
            t2r[g + 1] = r_lo;
            ++n_accepts[g];
        }
    }
}

}  // namespace pbc
