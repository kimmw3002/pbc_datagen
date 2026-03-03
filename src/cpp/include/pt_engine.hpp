// pt_engine.hpp — Parallel tempering inner-loop functions.
//
// Small, independently testable functions composed into pt_rounds().
// Each is bound via pybind11 and tested in isolation from Python.
//
// Temperature convention: temps sorted ascending, temps[0] = T_min (coldest).

#pragma once

#include <cmath>
#include <map>
#include <string>
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

// ── 3.2.2  pt_exchange_param ──────────────────────────────────────
// Param-direction exchange at fixed temperature.
// Δ = β × (param_i − param_j) × (dEdp_i − dEdp_j)
// where dEdp = dE/d(param) for each replica:
//   BC: dE/dD = Σ s_i²  (cached_sq_sum_)
//   AT: dE/dU = −cached_four_spin_
inline bool pt_exchange_param(double dEdp_i, double dEdp_j,
                              double T,
                              double param_i, double param_j,
                              Rng& rng) {
    double delta = (1.0 / T) * (param_i - param_j) * (dEdp_i - dEdp_j);
    if (delta >= 0.0) return true;
    return rng.uniform() < std::exp(delta);
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

// ── Label constants ────────────────────────────────────────────────
constexpr int LABEL_NONE = 0;
constexpr int LABEL_UP   = 1;   // last visited T_min (cold end)
constexpr int LABEL_DOWN = -1;  // last visited T_max (hot end)

// ── 1.5.3  pt_update_labels ───────────────────────────────────────
// Assign directional labels at the temperature extremes.
// Replica at coldest slot (0) → UP, replica at hottest slot (M-1) → DOWN.
// All other labels unchanged.
inline void pt_update_labels(
    std::vector<int>& labels,
    const std::vector<int>& t2r,
    int M)
{
    labels[t2r[0]]     = LABEL_UP;
    labels[t2r[M - 1]] = LABEL_DOWN;
}

// ── 1.5.4  pt_accumulate_histograms ───────────────────────────────
// For each T slot, increment n_up or n_down based on the directional
// label of the replica currently sitting there.  Skip LABEL_NONE.
inline void pt_accumulate_histograms(
    std::vector<int>& n_up,
    std::vector<int>& n_down,
    const std::vector<int>& labels,
    const std::vector<int>& t2r,
    int M)
{
    for (int t = 0; t < M; ++t) {
        int lbl = labels[t2r[t]];
        if (lbl == LABEL_UP)        ++n_up[t];
        else if (lbl == LABEL_DOWN) ++n_down[t];
    }
}

// ── 1.5.5  pt_count_round_trips ───────────────────────────────────
// A round trip completes when an UP-labeled replica reaches the hot end
// (slot M-1) and gets relabeled DOWN.  Check the replica at slot M-1:
// if prev was UP and curr is DOWN, that's one completed trip.
inline int pt_count_round_trips(
    const std::vector<int>& labels,
    const std::vector<int>& prev_labels,
    const std::vector<int>& t2r,
    int M)
{
    int r = t2r[M - 1];
    if (prev_labels[r] == LABEL_UP && labels[r] == LABEL_DOWN)
        return 1;
    return 0;
}

// ── Observable stream type ─────────────────────────────────────────
// obs_streams[obs_name][T_slot] → vector of values across rounds.
// E.g. obs_streams["energy"][0] = {E_round0, E_round1, ...}
using ObsStreams = std::map<std::string, std::vector<std::vector<double>>>;

// ── 1.5.6  pt_collect_obs ─────────────────────────────────────────
// For each T slot, read ALL observables from the replica sitting there
// and append each value to the corresponding stream.
// Uses the model's observables() method which returns name-value pairs.
template <typename Model>
void pt_collect_obs(
    ObsStreams& obs_streams,
    const std::vector<Model*>& replicas,
    const std::vector<int>& t2r,
    int M)
{
    for (int t = 0; t < M; ++t) {
        auto obs = replicas[t2r[t]]->observables();
        for (auto& [name, val] : obs) {
            obs_streams[name][t].push_back(val);
        }
    }
}

// ── 1.5.7  pt_rounds — thin composition loop ─────────────────────
// Result struct returned to Python as a dict.
// r2t, t2r, labels are NOT here — they're mutated in-place via
// reference parameters and written back by the pybind11 binding.
struct PTResult {
    std::vector<int> n_accepts;
    std::vector<int> n_attempts;
    std::vector<int> n_up;
    std::vector<int> n_down;
    int round_trip_count;
    ObsStreams obs_streams;
};

template <typename Model>
PTResult pt_rounds(
    std::vector<Model*>& replicas,
    const std::vector<double>& temps,
    std::vector<int>& r2t,
    std::vector<int>& t2r,
    std::vector<int>& labels,
    int n_rounds,
    Rng& rng,
    bool track_observables)
{
    int M = static_cast<int>(replicas.size());

    std::vector<int> n_accepts(M - 1, 0);
    std::vector<int> n_attempts(M - 1, 0);
    std::vector<int> n_up(M, 0);
    std::vector<int> n_down(M, 0);
    int round_trip_count = 0;

    // Pre-allocate obs_streams if tracking: discover keys from first replica
    ObsStreams obs_streams;
    if (track_observables) {
        auto obs = replicas[0]->observables();
        for (auto& [name, _] : obs) {
            obs_streams[name].resize(M);
        }
    }

    for (int round = 0; round < n_rounds; ++round) {
        // Sweep each replica at its current temperature.
        // Each replica has independent state (lattice, RNG, cache)
        // so iterations are fully independent — safe for OpenMP.
        // if(M >= 8): skip thread-pool overhead for tiny replica counts
        // (tests use M=3; production uses M=20–50).
        #pragma omp parallel for schedule(static) if(M >= 8)
        for (int r = 0; r < M; ++r) {
            replicas[r]->set_temperature(temps[r2t[r]]);
            replicas[r]->sweep(1);
        }

        // Save prev labels for round-trip detection
        std::vector<int> prev_labels(labels);

        pt_exchange_round(replicas, temps, r2t, t2r, n_accepts, n_attempts, rng);
        pt_update_labels(labels, t2r, M);
        pt_accumulate_histograms(n_up, n_down, labels, t2r, M);
        round_trip_count += pt_count_round_trips(labels, prev_labels, t2r, M);

        if (track_observables) {
            pt_collect_obs(obs_streams, replicas, t2r, M);
        }
    }

    return PTResult{
        n_accepts, n_attempts, n_up, n_down,
        round_trip_count, std::move(obs_streams)
    };
}

// ── 3.2.3  PT2DResult — result type for 2D parameter-space PT ─────
// Separate from PTResult because the 2D grid has no single "cold→hot"
// axis — 1D label tracking (n_up/n_down/round_trip_count) and per-gap
// acceptance counters (n_accepts/n_attempts) are not meaningful here.
// Future: add per-edge acceptance rates (T-dir and param-dir separately)
// and phase-crossing counts when those diagnostics are implemented.
struct PT2DResult {
    ObsStreams obs_streams;
    // T-direction: t_accepts[j*(n_T-1) + i] = edge (i,j)↔(i+1,j)
    std::vector<int> t_accepts;
    std::vector<int> t_attempts;
    // Param-direction: p_accepts[i*(n_P-1) + j] = edge (i,j)↔(i,j+1)
    std::vector<int> p_accepts;
    std::vector<int> p_attempts;
};

// ── 3.2.4  pt_rounds_2d — 2D parameter-space PT composition loop ──
// Same structure as pt_rounds but on an n_T × n_P 2D grid.
//
// Grid layout: slot(i,j) = i*n_P + j  (row = T index, col = param index)
// Total replicas M = n_T * n_P.
//
// Each round:
//   1. Sweep all M replicas at their current (T, param)  [OpenMP parallel]
//   2. T-direction exchanges (within each param column)
//   3. Param-direction exchanges (within each T row)
//   4. Observable collection (if tracking)
//
// Model must provide set_param(double) and dE_dparam() in addition to
// the standard set_temperature / sweep / energy / observables interface.
template <typename Model>
PT2DResult pt_rounds_2d(
    std::vector<Model*>& replicas,
    const std::vector<double>& temps,   // length n_T
    const std::vector<double>& params,  // length n_P
    std::vector<int>& r2s,              // replica → slot (flat)
    std::vector<int>& s2r,              // slot → replica
    int n_rounds,
    Rng& rng,
    bool track_observables)
{
    int n_T = static_cast<int>(temps.size());
    int n_P = static_cast<int>(params.size());
    int M = n_T * n_P;

    ObsStreams obs_streams;
    if (track_observables) {
        auto obs = replicas[0]->observables();
        for (auto& [name, _] : obs) {
            obs_streams[name].resize(M);
        }
    }

    // Per-edge acceptance counters.
    // T-direction: index j*(n_T-1) + i  = edge (i,j)↔(i+1,j)
    // P-direction: index i*(n_P-1) + j  = edge (i,j)↔(i,j+1)
    std::vector<int> t_accepts(n_P * (n_T - 1), 0);
    std::vector<int> t_attempts(n_P * (n_T - 1), 0);
    std::vector<int> p_accepts(n_T * (n_P - 1), 0);
    std::vector<int> p_attempts(n_T * (n_P - 1), 0);

    for (int round = 0; round < n_rounds; ++round) {
        // Sweep each replica at its current (T, param).
        #pragma omp parallel for schedule(static) if(M >= 8)
        for (int r = 0; r < M; ++r) {
            int s = r2s[r];
            int i_t = s / n_P;
            int j_p = s % n_P;
            replicas[r]->set_temperature(temps[i_t]);
            replicas[r]->set_param(params[j_p]);
            replicas[r]->sweep(1);
        }

        // T-direction exchanges (within each param column)
        for (int j = 0; j < n_P; ++j) {
            for (int i = 0; i < n_T - 1; ++i) {
                int edge = j * (n_T - 1) + i;
                int s_lo = i * n_P + j;
                int s_hi = (i + 1) * n_P + j;
                int r_lo = s2r[s_lo];
                int r_hi = s2r[s_hi];
                double E_lo = replicas[r_lo]->energy();
                double E_hi = replicas[r_hi]->energy();
                ++t_attempts[edge];
                if (pt_exchange(E_lo, E_hi, temps[i], temps[i + 1], rng)) {
                    r2s[r_lo] = s_hi;
                    r2s[r_hi] = s_lo;
                    s2r[s_lo] = r_hi;
                    s2r[s_hi] = r_lo;
                    ++t_accepts[edge];
                }
            }
        }

        // Param-direction exchanges (within each T row)
        for (int i = 0; i < n_T; ++i) {
            for (int j = 0; j < n_P - 1; ++j) {
                int edge = i * (n_P - 1) + j;
                int s_lo = i * n_P + j;
                int s_hi = i * n_P + j + 1;
                int r_lo = s2r[s_lo];
                int r_hi = s2r[s_hi];
                double dEdp_lo = replicas[r_lo]->dE_dparam();
                double dEdp_hi = replicas[r_hi]->dE_dparam();
                ++p_attempts[edge];
                if (pt_exchange_param(dEdp_lo, dEdp_hi, temps[i],
                                      params[j], params[j + 1], rng)) {
                    r2s[r_lo] = s_hi;
                    r2s[r_hi] = s_lo;
                    s2r[s_lo] = r_hi;
                    s2r[s_hi] = r_lo;
                    ++p_accepts[edge];
                }
            }
        }

        // After exchanges, sync each replica's param to its current slot.
        // Param-direction exchanges move replicas between D/U slots without
        // updating their internal D_ or U_.  Energy depends on the param
        // (BC: cached_energy_ includes D*sq_sum; AT: energy() uses U_),
        // so we must set_param() before reading observables.
        for (int r = 0; r < M; ++r) {
            int j_p = r2s[r] % n_P;
            replicas[r]->set_param(params[j_p]);
        }

        if (track_observables) {
            pt_collect_obs(obs_streams, replicas, s2r, M);
        }
    }

    return PT2DResult{
        std::move(obs_streams),
        std::move(t_accepts), std::move(t_attempts),
        std::move(p_accepts), std::move(p_attempts)
    };
}

}  // namespace pbc
