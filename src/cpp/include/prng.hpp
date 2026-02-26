#pragma once
// PRNG wrapper — thin interface over Xoshiro256++ for Monte Carlo use.

#include "xoshiro256pp.hpp"

#include <cstdint>

namespace pbc {

class Rng {
public:
    explicit Rng(uint64_t seed) : engine_(seed) {}

    // Uniform double in [0, 1).
    // Uses the upper 53 bits of a 64-bit output (doubles have 53
    // bits of mantissa), then divides by 2^53 so the result is
    // always strictly less than 1.0.
    double uniform() {
        return static_cast<double>(engine_.next() >> 11) * (1.0 / (uint64_t{1} << 53));
    }

    // Uniform integer in [0, n).
    // Uses simple modulo reduction.  The bias is negligible when n
    // is tiny compared to 2^64 (true for all our lattice sizes).
    uint64_t rand_below(uint64_t n) { return engine_.next() % n; }

    // Advance the state by 2^128 steps (for parallel streams).
    void jump() { engine_.jump(); }

private:
    Xoshiro256pp engine_;
};

}  // namespace pbc
