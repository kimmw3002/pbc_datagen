#pragma once
// Xoshiro256++ — high-quality, fast PRNG by Blackman & Vigna (2018).
// Reference: https://prng.di.unimi.it/xoshiro256plusplus.c
// License: public domain (CC0).
//
// Vendored here as a header-only file.  This is the canonical
// implementation, NOT hand-rolled.

#include <array>
#include <cstdint>

namespace pbc {

class Xoshiro256pp {
public:
    // Seed the 256-bit state from a single 64-bit integer
    // using splitmix64 to spread entropy across all four words.
    explicit Xoshiro256pp(uint64_t seed) {
        uint64_t sm = seed;
        state_[0] = splitmix64(sm);
        state_[1] = splitmix64(sm);
        state_[2] = splitmix64(sm);
        state_[3] = splitmix64(sm);
    }

    // Generate the next 64-bit pseudorandom number.
    uint64_t next() {
        const uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
        const uint64_t t = state_[1] << 17;

        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);

        return result;
    }

    // Jump ahead by 2^128 steps.  After calling jump(), the generator
    // behaves as if next() had been called 2^128 times — useful for
    // creating independent streams in parallel simulations.
    void jump() {
        static constexpr uint64_t JUMP[] = {
            0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
            0xa9582618e03fc9aa, 0x39abdc4529b1661c};

        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (auto jmp : JUMP) {
            for (int b = 0; b < 64; ++b) {
                if (jmp & (uint64_t{1} << b)) {
                    s0 ^= state_[0];
                    s1 ^= state_[1];
                    s2 ^= state_[2];
                    s3 ^= state_[3];
                }
                next();
            }
        }
        state_[0] = s0;
        state_[1] = s1;
        state_[2] = s2;
        state_[3] = s3;
    }

private:
    std::array<uint64_t, 4> state_;

    static constexpr uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    // Splitmix64 — used only during seeding to expand a single
    // 64-bit seed into four independent-looking 64-bit words.
    static uint64_t splitmix64(uint64_t& state) {
        uint64_t z = (state += 0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }
};

}  // namespace pbc
