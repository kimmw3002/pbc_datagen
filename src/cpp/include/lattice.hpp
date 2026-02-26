#pragma once
// Flat 1D-backed 2D lattice with precomputed PBC neighbor table.
//
// Site (r, c) on an L×L grid is stored at index  r * L + c.
// The neighbor table stores 4 neighbors per site:
//   [north, south, east, west]
// with periodic boundary conditions (PBC) so edges wrap around.

#include <cstdint>
#include <vector>

namespace pbc {

// Build the PBC neighbor table for an L×L square lattice.
// Returns a flat vector of length L*L*4.  For site i, the four
// neighbors are at offsets [i*4+0 .. i*4+3] = {N, S, E, W}.
inline std::vector<int32_t> make_neighbor_table(int L) {
    const int N = L * L;
    std::vector<int32_t> table(static_cast<size_t>(N) * 4);

    for (int i = 0; i < N; ++i) {
        int r = i / L;
        int c = i % L;
        table[i * 4 + 0] = ((r - 1 + L) % L) * L + c;   // north
        table[i * 4 + 1] = ((r + 1) % L) * L + c;        // south
        table[i * 4 + 2] = r * L + (c + 1) % L;           // east
        table[i * 4 + 3] = r * L + (c - 1 + L) % L;       // west
    }
    return table;
}

}  // namespace pbc
