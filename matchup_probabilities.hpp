#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Team names, indexed 0..15 (team i has initial seed i+1). Edit these to match
// the real field; masks in the CSV use bit i for this team.
// ---------------------------------------------------------------------------

inline const std::vector<std::string> TEAM_NAMES = {
    "team 1",  "team 2",  "team 3",  "team 4",  "team 5",  "team 6",
    "team 7",  "team 8",  "team 9",  "team 10", "team 11", "team 12",
    "team 13", "team 14", "team 15", "team 16",
};

// Render a 16-bit team mask as a comma-separated list of team names.
inline std::string mask_to_team_names(uint16_t mask) {
    std::string out;
    bool first = true;
    for (int i = 0; i < 16; ++i) {
        if (mask & (uint16_t(1) << i)) {
            if (!first) out += ", ";
            first = false;
            out += TEAM_NAMES[i];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Matchup probabilities.
//
// Two independent 16x16 matrices: P_bo1[i][j] = P(seed i+1 beats seed j+1 in a
// best-of-one) and P_bo3[i][j] = P(seed i+1 wins a best-of-three series). Teams
// are indexed 0..15 (initial seed = index + 1).
//
// The values below are a placeholder strength model. Replace build_probabilities()
// (or assign the matrices directly) with measured numbers as needed. The only
// requirement for the outcome tree to sum to 1 is P[i][j] + P[j][i] = 1.
// ---------------------------------------------------------------------------

inline double P_bo1[16][16];
inline double P_bo3[16][16];

inline void build_probabilities() {
    double s[16];
    for (int t = 0; t < 16; ++t) s[t] = double(16 - t); // seed 1 strongest
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            if (i == j) { P_bo1[i][j] = P_bo3[i][j] = 0.0; continue; }
            P_bo1[i][j] = s[i] / (s[i] + s[j]);
            // Independent, sharper curve for series play.
            P_bo3[i][j] = (s[i] * s[i]) / (s[i] * s[i] + s[j] * s[j]);
        }
    }
}
