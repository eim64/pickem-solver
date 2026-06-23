#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static int popcnt(uint32_t x) { return __builtin_popcount(x); }

static std::vector<uint16_t> masks_of_size(int n, int k) {
    std::vector<uint16_t> out;
    const uint32_t lim = 1u << n;
    for (uint32_t m = 0; m < lim; ++m) {
        if (popcnt(m) == k) out.push_back(static_cast<uint16_t>(m));
    }
    return out;
}

static std::vector<int> bit_positions(uint16_t m) {
    std::vector<int> p;
    for (int i = 0; i < 16; ++i) if (m & (1u << i)) p.push_back(i);
    return p;
}

static uint8_t extreme_score_from_extr_roles(uint16_t E, uint16_t A_pick, uint16_t B_pick, int extr_roles) {
    int k = 0;
    int shift = 0;
    for (int t = 0; t < 16; ++t) {
        uint16_t bit = uint16_t(1u << t);
        if (!(E & bit)) continue;
        int status = (extr_roles >> shift) & 3; // 0=A_out, 1=B_out, 2=C_out, 3=other
        if ((A_pick & bit) && status == 0) ++k;
        if ((B_pick & bit) && status == 1) ++k;
        shift += 2;
    }
    return static_cast<uint8_t>(k);
}

template <class T>
static void write_flat_array(std::ofstream& f, const std::string& type, const std::string& name,
                             const std::vector<T>& v, int per_line = 16) {
    f << "inline constexpr std::array<" << type << ", " << v.size() << "> " << name << " = {\n";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i % per_line == 0) f << "    ";
        f << +v[i];
        if (i + 1 != v.size()) f << ",";
        if (i % per_line == per_line - 1 || i + 1 == v.size()) f << "\n";
        else f << " ";
    }
    f << "};\n\n";
}

int main(int argc, char** argv) {
    std::string out_path = argc >= 2 ? argv[1] : "precomputed.hpp";
    const uint16_t ALL = 0xFFFFu;

    auto masks2 = masks_of_size(16, 2);
    auto masks4 = masks_of_size(16, 4);
    auto local6 = masks_of_size(12, 6);

    std::vector<uint16_t> e_masks = masks4;
    std::vector<uint8_t> e_team_pos;      // [eid * 4 + j]
    std::vector<uint8_t> e_free_pos;      // [eid * 12 + j]
    std::vector<uint16_t> e_role_a;       // [eid * 6 + role]
    std::vector<uint8_t> extr_score_from_roles; // [(eid * 6 + role) * 256 + extr_roles]

    e_team_pos.reserve(e_masks.size() * 4);
    e_free_pos.reserve(e_masks.size() * 12);
    e_role_a.reserve(e_masks.size() * 6);
    extr_score_from_roles.reserve(e_masks.size() * 6 * 256);

    for (uint16_t E : e_masks) {
        auto ep = bit_positions(E);
        for (int x : ep) e_team_pos.push_back(static_cast<uint8_t>(x));

        uint16_t R = ALL ^ E;
        auto rp = bit_positions(R);
        for (int x : rp) e_free_pos.push_back(static_cast<uint8_t>(x));

        std::vector<uint16_t> roles;
        for (uint16_t A : masks2) {
            if ((A & ~E) == 0) roles.push_back(A);
        }
        if (roles.size() != 6) {
            std::cerr << "internal error: bad role count\n";
            return 1;
        }
        for (uint16_t A : roles) e_role_a.push_back(A);

        for (uint16_t A : roles) {
            uint16_t B = E ^ A;
            for (int extr_roles = 0; extr_roles < 256; ++extr_roles) {
                extr_score_from_roles.push_back(extreme_score_from_extr_roles(E, A, B, extr_roles));
            }
        }
    }

    // ------------------------------------------------------------------
    // Swiss-bracket pairing precompute (used by the simulator).
    // ------------------------------------------------------------------

    // Priority table for the 6-team groups (rounds 4-5), rulebook priority
    // rows. Positions are seed ranks within the group, 0 = highest seed.
    static const int ROWS[15][3][2] = {
        {{0, 5}, {1, 4}, {2, 3}}, {{0, 5}, {1, 3}, {2, 4}}, {{0, 4}, {1, 5}, {2, 3}},
        {{0, 4}, {1, 3}, {2, 5}}, {{0, 3}, {1, 5}, {2, 4}}, {{0, 3}, {1, 4}, {2, 5}},
        {{0, 5}, {1, 2}, {3, 4}}, {{0, 4}, {1, 2}, {3, 5}}, {{0, 2}, {1, 5}, {3, 4}},
        {{0, 2}, {1, 4}, {3, 5}}, {{0, 3}, {1, 2}, {4, 5}}, {{0, 2}, {1, 3}, {4, 5}},
        {{0, 1}, {2, 5}, {3, 4}}, {{0, 1}, {2, 4}, {3, 5}}, {{0, 1}, {2, 3}, {4, 5}},
    };

    // PAIR_BIT[a*6+b] = bit index 0..14 of the unordered position pair {a,b}.
    std::vector<uint8_t> pair_bit(36, 0xFF);
    {
        int idx = 0;
        for (int a = 0; a < 6; ++a)
            for (int b = a + 1; b < 6; ++b) {
                pair_bit[a * 6 + b] = uint8_t(idx);
                pair_bit[b * 6 + a] = uint8_t(idx);
                ++idx;
            }
    }

    // ROW_PARTNER[r*6+pos] = partner position of pos in priority row r.
    // ROW_MASK[r] = OR of the three position-pair bits used by row r.
    std::vector<uint8_t> row_partner(15 * 6, 0);
    std::vector<uint16_t> row_mask(15, 0);
    for (int r = 0; r < 15; ++r) {
        uint16_t m = 0;
        for (int k = 0; k < 3; ++k) {
            int a = ROWS[r][k][0], b = ROWS[r][k][1];
            row_partner[r * 6 + a] = uint8_t(b);
            row_partner[r * 6 + b] = uint8_t(a);
            m |= uint16_t(1u << pair_bit[a * 6 + b]);
        }
        row_mask[r] = m;
    }

    // Note: the larger dense LUTs (PRIORITY_LUT[1<<15], EID_FROM_E[1<<16]) are
    // intentionally NOT emitted. The simulator computes the priority row from
    // the tiny ROW_MASK table and the E-set index from a colex rank, which keeps
    // those derivations out of the cache working set. Only MID_INDEX is kept
    // as a LUT because it is read once per phase-2 leaf (the hottest path).

    // MID_INDEX: inverse of MID_MASKS (12-bit popcount-6 mask -> 0..923).
    std::vector<uint16_t> local6_index(1u << 12, 0xFFFF);
    for (size_t i = 0; i < local6.size(); ++i) local6_index[local6[i]] = uint16_t(i);

    // R1_PAIRS: fixed initial matchups, team i (seed i+1) vs team i+8.
    std::vector<uint8_t> r1_pairs;
    for (int i = 0; i < 8; ++i) {
        r1_pairs.push_back(uint8_t(i));
        r1_pairs.push_back(uint8_t(i + 8));
    }

    std::ofstream f(out_path);
    if (!f) {
        std::cerr << "could not open " << out_path << "\n";
        return 1;
    }

    f << "#pragma once\n";
    f << "#include <array>\n";
    f << "#include <cstdint>\n\n";
    f << "namespace swiss_precomp {\n";
    f << "inline constexpr int NUM_TEAMS = 16;\n";
    f << "inline constexpr uint16_t ALL_MASK = 0xFFFFu;\n";
    f << "inline constexpr int NUM_EXCL_SETS = 1820;\n";
    f << "inline constexpr int NUM_ROLES = 6;\n";
    f << "inline constexpr int NUM_EXTR_ROLES = 256;\n";
    f << "inline constexpr int NUM_QUAL_MASKS = 4096;\n";
    f << "inline constexpr int NUM_MID_CHOICES = 924;\n\n";

    write_flat_array<uint16_t>(f, "uint16_t", "EXCL_SET_MASKS", e_masks, 12);
    write_flat_array<uint16_t>(f, "uint16_t", "MID_MASKS", local6, 12);
    write_flat_array<uint8_t>(f, "uint8_t", "EXCL_TEAM_POS", e_team_pos, 24);
    write_flat_array<uint8_t>(f, "uint8_t", "MID_TEAM_POS", e_free_pos, 24);
    write_flat_array<uint16_t>(f, "uint16_t", "ROLE_A_PICK", e_role_a, 12);
    write_flat_array<uint8_t>(f, "uint8_t", "EXTR_SCORE_FROM_ROLES", extr_score_from_roles, 32);

    write_flat_array<uint8_t>(f, "uint8_t", "PAIR_BIT", pair_bit, 6);
    write_flat_array<uint8_t>(f, "uint8_t", "ROW_PARTNER", row_partner, 6);
    write_flat_array<uint16_t>(f, "uint16_t", "ROW_MASK", row_mask, 15);
    write_flat_array<uint16_t>(f, "uint16_t", "MID_INDEX", local6_index, 16);
    write_flat_array<uint8_t>(f, "uint8_t", "R1_PAIRS", r1_pairs, 16);

    f << "inline uint16_t expand_mid_to_global(int excl_id, uint16_t mid) {\n";
    f << "    uint16_t out = 0;\n";
    f << "    for (int i = 0; i < 12; ++i) {\n";
    f << "        if (mid & (uint16_t(1) << i)) out |= uint16_t(1) << MID_TEAM_POS[excl_id * 12 + i];\n";
    f << "    }\n";
    f << "    return out;\n";
    f << "}\n\n";

    f << "inline uint16_t compress_global_to_mid(int excl_id, uint16_t global) {\n";
    f << "    uint16_t out = 0;\n";
    f << "    for (int i = 0; i < 12; ++i) {\n";
    f << "        if (global & (uint16_t(1) << MID_TEAM_POS[excl_id * 12 + i])) out |= uint16_t(1) << i;\n";
    f << "    }\n";
    f << "    return out;\n";
    f << "}\n\n";

    f << "inline uint8_t extr_score_from_roles(int excl_id, int role, int extr_roles) {\n";
    f << "    return EXTR_SCORE_FROM_ROLES[(excl_id * NUM_ROLES + role) * NUM_EXTR_ROLES + extr_roles];\n";
    f << "}\n\n";
    f << "} // namespace swiss_precomp\n";

    std::cerr << "wrote " << out_path << "\n";
    return 0;
}
