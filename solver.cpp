#include "matchup_probabilities.hpp"
#include "precomputed.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace swiss_precomp;

struct Outcome {
    uint16_t A = 0;
    uint16_t B = 0;
    uint16_t C = 0;
    uint32_t status_bits = 0; // two bits per team: (3-0 / 0-3 / middle / other)
    double p = 0.0;
};

struct PassList {
    std::vector<uint32_t> offsets; // size 6 * 4096 + 1. key = required * 4096 + qual_mask, required in [0,5]
    std::vector<uint16_t> ids;     // local6 ids satisfying popcount(mid_mask & qual_mask) >= required
};

static inline int popcnt(uint32_t x) { return __builtin_popcount(x); }

// Packs each team's result (3-0 / 0-3 / middle / other) into two status bits per team.
static uint32_t make_status_bits(uint16_t A, uint16_t B, uint16_t C) {
    uint32_t s = 0;
    for (int t = 0; t < 16; ++t) {
        uint32_t bit = 1u << t;
        uint32_t code = 3;
        if (A & bit) code = 0;
        else if (B & bit) code = 1;
        else if (C & bit) code = 2;
        s |= code << (2 * t);
    }
    return s;
}

static inline uint8_t team_status(uint32_t status_bits, int team) {
    return uint8_t((status_bits >> (2 * team)) & 3u);
}

// Extracts the four excluded teams' result roles for this excluded set into one packed value.
static inline uint8_t extr_roles_for_eid(int excl_id, uint32_t status_bits) {
    uint8_t extr_roles = 0;
    for (int j = 0; j < 4; ++j) {
        int team = EXCL_TEAM_POS[excl_id * 4 + j];
        extr_roles |= uint8_t(team_status(status_bits, team) << (2 * j));
    }
    return extr_roles;
}

static std::vector<Outcome> read_outcomes_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("could not open input file: " + path);

    std::vector<Outcome> out;
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (first) {
            first = false;
            if (line.find("A3_0") != std::string::npos) continue;
        }
        std::stringstream ss(line);
        std::string a, b, c, p;
        if (!std::getline(ss, a, ',')) continue;
        if (!std::getline(ss, b, ',')) continue;
        if (!std::getline(ss, c, ',')) continue;
        if (!std::getline(ss, p, ',')) continue;

        Outcome o;
        o.A = uint16_t(std::stoul(a));
        o.B = uint16_t(std::stoul(b));
        o.C = uint16_t(std::stoul(c));
        o.p = std::stod(p);
        o.status_bits = make_status_bits(o.A, o.B, o.C);
        out.push_back(o);
    }
    return out;
}

// Precomputes, per (required hits, qualifier mask), the middle choices that have 5 or more correct
static PassList build_pass_lists() {
    PassList pl;
    const int keys = 6 * NUM_QUAL_MASKS;
    pl.offsets.assign(keys + 1, 0);

    std::vector<std::vector<uint16_t>> tmp(keys);
    for (int required = 0; required <= 5; ++required) {
        for (int qual_mask = 0; qual_mask < NUM_QUAL_MASKS; ++qual_mask) {
            int key = required * NUM_QUAL_MASKS + qual_mask;
            for (int mid_id = 0; mid_id < NUM_MID_CHOICES; ++mid_id) {
                uint16_t mid_mask = MID_MASKS[mid_id];
                if (required <= 0 || popcnt(uint32_t(mid_mask & qual_mask)) >= required) {
                    tmp[key].push_back(uint16_t(mid_id));
                }
            }
        }
    }

    uint32_t total = 0;
    for (int key_idx = 0; key_idx < keys; ++key_idx) {
        pl.offsets[key_idx] = total;
        total += uint32_t(tmp[key_idx].size());
    }
    pl.offsets[keys] = total;
    pl.ids.reserve(total);
    for (int key_idx = 0; key_idx < keys; ++key_idx) {
        pl.ids.insert(pl.ids.end(), tmp[key_idx].begin(), tmp[key_idx].end());
    }
    return pl;
}

static std::string mask_to_string(uint16_t m) {
    std::ostringstream os;
    os << "{";
    bool first = true;
    for (int i = 0; i < 16; ++i) {
        if (m & (uint16_t(1) << i)) {
            if (!first) os << ' ';
            first = false;
            os << i;
        }
    }
    os << "}";
    return os.str();
}

struct Result {
    double value = -std::numeric_limits<double>::infinity();
    uint16_t A = 0;
    uint16_t B = 0;
    uint16_t C = 0;
};

// Fixed-capacity descending list of the best results seen so far.
struct TopResults {
    std::vector<Result> items; // sorted by value, descending
    int capacity = 1;
    int count = 0;

    void init(int n) {
        capacity = n < 1 ? 1 : n;
        items.assign(capacity, Result{});
        count = 0;
    }

    // Insert value at its sorted position, dropping the current worst when full.
    inline void consider_result(double value, uint16_t A, uint16_t B, uint16_t C) {
        if (count == capacity && value <= items[count - 1].value) return;
        int i = (count < capacity) ? count++ : count - 1;
        for (; i > 0 && items[i - 1].value < value; --i) items[i] = items[i - 1];
        items[i] = Result{value, A, B, C};
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: ./solver outcomes.csv [--all output.csv] [--top N]\n";
        return 2;
    }
    const std::string input_path = argv[1];
    bool write_all = false;
    std::string all_path;
    int top_n = 1;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--all" && i + 1 < argc) {
            write_all = true;
            all_path = argv[++i];
        } else if (arg == "--top" && i + 1 < argc) {
            top_n = std::atoi(argv[++i]);
            if (top_n < 1) top_n = 1;
        }
    }

    auto outcomes = read_outcomes_csv(input_path);
    std::cerr << "loaded outcomes: " << outcomes.size() << "\n";

    auto pass_lists = build_pass_lists();
    std::cerr << "built pass lists, entries: " << pass_lists.ids.size() << "\n";

    std::ofstream all_file;
    if (write_all) {
        all_file.open(all_path);
        if (!all_file) {
            std::cerr << "could not open output file: " << all_path << "\n";
            return 1;
        }
        all_file << "A_pick,B_pick,C_pick,pass_probability\n";
        all_file << std::setprecision(17);
    }

    // Probability mass keyed by (excluded-set roles, middle qualifier mask).
    std::vector<double> excl_buckets(NUM_EXTR_ROLES * NUM_QUAL_MASKS, 0.0);
    std::vector<int> touched_excl_buckets;
    touched_excl_buckets.reserve(200000);

    // Probability mass keyed by (extreme score, middle qualifier mask).
    std::vector<double> extr_buckets(5 * NUM_QUAL_MASKS, 0.0);
    std::vector<int> touched_extr_buckets;
    touched_extr_buckets.reserve(50000);

    std::array<double, NUM_MID_CHOICES> score{};
    TopResults top;
    top.init(top_n);

    // Local compression table for current excluded set: maps a 16-bit global mask to a 12-bit R-local mask.
    std::vector<uint16_t> local_id(1u << 16);

    for (int excl_id = 0; excl_id < NUM_EXCL_SETS; ++excl_id) {
        const uint16_t excl_set = EXCL_SET_MASKS[excl_id]; // 16-bit mask of the four excluded teams

        // Build global -> local projection for this excluded set once.
        for (uint32_t gm = 0; gm < (1u << 16); ++gm) {
            local_id[gm] = compress_global_to_mid(excl_id, uint16_t(gm));
        }

        touched_excl_buckets.clear();
        for (const Outcome& o : outcomes) {
            uint16_t qual_mask = local_id[o.C]; // local 12-bit mask of middle teams that finished 3-1/3-2
            uint8_t extr_roles = extr_roles_for_eid(excl_id, o.status_bits);
            int excl_idx = int(extr_roles) * NUM_QUAL_MASKS + int(qual_mask);
            if (excl_buckets[excl_idx] == 0.0) touched_excl_buckets.push_back(excl_idx);
            excl_buckets[excl_idx] += o.p;
        }

        for (int role = 0; role < NUM_ROLES; ++role) {
            std::fill(score.begin(), score.end(), 0.0);
            touched_extr_buckets.clear();

            // excl_buckets[extr_roles, qual_mask] -> extr_buckets[extr_score, qual_mask]
            for (int excl_idx : touched_excl_buckets) {
                double weight = excl_buckets[excl_idx];
                if (weight == 0.0) continue;
                int extr_roles = excl_idx / NUM_QUAL_MASKS;
                int qual_mask = excl_idx & (NUM_QUAL_MASKS - 1);
                int extr_score = extr_score_from_roles(excl_id, role, extr_roles); // picked extreme teams that hit their predicted result
                int extr_idx = extr_score * NUM_QUAL_MASKS + qual_mask;
                if (extr_buckets[extr_idx] == 0.0) touched_extr_buckets.push_back(extr_idx);
                extr_buckets[extr_idx] += weight;
            }

            // extr_buckets[extr_score, qual_mask] -> scores for all 924 middle choices.
            for (int extr_idx : touched_extr_buckets) {
                double weight = extr_buckets[extr_idx];
                if (weight == 0.0) continue;
                int extr_score = extr_idx / NUM_QUAL_MASKS;
                int qual_mask = extr_idx & (NUM_QUAL_MASKS - 1);
                int required = 5 - extr_score;
                if (required < 0) required = 0;
                int key = required * NUM_QUAL_MASKS + qual_mask;
                uint32_t begin = pass_lists.offsets[key];
                uint32_t end = pass_lists.offsets[key + 1];
                for (uint32_t ptr = begin; ptr < end; ++ptr) {
                    score[pass_lists.ids[ptr]] += weight;
                }
            }

            uint16_t A_pick = ROLE_A_PICK[excl_id * NUM_ROLES + role];
            uint16_t B_pick = excl_set ^ A_pick;

            // Index over the 924 middle 6-of-12 choices.
            for (int mid_id = 0; mid_id < NUM_MID_CHOICES; ++mid_id) {
                double value = score[mid_id];
                uint16_t C_pick = expand_mid_to_global(excl_id, MID_MASKS[mid_id]);

                if (write_all) {
                    all_file << A_pick << ',' << B_pick << ',' << C_pick << ',' << value << '\n';
                }
                top.consider_result(value, A_pick, B_pick, C_pick);
            }

            for (int extr_idx : touched_extr_buckets) extr_buckets[extr_idx] = 0.0;
        }

        for (int excl_idx : touched_excl_buckets) excl_buckets[excl_idx] = 0.0;

        if ((excl_id + 1) % 100 == 0) {
            std::cerr << "processed E " << (excl_id + 1) << "/" << NUM_EXCL_SETS
                      << ", current best=" << std::setprecision(12) << top.items[0].value << "\n";
        }
    }

    std::cout << std::setprecision(17);
    if (top_n == 1) {
        const Result& b = top.items[0];
        std::cout << "best_pass_probability=" << b.value << "\n";
        std::cout << "A_3_0: " << mask_to_team_names(b.A) << "\n";
        std::cout << "B_0_3: " << mask_to_team_names(b.B) << "\n";
        std::cout << "C_3_1_3_2: " << mask_to_team_names(b.C) << "\n";
    } else {
        std::cout << "top " << top.count << " pass probabilities:\n";
        for (int r = 0; r < top.count; ++r) {
            const Result& b = top.items[r];
            std::cout << "rank " << (r + 1) << ": pass_probability=" << b.value << "\n"
                      << "  A_3_0: " << mask_to_team_names(b.A) << "\n"
                      << "  B_0_3: " << mask_to_team_names(b.B) << "\n"
                      << "  C_3_1_3_2: " << mask_to_team_names(b.C) << "\n";
        }
    }
    return 0;
}
