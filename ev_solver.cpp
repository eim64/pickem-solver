#include "matchup_probabilities.hpp"
#include "precomputed.hpp"

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
    double p = 0.0;
};

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

static inline int popcnt(uint32_t x) { return __builtin_popcount(x); }

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
        out.push_back(o);
    }
    return out;
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

static uint16_t best_middle_top6(uint16_t mid_pool, const std::array<double, 16>& prob_mid) {
    // Pick the six teams inside mid_pool with largest prob_mid. Ties are resolved by lower team id.
    std::array<int, 16> teams{};
    int n = 0;
    for (int t = 0; t < 16; ++t) {
        if (mid_pool & (uint16_t(1) << t)) teams[n++] = t;
    }

    for (int i = 0; i < n; ++i) {
        int best = i;
        for (int j = i + 1; j < n; ++j) {
            int tj = teams[j];
            int tb = teams[best];
            if (prob_mid[tj] > prob_mid[tb] || (prob_mid[tj] == prob_mid[tb] && tj < tb)) {
                best = j;
            }
        }
        std::swap(teams[i], teams[best]);
    }

    uint16_t mid_mask = 0;
    for (int i = 0; i < 6; ++i) mid_mask |= uint16_t(1) << teams[i];
    return mid_mask;
}

static double sum_mask_values(uint16_t mask, const std::array<double, 16>& values) {
    double s = 0.0;
    for (int t = 0; t < 16; ++t) {
        if (mask & (uint16_t(1) << t)) s += values[t];
    }
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: ./ev_solver outcomes.csv [--all output.csv] [--top N]\n";
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

    std::array<double, 16> prob_3_0{}; // P(team is actual 3-0)
    std::array<double, 16> prob_0_3{}; // P(team is actual 0-3)
    std::array<double, 16> prob_mid{}; // P(team is actual 3-1 / 3-2)

    double total_prob = 0.0;
    for (const Outcome& o : outcomes) {
        total_prob += o.p;
        for (int t = 0; t < 16; ++t) {
            uint16_t bit = uint16_t(1) << t;
            if (o.A & bit) prob_3_0[t] += o.p;
            if (o.B & bit) prob_0_3[t] += o.p;
            if (o.C & bit) prob_mid[t] += o.p;
        }
    }

    std::cerr << std::setprecision(17) << "total probability: " << total_prob << "\n";

    std::ofstream all_file;
    if (write_all) {
        all_file.open(all_path);
        if (!all_file) {
            std::cerr << "could not open output file: " << all_path << "\n";
            return 1;
        }
        all_file << "A_pick,B_pick,C_pick,expected_correct\n";
        all_file << std::setprecision(17);
    }

    TopResults top;
    top.init(top_n);

    // The greedy single-C shortcut only finds the single best result, so any
    // request for more than one result must enumerate all middle choices.
    const bool enumerate_all = write_all || top_n > 1;

    // Enumerate by excluded set and role split, matching the pass-probability solver.
    // For max EV, the best middle pick for fixed A/B is simply the top six prob_mid in mid_pool.
    // Index over all excluded 4-sets.
    for (int excl_id = 0; excl_id < NUM_EXCL_SETS; ++excl_id) {
        uint16_t excl_set = EXCL_SET_MASKS[excl_id];   // 16-bit mask of the four excluded teams
        uint16_t mid_pool = ALL_MASK ^ excl_set; // 12-team remaining pool outside the excluded set

        for (int role = 0; role < NUM_ROLES; ++role) {
            uint16_t A_pick = ROLE_A_PICK[excl_id * NUM_ROLES + role];
            uint16_t B_pick = excl_set ^ A_pick;
            double base = sum_mask_values(A_pick, prob_3_0) + sum_mask_values(B_pick, prob_0_3);

            if (enumerate_all) {
                // Index over the 924 middle 6-of-12 choices.
                for (int mid_id = 0; mid_id < NUM_MID_CHOICES; ++mid_id) {
                    uint16_t C_pick = expand_mid_to_global(excl_id, MID_MASKS[mid_id]);
                    double ev = base + sum_mask_values(C_pick, prob_mid);
                    if (write_all) {
                        all_file << A_pick << ',' << B_pick << ',' << C_pick << ',' << ev << '\n';
                    }
                    top.consider_result(ev, A_pick, B_pick, C_pick);
                }
            } else {
                uint16_t C_pick = best_middle_top6(mid_pool, prob_mid);
                double ev = base + sum_mask_values(C_pick, prob_mid);
                top.consider_result(ev, A_pick, B_pick, C_pick);
            }
        }
    }

    std::cout << std::setprecision(17);
    if (top_n == 1) {
        const Result& b = top.items[0];
        std::cout << "best_expected_correct=" << b.value << "\n";
        std::cout << "A_3_0: " << mask_to_team_names(b.A) << "\n";
        std::cout << "B_0_3: " << mask_to_team_names(b.B) << "\n";
        std::cout << "C_3_1_3_2: " << mask_to_team_names(b.C) << "\n";
    } else {
        std::cout << "top " << top.count << " expected_correct values:\n";
        for (int r = 0; r < top.count; ++r) {
            const Result& b = top.items[r];
            std::cout << "rank " << (r + 1) << ": expected_correct=" << b.value << "\n"
                      << "  A_3_0: " << mask_to_team_names(b.A) << "\n"
                      << "  B_0_3: " << mask_to_team_names(b.B) << "\n"
                      << "  C_3_1_3_2: " << mask_to_team_names(b.C) << "\n";
        }
    }

    std::cout << "\nteam_marginals:\n";
    std::cout << "team,alpha_3_0,beta_0_3,gamma_mid\n";
    for (int t = 0; t < 16; ++t) {
        std::cout << t << ',' << prob_3_0[t] << ',' << prob_0_3[t] << ',' << prob_mid[t] << "\n";
    }

    return 0;
}
