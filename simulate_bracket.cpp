#include "matchup_probabilities.hpp"
#include "precomputed.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace swiss_precomp;

// Binomial coefficients for the colex rank used to map a 4-set E mask to its
// index in EXCL_SET_MASKS without a 128 KB lookup table.
static int BINOM[16][5];

static void build_binom() {
    for (int n = 0; n < 16; ++n) {
        BINOM[n][0] = 1;
        for (int k = 1; k < 5; ++k)
            BINOM[n][k] = (n == 0) ? 0 : BINOM[n - 1][k - 1] + BINOM[n - 1][k];
    }
}

// ---------------------------------------------------------------------------
// Per-path bracket state.
// ---------------------------------------------------------------------------

struct State {
    uint8_t wins[16];
    uint8_t losses[16];
    uint16_t oppmask[16];
};

static inline int difficulty(const State& st, int t) {
    int d = 0;
    uint16_t m = st.oppmask[t];
    while (m) {
        int j = __builtin_ctz(m);
        m &= uint16_t(m - 1);
        d += int(st.wins[j]) - int(st.losses[j]);
    }
    return d;
}

struct Match { uint8_t a, b; };

static inline void apply_match(State& s, double& w, const Match& mt, bool bo3, int bit) {
    const double (*P)[16] = bo3 ? P_bo3 : P_bo1;
    int wn, ls;
    if (!bit) { wn = mt.a; ls = mt.b; w *= P[mt.a][mt.b]; }
    else      { wn = mt.b; ls = mt.a; w *= P[mt.b][mt.a]; }
    s.wins[wn]++;
    s.losses[ls]++;
    s.oppmask[wn] |= uint16_t(1u << ls);
    s.oppmask[ls] |= uint16_t(1u << wn);
}

// Collect teams with record (w,l) and sort by seed order:
// difficulty descending, then initial seed (team index) ascending.
static inline int collect_sorted_group(const State& st, int w, int l, int out[8]) {
    int n = 0;
    int diff[8];
    for (int t = 0; t < 16; ++t) {
        if (st.wins[t] == w && st.losses[t] == l) {
            out[n] = t;
            diff[n] = difficulty(st, t);
            ++n;
        }
    }
    for (int i = 1; i < n; ++i) { // insertion sort, stable on (diff desc, seed asc)
        int t = out[i], d = diff[i], j = i - 1;
        while (j >= 0 && (diff[j] < d || (diff[j] == d && out[j] > t))) {
            out[j + 1] = out[j];
            diff[j + 1] = diff[j];
            --j;
        }
        out[j + 1] = t;
        diff[j + 1] = d;
    }
    return n;
}

// ---------------------------------------------------------------------------
// Pairing.
// ---------------------------------------------------------------------------

// Rounds 2-3 (groups of 8/4): Valve-style highest-vs-lowest with backtracking,
// guaranteeing a rematch-free pairing whenever one exists.
static bool rec_pair(const int g[], int n, const State& st, char used[],
                     Match out[], int& cnt) {
    int h = -1;
    for (int i = 0; i < n; ++i) if (!used[i]) { h = i; break; }
    if (h < 0) return true;
    used[h] = 1;
    for (int c = n - 1; c > h; --c) { // lowest seed first
        if (used[c]) continue;
        if (st.oppmask[g[h]] & (uint16_t(1u << g[c]))) continue; // rematch
        used[c] = 1;
        out[cnt].a = uint8_t(g[h]);
        out[cnt].b = uint8_t(g[c]);
        ++cnt;
        if (rec_pair(g, n, st, used, out, cnt)) return true;
        --cnt;
        used[c] = 0;
    }
    used[h] = 0;
    return false;
}

static inline void pair_group_rounds23(const int g[], int n, const State& st,
                                        Match out[], bool bo3[], bool bo3val,
                                        int& idx) {
    char used[8] = {0};
    Match tmp[4];
    int cnt = 0;
    if (!rec_pair(g, n, st, used, tmp, cnt) || cnt != n / 2) {
        // Forced rematch fallback: highest-vs-lowest ignoring history.
        cnt = 0;
        for (int i = 0; i < n / 2; ++i) {
            tmp[cnt].a = uint8_t(g[i]);
            tmp[cnt].b = uint8_t(g[n - 1 - i]);
            ++cnt;
        }
    }
    for (int i = 0; i < cnt; ++i) {
        out[idx] = tmp[i];
        bo3[idx] = bo3val;
        ++idx;
    }
}

// Rounds 4-5 (groups of 6): priority table. Fills exactly 3 matches.
static inline void pair_group_priority(const int g[6], const State& st, Match out[3]) {
    uint32_t forbidden = 0;
    for (int a = 0; a < 6; ++a)
        for (int b = a + 1; b < 6; ++b)
            if (st.oppmask[g[a]] & (uint16_t(1u << g[b])))
                forbidden |= uint32_t(1u << PAIR_BIT[a * 6 + b]);
    // First priority row with no forbidden pair (scan the tiny ROW_MASK table).
    // Falls back to row 0 if no rematch-free matching exists.
    int row = 0;
    for (int r = 0; r < 15; ++r)
        if ((ROW_MASK[r] & forbidden) == 0) { row = r; break; }
    int k = 0;
    for (int pos = 0; pos < 6; ++pos) {
        int partner = ROW_PARTNER[row * 6 + pos];
        if (pos < partner) {
            out[k].a = uint8_t(g[pos]);
            out[k].b = uint8_t(g[partner]);
            ++k;
        }
    }
}

// ---------------------------------------------------------------------------
// Round-1..3 match construction (exactly 8 matches per round).
// ---------------------------------------------------------------------------

static int build_matches(const State& st, int round, Match out[8], bool bo3[8]) {
    int idx = 0;
    if (round == 1) {
        for (int i = 0; i < 8; ++i) {
            out[i].a = R1_PAIRS[2 * i];
            out[i].b = R1_PAIRS[2 * i + 1];
            bo3[i] = false;
        }
        return 8;
    }
    int g[8];
    if (round == 2) {
        int n = collect_sorted_group(st, 1, 0, g); pair_group_rounds23(g, n, st, out, bo3, false, idx);
        n = collect_sorted_group(st, 0, 1, g);     pair_group_rounds23(g, n, st, out, bo3, false, idx);
        return idx;
    }
    // round == 3: advancement (2-0) and elimination (0-2) groups are best-of-3.
    int n = collect_sorted_group(st, 2, 0, g); pair_group_rounds23(g, n, st, out, bo3, true, idx);
    n = collect_sorted_group(st, 1, 1, g);     pair_group_rounds23(g, n, st, out, bo3, false, idx);
    n = collect_sorted_group(st, 0, 2, g);     pair_group_rounds23(g, n, st, out, bo3, true, idx);
    return idx;
}

// ---------------------------------------------------------------------------
// Accumulation (per thread), with Neumaier/Kahan compensated summation.
// ---------------------------------------------------------------------------

static inline void kadd(double& sum, double& comp, double x) {
    double t = sum + x;
    if (std::fabs(sum) >= std::fabs(x)) comp += (sum - t) + x;
    else                                comp += (x - t) + sum;
    sum = t;
}

struct Ctx {
    double* G;   // size NUM_EXCL_SETS*NUM_ROLES*NUM_MID_CHOICES
    double* Gc;  // companion compensation array
    double tot = 0.0, totc = 0.0; // total probability mass (should reach 1)
};

// ---------------------------------------------------------------------------
// Phase 2: enumerate rounds 4-5 for the 12 middle teams of a fixed prefix.
// ---------------------------------------------------------------------------

static void phase2(const State& post_r3, double prefix_w, int eid, int role, Ctx& ctx) {
    const long base = (long(eid) * NUM_ROLES + role) * NUM_MID_CHOICES;

    int g21[8], g12[8];
    collect_sorted_group(post_r3, 2, 1, g21); // 6 teams (advancement group)
    collect_sorted_group(post_r3, 1, 2, g12); // 6 teams (elimination group)

    Match m21[3], m12[3];
    pair_group_priority(g21, post_r3, m21);
    pair_group_priority(g12, post_r3, m12);

    for (int r4 = 0; r4 < 64; ++r4) {
        State s4 = post_r3;
        double w4 = prefix_w;
        uint16_t c31 = 0; // 3-1 teams (advance via the 2-1 group)
        for (int k = 0; k < 3; ++k) {
            int bit = (r4 >> k) & 1;
            int winner = bit ? m21[k].b : m21[k].a;
            apply_match(s4, w4, m21[k], true, bit);
            c31 |= uint16_t(1u << winner);
        }
        for (int k = 0; k < 3; ++k) {
            int bit = (r4 >> (3 + k)) & 1;
            apply_match(s4, w4, m12[k], true, bit); // loser -> 1-3 (eliminated)
        }

        int g22[8];
        collect_sorted_group(s4, 2, 2, g22); // 6 teams (2-2 group)
        Match m22[3];
        pair_group_priority(g22, s4, m22);

        for (int r5 = 0; r5 < 8; ++r5) {
            double w5 = w4;
            uint16_t C = c31;
            for (int k = 0; k < 3; ++k) {
                int bit = (r5 >> k) & 1;
                int winner = bit ? m22[k].b : m22[k].a;
                w5 *= (bit ? P_bo3[m22[k].b][m22[k].a] : P_bo3[m22[k].a][m22[k].b]);
                C |= uint16_t(1u << winner); // 3-2 team (advances)
            }
            uint16_t local = compress_global_to_mid(eid, C);
            uint16_t cidx = MID_INDEX[local];
            long idx = base + cidx;
            kadd(ctx.G[idx], ctx.Gc[idx], w5);
            kadd(ctx.tot, ctx.totc, w5);
        }
    }
}

// ---------------------------------------------------------------------------
// Prefix handling: classify A/B and dispatch to phase 2.
// ---------------------------------------------------------------------------

static inline void handle_prefix(const State& st, double w, Ctx& ctx) {
    uint16_t A = 0, B = 0;
    for (int t = 0; t < 16; ++t) {
        if (st.wins[t] == 3) A |= uint16_t(1u << t);       // 3-0
        else if (st.losses[t] == 3) B |= uint16_t(1u << t); // 0-3
    }
    uint16_t E = uint16_t(A | B);
    // Colex rank of the 4-set E within EXCL_SET_MASKS (which is in increasing-mask /
    // colex order): eid = C(e0,1) + C(e1,2) + C(e2,3) + C(e3,4).
    int eid;
    {
        uint16_t m = E;
        int e0 = __builtin_ctz(m); m &= uint16_t(m - 1);
        int e1 = __builtin_ctz(m); m &= uint16_t(m - 1);
        int e2 = __builtin_ctz(m); m &= uint16_t(m - 1);
        int e3 = __builtin_ctz(m);
        eid = BINOM[e0][1] + BINOM[e1][2] + BINOM[e2][3] + BINOM[e3][4];
    }
    int role = 0;
    for (int r = 0; r < NUM_ROLES; ++r)
        if (ROLE_A_PICK[eid * NUM_ROLES + r] == A) { role = r; break; }
    phase2(st, w, eid, role, ctx);
}

// ---------------------------------------------------------------------------
// One parallel task: a fixed round-1 and round-2 outcome (16 bits), then a
// full sweep of the 256 round-3 outcomes.
// ---------------------------------------------------------------------------

static void run_task(int combo16, Ctx& ctx) {
    State st0;
    std::memset(&st0, 0, sizeof(st0));

    Match m1[8], m2[8], m3[8];
    bool b1[8], b2[8], b3[8];

    build_matches(st0, 1, m1, b1);
    State st = st0;
    double w = 1.0;
    for (int k = 0; k < 8; ++k) apply_match(st, w, m1[k], b1[k], (combo16 >> k) & 1);

    build_matches(st, 2, m2, b2);
    for (int k = 0; k < 8; ++k) apply_match(st, w, m2[k], b2[k], (combo16 >> (8 + k)) & 1);

    build_matches(st, 3, m3, b3);
    for (int combo3 = 0; combo3 < 256; ++combo3) {
        State s3 = st;
        double w3 = w;
        for (int k = 0; k < 8; ++k) apply_match(s3, w3, m3[k], b3[k], (combo3 >> k) & 1);
        handle_prefix(s3, w3, ctx);
    }
}

// ---------------------------------------------------------------------------
// Driver.
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::string out_path = argc >= 2 ? argv[1] : "simulated_outcomes.csv";
    unsigned nthreads = std::thread::hardware_concurrency();
    if (argc >= 3) nthreads = unsigned(std::stoul(argv[2]));
    if (nthreads == 0) nthreads = 1;

    build_probabilities();
    build_binom();

    const long N = long(NUM_EXCL_SETS) * NUM_ROLES * NUM_MID_CHOICES; // 10,090,080
    std::vector<double> master(N, 0.0), masterc(N, 0.0);
    double global_tot = 0.0, global_totc = 0.0;
    std::mutex reduce_mtx;
    std::atomic<int> next_chunk{0};
    const int CHUNK = 8;
    const int NUM_TASKS = 65536;

    std::cerr << "running with " << nthreads << " threads\n";

    auto worker = [&]() {
        std::vector<double> G(N, 0.0), Gc(N, 0.0);
        Ctx ctx;
        ctx.G = G.data();
        ctx.Gc = Gc.data();

        for (;;) {
            int start = next_chunk.fetch_add(CHUNK);
            if (start >= NUM_TASKS) break;
            int end = std::min(start + CHUNK, NUM_TASKS);
            for (int combo16 = start; combo16 < end; ++combo16)
                run_task(combo16, ctx);
        }

        std::lock_guard<std::mutex> lock(reduce_mtx);
        for (long i = 0; i < N; ++i) kadd(master[i], masterc[i], G[i] + Gc[i]);
        kadd(global_tot, global_totc, ctx.tot + ctx.totc);
    };

    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    double total = global_tot + global_totc;
    std::cerr << "total probability mass = " << (total) << " (|1 - sum| = "
              << std::fabs(1.0 - total) << ")\n";

    FILE* f = std::fopen(out_path.c_str(), "w");
    if (!f) {
        std::cerr << "could not open " << out_path << "\n";
        return 1;
    }
    std::fprintf(f, "A3_0,B0_3,C3_1_3_2,prob\n");
    long rows = 0;
    for (int eid = 0; eid < NUM_EXCL_SETS; ++eid) {
        uint16_t E = EXCL_SET_MASKS[eid];
        for (int role = 0; role < NUM_ROLES; ++role) {
            uint16_t A = ROLE_A_PICK[eid * NUM_ROLES + role];
            uint16_t B = uint16_t(E ^ A);
            long slice = (long(eid) * NUM_ROLES + role) * NUM_MID_CHOICES;
            for (int cid = 0; cid < NUM_MID_CHOICES; ++cid) {
                double p = master[slice + cid] + masterc[slice + cid];
                if (p <= 0.0) continue;
                uint16_t C = expand_mid_to_global(eid, MID_MASKS[cid]);
                std::fprintf(f, "%u,%u,%u,%.17g\n", unsigned(A), unsigned(B), unsigned(C), p);
                ++rows;
            }
        }
    }
    std::fclose(f);
    std::cerr << "wrote " << rows << " rows to " << out_path << "\n";
    return 0;
}
