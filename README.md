# CS Major bracket simulator and pickem solver

Takes matchup probabilities between teams as input, exactly simulates the swiss bracket and solves for top N optimal pickems either in probability to pass (at least 5 correct), or in highest expected number of correct points.

Both simulator and solver should not take more than a minute or two to run on CPU, even on single core. Need to generate precomputed.hpp before compiling the solvers.

Algorithms are brain-made (about 2 years old now actually), most code is AI.

## Build

```bash
g++ -O3 -std=c++20 generate_precomp.cpp -o generate_precomp
./generate_precomp precomputed.hpp

g++ -O3 -std=c++20 generate_example_input.cpp -o generate_example_input
g++ -O3 -std=c++20 solver.cpp -o solver
g++ -O3 -std=c++20 ev_solver.cpp -o ev_solver
```

## Simulate swiss bracket outcomes (use multithreading)

`simulate_bracket` computes the *exact* probability of every Swiss outcome by driven by two hardcoded 16x16 matrices `P_bo1` / `P_bo3` (edit `build_probabilities()` to plug in real numbers). Output matches format used by solvers.

```bash
g++ -O3 -std=c++20 generate_precomp.cpp -o generate_precomp
./generate_precomp precomputed.hpp        # emits the bracket pairing tables too

g++ -O3 -std=c++20 -pthread simulate_bracket.cpp -o simulate_bracket
./simulate_bracket simulated_outcomes.csv          # all hardware threads
./simulate_bracket simulated_outcomes.csv 1        # optional thread-count override
```

## Run pass-probability solver

Find only the best pickem:

```bash
./solver example_outcomes.csv
```

Write all pickem pass probabilities as CSV:

```bash
./solver example_outcomes.csv --all all_scores.csv
```

Find the top `N` pickems, ranked best first:

```bash
./solver example_outcomes.csv --top 10
```

## Run EV solver

Find the maximum expected-correct pickem, is very fast in comparison.

```bash
./ev_solver example_outcomes.csv
```

Write every pickem expected-correct value as CSV:

```bash
./ev_solver example_outcomes.csv --all all_ev.csv
```

Find the top `N` pickems, ranked best first:

```bash
./ev_solver example_outcomes.csv --top 10
```

Note: `--top N` with `N > 1` disables the greedy single-`C` shortcut and
enumerates every middle choice, so it runs slower than the default search.

## Algorithm Overview

## Probability of passing

Let one full pickem choice be $\pi=(E,r,C)$:

- $E$: excluded-location set (the 3-0 / 0-3 teams),
- $r$: A/B role split on $E$,
- $C$: middle set (mask $m_C$).

why $E$, $r$ is split like this instead of just having 3-0, 0-3 sets will become apparent.  

For fixed $\pi$, the pass probability is:

```math
P(\text{pass}\mid \pi)=\sum_{\omega} p(\omega)\,\mathbf{1}[\text{score}_{\pi}(\omega)\ge 5].
```

Here $p(\omega)$ is the model probability of outcome $\omega$ (independent of the pickem), so $p(\omega\mid\pi)=p(\omega)$.

Write the score in outcome $\omega$ as:

```math
\text{score}_{\pi}(\omega)=k_E\!\big(r,\sigma_E(\omega)\big)+|m_C\cap q_E(\omega)|,
```

where $\sigma_E(\omega)$ is the extremity result pattern (`extr_roles`), $q_E(\omega)$ is the 12-bit qualifier mask (`qual_mask`), and $k_E(r,\sigma)\in\{0,\dots,4\}$ is the number correct from 3-0/0-3 picks.

If we build $V_{E,r}$ directly (without $W_E$), we can bucket outcome mass by extremity score $k$ and qualifier mask $q$:

```math
V^{\mathrm{direct}}_{E,r}(k,q)=\sum_{\omega} p(\omega)\,\mathbf{1}[k_E(r,\sigma_E(\omega))=k]\mathbf{1}[q_E(\omega)=q].
```

Then:

```math
P(\text{pass}\mid \pi)=\sum_{q}\sum_{k=0}^{4} V^{\mathrm{direct}}_{E,r}(k,q)\,\mathbf{1}\!\left[|m_C\cap q|\ge 5-k\right],
```

which is roughly $\binom{16}{4}\binom{4}{2}N=10{,}920N$ work across all $(E,r)$.

To reduce this, first compute a shared table for each excluded-location set $E$ (independent of the 6 role splits):

```math
W_E(\sigma,q)=\sum_{\omega:\,\sigma_E(\omega)=\sigma,\ q_E(\omega)=q} p(\omega).
```

Then derive each role-specific table by remapping $\sigma\to k$:

```math
V_{E,r}(k,q)=\sum_{\sigma} W_E(\sigma,q)\,\mathbf{1}[k_E(r,\sigma)=k].
```

Now the same final pass expression applies:

```math
P(\text{pass}\mid \pi)=\sum_{q}\sum_{k=0}^{4} V_{E,r}(k,q)\,\mathbf{1}\!\left[|m_C\cap q|\ge 5-k\right],
```

and the dominant outcome pass drops to around $\binom{16}{4}N=1{,}820N$.

## EV Solver

For EV we write one pickem as $\pi=(A,B,C)$:

- $A$: predicted 3-0 teams,
- $B$: predicted 0-3 teams,
- $C$: predicted middle teams.

For outcome $\omega$, let $A^{*}(\omega),B^{*}(\omega),C^{*}(\omega)$ be the true 3-0 / 0-3 / middle sets. Then:

```math
\mathbb{E}(\pi)=\sum_{\omega} p(\omega)\Big(|A\cap A^{*}(\omega)|+|B\cap B^{*}(\omega)|+|C\cap C^{*}(\omega)|\Big).
```

Expand cardinalities as team sums (over teams $i$):

```math
\mathbb{E}(\pi)=
\sum_{\omega}\sum_i p(\omega)\,\mathbf{1}[i\in A]\mathbf{1}[i\in A^{*}(\omega)]
+\sum_{\omega}\sum_i p(\omega)\,\mathbf{1}[i\in B]\mathbf{1}[i\in B^{*}(\omega)]
+\sum_{\omega}\sum_i p(\omega)\,\mathbf{1}[i\in C]\mathbf{1}[i\in C^{*}(\omega)].
```

Reorder so team indices are outermost, and collect the outcome sums:

```math
\mathbb{E}(\pi)=
\sum_i \mathbf{1}[i\in A]\underbrace{\sum_{\omega} p(\omega)\mathbf{1}[i\in A^{*}(\omega)]}_{\alpha_i}
+\sum_i \mathbf{1}[i\in B]\underbrace{\sum_{\omega} p(\omega)\mathbf{1}[i\in B^{*}(\omega)]}_{\beta_i}
+\sum_i \mathbf{1}[i\in C]\underbrace{\sum_{\omega} p(\omega)\mathbf{1}[i\in C^{*}(\omega)]}_{\gamma_i}.
```

So EV reduces to collected team terms times membership indicators:

```math
\mathbb{E}(\pi)=\sum_i \alpha_i\mathbf{1}[i\in A]+\sum_i \beta_i\mathbf{1}[i\in B]+\sum_i \gamma_i\mathbf{1}[i\in C].
```

So after computing $\alpha_i$, $\beta_i$, $\gamma_i$ (one pass over $N$), they can be reused to compute EV almost instantly. Furthermore, we can iterate over $A$, $B$ and greedily choose the best 6 3-1 / 3-2 teams.

## Simulation

Enumerates all $2^{33}$ paths very efficiently, uses compensated summation for accuracy.

- round-3 boundary factorization: expand each post-round-3 state through all $2^9$ round-4/5 suffixes in one pass, keeping write area small/cached.
- compact hot tables (`ROW_MASK`, `ROW_PARTNER`, `MID_INDEX`, colex-rank `BINOM`);
- bitmask state with fixed-size loops for low per-path overhead;
- thread-local aggregation (`G`, `Gc`) with compensated summation, then one reduction.
