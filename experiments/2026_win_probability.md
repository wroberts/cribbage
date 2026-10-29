# Score-aware play: does maximising win probability beat maximising expected points? (28 Oct 2026)

Question: everything so far maximises expected points per hand; the objective is winning a game to 121. Plan, cheap
to expensive: (1) a win-probability table V, (2) headroom for score-aware choices, (3) score-aware players in games.

## 1. The win-probability table (28 Oct)

**Player.** `model:data/runs/two_k100.pt` discards + `net:data/pegging/rl/net_w100_post.pt` pegging, both seats.

**Choice: Markov chain, not visit counts.** `V[sd, sn]` = P(dealer of the next hand wins), sd/sn the dealer's and
non-dealer's scores (121 × 121; the other dealer case is 1 − V transposed). Reasons: self-play visit counts are sparse
in the cells that matter (late, lopsided) and noisy; the chain is smooth, solved exactly in 2 s, and every hand
simulated serves every state. The price is an approximation, so I validated it against games (below).

**Method** (`cribbage/win_prob.py`). 2 000 000 hands (seed 0, 19 min) record five numbers each: dealer's nibs+pegging,
non-dealer's pegging, non-dealer's hand, dealer's hand, crib. Means 3.59, 2.23, 8.13, 8.01, 4.76. The chain applies the
stages in scoring order (peg, non-dealer's hand, dealer's hand, crib) and stops the moment a score reaches 121, which
captures the non-dealer-counts-first rule. Approximations: the pegging pair is the empirical joint, but the three show
scores are independent of each other and of the pegging; if both pass 121 in pegging the player needing fewer points
wins (order within pegging is not tracked); scores do not change how the player chooses (true of this player by
construction, and exactly what step 2 will test).

**Validation.** 100 000 self-play games (898 k hands, 502 s), visiting each round's start state:

| Check | Model | Games |
|---|---|---|
| first dealer wins | 0.5567 | 0.5579 ± 0.0031 |
| hands both in 80–115, mean P(dealer wins) | 0.5410 | 0.5412 ± 0.0024 |
| Brier score, all hands | 0.1766 | (0.25 for always 0.5) |

Calibration by predicted-probability bin is within about 0.005 everywhere (log in `data/winprob/validate.log`); the
biggest gaps are at the extremes (predicted 0.0062 / observed 0.0053, predicted 0.9941 / observed 0.9907), i.e. the
chain is slightly over-confident where the game is nearly decided. Intervals treat hands as independent, but hands of
one game are correlated, so they are somewhat too tight.

**Reading.** The dealer's edge at the start is 55.7 % (the dealer's crib). V is not smooth near the end: with level
scores the dealer wins 0.54 at 110, 0.26 at 115, 0.78 at 100 and 105, because who gets to count first, and how many
hands remain, decide the finish. Those kinks are where score-aware play could differ from points play; step 2 measures
how much.

Files: `data/winprob/{hands.npy,v.npy,visits.npy,*.log}`; commands `python -m cribbage.win_prob hands|solve|validate`.

## 2. Headroom: how much would a win-probability choice gain? (28 Oct)

**Question.** In mid-game states, how often does the choice that maximises P(win) differ from the expected-points
choice, and what does it gain? Stop if the gain is under about 0.5 pp of win rate.

**Method** (`cribbage/score_aware.py`). The value of a choice is the mean, over simulated continuations of the hand, of
P(win) at the end of the hand from the table V of §1, with the game ending the moment a score reaches 121. A continuation
is the ordered list of scoring events: nibs, each pegging score as it happens, the non-dealer's hand, the dealer's hand,
the crib. So crossing 121 mid-pegging or in the show is handled in the right order. The same simulation gives the
points margin, so the two objectives are compared on identical samples (common random numbers).

* **Discard.** 600 round-start states drawn at random from the self-play visits of §1 with both scores in 80–115, each
  evaluated for both seats (1 200 evaluations). A random six-card hand; for each of the 15 splits, 300 samples ×
  2 independent replicates, the opponent's six, the starter and the pegging shared by all splits; the opponent keeps
  what `two_k100` keeps; both peg with `net_w100_post`.
* **Pegging.** 1 500 states × 2 seats from the same zone (2 997 had a real choice), one pegging decision with at least
  two differently-valued legal cards chosen at random in a hand started at those scores (nibs and the pegging so far
  are added to the score). 300 worlds × 2 replicates drawn from the discard-network posterior (`NetOppHandModel`,
  as in `pimc-net:W:PATH@DISCARD`); each candidate is played and the hand finished by the network on both sides, with the
  show scored and the unseen crib cards guessed.
* **Selection bias.** Choose on replicate A, score on replicate B (and the reverse, averaged), as in §15.2 of the
  pegging book. This estimate is not inflated by picking noise, but it is **attenuated by it**: with 300 samples the
  per-choice noise is still comparable to the true differences, so these gains are **lower bounds**.
* Three comparisons: WP pick vs the model's own choice (`two_k100` discard, `net_w100_post` play); points pick (best
  mean margin on the same samples, i.e. pegging-aware for discards and a net-rollout search for play) vs the model; and
  WP pick vs that points pick, which isolates what *score awareness* adds. Intervals are across states.

**Results**, win-probability gain in pp per decision (95 % interval):

| Comparison | Discard | Pegging |
|---|---|---|
| WP pick vs the model's choice | +0.376 ± 0.084 | +0.410 ± 0.070 |
| points pick vs the model's choice | +0.066 ± 0.072 | +0.158 ± 0.060 |
| **WP pick vs points pick (score awareness)** | **+0.311 ± 0.085** | **+0.253 ± 0.048** |
| WP pick differs from the model's | 34 % | 35 % |
| WP pick differs from the points pick (replicate noise floor) | 27 % (11 %) | 24 % (11 %) |

By seat, score awareness is worth +0.52 ± 0.16 pp (non-dealer) and +0.11 ± 0.07 (dealer) at discard, and
+0.24 ± 0.07 and +0.26 ± 0.07 at pegging. The "differs" rates are inflated by noise; the noise floor (two replicates
disagreeing about the points pick) is 11 %.

**Per hand and per game.** At pegging a hand has on average more than one real choice; weighting each sampled decision by
the number of such decisions in its hand, score awareness is worth **+0.65 ± 0.13 pp per hand** in the zone
(WP pick vs the model: +1.08 ± 0.20, which includes about +0.43 from searching rather than the net's one-step choice). A
hand has both scores in 80–115 in 18.1 % of the round starts, **1.63 hands per game**. To first order, and treating the
gains as additive:

| | pp of win rate per game |
|---|---|
| discard, score-aware part | 0.31 × 1.63 ≈ **0.51 ± 0.14** |
| pegging, score-aware part | 0.65 × 1.63 ≈ **1.06 ± 0.21** |
| both | ≈ **1.6 ± 0.3** (zone only) |

**Reading.** The headroom is real, and above the 0.5 pp stopping line for each piece on its own and clearly for both.
It is a lower bound in three ways: noise attenuation as above; states outside 80–115 were not measured (they will
add something, since V already bends from about 60); and V was built for points players, so a score-aware player's
own V would be a little more favourable. It is an upper bound in one: gains were taken from a simulation of the same
policies used to build V, with the opponent held at its points play, and adding up per-hand gains ignores that the
choices interact. The comparison to make is the full games of step 3, which are the only measurement that includes
all of this.

## 3. Score-aware players in full games (28–29 Oct)

**Players** (`ScoreAwareDiscardPolicy`, `ScoreAwarePeggingPolicy` in `cribbage/score_aware.py`; specs `score:PATH[,...]` and
`score-net:PATH[,...]`). Each plays exactly what the expected-points player plays (`two_k100` discard, `net_w100_post`
pegging) unless the larger of the two scores is at least 75 and a short simulation says another choice is better for the
chance of winning. The simulation values a choice by the mean P(win) from V at the end of the hand, as in §2. The default
is overruled only if the mean paired difference exceeds `z` standard errors, since choosing the best of several noisy
estimates loses more than score awareness gains.

* **Discard:** the network's 3 best splits, simulated with 40 samples (opponent's six, starter and pegging shared by the 3).
* **Pegging:** 80 worlds drawn from the discard network's posterior, each candidate card played and the hand finished by
  the network on both sides, the unseen crib cards guessed.
* **Tuning** (`score_aware tune`; 800 and 1 600 states in 80–115, a seed disjoint from §2, scored against 300 / 600
  independent samples; `data/winprob/tune_*.log`). Gain over the default choice in pp, z = 1: discard 0.12 / 0.26 / 0.25 for
  20 / 40 / 80 samples, pegging 0.58 / 0.77 / 0.97 per hand for 20 / 40 / 80 worlds. z between 0.5 and 1 was best; z = 0
  (always take the best estimate) was worse or equal at every size, and for discards at 20 samples it lost 0.16. I chose
  discard 40 samples and pegging 80 worlds, both z = 1, a compromise with cost (about 0.38 s a game for both, against
  a few ms for the base player).

**Games** (`pegging_bench match`; A = score-aware, B = `model:data/runs/two_k100.pt+net:data/pegging/rl/net_w100_post.pt`;
the first dealer is random; logs `data/winprob/match_*.log`, script `run_matches.sh`):

| A | games | A's win rate (95 %) | gain over 50 % |
|---|---|---|---|
| both policies | 20 000 | **51.3 % ± 0.7** | +1.3 pp |
| pegging only (default discards) | 20 000 | **51.3 % ± 0.7** | +1.3 pp |
| discard only (default pegging) | 40 000 | **50.7 % ± 0.5** | +0.7 pp |
| *control:* pegging search maximising points (same 80 worlds, z = 1, threshold 75) | 20 000 | **50.0 % ± 0.7** | +0.0 pp |

An earlier 800-game trial of the "both" player (53.9 % ± 3.5) was only a speed check and is not evidence.

**Reading.**

* **Score-aware play wins more, by a small amount.** All three are above 50 %, by about 2.6σ for pegging and both, and
  2.8σ for discard alone. The point estimate for the combined player is +1.3 pp, interval roughly +0.6 to +2.0. §2 predicted
  about +1.6 pp (zone only) for the ideal choices, +0.4 (discard) and +1.5 (pegging) for the tuned policies by the
  per-hand measure; each observed value is within about one interval of its prediction.
* **The pieces do not visibly add.** Both together (+1.3) is not above pegging alone (+1.3); the difference of the
  predictions (about 1.9 against 1.3) is inside the ±0.7 interval, so I would not read anything into it, but I cannot
  claim the discard gain adds to the pegging gain either. Resolving a 0.5 pp difference between the arms needs about
  100 000 games each (about 11 hours of compute for the combined player).
* **For pegging the control says the gain is score awareness, not search.** The same search with the same rollouts,
  beliefs, threshold and z, but maximising the points margin, wins **50.0 % ± 0.7** of 20 000 games (`match_peg_points.log`,
  seed 14). §2 had predicted some search benefit (+0.43 pp per hand in the zone, from the 300-world figures); in games
  it is not visible, which is plausible since the net already plays near its teacher (§13, §15.1 of the pegging
  book: search with the network is level with the network). The win-probability version is 1.3 pp above the control.
  That difference has an interval of about ±1.0 (two independent 20 000-game matches), so on its own it is only
  1.3σ; the stronger statement is that the win-probability arm is 2.6σ above 50 % while the matched points arm sits at
  50.0. A direct head-to-head of the two arms would need about 100 000 games to settle it.
* **Meaning.** About one extra win in 75 games. It is real but a minor effect next to what pegging and discard quality
  already contribute (§5 of the pegging book: 1.8 pp per 0.1 pts/hand).

**Limits.**

* One opponent (the points player), no adaptation to the score-aware player's tendencies.
* V comes from points players; a score-aware player's own V would be slightly more favourable and could be iterated
  (not done).
* Tuned on states with a disjoint seed, but on the same simulator that measures the gain; the games are the independent
  check.
* Policies only act above 75, based on the headroom zone; the thresholds 70 and 60 were not tried.
* Only one run of each match, one seed.

**Where a learned value function would slot in** (not built; the longer-term question).

* V is a table over (score, score, dealer), built for the points player. Replace it with a learned value
  `V(s_me, s_opp, dealer, features)` trained by self-play of the score-aware player itself (policy iteration, the
  fixed point where the table and the policy agree), or a function of the cards still unseen and the pegging
  position so that it can score mid-hand states.
* The simulation of the rest of the hand (the 40–80 samples) is the expensive part: 0.4 s a game. A Q-function
  `Q(state, action)` for discards and cards, regressed on the sampled win probabilities as the pegging network
  was regressed on search margins (§8 of the pegging book), would remove it and could be trained with the labels these
  policies generate: `peg_values` and `ScoreAwareDiscardPolicy.simulate` return the per-sample win probabilities.
* `dqlearning.py` / `netbuilder.py` could provide the learner; the terminal reward is the win, with `None` as the
  terminal state as in the replay memory there.
