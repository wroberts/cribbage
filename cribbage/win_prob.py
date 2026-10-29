#!/usr/bin/env python

"""
win_prob.py

The probability of winning a game from a score, for the expected-points
player playing itself, and the experiments that ask whether play that
maximises it beats play that maximises expected points.

`V[dealer score, non-dealer score]` is the probability that the dealer of
the hand about to be dealt goes on to win.  It comes from a Markov chain on
the scores: a hand's scoring is split into stages in the order the rules
score them (nibs and pegging, the non-dealer's hand, the dealer's hand, the
crib), the points in each stage are drawn from the distributions seen in
simulated hands, and the game ends the moment a score reaches 121.  `solve`
runs the chain backwards.  `validate` checks the table against full
self-played games.

Commands (`python -m cribbage.win_prob --help`):

- `hands`: simulate hands and save the per-stage points
- `solve`: build V from the saved hands
- `validate`: play games and compare V with how they ended
- `plot`: write V as a heatmap, a self-contained HTML page or (`--png`) a PNG
"""

from __future__ import annotations

import argparse
import multiprocessing
import random
import time
from collections.abc import Sequence

import numpy as np

from cribbage.cards import split_card
from cribbage.discard_policy import DiscardPolicy
from cribbage.game import Game
from cribbage.pegging import PegState
from cribbage.pegging_model import NetPeggingPolicy
from cribbage.player import DiscardView
from cribbage.scoring import score_hand

TARGET = 121
DISCARD = "data/runs/two_k100.pt"
PEGGING = "data/pegging/rl/net_w100_post.pt"
PLAYER = f"model:{DISCARD}+net:{PEGGING}"

# ------------------------------------------------------------
#  Simulated hands


def simulate_hand(
    deal: np.ndarray, discard: DiscardPolicy, peg: NetPeggingPolicy
) -> tuple[int, int, int, int, int]:
    """
    Plays one hand, the dealer being seat 0 and both seats using the same
    policies.  `deal` is (13,): six cards each and the starter.  Returns the
    dealer's points from nibs and pegging, the non-dealer's pegging points,
    and the non-dealer's hand, the dealer's hand and the crib.
    """
    dealt = [tuple(int(c) for c in deal[6 * s : 6 * s + 6]) for s in (0, 1)]
    starter = int(deal[12])
    kept, thrown = [], []
    for seat in (0, 1):
        out = discard.choose(DiscardView(dealt[seat], seat == 0, 0, 0))
        thrown.append(tuple(out))
        kept.append([c for c in dealt[seat] if c not in out])
    state = PegState.start(kept, 0)
    while not state.done:
        seat = state.turn
        view = state.view(seat, thrown[seat], starter, seat == 0, 0, 0)
        state.play_card(peg.choose(view))
    nibs = 2 if split_card(starter)[0] == 10 else 0
    return (
        nibs + state.points[0],
        state.points[1],
        score_hand(kept[1], starter),
        score_hand(kept[0], starter),
        score_hand([*thrown[0], *thrown[1]], starter, crib=True),
    )


_WORKER: dict[str, object] = {}


def _init_hands(discard_path: str, peg_path: str) -> None:
    import torch

    from cribbage.discard_model import NetDiscardPolicy
    from cribbage.pegging_model import NetPeggingPolicy

    torch.set_num_threads(1)
    _WORKER["discard"] = NetDiscardPolicy.load(discard_path)
    _WORKER["peg"] = NetPeggingPolicy.load(peg_path)


def _hands_task(task: tuple[int, int, int]) -> np.ndarray:
    first, num, seed = task
    out = np.zeros((num, 5), dtype=np.int16)
    peg = _WORKER["peg"]
    assert isinstance(peg, NetPeggingPolicy)
    discard = _WORKER["discard"]
    assert isinstance(discard, DiscardPolicy)
    for i in range(num):
        rng = np.random.default_rng([seed, first + i])
        peg.reseed(f"{seed}/{first + i}")
        out[i] = simulate_hand(rng.permutation(52)[:13], discard, peg)
    return out


def run_hands(args: argparse.Namespace) -> None:
    start = time.time()
    chunk = 500
    tasks = [(i, min(chunk, args.hands - i), args.seed) for i in range(0, args.hands, chunk)]
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        args.workers, initializer=_init_hands, initargs=(args.discard, args.peg)
    ) as pool:
        parts = list(pool.imap(_hands_task, tasks))
    rows = np.concatenate(parts)
    np.save(args.out, rows)
    print(f"{len(rows)} hands in {time.time() - start:.0f}s -> {args.out}")
    names = ["dealer peg+nibs", "non-dealer peg", "non-dealer hand", "dealer hand", "crib"]
    for name, col in zip(names, rows.T):
        print(f"  {name:16} mean {col.mean():.3f}  sd {col.std():.3f}  max {col.max()}")


# ------------------------------------------------------------
#  The Markov chain


def pmf(values: np.ndarray, size: int = 40) -> np.ndarray:
    return np.bincount(values, minlength=size).astype(float) / len(values)


def solve(rows: np.ndarray, sweeps: int = 200) -> np.ndarray:
    """
    V[sd, sn] for sd, sn in 0..120: the probability that the dealer wins when
    the hand about to be dealt has the dealer on `sd` and the non-dealer on
    `sn`.  Stages follow the order of scoring.  The rows are
    (dealer nibs+peg, non-dealer peg, non-dealer hand, dealer hand, crib); the
    last three are treated as independent of each other and of the pegging.
    If pegging takes both players past 121 (rare), the dealer, who scores
    first in the show, is not given the win; the player who needs fewer points
    gets it.
    """
    n = TARGET
    peg_pairs, peg_counts = np.unique(rows[:, :2].astype(int), axis=0, return_counts=True)
    peg_p = peg_counts / peg_counts.sum()
    p_hn, p_hd, p_hc = (pmf(rows[:, k].astype(int)) for k in (2, 3, 4))
    sd = np.arange(n)[:, None] * np.ones((1, n), dtype=int)
    sn = np.arange(n)[None, :] * np.ones((n, 1), dtype=int)
    v = np.full((n, n), 0.5)
    for _ in range(sweeps):
        # crib: the dealer scores last; then the other player deals
        c = np.zeros((n, n))
        for pts, p in enumerate(p_hc):
            if p == 0:
                continue
            nd = sd + pts
            nxt = 1.0 - v[sn, np.minimum(nd, n - 1)]
            c += p * np.where(nd >= TARGET, 1.0, nxt)
        # dealer's hand
        b = np.zeros((n, n))
        for pts, p in enumerate(p_hd):
            if p == 0:
                continue
            nd = sd + pts
            b += p * np.where(nd >= TARGET, 1.0, c[np.minimum(nd, n - 1), sn])
        # non-dealer's hand
        a = np.zeros((n, n))
        for pts, p in enumerate(p_hn):
            if p == 0:
                continue
            nn = sn + pts
            a += p * np.where(nn >= TARGET, 0.0, b[sd, np.minimum(nn, n - 1)])
        # nibs and pegging
        new = np.zeros((n, n))
        for (pd, pn), p in zip(peg_pairs, peg_p):
            nd, nn = sd + pd, sn + pn
            both = (nd >= TARGET) & (nn >= TARGET)
            dealer_first = (TARGET - sd) <= (TARGET - sn)
            win = np.where(both, dealer_first, nd >= TARGET)
            lose = (nn >= TARGET) & ~win
            cont = a[np.minimum(nd, n - 1), np.minimum(nn, n - 1)]
            new += p * np.where(win, 1.0, np.where(lose, 0.0, cont))
        if np.abs(new - v).max() < 1e-12:
            v = new
            break
        v = new
    return v


def win_prob(v: np.ndarray, score: int, opp_score: int, is_dealer: bool) -> float:
    """The probability that a player on `score` wins against `opp_score`, before the deal."""
    if is_dealer:
        return float(v[score, opp_score])
    return float(1.0 - v[opp_score, score])


def run_solve(args: argparse.Namespace) -> None:
    rows = np.load(args.hands)
    v = solve(rows)
    np.save(args.out, v)
    print(f"V from {len(rows)} hands -> {args.out}")
    print(f"  start of game: dealer wins {v[0, 0]:.4f}")
    for s in (60, 90, 100, 110, 115):
        print(f"  level at {s}: dealer wins {v[s, s]:.4f}")


# ------------------------------------------------------------
#  Validation against full games


def _games_task(task: tuple[str, int, int]) -> np.ndarray:
    """Plays games; returns one row per round: (dealer score, non-dealer score, dealer won)."""
    import torch

    from cribbage.specs import make_player

    spec, games, seed = task
    torch.set_num_threads(1)
    random.seed(seed)
    player = make_player(spec)
    player.reseed(seed)
    rows = []
    for _ in range(games):
        game = Game([player, player])
        visits = []
        while True:
            dealer = game.dealer_idx
            visits.append((game.scores[dealer], game.scores[1 - dealer], dealer))
            if not game.play_round():
                break
        for sd, sn, dealer in visits:
            rows.append((sd, sn, int(game.winner == dealer)))
    return np.array(rows, dtype=np.int16)


def play_games(spec: str, games: int, seed: int, workers: int) -> np.ndarray:
    chunk = 100
    tasks = [(spec, chunk, seed * 100000 + i) for i in range(games // chunk)]
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers) as pool:
        return np.concatenate(pool.map(_games_task, tasks))


def run_validate(args: argparse.Namespace) -> None:
    start = time.time()
    v = np.load(args.v)
    rows = play_games(args.player, args.games, args.seed, args.workers).astype(int)
    np.save(args.out, rows)
    sd, sn, won = rows.T
    pred = v[sd, sn]
    print(f"{args.games} self-play games, {len(rows)} hands ({time.time() - start:.0f}s)")
    print(f"  hands per game {len(rows) / args.games:.2f}")
    first = rows[(sd == 0) & (sn == 0)]
    print(f"  first dealer wins {first[:, 2].mean():.4f} ± "
          f"{1.96 * first[:, 2].std() / np.sqrt(len(first)):.4f}   model {v[0, 0]:.4f}")
    print(f"  Brier: model {np.mean((pred - won) ** 2):.5f}   always 0.5 {0.25:.5f}")
    print("  calibration (hands pooled by predicted P(dealer wins)):")
    edges = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.sum():
            print(f"    [{lo:.2f},{hi:.2f}) n={m.sum():7d}  predicted {pred[m].mean():.4f}  "
                  f"observed {won[m].mean():.4f} ± {1.96 * np.sqrt(max(won[m].mean() * (1 - won[m].mean()), 1e-9) / m.sum()):.4f}")
    print("  late-game cells (both between 80 and 115):")
    m = (sd >= 80) & (sd <= 115) & (sn >= 80) & (sn <= 115)
    print(f"    n={m.sum()}  predicted {pred[m].mean():.4f}  observed {won[m].mean():.4f} ± "
          f"{1.96 * won[m].std() / np.sqrt(m.sum()):.4f}   Brier {np.mean((pred[m] - won[m]) ** 2):.5f}")


# ------------------------------------------------------------
#  Picture

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>V: P(dealer wins)</title>
<style>
  body { font: 14px system-ui, sans-serif; margin: 16px; background: #fff; color: #222; }
  @media (prefers-color-scheme: dark) { body { background: #16181c; color: #ddd; } }
  .row { display: flex; flex-wrap: wrap; gap: 24px; }
  canvas { image-rendering: pixelated; border: 1px solid #888; }
  #readout { height: 1.5em; margin: 8px 0; font-weight: 600; }
  .bar { width: 260px; height: 14px; border: 1px solid #888;
         background: linear-gradient(to right, rgb(33,102,172), rgb(247,247,247), rgb(178,24,43)); }
  .scale { display: flex; justify-content: space-between; width: 260px; font-size: 12px; }
</style>
<h3>V: probability that the dealer of the next hand wins the game</h3>
<div id="readout">hover a cell</div>
<div class="bar"></div><div class="scale"><span>0 (non-dealer wins)</span><span>0.5</span><span>1 (dealer wins)</span></div>
<div class="row">
  <div><div>all scores</div><canvas id="full" width="605" height="605"></canvas></div>
  <div><div>zoom: both scores 80-120</div><canvas id="zoom" width="605" height="605"></canvas></div>
</div>
<p>x: non-dealer's score, y: dealer's score (up). Dark outline: cells within 0.02 of 0.5.</p>
<script>
const V = __DATA__;
const colour = p => {
  const lo = [33, 102, 172], mid = [247, 247, 247], hi = [178, 24, 43];
  const t = p < 0.5 ? p / 0.5 : (p - 0.5) / 0.5, a = p < 0.5 ? lo : mid, b = p < 0.5 ? mid : hi;
  return a.map((x, i) => Math.round(x + (b[i] - x) * t));
};
function draw(id, lo, hi) {
  const c = document.getElementById(id), g = c.getContext("2d"), n = hi - lo + 1, size = c.width / n;
  for (let d = lo; d <= hi; d++) for (let nd = lo; nd <= hi; nd++) {
    const p = V[d][nd];
    g.fillStyle = "rgb(" + colour(p).join(",") + ")";
    g.fillRect((nd - lo) * size, (hi - d) * size, size + 0.5, size + 0.5);
    if (Math.abs(p - 0.5) < 0.02 && size >= 4) {
      g.strokeStyle = "rgba(0,0,0,0.35)"; g.strokeRect((nd - lo) * size, (hi - d) * size, size, size);
    }
  }
  c.addEventListener("mousemove", e => {
    const r = c.getBoundingClientRect(), x = (e.clientX - r.left) * c.width / r.width, y = (e.clientY - r.top) * c.height / r.height;
    const nd = lo + Math.floor(x / size), d = hi - Math.floor(y / size);
    if (nd >= lo && nd <= hi && d >= lo && d <= hi)
      document.getElementById("readout").textContent =
        "dealer " + d + ", non-dealer " + nd + ": P(dealer wins) = " + V[d][nd].toFixed(3);
  });
}
draw("full", 0, 120); draw("zoom", 80, 120);
</script>
"""


def plot_png(v: np.ndarray, path: str) -> None:
    """V as two heatmaps, all scores and 80-120, with the 0.5 contour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    for ax, (lo, hi) in zip(axes, ((0, 120), (80, 120))):
        sub = v[lo : hi + 1, lo : hi + 1]
        image = ax.imshow(
            sub, origin="lower", cmap="RdBu_r", vmin=0, vmax=1,
            extent=(lo - 0.5, hi + 0.5, lo - 0.5, hi + 0.5), interpolation="nearest",
        )
        ax.contour(np.arange(lo, hi + 1), np.arange(lo, hi + 1), sub, levels=[0.5],
                   colors="k", linewidths=0.8)
        ax.set_xlabel("non-dealer's score")
        ax.set_ylabel("dealer's score")
        ax.set_title(f"scores {lo}-{hi}")
    fig.colorbar(image, ax=axes, shrink=0.85, label="V: probability the dealer wins")
    fig.savefig(path, dpi=130)


def run_plot(args: argparse.Namespace) -> None:
    import json

    v = np.load(args.v)
    if args.png:
        plot_png(v, args.png)
        print(f"wrote {args.png}")
        return
    data = json.dumps(np.round(v, 4).tolist())
    with open(args.out, "w") as out:
        out.write(PAGE.replace("__DATA__", data))
    print(f"wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("hands", help="simulate hands")
    p.add_argument("--hands", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--discard", default=DISCARD)
    p.add_argument("--peg", default=PEGGING)
    p.add_argument("--out", default="data/winprob/hands.npy")
    p.set_defaults(run=run_hands)
    p = sub.add_parser("solve", help="solve the chain")
    p.add_argument("--hands", default="data/winprob/hands.npy")
    p.add_argument("--out", default="data/winprob/v.npy")
    p.set_defaults(run=run_solve)
    p = sub.add_parser("validate", help="compare V with self-played games")
    p.add_argument("--v", default="data/winprob/v.npy")
    p.add_argument("--player", default=PLAYER)
    p.add_argument("--games", type=int, default=40000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="data/winprob/visits.npy")
    p.set_defaults(run=run_validate)
    p = sub.add_parser("plot", help="write V as an HTML heatmap")
    p.add_argument("--v", default="data/winprob/v.npy")
    p.add_argument("--out", default="data/winprob/v_heatmap.html")
    p.add_argument("--png", help="write a PNG with matplotlib instead of the HTML page")
    p.set_defaults(run=run_plot)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
