"""
pegging_bench.py

How much is the pegging phase (`play_card`) worth, and does search help?

The discard policy is held fixed (a trained `NetDiscardPolicy`,
`model:PATH`) and only the pegging policy varies, along the ladder of
strength described in `pegging_policy` and `pegging_search` (`random`,
`greedy`, `greedy2`, `pimc-roll`, `pimc-solve`, `ismcts`, `oracle`; see
`specs.make_pegging_policy` for the spec strings).

Pegging is measured in isolation (`peg-bench`, `ladder`, `cost`): fixed
deals are made once, both players keep the cards the discard model would
keep, and each deal is played twice with the seats swapped, so that hand
luck and the dealer/non-dealer asymmetry cancel.  The score is the
pegging-points margin per hand.  `match` plays full games between
`DISCARD+POLICY` players, and `calibrate` relates points per hand to win
rate.

Run `python -m cribbage.pegging_bench --help` for the commands.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from cribbage.cards import Card
from cribbage.discard_model import (
    DISCARD_IDX,
    KEEP_IDX,
    encode_hands,
    load_model,
    predict_values,
)
from cribbage.game import Game, compare_players, count_wins, win_ci
from cribbage.pegging import PegView
from cribbage.pegging_policy import PeggingPolicy
from cribbage.player import PolicyPlayer
from cribbage.round import Deal, Round
from cribbage.specs import make_discard_policy, make_pegging_policy, make_player

# ------------------------------------------------------------
#  The player


class BenchPlayer(PolicyPlayer):
    """
    A `PolicyPlayer` that times its pegging decisions, and that can show
    an oracle policy the opponent's hand (`peek`, set by the harness).
    """

    def __init__(self, discard_spec: str, pegging_policy: PeggingPolicy) -> None:
        super().__init__(make_discard_policy(discard_spec), pegging_policy)
        # reads the opponent's hand, for the oracle (set by the harness)
        self.peek: Callable[[], Sequence[Card]] | None = None
        self.seconds = 0.0
        self.decisions = 0

    def play_card(self, view: PegView) -> Card:
        if self.pegging_policy.needs_oracle:
            assert self.peek is not None
            view = replace(view, opp_hand=tuple(self.peek()))
        start = time.perf_counter()
        card = super().play_card(view)
        self.seconds += time.perf_counter() - start
        self.decisions += 1
        return card


def make_bench_player(peg_spec: str, discard_spec: str = "random") -> BenchPlayer:
    """A BenchPlayer for pegging policy `peg_spec`; the discard is irrelevant by default."""
    return BenchPlayer(discard_spec, make_pegging_policy(peg_spec))


# ------------------------------------------------------------
#  Fixed deals


@dataclass
class Deals:
    """
    `num` deals.  `kept[i, 0]` and `kept[i, 1]` are the four cards kept by
    the non-dealer and the dealer, `thrown` the two they discarded, and
    `starter` the cut card.
    """

    kept: np.ndarray
    thrown: np.ndarray
    starter: np.ndarray

    def __len__(self) -> int:
        return len(self.starter)

    def slice(self, start: int, stop: int) -> Deals:
        return Deals(
            self.kept[start:stop], self.thrown[start:stop], self.starter[start:stop]
        )


def make_deals(num: int, seed: int, discard: str = "random") -> Deals:
    """
    Deals `num` rounds.  Both players discard with `discard`: `random`, or
    `model:PATH` for a saved DiscardNet (the non-dealer and dealer values
    differ by the crib term).
    """
    rng = np.random.default_rng([seed, 1])
    cards = np.argsort(rng.random((num, 52)), axis=1)[:, :13]
    starter = cards[:, 12]
    kept = np.zeros((num, 2, 4), dtype=np.int64)
    thrown = np.zeros((num, 2, 2), dtype=np.int64)
    model = None
    if discard != "random":
        if not discard.startswith("model:"):
            raise ValueError(f"unknown discard {discard!r}")
        model = load_model(discard[len("model:") :]).eval()
    for seat in range(2):
        dealt = cards[:, 6 * seat : 6 * seat + 6]
        if model is None:
            best = rng.integers(0, len(KEEP_IDX), size=num)
        else:
            best = np.zeros(num, dtype=np.int64)
            with torch.no_grad():
                for lo in range(0, num, 20000):
                    chunk = torch.as_tensor(dealt[lo : lo + 20000])
                    values = predict_values(model, encode_hands(chunk), bool(seat))
                    best[lo : lo + 20000] = values.argmax(dim=1).numpy()
        rows = np.arange(num)[:, None]
        kept[:, seat] = dealt[rows, KEEP_IDX.numpy()[best]]
        thrown[:, seat] = dealt[rows, DISCARD_IDX.numpy()[best]]
    return Deals(kept, thrown, starter)


# ------------------------------------------------------------
#  Pegging-only matches


def _hand_reader(rnd: Round, seat: int) -> Callable[[], Sequence[Card]]:
    def read() -> Sequence[Card]:
        assert rnd.peg is not None
        return rnd.peg.hands[seat]

    return read


def play_pegging(
    players: Sequence[BenchPlayer],
    kept: Sequence[Sequence[Card]],
    thrown: Sequence[Sequence[Card]],
    starter: Card,
) -> tuple[int, int]:
    """
    Plays out the pegging of one deal with the engine's own `Round`.
    `players[0]` is the non-dealer; `kept` and `thrown` are indexed the
    same way.  Returns the pegging points of both players.
    """
    game = Game(list(players))
    game.target_score = 10**9
    rnd = Round.from_deal(game, list(players), Deal.from_kept(1, kept, thrown, starter))
    for seat, player in enumerate(players):
        if player.pegging_policy.needs_oracle:
            player.peek = _hand_reader(rnd, 1 - seat)
    rnd.play()
    return int(game.scores[0]), int(game.scores[1])


@dataclass
class PegResult:
    """
    Per-deal pegging points for players A and B, each deal being played
    twice: `a_nd`, `b_d` (A non-dealer) and `b_nd`, `a_d` (A dealer).
    """

    a_nd: np.ndarray
    b_d: np.ndarray
    b_nd: np.ndarray
    a_d: np.ndarray
    seconds: tuple[float, float] = (0.0, 0.0)
    decisions: tuple[int, int] = (0, 0)

    @property
    def pair_margin(self) -> np.ndarray:
        """A's margin over the two plays of each deal."""
        return (self.a_nd - self.b_d) + (self.a_d - self.b_nd)

    def per_hand(self) -> tuple[float, float]:
        """A's mean margin per hand played, and its 95% half-width."""
        m = self.pair_margin
        return m.mean() / 2, 1.96 * m.std(ddof=1) / math.sqrt(len(m)) / 2

    def a_points(self) -> float:
        return float((self.a_nd + self.a_d).mean() / 2)

    def b_points(self) -> float:
        return float((self.b_d + self.b_nd).mean() / 2)

    def role_margins(self) -> tuple[float, float]:
        """A's mean margin as non-dealer and as dealer."""
        return float((self.a_nd - self.b_d).mean()), float(
            (self.a_d - self.b_nd).mean()
        )

    def ms_per_hand(self) -> tuple[float, float]:
        n = 2 * len(self.a_nd)
        return 1000 * self.seconds[0] / n, 1000 * self.seconds[1] / n

    @staticmethod
    def concat(results: Sequence[PegResult]) -> PegResult:
        a_nd, b_d, b_nd, a_d = (
            np.concatenate([getattr(r, f) for r in results])
            for f in ("a_nd", "b_d", "b_nd", "a_d")
        )
        return PegResult(
            a_nd,
            b_d,
            b_nd,
            a_d,
            seconds=(
                sum(r.seconds[0] for r in results),
                sum(r.seconds[1] for r in results),
            ),
            decisions=(
                sum(r.decisions[0] for r in results),
                sum(r.decisions[1] for r in results),
            ),
        )


def _peg_chunk(task: tuple[str, str, Deals, int, int]) -> PegResult:
    """Plays one chunk of deals, each with the seats swapped."""
    spec_a, spec_b, deals, seed, first = task
    a, b = make_bench_player(spec_a), make_bench_player(spec_b)
    cols = np.zeros((len(deals), 4), dtype=np.int64)
    for i in range(len(deals)):
        for player, who in ((a, "a"), (b, "b")):
            player.pegging_policy.reseed(f"{seed}/{first + i}/{who}")
        kept, thrown, starter = deals.kept[i], deals.thrown[i], deals.starter[i]
        # A is the non-dealer with hand X, B the dealer with hand Y ...
        cols[i, 0], cols[i, 1] = play_pegging([a, b], kept, thrown, starter)
        # ... then B is the non-dealer with X, and A the dealer with Y
        cols[i, 2], cols[i, 3] = play_pegging([b, a], kept, thrown, starter)
    return PegResult(
        cols[:, 0],
        cols[:, 1],
        cols[:, 2],
        cols[:, 3],
        seconds=(a.seconds, b.seconds),
        decisions=(a.decisions, b.decisions),
    )


def run_pegging(
    spec_a: str, spec_b: str, deals: Deals, seed: int = 0, workers: int = 1
) -> PegResult:
    """
    Plays every deal twice (seats swapped) between pegging policies A and
    B.  The result does not depend on `workers`.
    """
    size = max(1, min(500, math.ceil(len(deals) / max(1, 4 * workers))))
    tasks = [
        (spec_a, spec_b, deals.slice(lo, lo + size), seed, lo)
        for lo in range(0, len(deals), size)
    ]
    if workers > 1:
        # forked workers that use torch deadlock once the parent has, so
        # policies that load a network get fresh processes
        method = "spawn" if "net:" in spec_a + spec_b else None  # also pimc-net:
        with multiprocessing.get_context(method).Pool(workers) as pool:
            parts = pool.map(_peg_chunk, tasks)
    else:
        parts = [_peg_chunk(t) for t in tasks]
    return PegResult.concat(parts)


def describe(spec_a: str, spec_b: str, result: PegResult) -> str:
    margin, half = result.per_hand()
    nd, d = result.role_margins()
    ms_a, ms_b = result.ms_per_hand()
    return (
        f"{spec_a} vs {spec_b}, {len(result.a_nd)} deals x2 seats\n"
        f"  margin per hand: {margin:+.3f} +/- {half:.3f} (95% CI)"
        f"   [as non-dealer {nd:+.3f}, as dealer {d:+.3f}]\n"
        f"  pegging points per hand: {result.a_points():.3f} vs {result.b_points():.3f}\n"
        f"  search time per hand: {ms_a:.3f} ms vs {ms_b:.3f} ms"
    )


# ------------------------------------------------------------
#  Full games


def _match_chunk(task: tuple[str, str, int, int]) -> int:
    spec_a, spec_b, games, seed = task
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    players = [make_player(spec_a), make_player(spec_b)]
    players[0].reseed(seed)
    players[1].reseed(seed + 1)
    return compare_players(list(players), games)[0]


def play_match(spec_a: str, spec_b: str, games: int, seed: int, workers: int) -> float:
    """A's win rate over `games` full games (see `count_wins`)."""
    return count_wins(_match_chunk, spec_a, spec_b, games, seed, workers) / games


# ------------------------------------------------------------
#  Command line


def _deals_for(args: argparse.Namespace) -> Deals:
    return make_deals(args.deals, args.seed, args.discard)


def run_peg_bench(args: argparse.Namespace) -> None:
    """Pegging-only duplicate match between two pegging policies."""
    start = time.time()
    result = run_pegging(args.a, args.b, _deals_for(args), args.seed, args.workers)
    print(describe(args.a, args.b, result))
    print(f"  ({time.time() - start:.0f}s)")


def run_ladder(args: argparse.Namespace) -> None:
    """Round-robin of pegging policies on the same deals."""
    specs = args.policies.split(",")
    deals = _deals_for(args)
    margins: dict[tuple[str, str], tuple[float, float]] = {}
    ms: dict[str, float] = {}
    for i, a in enumerate(specs):
        for b in specs[i + 1 :]:
            result = run_pegging(a, b, deals, args.seed, args.workers)
            margins[a, b] = result.per_hand()
            m_a, m_b = result.ms_per_hand()
            ms[a], ms[b] = max(ms.get(a, 0), m_a), max(ms.get(b, 0), m_b)
            print(f"  {a} vs {b}: {margins[a, b][0]:+.3f}", flush=True)
    print(
        f"\nrow's pegging-points margin over column, per hand "
        f"({args.deals} deals, 95% half-width ~{max(h for _, h in margins.values()):.3f})"
    )
    width = max(len(s) for s in specs) + 2
    print(" " * width + "".join(f"{s:>{width}}" for s in specs) + "   ms/hand")
    for a in specs:
        cells = []
        for b in specs:
            if a == b:
                cells.append("-")
            elif (a, b) in margins:
                cells.append(f"{margins[a, b][0]:+.3f}")
            else:
                cells.append(f"{-margins[b, a][0]:+.3f}")
        print(
            f"{a:<{width}}"
            + "".join(f"{c:>{width}}" for c in cells)
            + f"   {ms.get(a, 0):.3f}"
        )


def run_cost(args: argparse.Namespace) -> None:
    """Margin against a reference policy as a family's search budget grows."""
    deals = _deals_for(args)
    print(f"{args.family}:N vs {args.ref}, {args.deals} deals")
    print("      N   margin/hand   +/-     ms/hand")
    for n in (int(x) for x in args.budgets.split(",")):
        result = run_pegging(
            f"{args.family}:{n}{args.flags}", args.ref, deals, args.seed, args.workers
        )
        margin, half = result.per_hand()
        print(
            f"{n:7d}   {margin:+.3f}    {half:.3f}    {result.ms_per_hand()[0]:.3f}",
            flush=True,
        )


def run_match(args: argparse.Namespace) -> None:
    """Full games between DISCARD+POLICY players."""
    start = time.time()
    rate = play_match(args.a, args.b, args.games, args.seed, args.workers)
    print(
        f"{args.a} beat {args.b}: {100 * rate:.1f}% +/- "
        f"{100 * win_ci(rate, args.games):.1f}% of {args.games} games,"
        f" {time.time() - start:.0f}s"
    )


def run_calibrate(args: argparse.Namespace) -> None:
    """Pegging margin per hand against full-game win rate, versus random pegging."""
    deals = _deals_for(args)
    xs, ys = [], []
    print(f"discard {args.discard}; each policy vs random pegging")
    print("policy               margin/hand   win rate")
    for spec in args.policies.split(","):
        margin, half = run_pegging(
            spec, "random", deals, args.seed, args.workers
        ).per_hand()
        rate = play_match(
            f"{args.discard}+{spec}",
            f"{args.discard}+random",
            args.games,
            args.seed,
            args.workers,
        )
        xs.append(margin)
        ys.append(rate - 0.5)
        print(
            f"{spec:<20} {margin:+.3f}+/-{half:.3f}   {100 * rate:.1f}% +/- {100 * win_ci(rate, args.games):.1f}%",
            flush=True,
        )
    x, y = np.array(xs), np.array(ys)
    slope = float((x * y).sum() / (x * x).sum())
    print(
        f"least-squares slope through 50%: {100 * 0.1 * slope:.2f} pp of win rate per 0.1 pt/hand"
    )


def main() -> None:
    """Experiments on the value of pegging skill and of search."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    bench = commands.add_parser("peg-bench", help=run_peg_bench.__doc__)
    bench.add_argument("--a", required=True, help="pegging policy, see make_policy")
    bench.add_argument("--b", required=True, help="pegging policy, see make_policy")
    bench.set_defaults(func=run_peg_bench)

    ladder = commands.add_parser("ladder", help=run_ladder.__doc__)
    ladder.add_argument(
        "--policies",
        default="random,greedy,greedy2,pimc-roll:10,pimc-solve:10,ismcts:200,oracle",
    )
    ladder.set_defaults(func=run_ladder)

    cost = commands.add_parser("cost", help=run_cost.__doc__)
    cost.add_argument("--family", default="pimc-solve")
    cost.add_argument("--flags", default="", help="suffix such as /opp")
    cost.add_argument("--budgets", default="1,2,5,10,20,50")
    cost.add_argument("--ref", default="greedy")
    cost.set_defaults(func=run_cost)

    match = commands.add_parser("match", help=run_match.__doc__)
    match.add_argument("--a", required=True, help="DISCARD+POLICY")
    match.add_argument("--b", required=True, help="DISCARD+POLICY")
    match.add_argument("--games", type=int, default=2000)
    match.set_defaults(func=run_match)

    calibrate = commands.add_parser("calibrate", help=run_calibrate.__doc__)
    calibrate.add_argument("--policies", default="greedy,greedy2,pimc-roll:10")
    calibrate.add_argument("--games", type=int, default=4000)
    calibrate.set_defaults(func=run_calibrate)

    for sub in (bench, ladder, cost, calibrate):
        sub.add_argument("--deals", type=int, default=20000)
        sub.add_argument("--discard", default="model:data/runs/two_k100.pt")
    for sub in (bench, ladder, cost, match, calibrate):
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
