#!/usr/bin/env python

"""
score_aware.py

Would choices that maximise the probability of winning beat choices that
maximise expected points?  Step 2 of the score-aware study (see
`experiments/2026_win_probability.md`): measure the headroom before building
a player.

`V` (see `win_prob`) gives the chance of winning from a pair of scores
before a deal.  The value of a choice is the mean, over simulated
continuations of the hand, of the win probability at the end of the hand,
with the game ending the moment a score reaches 121.  A continuation is an
ordered list of (seat, points): nibs, the pegging, the non-dealer's hand,
the dealer's hand and the crib.

`discard-headroom` takes round-start states from self-played games, deals
random hands, and for each of the 15 splits simulates many continuations
(opponent hand, starter and pegging shared by all splits).  It runs two
independent replicates.  Choosing a split on one replicate and scoring it on
the other gives a gain that is not inflated by picking the noisiest split.

Commands: `python -m cribbage.score_aware --help`.
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing
import random
import time

import numpy as np

from cribbage.cards import face, split_card
from cribbage.discard_model import DISCARD_IDX, NetDiscardPolicy, encode_hands, predict_values
from cribbage.discard_policy import DiscardPolicy
from cribbage.pegging import PegState, PegView, Play
from cribbage.pegging_model import NetOppHandModel, NetPeggingPolicy, NetRollout
from cribbage.pegging_policy import PeggingPolicy
from cribbage.pegging_search import WorldSampler, distinct_moves
from cribbage.player import DiscardView
from cribbage.scoring import score_hand

TARGET = 121
SPLITS = list(itertools.combinations(range(6), 2))

# ------------------------------------------------------------
#  Hands as ordered scoring events


def hand_events(
    kept: list[list[int]],
    thrown: list[tuple[int, ...]],
    starter: int,
    dealer: int,
    peg: NetPeggingPolicy,
) -> list[tuple[int, int]]:
    """The (seat, points) events of one hand, in the order the rules score them."""
    events: list[tuple[int, int]] = []
    if split_card(starter)[0] == 10:
        events.append((dealer, 2))
    state = PegState.start(kept, dealer)
    while not state.done:
        seat = state.turn
        before = state.points[seat]
        view = state.view(seat, thrown[seat], starter, seat == dealer, 0, 0)
        state.play_card(peg.choose(view))
        gained = state.points[seat] - before
        if gained:
            events.append((seat, gained))
    # the pegging events above are in play order; closing points go to the
    # mover, so no seat other than `seat` scores within a play
    other = 1 - dealer
    for seat, cards, crib in (
        (other, kept[other], False),
        (dealer, kept[dealer], False),
        (dealer, [*thrown[0], *thrown[1]], True),
    ):
        events.append((seat, score_hand(cards, starter, crib=crib)))
    return events


def win_after(
    v: np.ndarray, score: int, opp_score: int, is_dealer: bool, events: list[tuple[int, int]], me: int
) -> float:
    """
    The probability that seat `me` wins, given the scores before the hand and the
    events of the hand.  `is_dealer` says whether `me` deals this hand.
    """
    scores = [0, 0]
    scores[me], scores[1 - me] = score, opp_score
    for seat, points in events:
        scores[seat] += points
        if scores[seat] >= TARGET:
            return 1.0 if seat == me else 0.0
    mine, theirs = scores[me], scores[1 - me]
    # the other player deals next
    return float(1.0 - v[theirs, mine]) if is_dealer else float(v[mine, theirs])


def margin(events: list[tuple[int, int]], me: int) -> int:
    return sum(p if s == me else -p for s, p in events)


# ------------------------------------------------------------
#  Discard headroom

_WORKER: dict[str, object] = {}


def _init(discard_path: str, peg_path: str, v_path: str) -> None:
    import torch

    torch.set_num_threads(1)
    _WORKER["discard"] = NetDiscardPolicy.load(discard_path)
    _WORKER["peg"] = NetPeggingPolicy.load(peg_path)
    _WORKER["v"] = np.load(v_path)


def evaluate_splits(
    seed: tuple[int, ...],
    hand: tuple[int, ...],
    me_dealer: bool,
    score: int,
    opp_score: int,
    samples: int,
    replicates: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Mean win probability and mean points margin of each of the 15 splits of `hand`,
    per replicate: arrays (replicates, 15), and the index of the split the discard
    network picks.  Seat 0 is the player; the starter, the opponent's hand and its
    discard are shared by the splits.
    """
    discard = _WORKER["discard"]
    peg = _WORKER["peg"]
    v = _WORKER["v"]
    assert isinstance(discard, DiscardPolicy) and isinstance(peg, NetPeggingPolicy)
    assert isinstance(v, np.ndarray)
    rng = np.random.default_rng(seed)
    dealer = 0 if me_dealer else 1
    others = np.array([c for c in range(52) if c not in hand])
    wins = np.zeros((replicates, 15))
    margins = np.zeros((replicates, 15))
    for rep in range(replicates):
        for _ in range(samples):
            draw = rng.permutation(others)
            opp_hand = tuple(int(c) for c in draw[:6])
            starter = int(draw[6])
            out = discard.choose(DiscardView(opp_hand, not me_dealer, opp_score, score))
            opp_thrown = tuple(out)
            opp_kept = [c for c in opp_hand if c not in out]
            peg_seed = int(rng.integers(2**31))
            for k, (i, j) in enumerate(SPLITS):
                thrown = (hand[i], hand[j])
                kept = [c for c in hand if c not in thrown]
                peg.reseed(peg_seed)
                events = hand_events(
                    [kept, opp_kept], [thrown, opp_thrown], starter, dealer, peg
                )
                wins[rep, k] += win_after(v, score, opp_score, me_dealer, events, 0)
                margins[rep, k] += margin(events, 0)
    pick = discard.choose(DiscardView(hand, me_dealer, score, opp_score))
    model_idx = SPLITS.index(
        (min(hand.index(c) for c in pick), max(hand.index(c) for c in pick))
    )
    return wins / samples, margins / samples, model_idx


def _task(task: tuple[int, int, int, int, int, int, int]) -> tuple[int, int, np.ndarray, np.ndarray, int]:
    index, seed, dealer_score, nondealer_score, role, samples, replicates = task
    rng = np.random.default_rng([seed, index, 99])
    hand = tuple(int(c) for c in rng.permutation(52)[:6])
    me_dealer = bool(role)
    score, opp = (dealer_score, nondealer_score) if me_dealer else (nondealer_score, dealer_score)
    w, m, model = evaluate_splits(
        (seed, index, role), hand, me_dealer, score, opp, samples, replicates
    )
    return index, role, w, m, model


def run_discard(args: argparse.Namespace) -> None:
    start = time.time()
    visits = np.load(args.visits).astype(int)
    sd, sn = visits[:, 0], visits[:, 1]
    keep = np.ones(len(visits), dtype=bool)
    if args.zone:
        lo, hi = args.zone
        keep = (sd >= lo) & (sd <= hi) & (sn >= lo) & (sn <= hi)
    candidates = np.flatnonzero(keep)
    pick = np.random.default_rng(args.seed).choice(
        candidates, size=args.states, replace=False
    )
    tasks = [
        (i, args.seed, int(sd[p]), int(sn[p]), role, args.samples, 2)
        for i, p in enumerate(pick)
        for role in (0, 1)
    ]
    context = multiprocessing.get_context("spawn")
    results = []
    with context.Pool(
        args.workers, initializer=_init, initargs=(args.discard, args.peg, args.v)
    ) as pool:
        for n, res in enumerate(pool.imap_unordered(_task, tasks, chunksize=2)):
            results.append(res)
            if (n + 1) % 200 == 0:
                print(f"  {n + 1}/{len(tasks)} ({time.time() - start:.0f}s)", flush=True)
    results.sort(key=lambda r: (r[0], r[1]))
    np.savez(
        args.out,
        dealer_score=np.array([sd[pick[r[0]]] for r in results]),
        nondealer_score=np.array([sn[pick[r[0]]] for r in results]),
        role=np.array([r[1] for r in results]),
        wins=np.array([r[2] for r in results]),
        margins=np.array([r[3] for r in results]),
        model=np.array([r[4] for r in results]),
        samples=args.samples,
    )
    print(f"{len(results)} evaluations in {time.time() - start:.0f}s -> {args.out}")
    analyse(args.out)


# ------------------------------------------------------------
#  Pegging headroom


def peg_decision(
    seed: tuple[int, ...],
    score: int,
    opp_score: int,
    me_dealer: bool,
    discard: DiscardPolicy,
    peg: NetPeggingPolicy,
) -> tuple[PegView, int] | None:
    """
    A pegging decision of the player (seat 0) with at least two different cards to
    choose from, from a random hand started at the given scores: nibs and the pegging
    so far are added to the scores.  Chosen uniformly among such decisions of the hand
    (None if there is none), with the number of such decisions in the hand.
    """
    rng = np.random.default_rng(seed)
    deal = rng.permutation(52)[:13]
    dealer = 0 if me_dealer else 1
    dealt = [tuple(int(c) for c in deal[6 * s : 6 * s + 6]) for s in (0, 1)]
    starter = int(deal[12])
    kept, thrown = [], []
    for seat in (0, 1):
        out = discard.choose(DiscardView(dealt[seat], seat == dealer, 0, 0))
        thrown.append(tuple(out))
        kept.append([c for c in dealt[seat] if c not in out])
    nibs = [0, 0]
    if split_card(starter)[0] == 10:
        nibs[dealer] = 2
    peg_seed = int(rng.integers(2**31))
    target = -1
    total = 0
    for stage in (0, 1):
        peg.reseed(peg_seed)
        state = PegState.start(kept, dealer)
        count = 0
        while not state.done:
            seat = state.turn
            view = state.view(seat, thrown[seat], starter, seat == dealer, 0, 0)
            if seat == 0 and len({face(c) for c in view.legal}) > 1:
                if stage == 1 and count == target:
                    return state.view(
                        0,
                        thrown[0],
                        starter,
                        me_dealer,
                        score + nibs[0] + state.points[0],
                        opp_score + nibs[1] + state.points[1],
                    ), total
                count += 1
            state.play_card(peg.choose(view))
        if stage == 0:
            if count == 0:
                return None
            total = count
            target = int(rng.integers(count))
    return None


def peg_values(
    view: PegView,
    worlds: int,
    net: NetPeggingPolicy,
    opp_model: NetOppHandModel,
    v: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Win probability and points margin of each distinct legal card, per sampled world:
    arrays (worlds, cards), and the faces of the cards in column order.  A candidate
    is played, the rest of the hand is played out by the network on both sides, and
    the show is scored (the unseen crib cards are guessed at random).
    """
    beliefs = opp_model.kept_hands(view, rng)
    sampler = WorldSampler(view, beliefs=beliefs)
    unseen = view.unseen()
    mine = [
        e.card
        for seq in view.sequences
        for e in seq.events
        if isinstance(e, Play) and e.player == 0
    ]
    opp_played = [card for _, card in view.opp_plays]
    me_kept = [*view.hand, *mine]
    dealer = 0 if view.is_dealer else 1
    wins: list[list[float]] = []
    margins: list[list[float]] = []
    faces: list[int] = []
    for _ in range(worlds):
        world = sampler.sample(rng)
        moves = distinct_moves(world)
        faces = list(moves)
        others = [c for c in unseen if c not in world.hands[1]]
        opp_thrown = tuple(rng.sample(others, 2))
        ctx = NetRollout(
            net,
            world.hands,
            set(view.played) | {view.starter},
            [view.discards, opp_thrown],
            dealer,
        )
        opp_kept = [*world.hands[1], *opp_played]
        shows = []
        other = 1 - dealer
        kept = [me_kept, opp_kept]
        for seat, cards, crib in (
            (other, kept[other], False),
            (dealer, kept[dealer], False),
            (dealer, [*view.discards, *opp_thrown], True),
        ):
            shows.append((seat, score_hand(cards, view.starter, crib=crib)))
        row_w: list[float] = []
        row_m: list[float] = []
        for idx in moves.values():
            child = world.copy()
            events: list[tuple[int, int]] = []
            move = idx
            while not child.done:
                seat = child.turn
                before = child.points[seat]
                child.play(move)
                if child.points[seat] > before:
                    events.append((seat, child.points[seat] - before))
                if not child.done:
                    move = ctx.move(child, rng)
            events.extend(shows)
            row_w.append(win_after(v, view.score, view.opp_score, view.is_dealer, events, 0))
            row_m.append(margin(events, 0))
        wins.append(row_w)
        margins.append(row_m)
    return np.array(wins), np.array(margins), faces


MAX_CANDS = 4


def _peg_init(discard_path: str, peg_path: str, v_path: str) -> None:
    _init(discard_path, peg_path, v_path)
    _WORKER["opp"] = NetOppHandModel.load(discard_path)


def _peg_task(
    task: tuple[int, int, int, int, int, int]
) -> tuple[int, int, np.ndarray, np.ndarray, int, int] | None:
    index, seed, dealer_score, nondealer_score, role, worlds = task
    discard, peg = _WORKER["discard"], _WORKER["peg"]
    v, opp = _WORKER["v"], _WORKER["opp"]
    assert isinstance(discard, DiscardPolicy) and isinstance(peg, NetPeggingPolicy)
    assert isinstance(v, np.ndarray) and isinstance(opp, NetOppHandModel)
    me_dealer = bool(role)
    score, opp_score = (
        (dealer_score, nondealer_score) if me_dealer else (nondealer_score, dealer_score)
    )
    found = peg_decision((seed, index, role, 7), score, opp_score, me_dealer, discard, peg)
    if found is None:
        return None
    view, decisions = found
    rng = random.Random(f"{seed}/{index}/{role}")
    wins, margins, faces = peg_values(view, 2 * worlds, peg, opp, v, rng)
    peg.reseed(f"{seed}/{index}")
    chosen = face(peg.choose(view))
    out_w = np.full((2, MAX_CANDS), np.nan)
    out_m = np.full((2, MAX_CANDS), np.nan)
    for rep in range(2):
        half = slice(rep * worlds, (rep + 1) * worlds)
        out_w[rep, : len(faces)] = wins[half].mean(axis=0)
        out_m[rep, : len(faces)] = margins[half].mean(axis=0)
    return index, role, out_w, out_m, faces.index(chosen), decisions


def run_peg(args: argparse.Namespace) -> None:
    start = time.time()
    visits = np.load(args.visits).astype(int)
    sd, sn = visits[:, 0], visits[:, 1]
    keep = np.ones(len(visits), dtype=bool)
    if args.zone:
        lo, hi = args.zone
        keep = (sd >= lo) & (sd <= hi) & (sn >= lo) & (sn <= hi)
    candidates = np.flatnonzero(keep)
    pick = np.random.default_rng(args.seed).choice(
        candidates, size=args.states, replace=False
    )
    tasks = [
        (i, args.seed, int(sd[p]), int(sn[p]), role, args.worlds)
        for i, p in enumerate(pick)
        for role in (0, 1)
    ]
    context = multiprocessing.get_context("spawn")
    results = []
    with context.Pool(
        args.workers, initializer=_peg_init, initargs=(args.discard, args.peg, args.v)
    ) as pool:
        for n, res in enumerate(pool.imap_unordered(_peg_task, tasks, chunksize=2)):
            if res is not None:
                results.append(res)
            if (n + 1) % 200 == 0:
                print(f"  {n + 1}/{len(tasks)} ({time.time() - start:.0f}s)", flush=True)
    results.sort(key=lambda r: (r[0], r[1]))
    np.savez(
        args.out,
        dealer_score=np.array([sd[pick[r[0]]] for r in results]),
        nondealer_score=np.array([sn[pick[r[0]]] for r in results]),
        role=np.array([r[1] for r in results]),
        wins=np.array([r[2] for r in results]),
        margins=np.array([r[3] for r in results]),
        model=np.array([r[4] for r in results]),
        decisions=np.array([r[5] for r in results]),
        attempted=len(tasks),
        samples=args.worlds,
    )
    print(f"{len(results)} decisions in {time.time() - start:.0f}s -> {args.out}")
    analyse(args.out)



# ------------------------------------------------------------
#  Score-aware players
#
# Both policies play what the expected-points player plays unless, late in the
# game, a short simulation says that another choice is better for the chance
# of winning by more than its noise: the mean win-probability difference from
# the default choice (paired over shared samples) must exceed `z` standard
# errors.  Without that test, picking among noisy estimates would lose more
# than score-awareness gains.

DEFAULT_V = "data/winprob/v.npy"
DEFAULT_PEG = "data/pegging/rl/net_w100_post.pt"


def parse_options(spec: str, defaults: dict[str, float | str]) -> tuple[str, dict[str, float | str]]:
    """`PATH,key=value,...` -> (PATH, options), values typed like the defaults."""
    path, *items = spec.split(",")
    options = dict(defaults)
    for item in items:
        key, _, value = item.partition("=")
        if key not in defaults:
            raise ValueError(f"unknown option {key!r} in {spec!r}")
        options[key] = type(defaults[key])(value)
    return path, options


def pick_candidate(wins: np.ndarray, base: int, z: float) -> int:
    """
    The column of `wins` (samples, candidates) to play: the best mean difference
    from column `base`, if it is more than `z` standard errors above zero, else `base`.
    """
    n = len(wins)
    diff = wins - wins[:, [base]]
    mean = diff.mean(axis=0)
    se = diff.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(diff.shape[1], np.inf)
    gain = mean - z * se
    gain[base] = 0.0
    best = int(gain.argmax())
    return best if gain[best] > 0 and mean[best] > 0 else base


class ScoreAwareDiscardPolicy(DiscardPolicy):
    """
    Discards what the network discards, unless a simulation of the few best splits
    says another gives a better chance of winning.  Spec:
    `score:DISCARDPATH[,samples=N][,z=Z][,cands=K][,min=S][,v=PATH][,peg=PATH]`.
    """

    DEFAULTS: dict[str, float | str] = {
        "samples": 40, "z": 1.0, "cands": 3, "min": 75, "v": DEFAULT_V, "peg": DEFAULT_PEG
    }

    def __init__(
        self,
        base: NetDiscardPolicy,
        peg: NetPeggingPolicy,
        v: np.ndarray,
        samples: int = 40,
        z: float = 1.0,
        cands: int = 3,
        min_score: int = 75,
    ) -> None:
        super().__init__()
        self.base, self.peg, self.v = base, peg, v
        self.samples, self.z, self.cands, self.min_score = samples, z, cands, min_score

    @classmethod
    def load(cls, spec: str) -> ScoreAwareDiscardPolicy:
        path, o = parse_options(spec, cls.DEFAULTS)
        return cls(
            NetDiscardPolicy.load(path),
            NetPeggingPolicy.load(str(o["peg"])),
            np.load(str(o["v"])),
            int(o["samples"]), float(o["z"]), int(o["cands"]), int(o["min"]),
        )

    def reseed(self, seed: int | str) -> None:
        super().reseed(seed)
        self.peg.reseed(f"{seed}/sim")

    def candidates(self, view: DiscardView) -> list[tuple[int, int]]:
        """The throws of the network's best splits, best first."""
        import torch

        with torch.no_grad():
            values = predict_values(
                self.base.model, encode_hands(torch.tensor([view.hand])), view.is_dealer
            )[0]
        order = values.argsort(descending=True)[: self.cands]
        return [
            (view.hand[i], view.hand[j]) for i, j in DISCARD_IDX[order].tolist()
        ]

    def simulate(self, view: DiscardView, throws: list[tuple[int, int]], samples: int) -> np.ndarray:
        """Win probability of each throw in each sample: (samples, throws)."""
        hand = view.hand
        others = [c for c in range(52) if c not in hand]
        dealer = 0 if view.is_dealer else 1
        wins = np.zeros((samples, len(throws)))
        for n in range(samples):
            draw = self.rng.sample(others, 7)
            opp_hand, starter = tuple(draw[:6]), draw[6]
            out = tuple(
                self.base.choose(DiscardView(opp_hand, not view.is_dealer, view.opp_score, view.score))
            )
            opp_kept = [c for c in opp_hand if c not in out]
            seed = self.rng.getrandbits(31)
            for k, thrown in enumerate(throws):
                kept = [c for c in hand if c not in thrown]
                self.peg.reseed(seed)
                events = hand_events([kept, opp_kept], [thrown, out], starter, dealer, self.peg)
                wins[n, k] = win_after(self.v, view.score, view.opp_score, view.is_dealer, events, 0)
        return wins

    def choose(self, view: DiscardView) -> tuple[int, int]:
        throws = self.candidates(view)
        if max(view.score, view.opp_score) < self.min_score or len(throws) < 2:
            return throws[0]
        wins = self.simulate(view, throws, self.samples)
        return throws[pick_candidate(wins, 0, self.z)]


class ScoreAwarePeggingPolicy(PeggingPolicy):
    """
    Plays what the pegging network plays, unless a simulation of the rest of the
    hand (opponent's hand from the discard network's posterior, rollouts played
    by the network) says another card gives a better chance of winning.  Spec:
    `score-net:PEGPATH[,opp=DISCARDPATH][,worlds=N][,z=Z][,min=S][,v=PATH][,obj=win|points]`.
    With `obj=points` the same search maximises the points margin instead, as a control
    that separates the value of searching from the value of knowing the score.
    """

    DEFAULTS: dict[str, float | str] = {
        "opp": "data/runs/two_k100.pt", "worlds": 40, "z": 1.0, "min": 75,
        "v": DEFAULT_V, "obj": "win",
    }

    def __init__(
        self,
        net: NetPeggingPolicy,
        opp_model: NetOppHandModel,
        v: np.ndarray,
        worlds: int = 40,
        z: float = 1.0,
        min_score: int = 75,
        objective: str = "win",
    ) -> None:
        super().__init__()
        if objective not in ("win", "points"):
            raise ValueError(f"unknown objective {objective!r}")
        self.net, self.opp_model, self.v = net, opp_model, v
        self.worlds, self.z, self.min_score = worlds, z, min_score
        self.objective = objective

    @classmethod
    def load(cls, spec: str) -> ScoreAwarePeggingPolicy:
        path, o = parse_options(spec, cls.DEFAULTS)
        return cls(
            NetPeggingPolicy.load(path),
            NetOppHandModel.load(str(o["opp"])),
            np.load(str(o["v"])),
            int(o["worlds"]), float(o["z"]), int(o["min"]), str(o["obj"]),
        )

    def reseed(self, seed: int | str) -> None:
        super().reseed(seed)
        self.net.reseed(f"{seed}/net")

    def choose(self, view: PegView) -> int:
        default = self.net.choose(view)
        if (
            max(view.score, view.opp_score) < self.min_score
            or len({face(c) for c in view.legal}) < 2
        ):
            return default
        wins, margins, faces = peg_values(
            view, self.worlds, self.net, self.opp_model, self.v, self.rng
        )
        values = wins if self.objective == "win" else margins
        pick = faces[pick_candidate(values, faces.index(face(default)), self.z)]
        if pick == face(default):
            return default
        return next(c for c in view.legal if face(c) == pick)



# ------------------------------------------------------------
#  Tuning the number of samples and the threshold, on states with a known answer

GRID_N = (20, 40, 80)
GRID_Z = (0.0, 0.5, 1.0, 1.5, 2.0)


def _tune_init(discard_path: str, peg_path: str, v_path: str) -> None:
    _peg_init(discard_path, peg_path, v_path)
    discard, peg, v = _WORKER["discard"], _WORKER["peg"], _WORKER["v"]
    opp = _WORKER["opp"]
    assert isinstance(discard, NetDiscardPolicy) and isinstance(peg, NetPeggingPolicy)
    assert isinstance(v, np.ndarray) and isinstance(opp, NetOppHandModel)
    _WORKER["sa_discard"] = ScoreAwareDiscardPolicy(discard, peg, v, min_score=0)
    _WORKER["sa_peg"] = ScoreAwarePeggingPolicy(peg, opp, v, min_score=0)


def _tune_task(task: tuple[str, int, int, int, int, int]) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Gains (pp) and switch indicators over GRID_N x GRID_Z for one state, and the weight."""
    kind, index, seed, dealer_score, nondealer_score, role = task
    me_dealer = bool(role)
    score, opp_score = (
        (dealer_score, nondealer_score) if me_dealer else (nondealer_score, dealer_score)
    )
    rng = np.random.default_rng([seed, index, 5])
    gain = np.zeros((len(GRID_N), len(GRID_Z)))
    switched = np.zeros_like(gain)
    if kind == "discard":
        pol = _WORKER["sa_discard"]
        assert isinstance(pol, ScoreAwareDiscardPolicy)
        pol.reseed(f"{seed}/{index}/{role}")
        hand = tuple(int(c) for c in rng.permutation(52)[:6])
        view = DiscardView(hand, me_dealer, score, opp_score)
        throws = pol.candidates(view)
        wins = pol.simulate(view, throws, max(GRID_N))
        truth = pol.simulate(view, throws, 300).mean(axis=0)
        base, weight = 0, 1
    else:
        pol_p = _WORKER["sa_peg"]
        assert isinstance(pol_p, ScoreAwarePeggingPolicy)
        discard = _WORKER["discard"]
        assert isinstance(discard, DiscardPolicy)
        found = peg_decision((seed, index, role, 7), score, opp_score, me_dealer, discard, pol_p.net)
        if found is None:
            return None
        pview, weight = found
        pol_p.reseed(f"{seed}/{index}/{role}")
        rr = random.Random(f"{seed}/{index}/{role}")
        wins, _, faces = peg_values(pview, max(GRID_N), pol_p.net, pol_p.opp_model, pol_p.v, rr)
        truth = peg_values(pview, 600, pol_p.net, pol_p.opp_model, pol_p.v, rr)[0].mean(axis=0)
        base = faces.index(face(pol_p.net.choose(pview)))
    for i, n in enumerate(GRID_N):
        for j, z in enumerate(GRID_Z):
            k = pick_candidate(wins[:n], base, z)
            gain[i, j] = 100 * (truth[k] - truth[base])
            switched[i, j] = k != base
    return gain, switched, weight


def run_tune(args: argparse.Namespace) -> None:
    start = time.time()
    visits = np.load(args.visits).astype(int)
    sd, sn = visits[:, 0], visits[:, 1]
    lo, hi = args.zone
    candidates = np.flatnonzero((sd >= lo) & (sd <= hi) & (sn >= lo) & (sn <= hi))
    pick = np.random.default_rng(args.seed).choice(candidates, size=args.states, replace=False)
    tasks = [
        (args.kind, i, args.seed, int(sd[p]), int(sn[p]), role)
        for i, p in enumerate(pick)
        for role in (0, 1)
    ]
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        args.workers, initializer=_tune_init, initargs=(args.discard, args.peg, args.v)
    ) as pool:
        rows = [r for r in pool.imap_unordered(_tune_task, tasks, chunksize=2) if r is not None]
    gains = np.array([r[0] for r in rows])
    flips = np.array([r[1] for r in rows])
    weights = np.array([r[2] for r in rows], dtype=float)
    scale = weights[:, None, None] if args.kind == "peg" else 1.0
    total = len(tasks)
    print(f"{args.kind}: {len(rows)} decisions of {total} tried, zone {lo}-{hi} ({time.time() - start:.0f}s)")
    print("gain over the default choice, pp per decision (per hand for pegging); switch rate")
    print("  samples  " + "  ".join(f"z={z:<4}" + " " * 14 for z in GRID_Z))
    g = gains * scale
    if args.kind == "peg":
        g = np.concatenate([g, np.zeros((total - len(rows),) + gains.shape[1:])])
    for i, n in enumerate(GRID_N):
        cells = []
        for j in range(len(GRID_Z)):
            x = g[:, i, j]
            cells.append(f"{x.mean():+.3f}±{1.96 * x.std(ddof=1) / np.sqrt(len(x)):.3f} ({flips[:, i, j].mean():.2f})")
        print(f"  {n:7d}  " + "  ".join(cells))



def analyse(path: str) -> None:
    d = np.load(path)
    w, m, model = d["wins"], d["margins"], d["model"]
    n = len(model)
    idx = np.arange(n)

    out = {}
    both = (0, 1), (1, 0)
    wp_pick = [np.nanargmax(w[:, a], axis=1) for a, _ in both]
    ep_pick = [np.nanargmax(m[:, a], axis=1) for a, _ in both]
    for name, picks, base in (
        ("WP choice vs the model's choice", wp_pick, [model, model]),
        ("points choice (with pegging) vs the model's choice", ep_pick, [model, model]),
        ("WP choice vs points choice (the score-aware part)", wp_pick, ep_pick),
    ):
        g = np.mean(
            [w[idx, b, p] - w[idx, b, q] for (a, b), p, q in zip(both, picks, base)], axis=0
        )
        out[name] = g
    print(f"{n} (state, seat) evaluations, {int(d['samples'])} samples x 2 replicates each")
    print("win-probability gain per decision, pp (choose on one replicate, score on the other):")
    for name, g in out.items():
        print(f"  {name}: {100 * g.mean():+.3f} ± {196 * g.std(ddof=1) / np.sqrt(n):.3f}")
    differ = np.mean([wp_pick[0] != model, wp_pick[1] != model])
    differ_ep = np.mean([wp_pick[0] != ep_pick[0], wp_pick[1] != ep_pick[1]])
    noise = np.mean(ep_pick[0] != ep_pick[1])
    print(f"WP pick differs from the model's: {differ:.3f}")
    print(f"WP pick differs from the points pick: {differ_ep:.3f}  "
          f"(points pick differs between replicates, pure noise: {noise:.3f})")
    if "decisions" in d.files:
        # each hand contributes one random decision out of `decisions`; hands without
        # a choice contribute nothing
        per_hand = {}
        for name, g in out.items():
            x = np.concatenate([g * d["decisions"], np.zeros(int(d["attempted"]) - n)])
            per_hand[name] = x
            print(f"per hand (all decisions of a hand, hands with no choice count 0): "
                  f"{name}: {100 * x.mean():+.3f} ± {196 * x.std(ddof=1) / np.sqrt(len(x)):.3f} pp")
    for role, label in ((0, "non-dealer"), (1, "dealer")):
        r = d["role"] == role
        g = out["WP choice vs the model's choice"][r]
        h = out["WP choice vs points choice (the score-aware part)"][r]
        print(f"  {label}: WP vs model {100 * g.mean():+.3f} ± {196 * g.std(ddof=1) / np.sqrt(r.sum()):.3f} pp, "
              f"WP vs points {100 * h.mean():+.3f} ± {196 * h.std(ddof=1) / np.sqrt(r.sum()):.3f} pp")


def run_analyse(args: argparse.Namespace) -> None:
    analyse(args.inp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("discard-headroom", help="gain from a win-probability discard")
    p.add_argument("--states", type=int, default=500)
    p.add_argument("--samples", type=int, default=100, help="per replicate")
    p.add_argument("--zone", type=int, nargs=2, metavar=("LO", "HI"), help="both scores in LO..HI")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--visits", default="data/winprob/visits.npy")
    p.add_argument("--v", default="data/winprob/v.npy")
    p.add_argument("--discard", default="data/runs/two_k100.pt")
    p.add_argument("--peg", default="data/pegging/rl/net_w100_post.pt")
    p.add_argument("--out", default="data/winprob/discard_headroom.npz")
    p.set_defaults(run=run_discard)
    p = sub.add_parser("peg-headroom", help="gain from a win-probability pegging play")
    p.add_argument("--states", type=int, default=500)
    p.add_argument("--worlds", type=int, default=100, help="per replicate")
    p.add_argument("--zone", type=int, nargs=2, metavar=("LO", "HI"), help="both scores in LO..HI")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--visits", default="data/winprob/visits.npy")
    p.add_argument("--v", default="data/winprob/v.npy")
    p.add_argument("--discard", default="data/runs/two_k100.pt")
    p.add_argument("--peg", default="data/pegging/rl/net_w100_post.pt")
    p.add_argument("--out", default="data/winprob/peg_headroom.npz")
    p.set_defaults(run=run_peg)
    p = sub.add_parser("tune", help="gain of the policies for each sample count and threshold")
    p.add_argument("kind", choices=("discard", "peg"))
    p.add_argument("--states", type=int, default=300)
    p.add_argument("--zone", type=int, nargs=2, default=(80, 115), metavar=("LO", "HI"))
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--visits", default="data/winprob/visits.npy")
    p.add_argument("--v", default=DEFAULT_V)
    p.add_argument("--discard", default="data/runs/two_k100.pt")
    p.add_argument("--peg", default=DEFAULT_PEG)
    p.set_defaults(run=run_tune)
    p = sub.add_parser("analyse", help="report a saved run")
    p.add_argument("inp")
    p.set_defaults(run=run_analyse)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
