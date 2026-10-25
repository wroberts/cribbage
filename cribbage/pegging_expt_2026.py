#!/usr/bin/env python

"""
pegging_expt_2026.py

How much is the pegging phase (`play_card`) worth, and does search help?

The discard policy is held fixed (a trained DiscardNet, `model:PATH`) and
only the pegging policy varies, along a ladder of strength:

- `random`: any legal card.
- `greedy`: the card that scores most right now (Simple's rule).
- `greedy2`: greedy, less the opponent's expected best reply (analytic,
  under a uniform guess at the opponent's hand).
- `pimc-roll:N[:R]`: perfect-information Monte Carlo.  Sample N worlds
  (the opponent's hidden cards drawn from the unseen ones) and score each
  candidate card by R greedy-vs-greedy rollouts in each world.
- `pimc-solve:N`: as above, but each world is solved exactly (minimax over
  the perfect-information pegging game, with a transposition table).
- `ismcts:I`: single-observer information-set MCTS.  Each of I iterations
  samples a world, but the search tree is shared between worlds, and
  indexed by the moves made, so that statistics reflect the hidden
  information rather than any one world.
- `oracle`: minimax in the true world, i.e. it peeks at the opponent's
  hand.  Not a real player; it bounds what hidden information is worth.
- `first`: the first legal card (deterministic; used for testing).

A `/opp` suffix (e.g. `pimc-solve:20/opp`) makes the sampler assume the
opponent is a `greedy` player: worlds in which the opponent would have
played differently earlier in the round are excluded.  Sampling always
respects "said go" (the opponent cannot play any of their cards).

Pegging is measured in isolation (`peg-bench`, `ladder`, `cost`): fixed
deals are made once, both players keep the cards the discard model would
keep, and each deal is played twice with the seats swapped, so that hand
luck and the dealer/non-dealer asymmetry cancel.  The score is the
pegging-points margin per hand.  `match` plays full games between
`DISCARD+POLICY` players, and `calibrate` relates points per hand to win
rate.

Run `python -m cribbage.pegging_expt_2026 --help` for the commands.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from cribbage.cards import CARD_VALUES, Card, cards_worth
from cribbage.discard_expt_2026 import (
    DISCARD_IDX,
    KEEP_IDX,
    encode_hands,
    load_model,
    make_player,
    predict_values,
)
from cribbage.game import Game, compare_players
from cribbage.player import CribbagePlayer
from cribbage.round import Round

try:
    from cribbage._cribbage_score import score_play
except ImportError:
    from cribbage.cribbage_score import score_play

# the value of each card towards the count
WORTH = [CARD_VALUES[c % 13] for c in range(52)]

# ------------------------------------------------------------
#  The pegging game


class PegState:
    """
    The pegging phase of one round, as a mutable game state.  Players are
    numbered 0 and 1; a state is always either terminal (`done`) or waiting
    for `turn` to choose a card.  `play` and `advance` follow the logic of
    `Round.play` exactly (go, 31, last card), without the `Game` and
    `CribbagePlayer` machinery, so that search can copy and replay it
    quickly.  Cards are 0..51, so a hand never repeats a card.
    """

    __slots__ = ("hands", "seq", "count", "turn", "is_go", "last", "points", "done")

    def __init__(
        self,
        hands: list[list[Card]],
        turn: int,
        seq: Sequence[Card] = (),
        is_go: bool = False,
        last: int | None = None,
    ) -> None:
        self.hands = hands
        self.seq: list[Card] = list(seq)
        self.count = cards_worth(self.seq)
        self.turn = turn
        self.is_go = is_go
        self.last = last
        # pegging points scored so far by each player
        self.points = [0, 0]
        self.done = False

    @classmethod
    def start(cls, hands: Sequence[Sequence[Card]], dealer_idx: int) -> PegState:
        """The state at the start of a round's play: the non-dealer leads."""
        state = cls([list(hands[0]), list(hands[1])], 1 - dealer_idx)
        state.advance()
        return state

    def copy(self) -> PegState:
        other = PegState.__new__(PegState)
        other.hands = [self.hands[0][:], self.hands[1][:]]
        other.seq = self.seq[:]
        other.count = self.count
        other.turn = self.turn
        other.is_go = self.is_go
        other.last = self.last
        other.points = self.points[:]
        other.done = self.done
        return other

    def margin(self) -> int:
        """Points scored so far by player 0, less those of player 1."""
        return self.points[0] - self.points[1]

    def legal(self) -> list[int]:
        """Indices into the hand of `turn` of the cards that may be played."""
        room = 31 - self.count
        return [i for i, c in enumerate(self.hands[self.turn]) if WORTH[c] <= room]

    def advance(self) -> None:
        """Resolves go calls and the ends of sequences, up to the next choice."""
        while True:
            room = 31 - self.count
            can_play = any(WORTH[c] <= room for c in self.hands[self.turn])
            if (self.is_go and not can_play) or self.count == 31:
                # the sequence is over: the last player scores 2 for 31, else 1
                assert self.last is not None
                self.points[self.last] += 2 if self.count == 31 else 1
                self.seq = []
                self.count = 0
                self.is_go = False
                if not self.hands[0] and not self.hands[1]:
                    self.done = True
                    return
                self.turn = 1 - self.last
                self.last = None
            elif not can_play:
                # "go": the other player continues
                self.turn = 1 - self.turn
                self.is_go = True
            else:
                return

    def play(self, idx: int) -> None:
        """The player to move plays `hand[idx]` (which must be legal)."""
        turn = self.turn
        card = self.hands[turn].pop(idx)
        self.seq.append(card)
        self.count += WORTH[card]
        self.last = turn
        if not self.hands[1 - turn]:
            self.is_go = True
        self.points[turn] += score_play(self.seq)
        if not self.is_go and self.count != 31:
            self.turn = 1 - turn
        self.advance()


# ------------------------------------------------------------
#  What a player sees, and sampling the rest


@dataclass
class PegObs:
    """Everything a pegging policy may use at one decision."""

    hand: list[Card]
    seq: list[Card]
    played: set[Card]
    is_go: bool
    legal: list[int]
    discards: tuple[Card, ...]
    is_dealer: bool
    # (sequence so far, card) for each opponent play whose context is known
    opp_plays: list[tuple[tuple[Card, ...], Card]] = field(default_factory=list)
    # the opponent's actual hand; set only for the oracle
    opp_hand: list[Card] | None = None

    @property
    def num_opp(self) -> int:
        """Cards left in the opponent's hand (4 less their plays so far)."""
        return 9 - len(self.played) - len(self.hand)

    def unseen(self) -> list[Card]:
        """Cards that may be in the opponent's hand or in the crib."""
        known = self.played.union(self.hand, self.discards)
        return [c for c in range(52) if c not in known]


def distinct_moves(state: PegState) -> dict[int, int]:
    """
    The legal moves of the player to move, as {face: hand index}.  Equal
    faces are interchangeable in pegging, so each is listed once.
    """
    hand = state.hands[state.turn]
    moves: dict[int, int] = {}
    for i in state.legal():
        moves.setdefault(hand[i] % 13, i)
    return moves


class WorldSampler:
    """
    Draws worlds consistent with an observation: the opponent's hand is
    sampled from the unseen cards (the searcher is player 0).  Cards the
    opponent cannot hold are excluded: when they said go, any card that
    would have been playable, and with `opp_aware`, any card a greedy
    opponent would have played in preference at an earlier decision.
    """

    def __init__(self, obs: PegObs, opp_aware: bool = False) -> None:
        self.obs = obs
        self.num_opp = obs.num_opp
        pool = obs.unseen()
        if obs.is_go and self.num_opp:
            # the opponent said go, so none of their cards fits
            room = 31 - cards_worth(obs.seq)
            pool = [c for c in pool if WORTH[c] > room]
        self.excluded: set[Card] = set()
        if opp_aware:
            self.excluded = self._greedy_excluded(obs, obs.unseen())
        narrowed = [c for c in pool if c not in self.excluded]
        # fall back if the inferences contradict each other
        self.pool = narrowed if len(narrowed) >= self.num_opp else pool
        if len(self.pool) < self.num_opp:
            self.pool = obs.unseen()

    @staticmethod
    def _greedy_excluded(obs: PegObs, pool: list[Card]) -> set[Card]:
        excluded: set[Card] = set()
        for context, card in obs.opp_plays:
            room = 31 - cards_worth(context)
            chosen = score_play([*context, card])
            for c in pool:
                if WORTH[c] <= room and score_play([*context, c]) > chosen:
                    excluded.add(c)
        return excluded

    def build(self, opp_hand: Sequence[Card]) -> PegState:
        """The world in which the opponent holds `opp_hand`."""
        obs = self.obs
        last = None if not obs.seq else (0 if obs.is_go else 1)
        return PegState([obs.hand[:], list(opp_hand)], 0, obs.seq, obs.is_go, last)

    def sample(self, rng: random.Random) -> PegState:
        return self.build(rng.sample(self.pool, self.num_opp))


# ------------------------------------------------------------
#  Rollout move rules


def first_move(state: PegState, rng: random.Random) -> int:
    return state.legal()[0]


def random_move(state: PegState, rng: random.Random) -> int:
    return rng.choice(state.legal())


def greedy_move(state: PegState, rng: random.Random) -> int:
    """The card that scores most now; ties are broken at random."""
    seq = state.seq
    best = -1
    best_idxs: list[int] = []
    for i in state.legal():
        seq.append(state.hands[state.turn][i])
        value = score_play(seq)
        seq.pop()
        if value > best:
            best = value
            best_idxs = [i]
        elif value == best:
            best_idxs.append(i)
    return best_idxs[0] if len(best_idxs) == 1 else rng.choice(best_idxs)


def rollout(
    state: PegState, move: Callable[[PegState, random.Random], int], rng: random.Random
) -> None:
    """Plays `state` out to the end of the round with `move` for both players."""
    while not state.done:
        state.play(move(state, rng))


# ------------------------------------------------------------
#  Exact solution of the perfect-information game


class Solver:
    """
    Minimax value (player 0's margin from here on) of a pegging state in
    which both hands are known.  Only faces matter to the future, so
    states are keyed by faces and equal cards are tried once.  The table
    is valid across worlds and rounds.
    """

    MAX_ENTRIES = 3_000_000

    def __init__(self) -> None:
        self.memo: dict[tuple[object, ...], int] = {}

    def value(self, state: PegState) -> int:
        if state.done:
            return 0
        if len(self.memo) > self.MAX_ENTRIES:
            self.memo.clear()
        key = (
            tuple(sorted(c % 13 for c in state.hands[0])),
            tuple(sorted(c % 13 for c in state.hands[1])),
            tuple(c % 13 for c in state.seq),
            state.turn,
            state.is_go,
            state.last,
        )
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        maximise = state.turn == 0
        best = -99 if maximise else 99
        for idx in distinct_moves(state).values():
            best = self.after(state, idx, maximise, best)
        self.memo[key] = best
        return best

    def after(self, state: PegState, idx: int, maximise: bool, best: int) -> int:
        """Updates `best` with the value of playing `idx`."""
        child = state.copy()
        before = child.margin()
        child.play(idx)
        value = child.margin() - before + self.value(child)
        return max(best, value) if maximise else min(best, value)

    def move_values(self, state: PegState) -> dict[int, int]:
        """The value of each distinct legal move (by hand index)."""
        values = {}
        for idx in distinct_moves(state).values():
            child = state.copy()
            child.play(idx)
            values[idx] = child.margin() - state.margin() + self.value(child)
        return values


# ------------------------------------------------------------
#  Policies


class PegPolicy:
    """Chooses which card to play; `choose` returns an index into the hand."""

    needs_oracle = False

    def __init__(self) -> None:
        self.rng = random.Random(0)

    def reseed(self, seed: int | str) -> None:
        self.rng = random.Random(seed)

    def choose(self, obs: PegObs) -> int:
        raise NotImplementedError()

    def argmax(self, values: dict[int, float]) -> int:
        """The key with the highest value; ties broken at random."""
        top = max(values.values())
        best = [k for k, v in values.items() if v == top]
        return best[0] if len(best) == 1 else self.rng.choice(best)


class FirstPeg(PegPolicy):
    def choose(self, obs: PegObs) -> int:
        return obs.legal[0]


class RandomPeg(PegPolicy):
    def choose(self, obs: PegObs) -> int:
        return self.rng.choice(obs.legal)


class GreedyPeg(PegPolicy):
    """Simple's rule: the card with the best immediate score."""

    def choose(self, obs: PegObs) -> int:
        values = {i: float(score_play([*obs.seq, obs.hand[i]])) for i in obs.legal}
        return self.argmax(values)


class Greedy2Peg(PegPolicy):
    """
    Greedy, less the opponent's best immediate reply.  Their hand is
    n cards from the U unseen; the best playable reply of rank k (0 is
    best) is theirs to play with probability C(U-1-k, n-1) / C(U, n).
    """

    def choose(self, obs: PegObs) -> int:
        if len(obs.legal) == 1:
            return obs.legal[0]
        pool = obs.unseen()
        n = obs.num_opp
        total = math.comb(len(pool), n) if n else 1
        values: dict[int, float] = {}
        for i in obs.legal:
            card = obs.hand[i]
            immediate = score_play([*obs.seq, card])
            reply = 0.0
            if n and not obs.is_go:
                room = 31 - cards_worth(obs.seq) - WORTH[card]
                scores = sorted(
                    (score_play([*obs.seq, card, x]) for x in pool if WORTH[x] <= room),
                    reverse=True,
                )
                reply = (
                    sum(
                        s * math.comb(len(pool) - 1 - k, n - 1)
                        for k, s in enumerate(scores)
                    )
                    / total
                )
            values[i] = immediate - reply
        return self.argmax(values)


class PimcRollout(PegPolicy):
    """Scores each card by greedy rollouts in sampled worlds."""

    def __init__(self, worlds: int, rollouts: int = 1, opp_aware: bool = False) -> None:
        super().__init__()
        self.worlds = worlds
        self.rollouts = rollouts
        self.opp_aware = opp_aware

    def values(self, obs: PegObs) -> dict[int, float]:
        """Mean rollout margin of each distinct legal card (by hand index)."""
        sampler = WorldSampler(obs, self.opp_aware)
        totals: dict[int, float] = {}
        for _ in range(self.worlds):
            world = sampler.sample(self.rng)
            for idx in distinct_moves(world).values():
                for _ in range(self.rollouts):
                    child = world.copy()
                    child.play(idx)
                    rollout(child, greedy_move, self.rng)
                    totals[idx] = totals.get(idx, 0.0) + child.margin()
        runs = self.worlds * self.rollouts
        return {idx: total / runs for idx, total in totals.items()}

    def choose(self, obs: PegObs) -> int:
        if len(obs.legal) == 1:
            return obs.legal[0]
        return self.argmax(self.values(obs))


class PimcSolve(PegPolicy):
    """Averages the exact minimax value of each card over sampled worlds."""

    def __init__(self, worlds: int, opp_aware: bool = False) -> None:
        super().__init__()
        self.worlds = worlds
        self.opp_aware = opp_aware
        self.solver = Solver()

    def choose(self, obs: PegObs) -> int:
        if len(obs.legal) == 1:
            return obs.legal[0]
        sampler = WorldSampler(obs, self.opp_aware)
        totals: dict[int, float] = {}
        for _ in range(self.worlds):
            world = sampler.sample(self.rng)
            for idx, value in self.solver.move_values(world).items():
                totals[idx] = totals.get(idx, 0.0) + value
        return self.argmax(totals)


class OraclePeg(PegPolicy):
    """Solves the true world: it is shown the opponent's hand."""

    needs_oracle = True

    def __init__(self) -> None:
        super().__init__()
        self.solver = Solver()

    def choose(self, obs: PegObs) -> int:
        if len(obs.legal) == 1:
            return obs.legal[0]
        assert obs.opp_hand is not None
        world = WorldSampler(obs).build(obs.opp_hand)
        return self.argmax(dict(self.solver.move_values(world)))


class _Node:
    """A child in the ISMCTS tree: statistics of the move leading to it."""

    __slots__ = ("visits", "total", "avail", "children")

    def __init__(self) -> None:
        self.visits = 0
        # sum of rewards, from player 0's point of view
        self.total = 0.0
        # times this move was legal when its parent was visited
        self.avail = 0
        self.children: dict[int, _Node] = {}


class Ismcts(PegPolicy):
    """
    Single-observer information-set MCTS.  Moves are identified by card
    face, so a node stands for everything that could have led to it.
    UCB1 uses the number of times a move was available, as the legal moves
    vary from world to world.  Rewards are the final margin / `SCALE`.
    """

    SCALE = 10.0

    def __init__(
        self, iterations: int, explore: float = 0.7, opp_aware: bool = False
    ) -> None:
        super().__init__()
        self.iterations = iterations
        self.explore = explore
        self.opp_aware = opp_aware

    def choose(self, obs: PegObs) -> int:
        if len(obs.legal) == 1:
            return obs.legal[0]
        sampler = WorldSampler(obs, self.opp_aware)
        root = _Node()
        root_moves: dict[int, int] = {}
        for _ in range(self.iterations):
            world = sampler.sample(self.rng)
            if not root_moves:
                root_moves = distinct_moves(world)
            self._iterate(root, world)
        best = {root_moves[f]: float(c.visits) for f, c in root.children.items()}
        return self.argmax(best)

    def _iterate(self, root: _Node, state: PegState) -> None:
        node = root
        path: list[_Node] = []
        while not state.done:
            moves = distinct_moves(state)
            for face in moves:
                child = node.children.get(face)
                if child is not None:
                    child.avail += 1
            untried = [f for f in moves if f not in node.children]
            if untried:
                face = self.rng.choice(untried)
                node.children[face] = _Node()
                node.children[face].avail = 1
                path.append(node.children[face])
                state.play(moves[face])
                break
            sign = 1.0 if state.turn == 0 else -1.0
            best_face = -1
            best_value = -math.inf
            for face in moves:
                child = node.children[face]
                bonus = math.sqrt(math.log(child.avail) / child.visits)
                value = sign * child.total / child.visits + self.explore * bonus
                if value > best_value:
                    best_value = value
                    best_face = face
            node = node.children[best_face]
            path.append(node)
            state.play(moves[best_face])
        rollout(state, greedy_move, self.rng)
        reward = state.margin() / self.SCALE
        for node in path:
            node.visits += 1
            node.total += reward


def make_policy(spec: str) -> PegPolicy:
    """
    Builds a pegging policy from its name, e.g. `greedy`, `pimc-roll:20:2`,
    `pimc-solve:50/opp`, `ismcts:400`, `oracle`.
    """
    if spec.startswith("net:"):
        # a network trained by pegging_rl_expt_2026 (the path may contain "/")
        from cribbage.pegging_rl_expt_2026 import NetPeg

        return NetPeg.load(spec[len("net:") :])
    base, *flags = spec.split("/")
    if set(flags) - {"opp"}:
        raise ValueError(f"unknown flag in {spec!r}")
    opp = "opp" in flags
    name, *args = base.split(":")
    nums = [int(a) for a in args]
    if name in ("first", "random", "greedy", "greedy2", "oracle") and (nums or opp):
        raise ValueError(f"{name} takes no arguments: {spec!r}")
    if name == "first":
        return FirstPeg()
    if name == "random":
        return RandomPeg()
    if name == "greedy":
        return GreedyPeg()
    if name == "greedy2":
        return Greedy2Peg()
    if name == "oracle":
        return OraclePeg()
    if name == "pimc-roll" and 1 <= len(nums) <= 2:
        return PimcRollout(nums[0], nums[1] if len(nums) > 1 else 1, opp_aware=opp)
    if name == "pimc-solve" and len(nums) == 1:
        return PimcSolve(nums[0], opp_aware=opp)
    if name == "ismcts" and len(nums) == 1:
        return Ismcts(nums[0], opp_aware=opp)
    raise ValueError(f"unknown pegging policy {spec!r}")


# ------------------------------------------------------------
#  The player


class PegPlayer(CribbagePlayer):
    """
    Discards with another player and pegs with a `PegPolicy`.  Remembers
    its own discards (which `play_card` is not told) and reconstructs the
    opponent's earlier plays from the sequences it sees.
    """

    def __init__(self, discarder: CribbagePlayer, policy: PegPolicy) -> None:
        super().__init__()
        self.discarder = discarder
        self.policy = policy
        # reads the opponent's hand, for the oracle (set by the harness)
        self.peek: Callable[[], list[Card]] | None = None
        self.seconds = 0.0
        self.decisions = 0
        self.begin_round(())

    def begin_round(self, discards: Sequence[Card]) -> None:
        self.discards = tuple(discards)
        self.opp_plays: list[tuple[tuple[Card, ...], Card]] = []
        # the sequence as it stood after this player's last play
        self.after_mine: list[Card] = []

    def discard(
        self,
        is_dealer: bool,
        hand: list[Card],
        player_score: int,
        opponent_score: int,
    ) -> list[int]:
        idxs = self.discarder.discard(is_dealer, hand, player_score, opponent_score)
        self.begin_round([hand[i] for i in idxs])
        return idxs

    def play_card(
        self,
        is_dealer: bool,
        hand: list[Card],
        played_cards: set[Card],
        is_go: bool,
        linear_play: list[Card],
        player_score: int,
        opponent_score: int,
        legal_moves: list[int],
    ) -> int:
        # whatever follows our last play, up to now, was played by the
        # opponent; if the sequence has since ended, their order is lost
        mine = self.after_mine
        if linear_play[: len(mine)] == mine:
            for i in range(len(mine), len(linear_play)):
                self.opp_plays.append((tuple(linear_play[:i]), linear_play[i]))
        obs = PegObs(
            hand,
            linear_play,
            played_cards,
            is_go,
            legal_moves,
            self.discards,
            is_dealer,
            self.opp_plays,
        )
        if self.policy.needs_oracle:
            assert self.peek is not None
            obs.opp_hand = list(self.peek())
        start = time.perf_counter()
        idx = self.policy.choose(obs)
        self.seconds += time.perf_counter() - start
        self.decisions += 1
        self.after_mine = [*linear_play, hand[idx]]
        if cards_worth(self.after_mine) == 31:
            self.after_mine = []
        return idx


def make_peg_player(spec: str, discarder: CribbagePlayer | None = None) -> PegPlayer:
    """A PegPlayer for pegging policy `spec`; the discarder is irrelevant by default."""
    return PegPlayer(discarder or make_player("random"), make_policy(spec))


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


def _hand_reader(rnd: Round, seat: int) -> Callable[[], list[Card]]:
    return lambda: rnd.hands[seat]


def play_pegging(
    players: Sequence[PegPlayer],
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
    rnd = Round(game, list(players), dealer_idx=1)
    rnd.hands = [[int(c) for c in kept[0]], [int(c) for c in kept[1]]]
    rnd.played_cards = {int(starter)}
    for seat, player in enumerate(players):
        player.begin_round([int(c) for c in thrown[seat]])
        if player.policy.needs_oracle:
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
    a, b = make_peg_player(spec_a), make_peg_player(spec_b)
    cols = np.zeros((len(deals), 4), dtype=np.int64)
    for i in range(len(deals)):
        for player, who in ((a, "a"), (b, "b")):
            player.policy.reseed(f"{seed}/{first + i}/{who}")
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
        method = "spawn" if "net:" in spec_a + spec_b else None
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


def make_full_player(spec: str, seed: int) -> PegPlayer:
    """A player from `DISCARD+POLICY`, e.g. `model:data/runs/two_k100.pt+greedy`."""
    discard, plus, peg = spec.partition("+")
    if not plus:
        raise ValueError(f"expected DISCARD+POLICY: {spec!r}")
    player = PegPlayer(make_player(discard), make_policy(peg))
    player.policy.reseed(seed)
    return player


def _match_chunk(task: tuple[str, str, int, int]) -> int:
    spec_a, spec_b, games, seed = task
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    players = [make_full_player(spec_a, seed), make_full_player(spec_b, seed + 1)]
    return compare_players(list(players), games)[0]


def play_match(spec_a: str, spec_b: str, games: int, seed: int, workers: int) -> float:
    """A's win rate over `games` full games (chunked, so independent of `workers`)."""
    chunk = 100
    sizes = [chunk] * (games // chunk) + ([games % chunk] if games % chunk else [])
    tasks = [(spec_a, spec_b, n, seed * 100000 + i) for i, n in enumerate(sizes)]
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            wins = sum(pool.map(_match_chunk, tasks))
    else:
        wins = sum(_match_chunk(t) for t in tasks)
    return wins / games


def win_ci(rate: float, games: int) -> float:
    return 1.96 * math.sqrt(rate * (1 - rate) / games)


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
