"""
pegging_search.py

Pegging policies that search.  They sample worlds consistent with a
`PegView` (the opponent's hidden cards drawn from the unseen ones), play
them out with `PegState`, and pick the card that does best.

- `pimc-roll:N[:R]`: perfect-information Monte Carlo.  Sample N worlds and
  score each candidate card by R greedy-vs-greedy rollouts in each world.
- `pimc-solve:N`: as above, but each world is solved exactly (minimax over
  the perfect-information pegging game, with a transposition table).
- `ismcts:I`: single-observer information-set MCTS.  Each of I iterations
  samples a world, but the search tree is shared between worlds, and
  indexed by the moves made, so that statistics reflect the hidden
  information rather than any one world.
- `oracle`: minimax in the true world, i.e. it peeks at the opponent's
  hand.  Not a real player; it bounds what hidden information is worth.

A `/opp` suffix (e.g. `pimc-solve:20/opp`) makes the sampler assume the
opponent is a `greedy` player: worlds in which the opponent would have
played differently earlier in the round are excluded.  Sampling always
respects "said go" (the opponent cannot play any of their cards).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from cribbage.cards import WORTH, Card, cards_worth, face
from cribbage.pegging import PegState, PegView
from cribbage.pegging_policy import PeggingPolicy
from cribbage.scoring import score_play

# ------------------------------------------------------------
#  What a player sees, and sampling the rest


def distinct_moves(state: PegState) -> dict[int, int]:
    """
    The legal moves of the player to move, as {face: hand index}.  Equal
    faces are interchangeable in pegging, so each is listed once.
    """
    hand = state.hands[state.turn]
    moves: dict[int, int] = {}
    for i in state.legal():
        moves.setdefault(face(hand[i]), i)
    return moves


class WorldSampler:
    """
    Draws worlds consistent with a view: the opponent's hand is sampled
    from the unseen cards (the searcher is player 0).  Cards the opponent
    cannot hold are excluded: when they said go, any card that would have
    been playable, and with `opp_aware`, any card a greedy opponent would
    have played in preference at an earlier decision.
    """

    def __init__(self, view: PegView, opp_aware: bool = False) -> None:
        self.view = view
        self.num_opp = view.opp_cards_left
        pool = view.unseen()
        if view.opp_said_go and self.num_opp:
            # the opponent said go, so none of their cards fits
            room = 31 - view.count
            pool = [c for c in pool if WORTH[c] > room]
        self.excluded: set[Card] = set()
        if opp_aware:
            self.excluded = self._greedy_excluded(view, view.unseen())
        narrowed = [c for c in pool if c not in self.excluded]
        # fall back if the inferences contradict each other
        self.pool = narrowed if len(narrowed) >= self.num_opp else pool
        if len(self.pool) < self.num_opp:
            self.pool = view.unseen()

    @staticmethod
    def _greedy_excluded(view: PegView, pool: list[Card]) -> set[Card]:
        excluded: set[Card] = set()
        for context, card in view.opp_plays:
            room = 31 - cards_worth(context)
            chosen = score_play([*context, card])
            for c in pool:
                if WORTH[c] <= room and score_play([*context, c]) > chosen:
                    excluded.add(c)
        return excluded

    def build(self, opp_hand: Sequence[Card]) -> PegState:
        """The world in which the opponent holds `opp_hand`."""
        return PegState.from_view(self.view, opp_hand)

    def sample(self, rng: random.Random) -> PegState:
        return self.build(rng.sample(self.pool, self.num_opp))


# ------------------------------------------------------------
#  Rollout move rules


def first_move(state: PegState, rng: random.Random) -> int:
    return state.legal()[0]


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
            tuple(sorted(face(c) for c in state.hands[0])),
            tuple(sorted(face(c) for c in state.hands[1])),
            tuple(face(c) for c in state.seq),
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


class PimcRolloutPeggingPolicy(PeggingPolicy):
    """Scores each card by greedy rollouts in sampled worlds."""

    def __init__(self, worlds: int, rollouts: int = 1, opp_aware: bool = False) -> None:
        super().__init__()
        self.worlds = worlds
        self.rollouts = rollouts
        self.opp_aware = opp_aware

    def values(self, view: PegView) -> dict[Card, float]:
        """Mean rollout margin of each distinct legal card."""
        sampler = WorldSampler(view, self.opp_aware)
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
        return {view.hand[idx]: total / runs for idx, total in totals.items()}

    def choose(self, view: PegView) -> Card:
        if len(view.legal) == 1:
            return view.legal[0]
        return self.argmax(self.values(view))


class PimcSolvePeggingPolicy(PeggingPolicy):
    """Averages the exact minimax value of each card over sampled worlds."""

    def __init__(self, worlds: int, opp_aware: bool = False) -> None:
        super().__init__()
        self.worlds = worlds
        self.opp_aware = opp_aware
        self.solver = Solver()

    def choose(self, view: PegView) -> Card:
        if len(view.legal) == 1:
            return view.legal[0]
        sampler = WorldSampler(view, self.opp_aware)
        totals: dict[Card, float] = {}
        for _ in range(self.worlds):
            world = sampler.sample(self.rng)
            for idx, value in self.solver.move_values(world).items():
                card = view.hand[idx]
                totals[card] = totals.get(card, 0.0) + value
        return self.argmax(totals)


class OraclePeggingPolicy(PeggingPolicy):
    """Solves the true world: it is shown the opponent's hand."""

    needs_oracle = True

    def __init__(self) -> None:
        super().__init__()
        self.solver = Solver()

    def choose(self, view: PegView) -> Card:
        if len(view.legal) == 1:
            return view.legal[0]
        assert view.opp_hand is not None
        world = WorldSampler(view).build(view.opp_hand)
        values = self.solver.move_values(world)
        return self.argmax({view.hand[idx]: float(v) for idx, v in values.items()})


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


class IsmctsPeggingPolicy(PeggingPolicy):
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

    def choose(self, view: PegView) -> Card:
        if len(view.legal) == 1:
            return view.legal[0]
        sampler = WorldSampler(view, self.opp_aware)
        root = _Node()
        root_moves: dict[int, int] = {}
        for _ in range(self.iterations):
            world = sampler.sample(self.rng)
            if not root_moves:
                root_moves = distinct_moves(world)
            self._iterate(root, world)
        best = {
            view.hand[root_moves[f]]: float(c.visits) for f, c in root.children.items()
        }
        return self.argmax(best)

    def _iterate(self, root: _Node, state: PegState) -> None:
        node = root
        path: list[_Node] = []
        while not state.done:
            moves = distinct_moves(state)
            for f in moves:
                child = node.children.get(f)
                if child is not None:
                    child.avail += 1
            untried = [f for f in moves if f not in node.children]
            if untried:
                f = self.rng.choice(untried)
                node.children[f] = _Node()
                node.children[f].avail = 1
                path.append(node.children[f])
                state.play(moves[f])
                break
            sign = 1.0 if state.turn == 0 else -1.0
            best_face = -1
            best_value = -math.inf
            for f in moves:
                child = node.children[f]
                bonus = math.sqrt(math.log(child.avail) / child.visits)
                value = sign * child.total / child.visits + self.explore * bonus
                if value > best_value:
                    best_value = value
                    best_face = f
            node = node.children[best_face]
            path.append(node)
            state.play(moves[best_face])
        rollout(state, greedy_move, self.rng)
        reward = state.margin() / self.SCALE
        for node in path:
            node.visits += 1
            node.total += reward
