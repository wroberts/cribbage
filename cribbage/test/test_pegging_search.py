#!/usr/bin/env python

"""
test_pegging_search.py

Test the pegging search: the exact solver, world sampling, and the
policies built on them.
"""

import random
from dataclasses import replace

import pytest
from cribbage.cards import WORTH
from cribbage.game import Game
from cribbage.pegging import PegState
from cribbage.pegging_bench import BenchPlayer, make_bench_player
from cribbage.pegging_policy import PeggingPolicy
from cribbage.pegging_search import (
    Solver,
    WorldSampler,
    distinct_moves,
    first_move,
    greedy_move,
    rollout,
)
from cribbage.player import CribbagePlayer
from cribbage.specs import make_pegging_policy, make_player
from cribbage.test.helpers import make_round, random_deal


def brute_force(state):
    """Plain minimax of player 0's margin, with no tables or shortcuts."""
    if state.done:
        return 0
    values = []
    for idx in state.legal():
        child = state.copy()
        before = child.margin()
        child.play(idx)
        values.append(child.margin() - before + brute_force(child))
    return max(values) if state.turn == 0 else min(values)


def test_solver_matches_brute_force():
    rng = random.Random(2)
    solver = Solver()
    for trial in range(60):
        hands, _ = random_deal(rng)
        # three cards each keeps the plain search quick
        state = PegState.start([hands[0][:3], hands[1][:3]], trial % 2)
        assert solver.value(state) == brute_force(state)


def test_solver_known_value():
    # the non-dealer holds a five and the dealer a ten: 5, 10 makes fifteen
    # (2), then the dealer also takes the go
    state = PegState.start([[4], [9]], 1)
    assert state.turn == 0
    assert Solver().value(state) == -3


def test_distinct_moves_merge_faces():
    state = PegState.start([[0, 13, 5, 30], [1, 2, 3, 4]], 1)
    # faces ace, ace, six, five: three distinct plays
    assert len(distinct_moves(state)) == 3


class Recorder(PeggingPolicy):
    """Plays the first legal card, recording what the sampler could know."""

    needs_oracle = True

    def __init__(self, game, seat, opp_aware, records):
        super().__init__()
        self.game, self.seat = game, seat
        self.opp_aware, self.records = opp_aware, records

    def choose(self, view):
        sampler = WorldSampler(view, self.opp_aware)
        world = sampler.build(view.opp_hand)
        # a world with the true hand, played out by first-legal moves
        rollout(world, first_move, self.rng)
        scores = self.game.scores
        self.records.append(
            (
                set(view.opp_hand) <= set(sampler.pool),
                len(view.opp_hand) == sampler.num_opp,
                world.margin(),
                scores[self.seat] - scores[1 - self.seat],
                self.seat,
            )
        )
        return view.legal[0]


@pytest.mark.parametrize("opp_greedy", [False, True])
def test_sampler_world_is_consistent(opp_greedy):
    """
    At every decision the true opponent hand is a possible sample, the
    reconstructed state predicts the rest of the round exactly, and (with
    a greedy opponent) the opponent-aware exclusions never remove one of
    their cards.
    """
    rng = random.Random(3)
    seen = 0
    for trial in range(200):
        hands, starter = random_deal(rng)
        game = Game([make_player("random"), make_player("random")])
        game.target_score = 10**9
        records: list[tuple[bool, bool, int, int, int]] = []
        spec = "greedy" if opp_greedy else "first"
        # seat 0 records and plays first-legal; seat 1 is the opponent
        p0 = BenchPlayer("random", Recorder(game, 0, opp_greedy, records))
        p1 = make_bench_player(spec)
        if not opp_greedy:
            p1 = BenchPlayer("random", Recorder(game, 1, False, records))
        p1.pegging_policy.reseed(trial)
        players: list[CribbagePlayer] = [p0, p1]
        game.players = players
        rnd = make_round(players, hands, starter, trial % 2, game)
        p0.peek = lambda: rnd.peg.hands[1]  # type: ignore[union-attr]
        p1.peek = lambda: rnd.peg.hands[0]  # type: ignore[union-attr]
        rnd.play()
        final = game.scores[0] - game.scores[1]
        for ok_pool, ok_size, predicted, margin_then, seat in records:
            assert ok_pool and ok_size
            if not opp_greedy:
                # both play first-legal, so the prediction is exact
                assert margin_then + predicted == (final if seat == 0 else -final)
            seen += 1
    assert seen > 500


def test_oracle_picks_a_best_move():
    rng = random.Random(6)
    solver = Solver()
    for _ in range(40):
        hands, starter = random_deal(rng)
        state = PegState.start(hands, 1)
        view = replace(
            state.view(0, (), starter, False, 0, 0), opp_hand=tuple(state.hands[1])
        )
        card = make_pegging_policy("oracle").choose(view)
        values = solver.move_values(state)
        assert values[state.hands[0].index(card)] == max(values.values())


def test_sampler_respects_go():
    # the count is 29 and the opponent has said go: all they hold is too big
    # to play.  Player 0 holds 0S 9S AS 2S and player 1 JS QH KH JH; the
    # sequence so far is 0S JS 9S
    state = PegState.start([[9, 8, 0, 1], [10, 24, 25, 23]], 1)
    for card in (9, 10, 8):
        state.play_card(card)
    assert state.count == 29 and state.is_go
    view = state.view(0, [40, 41], 42, False, 0, 0)
    sampler = WorldSampler(view)
    assert all(WORTH[c] > 2 for c in sampler.pool)
    assert sampler.num_opp == 3
    assert set(state.hands[1]) <= set(sampler.pool)


def test_greedy_move_prefers_scoring_card():
    # player 1 leads a ten (0S), so the count is 10; player 0 holds 5S and
    # 9S: the five makes fifteen for two points, so play the five
    state = PegState.start([[4, 8], [9, 1]], 0)
    state.play_card(9)
    assert state.turn == 0
    assert state.hands[0][greedy_move(state, random.Random(0))] == 4
