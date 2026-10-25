#!/usr/bin/env python

"""
test_pegging_expt_2026.py

Test the pegging_expt_2026 module.
"""

import random
from dataclasses import replace

import numpy as np
import pytest
from cribbage.cards import WORTH
from cribbage.game import Game
from cribbage.player import CribbagePlayer
from cribbage.pegging_expt_2026 import (
    Deals,
    PegPlayer,
    PegPolicy,
    Solver,
    WorldSampler,
    distinct_moves,
    first_move,
    greedy_move,
    make_deals,
    make_peg_player,
    make_policy,
    play_pegging,
    rollout,
    run_pegging,
)
from cribbage.pegging import PegState
from cribbage.randomplayer import RandomCribbagePlayer
from cribbage.test.helpers import make_round

ALL_SPECS = [
    "first",
    "random",
    "greedy",
    "greedy2",
    "pimc-roll:3:2",
    "pimc-solve:3",
    "pimc-solve:3/opp",
    "ismcts:30",
    "ismcts:30/opp",
    "oracle",
]


def random_deal(rng):
    cards = rng.sample(range(52), 9)
    return [cards[:4], cards[4:8]], cards[8]


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


def test_state_matches_round():
    """The points Round awards for the play are those PegState adds up."""
    rng = random.Random(1)
    for trial in range(300):
        hands, starter = random_deal(rng)
        dealer = trial % 2
        players: list[CribbagePlayer] = [RandomCribbagePlayer(), RandomCribbagePlayer()]
        rnd = make_round(players, hands, starter, dealer)
        # record who played what, by wrapping the players' choices
        log = []
        for seat, player in enumerate(players):
            original = player.play_card

            def spy(view, original=original, seat=seat):
                card = original(view)
                log.append((seat, card))
                return card

            player.play_card = spy  # type: ignore[method-assign]
        random.seed(trial)
        rnd.play()
        state = PegState.start(hands, dealer)
        for seat, card in log:
            assert not state.done
            assert state.turn == seat
            state.play_card(card)
        assert state.done
        assert state.points == rnd.game.scores
        assert all(type(p) is int for p in state.points)
        assert rnd.peg is not None and rnd.peg.history() == state.history()


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


class Recorder(PegPolicy):
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
        game = Game([RandomCribbagePlayer(), RandomCribbagePlayer()])
        game.target_score = 10**9
        records: list[tuple[bool, bool, int, int, int]] = []
        spec = "greedy" if opp_greedy else "first"
        # seat 0 records and plays first-legal; seat 1 is the opponent
        p0 = PegPlayer(RandomCribbagePlayer(), Recorder(game, 0, opp_greedy, records))
        p1 = make_peg_player(spec)
        if not opp_greedy:
            p1 = PegPlayer(RandomCribbagePlayer(), Recorder(game, 1, False, records))
        p1.policy.reseed(trial)
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


def test_policies_return_legal_moves():
    rng = random.Random(4)
    for spec in ALL_SPECS:
        policy_players = [make_peg_player(spec), make_peg_player("random")]
        for trial in range(15):
            hands, starter = random_deal(rng)
            for p in policy_players:
                p.policy.reseed(trial)
            kept = [hands[0], hands[1]]
            thrown = [[0, 1], [2, 3]]
            points = play_pegging(policy_players, kept, thrown, starter)
            assert all(type(p) is int for p in points)


def test_policies_reproducible():
    deals = make_deals(30, 0)
    for spec in ["pimc-roll:3", "pimc-solve:3", "ismcts:20/opp"]:
        a = run_pegging(spec, "greedy", deals, seed=5)
        b = run_pegging(spec, "greedy", deals, seed=5)
        assert (a.pair_margin == b.pair_margin).all()


def test_duplicate_deals_cancel_for_deterministic_players():
    deals = make_deals(50, 1)
    result = run_pegging("first", "first", deals)
    assert (result.pair_margin == 0).all()
    # the seats are not symmetric: the dealer does better
    nd, dealer = result.role_margins()
    assert dealer > nd


def test_workers_do_not_change_results():
    deals = make_deals(40, 2)
    one = run_pegging("pimc-roll:2", "greedy", deals, seed=1, workers=1)
    two = run_pegging("pimc-roll:2", "greedy", deals, seed=1, workers=2)
    assert (one.pair_margin == two.pair_margin).all()


def test_oracle_picks_a_best_move():
    rng = random.Random(6)
    solver = Solver()
    for _ in range(40):
        hands, starter = random_deal(rng)
        state = PegState.start(hands, 1)
        view = replace(
            state.view(0, (), starter, False, 0, 0), opp_hand=tuple(state.hands[1])
        )
        card = make_policy("oracle").choose(view)
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


def test_make_deals():
    deals = make_deals(100, 3)
    assert isinstance(deals, Deals)
    assert deals.kept.shape == (100, 2, 4) and deals.thrown.shape == (100, 2, 2)
    for i in range(100):
        cards = np.concatenate([deals.kept[i].ravel(), deals.thrown[i].ravel(), [deals.starter[i]]])
        assert len(set(cards.tolist())) == 13
    assert (make_deals(100, 3).kept == deals.kept).all()


def test_make_policy_rejects_nonsense():
    for spec in ["nope", "greedy:3", "random/opp", "pimc-solve", "ismcts:5/foo"]:
        with pytest.raises(ValueError):
            make_policy(spec)


def test_greedy_move_prefers_scoring_card():
    # player 1 leads a ten (0S), so the count is 10; player 0 holds 5S and
    # 9S: the five makes fifteen for two points, so play the five
    state = PegState.start([[4, 8], [9, 1]], 0)
    state.play_card(9)
    assert state.turn == 0
    assert state.hands[0][greedy_move(state, random.Random(0))] == 4
