#!/usr/bin/env python

"""
test_pegging_expt_2026.py

Test the pegging_expt_2026 module.
"""

import random

import numpy as np
import pytest
from cribbage.cards import WORTH, cards_worth
from cribbage.game import Game
from cribbage.player import CribbagePlayer
from cribbage.pegging_expt_2026 import (
    Deals,
    PegObs,
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
from cribbage.round import Round

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
    """PegState scores and sequences plays exactly as Round.play does."""
    rng = random.Random(1)
    for trial in range(300):
        hands, starter = random_deal(rng)
        dealer = trial % 2
        players: list[CribbagePlayer] = [RandomCribbagePlayer(), RandomCribbagePlayer()]
        game = Game(players)
        game.target_score = 10**9
        rnd = Round(game, players, dealer)
        rnd.hands = [h[:] for h in hands]
        rnd.played_cards = {starter}
        # record who played what, by wrapping the players' choices
        log = []
        for seat, player in enumerate(players):
            original = player.play_card

            def spy(*args, original=original, seat=seat, **kwargs):
                idx = original(*args, **kwargs)
                log.append((seat, kwargs["hand"][idx]))
                return idx

            player.play_card = spy  # type: ignore[method-assign]
        random.seed(trial)
        rnd.play()
        state = PegState.start(hands, dealer)
        for seat, card in log:
            assert not state.done
            assert state.turn == seat
            state.play(state.hands[seat].index(card))
        assert state.done
        assert state.points == game.scores
        assert all(type(p) is int for p in state.points)


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

    def choose(self, obs):
        sampler = WorldSampler(obs, self.opp_aware)
        world = sampler.build(obs.opp_hand)
        # a world with the true hand, played out by first-legal moves
        rollout(world, first_move, self.rng)
        scores = self.game.scores
        self.records.append(
            (
                set(obs.opp_hand) <= set(sampler.pool),
                len(obs.opp_hand) == sampler.num_opp,
                world.margin(),
                scores[self.seat] - scores[1 - self.seat],
                self.seat,
            )
        )
        return obs.legal[0]


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
        rnd = Round(game, players, dealer_idx=trial % 2)
        rnd.hands = [h[:] for h in hands]
        rnd.played_cards = {starter}
        p0.begin_round([])
        p1.begin_round([])
        p0.peek = lambda: rnd.hands[1]
        p1.peek = lambda: rnd.hands[0]
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
        legal = state.legal()
        obs = PegObs(
            state.hands[0][:], [], {starter}, False, legal, (), False,
            opp_hand=state.hands[1][:],
        )
        policy = make_policy("oracle")
        idx = policy.choose(obs)
        values = solver.move_values(state)
        assert values[idx] == max(values.values())


def test_sampler_respects_go():
    # count 29 with a go called: any opponent card is too big to play
    seq = [9, 10, 4 + 13]  # 10, J, 5 -> 25 ... then a 4 makes 29
    seq = [9, 10, 3 + 13 * 2 + 0, 8 + 13]  # 10, J, 4, 9 -> 33? keep it legal below
    seq = [9, 10, 8]  # 10 + 10 + 9 = 29
    obs = PegObs([0], seq, {40, *seq}, True, [0], (), False)
    sampler = WorldSampler(obs)
    assert cards_worth(seq) == 29
    assert all(WORTH[c] > 2 for c in sampler.pool)
    assert sampler.num_opp == 9 - len(obs.played) - 1


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
    # the count is 10; holding a five (15 for two) and a nine, play the five
    state = PegState.from_cards([[4, 8], [0, 1]], 0, seq=[9])
    assert state.hands[0][greedy_move(state, random.Random(0))] == 4
