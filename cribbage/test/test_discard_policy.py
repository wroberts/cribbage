#!/usr/bin/env python

"""
test_discard_policy.py

Test the discard policies and PolicyPlayer.
"""

import random

from cribbage.cards import KEEP_COMBINATIONS, string_tocard
from cribbage.discard_policy import ExpectedHandDiscardPolicy, RandomDiscardPolicy
from cribbage.game import Game, compare_players
from cribbage.player import DiscardView
from cribbage.scoring import score_hand
from cribbage.specs import make_player


def view_of(*names):
    return DiscardView(tuple(string_tocard(n) for n in names), False, 0, 0)


HAND = ("5S", "5H", "JD", "KC", "AS", "9D")


def test_random_discards_two_cards_of_the_hand():
    policy = RandomDiscardPolicy()
    view = view_of(*HAND)
    seen = set()
    for _ in range(50):
        first, second = policy.choose(view)
        assert first != second and {first, second} <= set(view.hand)
        seen.add(frozenset((first, second)))
    assert len(seen) > 5


def test_expected_hand_keeps_the_best_four():
    # the pair of fives with the jack and king is worth more than any other
    # four, whatever the starter (and 15s with the 5s)
    view = view_of("5S", "5H", "5D", "JC", "AS", "9D")
    first, second = ExpectedHandDiscardPolicy(num_samples=200).choose(view)
    assert {first, second} == {string_tocard("AS"), string_tocard("9D")}


def test_expected_hand_is_reproducible_from_its_seed():
    view = view_of(*HAND)
    a, b = ExpectedHandDiscardPolicy(), ExpectedHandDiscardPolicy()
    a.reseed(3)
    b.reseed(3)
    assert [a.choose(view) for _ in range(5)] == [b.choose(view) for _ in range(5)]


def test_expected_hand_matches_exhaustive_values():
    # with many samples the choice is the keep with the best mean score
    view = view_of(*HAND)
    hand = list(view.hand)
    starters = [c for c in range(52) if c not in hand]

    def mean(keep):
        return sum(score_hand(keep, draw=s) for s in starters) / len(starters)

    best = max((mean([hand[i] for i in keep]), keep) for keep in KEEP_COMBINATIONS)[0]
    first, second = ExpectedHandDiscardPolicy(num_samples=2000).choose(view)
    kept = [c for c in hand if c not in (first, second)]
    assert mean(kept) >= best - 0.5


def test_policy_player_delegates_and_reseeds():
    player = make_player("random")
    view = view_of(*HAND)
    player.reseed(7)
    first = [player.discard(view) for _ in range(5)]
    player.reseed(7)
    assert [player.discard(view) for _ in range(5)] == first


def test_games_are_reproducible_when_the_players_are_reseeded():
    def play(seed):
        players = [make_player("simple"), make_player("random")]
        players[0].reseed(seed)
        players[1].reseed(seed + 1)
        game = Game(players, rng=random.Random(seed))
        game.play()
        return game.scores

    assert play(4) == play(4)


def test_simple_beats_random():
    random.seed(0)
    wins = compare_players([make_player("simple"), make_player("random")], 60)
    assert wins[0] > wins[1]
