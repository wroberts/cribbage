#!/usr/bin/env python

"""
test_pegging_bench.py

Test the pegging_bench module: fixed deals and duplicate matches.
"""

import random

import numpy as np
from cribbage.pegging_bench import (
    Deals,
    make_bench_player,
    make_deals,
    play_pegging,
    run_pegging,
)
from cribbage.test.helpers import random_deal

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


def test_policies_return_legal_moves():
    rng = random.Random(4)
    for spec in ALL_SPECS:
        policy_players = [make_bench_player(spec), make_bench_player("random")]
        for trial in range(15):
            hands, starter = random_deal(rng)
            for p in policy_players:
                p.pegging_policy.reseed(trial)
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


def test_make_deals():
    deals = make_deals(100, 3)
    assert isinstance(deals, Deals)
    assert deals.kept.shape == (100, 2, 4) and deals.thrown.shape == (100, 2, 2)
    for i in range(100):
        cards = np.concatenate([deals.kept[i].ravel(), deals.thrown[i].ravel(), [deals.starter[i]]])
        assert len(set(cards.tolist())) == 13
    assert (make_deals(100, 3).kept == deals.kept).all()
