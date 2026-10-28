#!/usr/bin/env python

"""
test_discard_pegging.py

Test the discard_pegging module.
"""

import random

import numpy as np
from cribbage.discard_model import TrainConfig
from cribbage.discard_pegging import analyse, pegging_values, split_values
from cribbage.pegging_model import NetPeggingPolicy, PegNet


def test_pegging_values_shape_and_determinism():
    torch_model = TrainConfig(True, width=8, depth=2).make_net()
    peg = NetPeggingPolicy(PegNet(16, 2))
    hand = random.Random(1).sample(range(52), 6)
    first = pegging_values(hand, True, peg, torch_model, 6, "a")
    again = pegging_values(hand, True, peg, torch_model, 6, "a")
    other = pegging_values(hand, True, peg, torch_model, 6, "b")
    assert first.shape == (15,)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    # a margin is at most a few dozen points
    assert np.abs(first).max() < 40


def test_split_values_signs():
    h = np.array([[1.0, 2.0]])
    c = np.array([[0.5, 0.5]])
    u = split_values(h, c)
    assert u.shape == (1, 2, 2)
    assert u[0, 0].tolist() == [0.5, 1.5] and u[0, 1].tolist() == [1.5, 2.5]


def test_analyse_recovers_a_known_gain():
    rng = np.random.default_rng(0)
    n = 4000
    u = rng.normal(size=(n, 2, 15))
    truth = rng.normal(scale=0.5, size=(n, 2, 15))
    noise = 0.3
    p = truth[:, :, None, :] + rng.normal(scale=noise, size=(n, 2, 2, 15))
    stats = analyse(u, p)
    # the gain is judged on an independent replicate, so it is unbiased: it
    # matches the gain of the choice made from one noisy replicate, measured
    # on the truth
    base = u.argmax(axis=2)
    pick = (u + p[:, :, 0]).argmax(axis=2)
    rows, seats = np.arange(n)[:, None], np.arange(2)[None, :]
    full = u + truth
    expected = (full[rows, seats, pick] - full[rows, seats, base]).mean(axis=0)
    assert np.allclose(stats["gain"], expected, atol=0.03)
    assert (stats["changed"] > 0).all()
    assert np.allclose(stats["noise"], noise, atol=0.02)


def test_peg_aware_policy_and_spec(tmp_path):
    from cribbage.discard_model import save_model
    from cribbage.player import DiscardView
    from cribbage.specs import make_discard_policy

    config = TrainConfig(True, width=8, depth=2)
    peg_config = TrainConfig(False, width=8, depth=2)
    base, peg = tmp_path / "base.pt", tmp_path / "peg.pt"
    save_model(config.make_net(), config, str(base))
    save_model(peg_config.make_net(), peg_config, str(peg))
    policy = make_discard_policy(f"model:{base}@{peg}")
    hand = tuple(random.Random(2).sample(range(52), 6))
    out = policy.choose(DiscardView(hand, True, 0, 0))
    assert len(set(out)) == 2 and set(out) <= set(hand)


def test_play_hand_returns_a_margin():
    from cribbage.discard_policy import RandomDiscardPolicy
    from cribbage.discard_pegging import play_hand

    peg = NetPeggingPolicy(PegNet(16, 2))
    deal = np.random.default_rng(3).permutation(52)[:13]
    policies = [RandomDiscardPolicy(), RandomDiscardPolicy()]
    margin = play_hand(deal, 0, policies, peg)
    assert isinstance(margin, int) and -60 < margin < 60
