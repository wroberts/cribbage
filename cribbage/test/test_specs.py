#!/usr/bin/env python

"""
test_specs.py

Test building policies and players from spec strings.
"""

import pytest
from cribbage.discard_policy import (
    ExpectedHandDiscardPolicy,
    RandomDiscardPolicy,
)
from cribbage.pegging_policy import (
    FirstPeggingPolicy,
    Greedy1PeggingPolicy,
    Greedy2PeggingPolicy,
    RandomPeggingPolicy,
)
from cribbage.pegging_search import (
    IsmctsPeggingPolicy,
    OraclePeggingPolicy,
    PimcRolloutPeggingPolicy,
    PimcSolvePeggingPolicy,
)
from cribbage.player import PolicyPlayer
from cribbage.specs import make_discard_policy, make_pegging_policy, make_player


def test_pegging_policy_specs():
    kinds = {
        "first": FirstPeggingPolicy,
        "random": RandomPeggingPolicy,
        "greedy": Greedy1PeggingPolicy,
        "greedy2": Greedy2PeggingPolicy,
        "oracle": OraclePeggingPolicy,
        "pimc-roll:5": PimcRolloutPeggingPolicy,
        "pimc-roll:5:2/opp": PimcRolloutPeggingPolicy,
        "pimc-solve:5": PimcSolvePeggingPolicy,
        "ismcts:10/opp": IsmctsPeggingPolicy,
    }
    for spec, kind in kinds.items():
        assert type(make_pegging_policy(spec)) is kind
    policy = make_pegging_policy("pimc-roll:5:2/opp")
    assert isinstance(policy, PimcRolloutPeggingPolicy)
    assert (policy.worlds, policy.rollouts, policy.opp_aware) == (5, 2, True)


def test_pegging_policy_specs_reject_nonsense():
    for spec in ["nope", "greedy:3", "random/opp", "pimc-solve", "ismcts:5/foo"]:
        with pytest.raises(ValueError):
            make_pegging_policy(spec)


def test_discard_policy_specs():
    assert type(make_discard_policy("random")) is RandomDiscardPolicy
    policy = make_discard_policy("expected")
    assert isinstance(policy, ExpectedHandDiscardPolicy) and policy.num_samples == 20
    fifty = make_discard_policy("expected:50")
    assert isinstance(fifty, ExpectedHandDiscardPolicy) and fifty.num_samples == 50
    for spec in ["nope", "random:3", "expected:1:2", "greedy"]:
        with pytest.raises(ValueError):
            make_discard_policy(spec)


def test_player_specs():
    player = make_player("expected+greedy")
    assert isinstance(player, PolicyPlayer)
    assert type(player.discard_policy) is ExpectedHandDiscardPolicy
    assert type(player.pegging_policy) is Greedy1PeggingPolicy
    # the shorthands
    for spec, discard, pegging in [
        ("random", RandomDiscardPolicy, RandomPeggingPolicy),
        ("simple", ExpectedHandDiscardPolicy, Greedy1PeggingPolicy),
        ("simple-discard", ExpectedHandDiscardPolicy, RandomPeggingPolicy),
    ]:
        player = make_player(spec)
        assert type(player.discard_policy) is discard
        assert type(player.pegging_policy) is pegging
    for spec in ["greedy", "random+nope", "nope+random"]:
        with pytest.raises(ValueError):
            make_player(spec)
