#!/usr/bin/env python

"""
test_pegging_model.py

Test the pegging_model module.
"""

import random

import numpy as np
import torch
from cribbage.cards import distinct_by_face
from cribbage.pegging import PegState
from cribbage.pegging_bench import make_deals, run_pegging
from cribbage.specs import make_pegging_policy
from cribbage.test.helpers import make_view
from cribbage.pegging_model import (
    CAND_DIM,
    MAX_CANDIDATES,
    STATE_DIM,
    Labels,
    NetPeggingPolicy,
    NetRollout,
    PimcNetPeggingPolicy,
    PegNet,
    centred,
    encode,
    generate_labels,
    load_net,
    regrets,
    save_net,
    to_tensors,
    train_net,
)


def test_encode_shapes_and_suit_invariance():
    # a ten, a five and a pair of fours; the count is 12
    view = make_view([9, 4, 3, 16], [8, 2], starter=20)
    state, cands, cards = encode(view)
    assert state.shape == (STATE_DIM,) and cands.shape == (3, CAND_DIM)
    # the two fours are one candidate
    assert cards == [9, 4, 3]
    assert distinct_by_face(view.legal) == cards

    def swap(c):
        return c % 13 + 13 * ((c // 13 + 1) % 4)

    # relabelling suits changes nothing
    swapped = make_view(
        [swap(c) for c in view.hand], [swap(c) for c in view.seq], starter=swap(20)
    )
    state2, cands2, _ = encode(swapped)
    assert np.allclose(state, state2) and np.allclose(cands, cands2)


def test_labels_are_consistent():
    deals = make_deals(30, 5)
    labels = generate_labels(deals, worlds=4, explore=0.3, seed=1, workers=1)
    assert len(labels) > 30
    assert labels.state.shape[1] == STATE_DIM
    assert labels.cand.shape[1:] == (MAX_CANDIDATES, CAND_DIM)
    # every decision has at least two candidates, packed first
    assert (labels.mask.sum(axis=1) >= 2).all()
    assert (labels.mask[:, 0]).all()
    assert (labels.cand[~labels.mask] == 0).all()
    # same labels from the same seed
    again = generate_labels(deals, worlds=4, explore=0.3, seed=1, workers=1)
    assert np.array_equal(labels.value, again.value)


def test_labels_round_trip(tmp_path):
    deals = make_deals(10, 6)
    labels = generate_labels(deals, 3, 0.2, 2, 1)
    labels.save(str(tmp_path / "labels.npz"))
    loaded = Labels.load(str(tmp_path / "labels.npz"))
    assert np.array_equal(loaded.state, labels.state)
    assert np.array_equal(loaded.mask, labels.mask)


def test_centred_ignores_masked_slots():
    x = torch.tensor([[1.0, 3.0, 99.0, 99.0]])
    mask = torch.tensor([[True, True, False, False]])
    assert centred(x, mask).tolist() == [[-1.0, 1.0, 0.0, 0.0]]


def test_regrets_and_training_reduce_error(tmp_path):
    deals = make_deals(300, 7)
    labels = generate_labels(deals, worlds=8, explore=0.3, seed=3, workers=1)
    data = to_tensors(labels)
    torch.manual_seed(0)
    before = regrets(PegNet(32, 2)(data["state"], data["cand"]), data)
    model, _ = train_net(labels, width=32, depth=2, epochs=30, log=False)
    after = regrets(model(data["state"], data["cand"]), data)
    # fitted to its own training data, the network loses less than chance
    assert after["net"] < before["net"]
    assert after["net"] < after["random"]
    # the saved network and the numpy policy agree with torch
    path = str(tmp_path / "net.pt")
    save_net(model, path)
    reloaded = load_net(path)
    assert torch.allclose(
        model(data["state"], data["cand"]), reloaded(data["state"], data["cand"])
    )
    policy = NetPeggingPolicy.load(path)
    state, cands, _ = encode(make_view([9, 4, 3], [8], starter=40))
    with torch.no_grad():
        expected = model(torch.from_numpy(state)[None], torch.from_numpy(cands)[None])[
            0
        ]
    assert np.allclose(policy.scores(state, cands), expected.numpy(), atol=1e-5)


def test_net_spec_plays_pegging(tmp_path):
    path = str(tmp_path / "net.pt")
    save_net(PegNet(16, 2), path)
    assert isinstance(make_pegging_policy(f"net:{path}"), NetPeggingPolicy)
    deals = make_deals(20, 8)
    result = run_pegging(f"net:{path}", "greedy", deals, seed=1)
    assert len(result.pair_margin) == 20
    assert all(type(p) is int for p in (int(result.a_nd[0]), int(result.b_d[0])))


def test_rollout_features_match_encode():
    """The rollout's fast features equal encode() on the mover's own view."""
    rng = random.Random(9)
    net = NetPeggingPolicy(PegNet(8, 1))
    for trial in range(60):
        cards = rng.sample(range(52), 13)
        hands = [cards[:4], cards[4:8]]
        discards = [cards[8:10], cards[10:12]]
        starter = cards[12]
        dealer = trial % 2
        ctx = NetRollout(net, hands, {starter}, discards, dealer)
        state = PegState.start(hands, dealer)
        while not state.done:
            seat = state.turn
            view = state.view(seat, discards[seat], starter, seat == dealer, 0, 0)
            want_state, want_cands, want_cards = encode(view)
            got_state, got_cands, got_idxs = ctx.features(state, state.legal())
            assert [state.hands[seat][i] for i in got_idxs] == want_cards
            assert np.allclose(got_state, want_state)
            assert np.allclose(got_cands, want_cands)
            state.play_card(rng.choice(view.legal))


def test_pimc_net_plays_pegging(tmp_path):
    path = str(tmp_path / "net.pt")
    save_net(PegNet(16, 2), path)
    policy = make_pegging_policy(f"pimc-net:3:{path}")
    assert isinstance(policy, PimcNetPeggingPolicy)
    deals = make_deals(10, 9)
    result = run_pegging(f"pimc-net:3:{path}", "greedy", deals, seed=1)
    assert len(result.pair_margin) == 10


def test_opp_hand_model_samples_consistent_hands(tmp_path):
    from cribbage.discard_model import TrainConfig, save_model
    from cribbage.pegging_model import NetOppHandModel
    from cribbage.pegging_search import WorldSampler

    path = str(tmp_path / "discard.pt")
    config = TrainConfig(True, width=8, depth=2)
    save_model(config.make_net(), config, path)
    model = NetOppHandModel.load(path, size=200)
    rng = random.Random(4)
    for trial in range(30):
        hands = [rng.sample(range(52), 4)]
        rest = [c for c in range(52) if c not in hands[0]]
        opp = rng.sample(rest, 4)
        discards = rng.sample([c for c in rest if c not in opp], 2)
        starter = rng.choice([c for c in rest if c not in opp and c not in discards])
        state = PegState.start([hands[0], opp], trial % 2)
        for _ in range(trial % 4):  # a few plays, whoever's they are
            state.play_card(rng.choice([state.hands[state.turn][i] for i in state.legal()]))
        view = state.view(0, discards, starter, trial % 2 == 0, 0, 0)
        played = {card for _, card in view.opp_plays}
        kept = model.kept_hands(view, rng)
        assert kept and all(len(k) == 4 and played <= k for k in kept)
        assert all(k <= set(view.unseen()) | played for k in kept)
        sampler = WorldSampler(view, beliefs=kept)
        for _ in range(5):
            world = sampler.sample(rng)
            assert frozenset(world.hands[1]) | played in kept


def test_labels_with_posterior_teacher(tmp_path):
    from cribbage.discard_model import TrainConfig, save_model

    path = str(tmp_path / "discard.pt")
    config = TrainConfig(True, width=8, depth=2)
    save_model(config.make_net(), config, path)
    deals = make_deals(20, 8)
    labels = generate_labels(deals, worlds=4, explore=0.3, seed=1, workers=1, post=path)
    assert len(labels) > 20 and (labels.mask.sum(axis=1) >= 2).all()
    parallel = generate_labels(deals, 4, 0.3, 1, workers=2, post=path)
    assert np.array_equal(labels.value, parallel.value)


def test_pimc_net_with_posterior_beliefs(tmp_path):
    from cribbage.discard_model import TrainConfig, save_model

    net_path = str(tmp_path / "net.pt")
    save_net(PegNet(16, 2), net_path)
    discard_path = str(tmp_path / "discard.pt")
    config = TrainConfig(True, width=8, depth=2)
    save_model(config.make_net(), config, discard_path)
    spec = f"pimc-net:3:{net_path}@{discard_path}"
    policy = make_pegging_policy(spec)
    assert isinstance(policy, PimcNetPeggingPolicy) and policy.opp_model is not None
    result = run_pegging(spec, "greedy", make_deals(8, 9), seed=1)
    assert len(result.pair_margin) == 8
