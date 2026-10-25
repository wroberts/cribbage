#!/usr/bin/env python

"""
test_pegging_rl_expt_2026.py

Test the pegging_rl_expt_2026 module.
"""

import random

import numpy as np
import torch
from cribbage.pegging import PegState
from cribbage.pegging_expt_2026 import (
    PegObs,
    make_deals,
    make_policy,
    play_pegging,
    run_pegging,
)
from cribbage.pegging_rl_expt_2026 import (
    CAND_DIM,
    MAX_CANDIDATES,
    STATE_DIM,
    Labels,
    NetPeg,
    NetRollout,
    PimcNet,
    PegNet,
    candidate_cards,
    centred,
    encode,
    generate_labels,
    load_net,
    regrets,
    save_net,
    to_tensors,
    train_net,
)


def make_obs(hand, seq, played, is_go=False, is_dealer=False, discards=()):
    return PegObs(hand, seq, played, is_go, list(range(len(hand))), discards, is_dealer)


def test_encode_shapes_and_suit_invariance():
    # a ten, a five and a pair of fours; the count is 12
    obs = make_obs([9, 4, 3, 16], [8, 2], {8, 2, 20, 9, 4, 3, 16})
    state, cands, idxs = encode(obs)
    assert state.shape == (STATE_DIM,) and cands.shape == (3, CAND_DIM)
    # the two fours are one candidate
    assert idxs == [0, 1, 2]
    assert candidate_cards(obs) == idxs
    # relabelling suits changes nothing
    swapped = make_obs(
        [c % 13 + 13 * ((c // 13 + 1) % 4) for c in obs.hand],
        [c % 13 + 13 * ((c // 13 + 1) % 4) for c in obs.seq],
        {c % 13 + 13 * ((c // 13 + 1) % 4) for c in obs.played},
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
    policy = NetPeg.load(path)
    state, cands, _ = encode(make_obs([9, 4, 3], [8], {8, 9, 4, 3, 40}))
    with torch.no_grad():
        expected = model(torch.from_numpy(state)[None], torch.from_numpy(cands)[None])[0]
    assert np.allclose(policy.scores(state, cands), expected.numpy(), atol=1e-5)


def test_net_spec_plays_pegging(tmp_path):
    path = str(tmp_path / "net.pt")
    save_net(PegNet(16, 2), path)
    assert isinstance(make_policy(f"net:{path}"), NetPeg)
    deals = make_deals(20, 8)
    result = run_pegging(f"net:{path}", "greedy", deals, seed=1)
    assert len(result.pair_margin) == 20
    assert all(type(p) is int for p in (int(result.a_nd[0]), int(result.b_d[0])))


def test_rollout_features_match_encode():
    """The rollout's fast features equal encode() on the mover's own view."""
    rng = random.Random(9)
    net = NetPeg(PegNet(8, 1))
    for trial in range(60):
        cards = rng.sample(range(52), 13)
        hands = [cards[:4], cards[4:8]]
        discards = [cards[8:10], cards[10:12]]
        starter = cards[12]
        dealer = trial % 2
        ctx = NetRollout(net, hands, {starter}, discards, dealer)
        state = PegState.start(hands, dealer)
        played = {starter}
        while not state.done:
            seat = state.turn
            legal = state.legal()
            obs = PegObs(
                state.hands[seat][:], state.seq[:], set(played), state.is_go,
                legal, tuple(discards[seat]), seat == dealer,
            )
            want_state, want_cands, want_idxs = encode(obs)
            got_state, got_cands, got_idxs = ctx.features(state, legal)
            assert got_idxs == want_idxs
            assert np.allclose(got_state, want_state)
            assert np.allclose(got_cands, want_cands)
            idx = rng.choice(legal)
            played.add(state.hands[seat][idx])
            state.play(idx)


def test_pimc_net_plays_pegging(tmp_path):
    path = str(tmp_path / "net.pt")
    save_net(PegNet(16, 2), path)
    policy = make_policy(f"pimc-net:3:{path}")
    assert isinstance(policy, PimcNet)
    deals = make_deals(10, 9)
    result = run_pegging(f"pimc-net:3:{path}", "greedy", deals, seed=1)
    assert len(result.pair_margin) == 10
