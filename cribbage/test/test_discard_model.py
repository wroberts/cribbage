#!/usr/bin/env python

"""
test_discard_model.py

Test the discard_model module.
"""

import numpy as np
import torch
from cribbage.cards import splits, unseen_cards
from cribbage.player import DiscardView
from cribbage.specs import make_player
from cribbage.cribbage_score import score_hand
from cribbage.discard_model import (
    NUM_FEATURES,
    NUM_KEEPS,
    SPLITS,
    Dataset,
    DISCARD_IDX,
    DiscardNet,
    NetDiscardPolicy,
    TrainConfig,
    UniformThrows,
    candidate_values,
    canonical_keys,
    choice_stats,
    crib_values_mc,
    deal_hands,
    encode_hands,
    evaluate,
    exact_labels,
    generate_dataset,
    h_only_stats,
    hand_values,
    load_model,
    play_match,
    predict_values,
    sample_split_hands,
    save_model,
    split_of,
    to_tensors,
    train_model,
)

HAND = [3, 17, 18, 30, 44, 51]


def test_splits_partition_the_hand():
    all_splits = splits(HAND)
    assert len(all_splits) == NUM_KEEPS == 15
    assert len({tuple(sorted(keep)) for keep, _ in all_splits}) == 15
    for keep, discard in all_splits:
        assert len(keep) == 4 and len(discard) == 2
        assert sorted(keep + discard) == sorted(HAND)


def test_hand_values_match_brute_force():
    unseen = unseen_cards(HAND)
    assert len(unseen) == 46
    expected = [
        np.mean([score_hand(keep, starter) for starter in unseen])
        for keep, _ in splits(HAND)
    ]
    assert np.allclose(hand_values(HAND), expected)


def test_uniform_support_is_a_distribution():
    unseen = unseen_cards(HAND)
    completions, weights = UniformThrows().support(unseen)
    assert completions.shape == (46 * 45 * 44 // 2, 3)
    assert np.isclose(weights.sum(), 1)
    assert set(completions.ravel()) <= set(unseen)
    assert all(len(set(row)) == 3 for row in completions[::997])


def test_uniform_sample_is_valid_and_reproducible():
    unseen = unseen_cards(HAND)
    sample = UniformThrows().sample(np.random.default_rng(1), unseen, 200)
    assert sample.shape == (200, 3)
    assert set(sample.ravel()) <= set(unseen)
    assert all(len(set(row)) == 3 for row in sample)
    again = UniformThrows().sample(np.random.default_rng(1), unseen, 200)
    assert np.array_equal(sample, again)


def test_monte_carlo_converges_to_exact():
    exact = exact_labels(HAND)
    estimate = crib_values_mc(HAND, 20000, np.random.default_rng(0))
    # a single crib score has a std of a few points, so the standard
    # error at K=20000 is about 0.02
    assert exact.c_std.min() > 1
    assert np.abs(estimate - exact.c).max() < 0.15


def test_candidate_values_sign():
    h, c = np.array([4.0, 2.0]), np.array([1.0, 3.0])
    assert list(candidate_values(h, c, True)) == [5.0, 5.0]
    assert list(candidate_values(h, c, False)) == [3.0, -1.0]


def relabel_suits(hand, perm):
    """Applies the suit permutation `perm` to a hand of card values."""
    return [card % 13 + 13 * perm[card // 13] for card in hand]


def test_canonical_key_ignores_suit_relabelling():
    hands = deal_hands(200, np.random.default_rng(3))
    perm = [2, 0, 3, 1]
    relabelled = np.array([relabel_suits(hand, perm) for hand in hands])
    assert np.array_equal(canonical_keys(hands), canonical_keys(relabelled))
    # but hands with different rank structure are different
    assert len(set(canonical_keys(hands))) > 190


def test_labels_are_invariant_to_suit_relabelling():
    perm = [3, 2, 0, 1]
    original = exact_labels(HAND)
    relabelled = exact_labels(relabel_suits(HAND, perm))
    # the order of the splits follows the sorted hand, so compare as sets
    assert sorted(np.round(original.h + original.c, 9)) == sorted(
        np.round(relabelled.h + relabelled.c, 9)
    )


def test_splits_follow_hand_class_and_proportions():
    hands = deal_hands(20000, np.random.default_rng(4))
    assignment = split_of(hands)
    fractions = np.bincount(assignment, minlength=3) / len(hands)
    assert np.allclose(fractions, [0.90, 0.05, 0.05], atol=0.01)
    relabelled = np.array([relabel_suits(hand, [1, 2, 3, 0]) for hand in hands[:500]])
    assert np.array_equal(split_of(relabelled), assignment[:500])


def test_sample_split_hands_stay_in_split():
    for split in range(3):
        hands = sample_split_hands(60, split, np.random.default_rng(split))
        assert hands.shape == (60, 6)
        assert (split_of(hands) == split).all()


def test_generate_dataset_is_deterministic_and_correct(tmp_path):
    kwargs = dict(split="val", num=30, k=10, seed=7)
    one = generate_dataset(workers=1, **kwargs)
    two = generate_dataset(workers=2, **kwargs)
    for a, b in ((one.hands, two.hands), (one.h, two.h), (one.c, two.c)):
        assert np.array_equal(a, b)
    assert one.hands.shape == (30, 6) and one.h.shape == one.c.shape == (30, 15)
    assert (split_of(one.hands) == SPLITS.index("val")).all()
    assert np.allclose(one.h[0], hand_values(one.hands[0].tolist()), atol=1e-5)
    # round trip through a file
    path = str(tmp_path / "data.npz")
    one.save(path)
    loaded = Dataset.load(path)
    assert (loaded.k, loaded.split, loaded.seed) == (10, "val", 7)
    assert np.array_equal(loaded.c, one.c) and np.array_equal(loaded.hands, one.hands)


def test_generate_exact_dataset_matches_exact_labels():
    dataset = generate_dataset("test", num=3, k=None, seed=0, workers=1)
    assert dataset.k is None
    exact = exact_labels(dataset.hands[1].tolist())
    assert np.allclose(dataset.c[1], exact.c, atol=1e-5)


def test_encode_hands_known_hand():
    # A, 2, 3, J of spades; A of hearts, A of diamonds
    features = encode_hands(torch.tensor([[0, 1, 2, 10, 13, 26]]))
    assert features.shape == (1, NUM_KEEPS, NUM_FEATURES)
    # keep the four spades and throw the two aces
    first = features[0, 0]
    keep_counts, discard_counts = first[:13], first[13:26]
    assert keep_counts.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    assert discard_counts.tolist() == [2] + [0] * 12
    flush, suited, nobs = first[26:].tolist()
    assert flush == 1 and suited == 0
    # nine spades are unseen, out of 46 unseen cards
    assert abs(nobs - 9 / 46) < 1e-6


def test_encoding_ignores_suit_relabelling():
    hands = deal_hands(50, np.random.default_rng(5))
    relabelled = np.array([relabel_suits(hand, [2, 3, 1, 0]) for hand in hands])
    one = encode_hands(torch.from_numpy(hands).long())
    two = encode_hands(torch.from_numpy(relabelled).long())
    # relabelling suits can reorder cards within the sorted hand, so the
    # splits come in a different order: compare the sorted rows
    def canonical_rows(x):
        return sorted(map(tuple, x.tolist()))

    for a, b in zip(one, two):
        assert canonical_rows(a) == canonical_rows(b)


def test_choice_stats():
    true = torch.tensor([[1.0, 3.0, 2.0], [0.0, 0.0, 5.0]])
    assert choice_stats(true, true) == (0.0, 1.0)
    regret, optimal = choice_stats(-true, true)
    assert regret > 0 and optimal < 1


def test_heads_agree_on_the_seat_sign():
    features = encode_hands(torch.tensor([[0, 1, 2, 10, 13, 26]]))
    model = DiscardNet(two_head=True, width=8, depth=2)
    dealer = predict_values(model, features, True)
    nondealer = predict_values(model, features, False)
    c_hat = model(features)[..., 1]
    assert torch.allclose(dealer - nondealer, 2 * c_hat, atol=1e-6)
    scalar = DiscardNet(two_head=False, width=8, depth=2)
    assert predict_values(scalar, features, True).shape == (1, NUM_KEEPS)


def test_train_smoke(tmp_path):
    train = generate_dataset("train", num=400, k=5, seed=0, workers=1)
    val = generate_dataset("val", num=100, k=5, seed=0, workers=1)
    for two_head in (True, False):
        config = TrainConfig(two_head, width=16, depth=2, epochs=2, batch_size=64)
        model, history = train_model(config, to_tensors(train), to_tensors(val), False)
        assert len(history) == 2 and all(np.isfinite(e["mse"]) for e in history)
        stats = evaluate(model, to_tensors(val))
        assert stats["dealer_regret"] >= 0 and 0 <= stats["nondealer_optimal"] <= 1
        path = str(tmp_path / "model.pt")
        save_model(model, config, path)
        features = encode_hands(to_tensors(val)[0][:3])
        with torch.no_grad():
            assert torch.allclose(
                predict_values(model, features, True),
                predict_values(load_model(path), features, True),
            )
    assert h_only_stats(to_tensors(val))["dealer_regret"] >= 0


def test_net_policy_discards_the_best_split():
    torch.manual_seed(0)
    model = DiscardNet(two_head=True, width=8, depth=2)
    policy = NetDiscardPolicy(model)
    for is_dealer in (True, False):
        for hand in deal_hands(5, np.random.default_rng(6)).tolist():
            discard = policy.choose(DiscardView(tuple(hand), is_dealer, 0, 0))
            assert len(set(discard)) == 2 and set(discard) <= set(hand)
            features = encode_hands(torch.tensor([hand]))
            best = int(predict_values(model, features, is_dealer).argmax())
            assert list(discard) == [hand[i] for i in DISCARD_IDX[best].tolist()]


def test_make_player_and_match(tmp_path):
    model = DiscardNet(two_head=True, width=8, depth=2)
    path = str(tmp_path / "model.pt")
    save_model(model, TrainConfig(True, width=8, depth=2), path)
    player = make_player(f"model:{path}")
    assert isinstance(player.discard_policy, NetDiscardPolicy)
    for spec in ("random", "simple", "simple-discard"):
        make_player(spec)
    wins, games = play_match(f"model:{path}", "random", games=6, seed=0, workers=1)
    assert games == 6 and 0 <= wins <= 6


def test_standardisation_uses_training_statistics():
    train = to_tensors(generate_dataset("train", num=300, k=5, seed=2, workers=1))
    config = TrainConfig(width=8, depth=1, standardize_inputs=True, standardize_targets=True)
    model = config.make_net()
    from cribbage.discard_model import fit_standardisation

    fit_standardisation(model, train)
    features = encode_hands(train[0][:300]).reshape(-1, NUM_FEATURES)
    assert torch.allclose(model.in_mean, features.mean(dim=0), atol=1e-5)
    assert torch.allclose(model.out_mean, torch.stack([train[1].mean(), train[2].mean()]))
    # an untrained network predicts about the mean of each target
    with torch.no_grad():
        out = model(encode_hands(train[0][:100]))
    assert abs(out[..., 0].mean() - train[1].mean()) < 1.5


def test_dropped_features_do_not_matter():
    model = DiscardNet(two_head=True, width=8, depth=2, drop=("nobs", "flush"))
    assert model.layers[0].in_features == NUM_FEATURES - 2
    features = encode_hands(torch.tensor([[0, 1, 2, 10, 13, 26]]))
    changed = features.clone()
    changed[..., 26] = 1 - changed[..., 26]  # flush
    changed[..., 28] += 0.5  # nobs
    assert torch.equal(model(features), model(changed))
    changed[..., 27] = 1 - changed[..., 27]  # suited is kept
    assert not torch.equal(model(features), model(changed))


