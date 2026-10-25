#!/usr/bin/env python

"""
discard_model.py

Supervised learning of cribbage discards.

For a dealt 6-card hand there are 15 ways to choose the four cards to
keep (`KEEP_COMBINATIONS`).  Each candidate discard has two expected
scores:

- H[i]: the expected score of the kept four cards, averaged over the
  starter card.  This is cheap enough to compute exactly.
- C[i]: the expected score of the crib, given the two cards thrown into
  it.  This depends on the starter and on the opponent's two throws, so
  it is estimated by Monte Carlo (or enumerated exactly, slowly).

The value of a candidate is H + C for the dealer and H - C for the
non-dealer: each player is maximising their own point differential, and
the opponent's hand score does not depend on our discard.

The label functions are here; `discard_study` has an experiment on how many
Monte Carlo samples K a good crib estimate needs, and an ablation study of
the network and training choices.

Labelled datasets (`python -m cribbage.discard_model
generate`).  Each row is a dealt hand with its H and C vectors; the
dealer flag is applied later, as the sign of C.  Hands are assigned to
train / val / test by a hash of their suit-canonical form, so a hand and
its suit-relabelled twins (which have identical labels) always land in
the same split.  Training sets use Monte Carlo C; val and test sets are
best generated with exact C.

A PyTorch model of the discard value (`python -m
cribbage.discard_model train`).  Each of the 15 splits of a hand is
encoded as rank counts of the kept and thrown cards plus a few suit
features, which are invariant to relabelling the suits and to the order
of the cards.  The network either predicts one value (with the dealer
flag as an input) or two heads, H and C, combined as H +/- C.  Quality
is measured on exact val / test labels by regret.

Head-to-head games (`python -m cribbage.discard_model
head-to-head`) between a trained model and the existing players.  The
model only chooses discards; it pegs at random, so that a match against
`simple-discard` (Simple's discard, random pegging) isolates discarding.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import multiprocessing
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cribbage.cards import KEEP_COMBINATIONS, Card, splits, unseen_cards
from cribbage.discard_policy import DiscardPolicy
from cribbage.game import compare_players, count_wins, win_ci
from cribbage.player import DiscardView
from cribbage.scoring import score_hand
from cribbage.specs import make_player

NUM_KEEPS = len(KEEP_COMBINATIONS)

# ------------------------------------------------------------
#  Hands, candidate discards


def deal_hands(num: int, rng: np.random.Generator) -> np.ndarray:
    """Returns a (num, 6) array of random dealt hands, each sorted."""
    return np.array(
        [np.sort(rng.choice(52, size=6, replace=False)) for _ in range(num)]
    )


# ------------------------------------------------------------
#  Opponent throw model


class OpponentThrowModel(Protocol):
    """
    A distribution over crib completions.  A completion is a triple
    `(starter, throw_a, throw_b)`: the starter card and the opponent's
    two crib cards, all drawn from the cards we have not seen.

    The uniform model below treats the three as a random subset of the
    unseen cards.  A model of a competent opponent would sample the
    opponent's six-card hand from the unseen cards, apply their discard
    policy to get the throws, and draw the starter from what remains.
    Nothing else in this module needs to change.
    """

    def sample(
        self, rng: np.random.Generator, unseen: Sequence[Card], k: int
    ) -> np.ndarray:
        """Draws k completions, as a (k, 3) array."""
        ...

    def support(self, unseen: Sequence[Card]) -> tuple[np.ndarray, np.ndarray]:
        """
        Enumerates every completion with non-zero probability: returns
        a (m, 3) array of completions and a length-m vector of
        probabilities that sums to 1.
        """
        ...


class UniformThrows:
    """The opponent throws two random cards; the starter is random."""

    def sample(
        self, rng: np.random.Generator, unseen: Sequence[Card], k: int
    ) -> np.ndarray:
        order = rng.random((k, len(unseen))).argsort(axis=1)[:, :3]
        return np.asarray(unseen)[order]

    def support(self, unseen: Sequence[Card]) -> tuple[np.ndarray, np.ndarray]:
        completions = [
            (starter, a, b)
            for starter in unseen
            for a, b in itertools.combinations([c for c in unseen if c != starter], 2)
        ]
        return np.array(completions), np.full(len(completions), 1 / len(completions))


# ------------------------------------------------------------
#  Labels


def hand_values(hand: Sequence[Card]) -> np.ndarray:
    """
    H: for each of the 15 splits, the exact expected score of the kept
    four cards, averaged over the 46 possible starters.
    """
    unseen = unseen_cards(hand)
    return np.array(
        [
            sum(score_hand(keep, starter) for starter in unseen) / len(unseen)
            for keep, _ in splits(hand)
        ]
    )


def crib_scores(hand: Sequence[Card], completions: np.ndarray) -> np.ndarray:
    """
    Scores the crib for every (split, completion) pair: returns a
    (15, m) array.  Every split sees the same completions (common
    random numbers), which makes comparisons between splits much less
    noisy than independent samples would.
    """
    comps = completions.tolist()
    return np.array(
        [
            [score_hand([d1, d2, a, b], starter, True) for starter, a, b in comps]
            for _, (d1, d2) in splits(hand)
        ],
        dtype=np.int16,
    )


def crib_values_mc(
    hand: Sequence[Card],
    k: int,
    rng: np.random.Generator,
    model: OpponentThrowModel = UniformThrows(),
) -> np.ndarray:
    """C estimated from k sampled completions."""
    completions = model.sample(rng, unseen_cards(hand), k)
    return crib_scores(hand, completions).mean(axis=1)


@dataclass
class ExactLabels:
    """Exact H and C for one hand, and the spread of a single crib score."""

    h: np.ndarray
    c: np.ndarray
    c_std: np.ndarray


def exact_labels(
    hand: Sequence[Card], model: OpponentThrowModel = UniformThrows()
) -> ExactLabels:
    """
    Exact H and C, by enumerating every completion.  For the uniform
    model that is 46 * C(45, 2) = 45,540 cribs per split.
    """
    completions, weights = model.support(unseen_cards(hand))
    scores = crib_scores(hand, completions)
    mean = scores @ weights
    var = (scores.astype(float) ** 2) @ weights - mean**2
    return ExactLabels(hand_values(hand), mean, np.sqrt(np.maximum(var, 0)))


# candidate_values works on numpy arrays and torch tensors alike
ArrayLike = TypeVar("ArrayLike", np.ndarray, torch.Tensor)


def candidate_values(h: ArrayLike, c: ArrayLike, is_dealer: bool) -> ArrayLike:
    """Expected point differential of each split: H + C dealing, H - C not."""
    return h + c if is_dealer else h - c


# ------------------------------------------------------------
#  Datasets

SPLITS = ("train", "val", "test")
# percentage of suit-canonical hand classes assigned to each split
SPLIT_PERCENT = (90, 5, 5)
CHUNK_SIZE = 500

# all 24 relabelings of the suits
_SUIT_PERMS = np.array(list(itertools.permutations(range(4))))


def canonical_keys(hands: np.ndarray) -> np.ndarray:
    """
    For each hand in a (n, 6) array, an integer that is the same for all
    24 suit-relabelings of the hand and different for hands that are not
    relabelings of each other.  H and C are invariant under relabelling
    the suits, so hands with the same key have the same labels.
    """
    ranks, suits = hands % 13, hands // 13
    place_values = 52 ** np.arange(6, dtype=np.int64)
    keys = []
    for perm in _SUIT_PERMS:
        relabelled = np.sort(ranks + 13 * perm[suits], axis=1)
        keys.append((relabelled.astype(np.int64) * place_values).sum(axis=1))
    return np.min(keys, axis=0)


def split_of(hands: np.ndarray) -> np.ndarray:
    """The split (index into SPLITS) of each hand in a (n, 6) array."""
    keys = canonical_keys(hands).astype(np.uint64)
    # multiplicative hash (wraps around in uint64) so splits do not
    # follow the order of the keys
    mixed = (keys * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)
    bucket = (mixed % np.uint64(100)).astype(int)
    return np.searchsorted(np.cumsum(SPLIT_PERCENT), bucket, side="right")


def sample_split_hands(num: int, split: int, rng: np.random.Generator) -> np.ndarray:
    """Deals random hands, keeping only those that belong to `split`."""
    kept: list[np.ndarray] = []
    have = 0
    while have < num:
        batch = deal_hands(2 * num + 100, rng)
        batch = batch[split_of(batch) == split]
        kept.append(batch)
        have += len(batch)
    return np.concatenate(kept)[:num]


@dataclass
class Dataset:
    """
    Labelled hands: `hands` is (N, 6) card values, `h` and `c` are (N, 15)
    expected scores for the splits of each hand.  `k` is the number of
    Monte Carlo samples behind `c`, or None if it is exact.
    """

    hands: np.ndarray
    h: np.ndarray
    c: np.ndarray
    k: int | None
    split: str
    seed: int
    opponent: str = "uniform"

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            hands=self.hands,
            h=self.h,
            c=self.c,
            k=-1 if self.k is None else self.k,
            split=self.split,
            seed=self.seed,
            opponent=self.opponent,
        )

    @classmethod
    def load(cls, path: str) -> "Dataset":
        data = np.load(path)
        k = int(data["k"])
        return cls(
            data["hands"],
            data["h"],
            data["c"],
            None if k < 0 else k,
            str(data["split"]),
            int(data["seed"]),
            str(data["opponent"]),
        )


def _label_chunk(
    task: tuple[int, int, int, int, int | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deals and labels one chunk of hands; the rng depends only on the task."""
    seed, split, chunk, num, k = task
    rng = np.random.default_rng([seed, 2, split, chunk])
    hands = sample_split_hands(num, split, rng)
    if k is None:
        exact = [exact_labels(hand.tolist()) for hand in hands]
        h = np.array([x.h for x in exact])
        c = np.array([x.c for x in exact])
    else:
        h = np.array([hand_values(hand.tolist()) for hand in hands])
        c = np.array([crib_values_mc(hand.tolist(), k, rng) for hand in hands])
    return hands.astype(np.int8), h.astype(np.float32), c.astype(np.float32)


def generate_dataset(
    split: str, num: int, k: int | None, seed: int, workers: int
) -> Dataset:
    """
    Generates `num` labelled hands for the given split, with C estimated
    from `k` Monte Carlo samples (None: exact).  The result depends on
    the arguments but not on `workers`.
    """
    sizes = [CHUNK_SIZE] * (num // CHUNK_SIZE)
    if num % CHUNK_SIZE:
        sizes.append(num % CHUNK_SIZE)
    tasks = [(seed, SPLITS.index(split), i, n, k) for i, n in enumerate(sizes)]
    parts = []
    with multiprocessing.Pool(workers) as pool:
        for done, part in enumerate(pool.imap(_label_chunk, tasks), 1):
            parts.append(part)
            if len(tasks) >= 10 and done % max(1, len(tasks) // 10) == 0:
                print(f"  {done}/{len(tasks)} chunks")
    hands, h, c = (np.concatenate(arrays) for arrays in zip(*parts))
    return Dataset(hands, h, c, k, split, seed)


# ------------------------------------------------------------
#  Features and model

KEEP_IDX = torch.tensor(KEEP_COMBINATIONS)
DISCARD_IDX = torch.tensor(
    [[i for i in range(6) if i not in keep] for keep in KEEP_COMBINATIONS]
)
JACK = 10  # rank index of the jack, in cards.CARD_FACES
# keep rank counts, discard rank counts, keep flush, discard suited, nobs chance
NUM_FEATURES = 13 + 13 + 3


def encode_hands(hands: torch.Tensor) -> torch.Tensor:
    """
    Encodes a (B, 6) long tensor of dealt hands as (B, 15, NUM_FEATURES)
    features, one row per split:

    - rank counts of the four kept cards (13) and of the two thrown cards (13)
    - 1 if the kept cards are all one suit (a four-card flush)
    - 1 if the thrown cards share a suit (the crib flush needs this)
    - the chance of scoring nobs: for each kept jack, the fraction of the
      46 unseen cards that would be a starter of the jack's suit
    """
    keep = hands[:, KEEP_IDX]  # (B, 15, 4)
    discard = hands[:, DISCARD_IDX]  # (B, 15, 2)
    keep_counts = F.one_hot(keep % 13, 13).sum(dim=2)
    discard_counts = F.one_hot(discard % 13, 13).sum(dim=2)
    keep_suits = keep // 13
    flush = (keep_suits == keep_suits[..., :1]).all(dim=-1)
    discard_suits = discard // 13
    suited = discard_suits[..., 0] == discard_suits[..., 1]
    unseen_per_suit = 13 - F.one_hot(hands // 13, 4).sum(dim=1)  # (B, 4)
    unseen_in_suit = torch.gather(
        unseen_per_suit[:, None, :].expand(-1, NUM_KEEPS, -1), 2, keep_suits
    )
    nobs = ((keep % 13 == JACK) * unseen_in_suit).sum(dim=-1) / 46
    return torch.cat(
        [
            keep_counts,
            discard_counts,
            flush[..., None],
            suited[..., None],
            nobs[..., None],
        ],
        dim=-1,
    ).float()


# the features that can be dropped, by name, and their column
FEATURE_GROUPS = {"flush": 26, "suited": 27, "nobs": 28}


class DiscardNet(nn.Module):
    """
    An MLP applied independently to each split's features.  With
    `two_head` it outputs (H, C) estimates; otherwise it takes the dealer
    flag as an extra input and outputs a single value.

    The remaining arguments are for ablation studies.  `init` is
    "default" (PyTorch's) or "he" (He normal weights, zero biases, for the
    hidden layers).  `drop` names features (see FEATURE_GROUPS) to leave
    out.  `standardize_inputs` and `standardize_targets` rescale the
    inputs inside the network, and the outputs back to points, using
    statistics that train_model fits; the buffers only exist when these
    are set, so models saved without them still load.
    """

    columns: torch.Tensor
    in_mean: torch.Tensor | None
    in_std: torch.Tensor | None
    out_mean: torch.Tensor | None
    out_std: torch.Tensor | None

    def __init__(
        self,
        two_head: bool,
        width: int,
        depth: int,
        init: str = "default",
        drop: Sequence[str] = (),
        standardize_inputs: bool = False,
        standardize_targets: bool = False,
    ) -> None:
        super().__init__()
        self.two_head = two_head
        dropped = {FEATURE_GROUPS[name] for name in drop}
        columns = [i for i in range(NUM_FEATURES) if i not in dropped]
        if not two_head:
            columns.append(NUM_FEATURES)  # the dealer flag
        self.register_buffer("columns", torch.tensor(columns), persistent=False)
        size = len(columns)
        num_outputs = 2 if two_head else 1
        self.register_buffer(
            "in_mean", torch.zeros(size) if standardize_inputs else None
        )
        self.register_buffer("in_std", torch.ones(size) if standardize_inputs else None)
        self.register_buffer(
            "out_mean", torch.zeros(num_outputs) if standardize_targets else None
        )
        self.register_buffer(
            "out_std", torch.ones(num_outputs) if standardize_targets else None
        )
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.ReLU()]
            size = width
        layers.append(nn.Linear(size, num_outputs))
        if init == "he":
            for layer in layers[:-1]:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)
        elif init != "default":
            raise ValueError(f"unknown init {init!r}")
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features[..., self.columns]
        if self.in_mean is not None and self.in_std is not None:
            x = (x - self.in_mean) / self.in_std
        out = self.layers(x)
        if self.out_mean is not None and self.out_std is not None:
            out = out * self.out_std + self.out_mean
        return out


def predict_values(
    model: DiscardNet, features: torch.Tensor, is_dealer: bool
) -> torch.Tensor:
    """Predicted value of each split, as a (B, 15) tensor."""
    if model.two_head:
        out = model(features)
        h, c = out[..., 0], out[..., 1]
        return h + c if is_dealer else h - c
    flag = torch.full_like(features[..., :1], float(is_dealer))
    return model(torch.cat([features, flag], dim=-1))[..., 0]


def loss_fn(
    model: DiscardNet, features: torch.Tensor, h: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Squared error against the labels; both seats for the scalar model."""
    if model.two_head:
        out = model(features)
        return F.mse_loss(out[..., 0], h) + F.mse_loss(out[..., 1], c)
    dealer_loss, nondealer_loss = (
        F.mse_loss(
            predict_values(model, features, dealer), candidate_values(h, c, dealer)
        )
        for dealer in (True, False)
    )
    return (dealer_loss + nondealer_loss) / 2


# ------------------------------------------------------------
#  Evaluation

Tensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def to_tensors(dataset: Dataset) -> Tensors:
    """The hands (long), H and C (float) of a dataset, as tensors."""
    return (
        torch.from_numpy(dataset.hands).long(),
        torch.from_numpy(dataset.h).float(),
        torch.from_numpy(dataset.c).float(),
    )


def choice_stats(pred: torch.Tensor, true: torch.Tensor) -> tuple[float, float]:
    """
    Regret and optimal rate of choosing argmax(pred) when `true` holds the
    real values: both (B, 15).  Ties in `pred` go to the first split.
    """
    chosen = true.gather(1, pred.argmax(dim=1, keepdim=True))[:, 0]
    best = true.max(dim=1).values
    return float((best - chosen).mean()), float((chosen >= best - 1e-6).float().mean())


@torch.no_grad()
def evaluate(
    model: DiscardNet, data: Tensors, batch_size: int = 4096
) -> dict[str, float]:
    """
    Mean squared error of the predicted values, and the regret and
    optimal rate of the discards the model would choose, for each seat,
    against the labels in `data`.  `mse` averages the two seats.
    """
    model.eval()
    hands, h, c = data
    sums: dict[str, float] = {}
    for dealer, seat in ((True, "dealer"), (False, "nondealer")):
        squared, regret, optimal = 0.0, 0.0, 0.0
        for start in range(0, len(hands), batch_size):
            chunk = slice(start, start + batch_size)
            true = candidate_values(h[chunk], c[chunk], dealer)
            pred = predict_values(model, encode_hands(hands[chunk]), dealer)
            n = len(true)
            squared += float(((pred - true) ** 2).mean()) * n
            chunk_regret, chunk_optimal = choice_stats(pred, true)
            regret += chunk_regret * n
            optimal += chunk_optimal * n
        sums[f"{seat}_mse"] = squared / len(hands)
        sums[f"{seat}_regret"] = regret / len(hands)
        sums[f"{seat}_optimal"] = optimal / len(hands)
    sums["mse"] = (sums["dealer_mse"] + sums["nondealer_mse"]) / 2
    return sums


def h_only_stats(data: Tensors) -> dict[str, float]:
    """The reference chooser that maximises H and ignores the crib."""
    _, h, c = data
    stats = {}
    for dealer, seat in ((True, "dealer"), (False, "nondealer")):
        regret, optimal = choice_stats(h, candidate_values(h, c, dealer))
        stats[f"{seat}_regret"], stats[f"{seat}_optimal"] = regret, optimal
    return stats


# ------------------------------------------------------------
#  Training


@dataclass
class TrainConfig:
    """Hyperparameters of a training run."""

    two_head: bool = True
    width: int = 256
    depth: int = 3
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 2e-3
    seed: int = 0
    # the remaining fields are for ablation studies
    schedule: str = "onecycle"  # or "constant"
    init: str = "default"  # or "he"
    standardize_inputs: bool = False
    standardize_targets: bool = False
    drop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.drop = tuple(self.drop)  # a list, if it was read from JSON

    def make_net(self) -> DiscardNet:
        """An untrained network with this configuration's architecture."""
        return DiscardNet(
            self.two_head,
            self.width,
            self.depth,
            self.init,
            self.drop,
            self.standardize_inputs,
            self.standardize_targets,
        )


BASE_CONFIG = TrainConfig()


@torch.no_grad()
def fit_standardisation(model: DiscardNet, train: Tensors) -> None:
    """
    Sets the input and target statistics of a network built with
    `standardize_inputs` / `standardize_targets` from the training data
    (inputs: a sample of 20,000 hands; the dealer flag is left alone).
    """
    hands, h, c = train
    if model.in_mean is not None and model.in_std is not None:
        sample = encode_hands(hands[:20000]).reshape(-1, NUM_FEATURES)
        feature_columns = model.columns[model.columns < NUM_FEATURES]
        x = sample[:, feature_columns]
        std = x.std(dim=0)
        model.in_mean[: len(feature_columns)] = x.mean(dim=0)
        model.in_std[: len(feature_columns)] = torch.where(
            std < 1e-6, torch.ones_like(std), std
        )
    if model.out_mean is not None and model.out_std is not None:
        if model.two_head:
            targets = [h, c]
        else:
            targets = [torch.cat([(h + c).flatten(), (h - c).flatten()])]
        model.out_mean[:] = torch.stack([t.mean() for t in targets])
        model.out_std[:] = torch.stack([t.std() for t in targets])


def train_model(
    config: TrainConfig, train: Tensors, val: Tensors, verbose: bool = True
) -> tuple[DiscardNet, list[dict[str, float]]]:
    """
    Trains a DiscardNet with AdamW and, by default, a one-cycle learning
    rate.  Returns the weights from the epoch with the lowest validation
    MSE, and the per-epoch history.
    """
    torch.manual_seed(config.seed)
    model = config.make_net()
    hands, h, c = train
    fit_standardisation(model, train)
    steps_per_epoch = (len(hands) + config.batch_size - 1) // config.batch_size
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr)
    schedule = (
        torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=config.lr, total_steps=config.epochs * steps_per_epoch
        )
        if config.schedule == "onecycle"
        else None
    )
    best_mse, best_state = float("inf"), copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(config.epochs):
        model.train()
        start = time.time()
        order = torch.randperm(len(hands))
        train_loss = 0.0
        for first in range(0, len(hands), config.batch_size):
            idx = order[first : first + config.batch_size]
            loss = loss_fn(model, encode_hands(hands[idx]), h[idx], c[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if schedule is not None:
                schedule.step()
            train_loss += loss.item() * len(idx)
        stats = evaluate(model, val)
        stats["train_loss"] = train_loss / len(hands)
        stats["epoch"] = epoch + 1
        history.append(stats)
        if stats["mse"] < best_mse:
            best_mse, best_state = stats["mse"], copy.deepcopy(model.state_dict())
        if verbose:
            print(
                f"epoch {epoch + 1:2d}  train {stats['train_loss']:.4f}"
                f"  val mse {stats['mse']:.4f}"
                f"  regret {stats['dealer_regret']:.4f}/{stats['nondealer_regret']:.4f}"
                f"  ({time.time() - start:.0f}s)",
                flush=True,
            )
    model.load_state_dict(best_state)
    return model, history


def save_model(model: DiscardNet, config: TrainConfig, path: str) -> None:
    """Writes the weights and the hyperparameters needed to rebuild them."""
    torch.save({"config": vars(config), "state": model.state_dict()}, path)


def load_model(path: str) -> DiscardNet:
    """Reads a model written by `save_model`."""
    saved = torch.load(path)
    config = TrainConfig(**saved["config"])
    model = config.make_net()
    model.load_state_dict(saved["state"])
    return model.eval()


# ------------------------------------------------------------
#  Head-to-head


class NetDiscardPolicy(DiscardPolicy):
    """Discards using a trained DiscardNet."""

    def __init__(self, model: DiscardNet) -> None:
        super().__init__()
        self.model = model.eval()

    @classmethod
    def load(cls, path: str) -> NetDiscardPolicy:
        return cls(load_model(path))

    def choose(self, view: DiscardView) -> tuple[Card, Card]:
        """Throws the two cards of the split the model values most."""
        with torch.no_grad():
            features = encode_hands(torch.tensor([view.hand]))
            best = int(predict_values(self.model, features, view.is_dealer).argmax())
        first, second = (view.hand[i] for i in DISCARD_IDX[best].tolist())
        return first, second


def _play_games(task: tuple[str, str, int, int]) -> int:
    """Plays `num` games of spec_a against spec_b; returns a's wins."""
    spec_a, spec_b, num, seed = task
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    return compare_players([make_player(spec_a), make_player(spec_b)], num)[0]


def play_match(
    spec_a: str, spec_b: str, games: int, seed: int, workers: int
) -> tuple[int, int]:
    """
    Plays `games` games of a against b (see `count_wins`).  The dealer of
    each game is random.  Returns a's wins and the number of games.
    """
    return count_wins(_play_games, spec_a, spec_b, games, seed, workers), games


# ------------------------------------------------------------
#  Command line


def run_generate(args: argparse.Namespace) -> None:
    """Generates a labelled dataset file."""
    k = None if args.k == "exact" else int(args.k)
    start = time.time()
    dataset = generate_dataset(args.split, args.hands, k, args.seed, args.workers)
    dataset.save(args.out)
    print(
        f"wrote {len(dataset.hands)} {args.split} hands (k={args.k}) to {args.out}"
        f" in {time.time() - start:.0f}s"
    )


def run_head_to_head(args: argparse.Namespace) -> None:
    """Plays games between two players, e.g. model:data/runs/two_k100.pt."""
    start = time.time()
    wins, games = play_match(args.a, args.b, args.games, args.seed, args.workers)
    rate = wins / games
    half_width = win_ci(rate, games)
    print(
        f"{args.a} beat {args.b} in {wins}/{games} games: "
        f"{100 * rate:.1f}% +/- {100 * half_width:.1f}% (95% CI),"
        f" {time.time() - start:.0f}s"
    )


def run_train(args: argparse.Namespace) -> None:
    """Trains a discard value network on a generated dataset."""
    torch.set_num_threads(args.threads)
    config = TrainConfig(
        args.head == "two",
        args.width,
        args.depth,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
    )
    train, val, test = (
        to_tensors(Dataset.load(path)) for path in (args.train, args.val, args.test)
    )
    model, history = train_model(config, train, val)
    result = {"config": vars(config), "train": args.train, "history": history}
    result["test"] = evaluate(model, test)
    result["test_h_only"] = h_only_stats(test)
    print("test:", json.dumps(result["test"]))
    print("test, H only:", json.dumps(result["test_h_only"]))
    if args.out:
        save_model(model, config, args.out)
    if args.results:
        with open(args.results, "w") as results_file:
            json.dump(result, results_file, indent=1)


def main() -> None:
    """Experiments on supervised cribbage discard models."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help=run_generate.__doc__)
    generate.add_argument("--split", choices=SPLITS, required=True)
    generate.add_argument("--hands", type=int, required=True)
    generate.add_argument("--k", default="20", help="Monte Carlo samples, or 'exact'")
    generate.add_argument("--out", required=True, help=".npz file to write")
    generate.set_defaults(func=run_generate)

    train = commands.add_parser("train", help=run_train.__doc__)
    train.add_argument("--train", required=True, help="training dataset .npz")
    train.add_argument("--val", default="data/discard_val_exact.npz")
    train.add_argument("--test", default="data/discard_test_exact.npz")
    train.add_argument("--head", choices=("two", "scalar"), default="two")
    train.add_argument("--width", type=int, default=256)
    train.add_argument("--depth", type=int, default=3)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--lr", type=float, default=2e-3)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--out", help="file to save the model to")
    train.add_argument("--results", help="file to write the results to (JSON)")
    train.set_defaults(func=run_train)

    match = commands.add_parser("head-to-head", help=run_head_to_head.__doc__)
    match.add_argument("--a", required=True, help="player spec, see make_player")
    match.add_argument("--b", required=True, help="player spec, see make_player")
    match.add_argument("--games", type=int, default=2000)
    match.set_defaults(func=run_head_to_head)

    for sub in (generate, train, match):
        sub.add_argument("--seed", type=int, default=0)
    for sub in (generate, match):
        sub.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
