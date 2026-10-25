#!/usr/bin/env python

"""
discard_expt_2026.py

Supervised-learning experiment for cribbage discards.

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

Stage 1: the label functions, and an experiment
(`python -m cribbage.discard_expt_2026 k-study`) measuring how many
Monte Carlo samples K are needed for a good crib estimate.  The measure
is regret: the exact expected points lost by choosing the discard that
the K-sample estimate prefers, relative to the exact best discard.

Stage 2: labelled datasets (`python -m cribbage.discard_expt_2026
generate`).  Each row is a dealt hand with its H and C vectors; the
dealer flag is applied later, as the sign of C.  Hands are assigned to
train / val / test by a hash of their suit-canonical form, so a hand and
its suit-relabelled twins (which have identical labels) always land in
the same split.  Training sets use Monte Carlo C; val and test sets are
best generated with exact C.

Stage 3: a PyTorch model of the discard value (`python -m
cribbage.discard_expt_2026 train`).  Each of the 15 splits of a hand is
encoded as rank counts of the kept and thrown cards plus a few suit
features, which are invariant to relabelling the suits and to the order
of the cards.  The network either predicts one value (with the dealer
flag as an input) or two heads, H and C, combined as H +/- C.  Quality
is measured on exact val / test labels by regret.

Stage 4: head-to-head games (`python -m cribbage.discard_expt_2026
head-to-head`) between a trained model and the existing players.  The
model only chooses discards; it pegs at random, so that a match against
`simple-discard` (Simple's discard, random pegging) isolates discarding.

Stage 5: an ablation study (`ablate`, `ablate-report`) of the network and
training choices: size, heads, learning rate and schedule, initialisation,
standardisation, features, label noise and amount of data.
"""

import argparse
import copy
import dataclasses
import functools
import itertools
import json
import math
import multiprocessing
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cribbage.cards import KEEP_COMBINATIONS, Card, splits, unseen_cards
from cribbage.game import compare_players
from cribbage.player import CribbagePlayer
from cribbage.randomplayer import RandomCribbagePlayer
from cribbage.scoring import score_hand
from cribbage.simpleplayer import SimpleCribbagePlayer

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
#  How many Monte Carlo samples do we need?


def argmax_random_tiebreak(values: np.ndarray, rng: np.random.Generator) -> int:
    """argmax, breaking ties at random (matters when K is small)."""
    return int(rng.choice(np.flatnonzero(values >= values.max() - 1e-9)))


def _exact_labels_task(hand: Sequence[Card]) -> ExactLabels:
    return exact_labels(hand)


def compute_exact_labels(
    hands: np.ndarray, workers: int, cache: str | None = None
) -> list[ExactLabels]:
    """
    Exact labels for each hand, in parallel.  If `cache` names an .npz
    file for the same hands, labels are read from it; otherwise they are
    computed and written to it.
    """
    if cache is not None and os.path.exists(cache):
        data = np.load(cache)
        if np.array_equal(data["hands"], hands):
            return [
                ExactLabels(h, c, s)
                for h, c, s in zip(data["h"], data["c"], data["c_std"])
            ]
    with multiprocessing.Pool(workers) as pool:
        labels = pool.map(_exact_labels_task, [list(h) for h in hands], chunksize=1)
    if cache is not None:
        np.savez(
            cache,
            hands=hands,
            h=np.array([x.h for x in labels]),
            c=np.array([x.c for x in labels]),
            c_std=np.array([x.c_std for x in labels]),
        )
    return labels


def k_study(
    hands: np.ndarray,
    labels: list[ExactLabels],
    ks: Sequence[int],
    reps: int,
    seed: int,
    model: OpponentThrowModel = UniformThrows(),
) -> dict[str, dict[str, float]]:
    """
    For each K, and for both seats, measures how well the discard chosen
    by a K-sample crib estimate does against the exact best discard.

    - `crib_rmse`: root mean squared error of the C estimate
    - `optimal`: fraction of choices whose exact value ties the best
    - `regret`: mean exact expected points lost against the best discard
    - `regret_se`: standard error of that mean, across hands

    Also reports two baselines, "H only" (ignore the crib entirely) and
    "random".  Each hand is estimated `reps` times at each K.
    """
    rng = np.random.default_rng([seed, 1])
    results: dict[str, dict[str, float]] = {}

    # regrets[name][is_dealer] -> per-hand mean regret; optimal likewise
    regrets: dict[str, dict[bool, list[float]]] = {}
    optimal: dict[str, dict[bool, list[float]]] = {}
    rmse_sq: dict[str, list[float]] = {}

    def record(
        name: str, is_dealer: bool, h: np.ndarray, c: np.ndarray, idxs: list[int]
    ) -> None:
        """Scores the chosen split indices against the exact values."""
        true = candidate_values(h, c, is_dealer)
        best = true.max()
        regrets.setdefault(name, {}).setdefault(is_dealer, []).append(
            float(np.mean([best - true[i] for i in idxs]))
        )
        optimal.setdefault(name, {}).setdefault(is_dealer, []).append(
            float(np.mean([true[i] >= best - 1e-9 for i in idxs]))
        )

    for hand, exact in zip(hands, labels):
        h, c = exact.h, exact.c
        for is_dealer in (True, False):
            record(
                "H only",
                is_dealer,
                h,
                c,
                [argmax_random_tiebreak(h, rng) for _ in range(reps)],
            )
            record(
                "random",
                is_dealer,
                h,
                c,
                [int(rng.integers(NUM_KEEPS)) for _ in range(reps)],
            )
        for k in ks:
            name = f"K={k}"
            estimates = [crib_values_mc(list(hand), k, rng, model) for _ in range(reps)]
            rmse_sq.setdefault(name, []).append(
                float(np.mean([np.mean((e - c) ** 2) for e in estimates]))
            )
            for is_dealer in (True, False):
                record(
                    name,
                    is_dealer,
                    h,
                    c,
                    [
                        argmax_random_tiebreak(candidate_values(h, e, is_dealer), rng)
                        for e in estimates
                    ],
                )

    for name in regrets:
        row: dict[str, float] = {}
        if name in rmse_sq:
            row["crib_rmse"] = float(np.sqrt(np.mean(rmse_sq[name])))
        for is_dealer, label in ((True, "dealer"), (False, "nondealer")):
            r = np.array(regrets[name][is_dealer])
            row[f"{label}_regret"] = float(r.mean())
            row[f"{label}_regret_se"] = float(r.std(ddof=1) / np.sqrt(len(r)))
            row[f"{label}_optimal"] = float(np.mean(optimal[name][is_dealer]))
        results[name] = row
    return results


def print_k_study(
    labels: list[ExactLabels], results: dict[str, dict[str, float]]
) -> None:
    """Prints the label spread and the K-study results as a table."""
    h = np.array([x.h for x in labels])
    c = np.array([x.c for x in labels])
    c_std = np.array([x.c_std for x in labels])
    print(f"hands: {len(labels)}")
    print(
        "mean spread across the 15 splits (max - min):"
        f" H {np.mean(h.max(axis=1) - h.min(axis=1)):.2f},"
        f" C {np.mean(c.max(axis=1) - c.min(axis=1)):.2f}"
    )
    print(f"mean std of a single crib score: {c_std.mean():.2f}")
    print()
    print(
        f"{'':9} {'crib':>6} | {'dealer':^24} | {'non-dealer':^24}\n"
        f"{'':9} {'rmse':>6} | {'regret':>10} {'se':>6} {'opt%':>6} |"
        f" {'regret':>10} {'se':>6} {'opt%':>6}"
    )
    for name, r in results.items():
        rmse = f"{r['crib_rmse']:6.3f}" if "crib_rmse" in r else f"{'':6}"
        print(
            f"{name:9} {rmse} | "
            f"{r['dealer_regret']:10.4f} {r['dealer_regret_se']:6.4f}"
            f" {100 * r['dealer_optimal']:6.1f} | "
            f"{r['nondealer_regret']:10.4f} {r['nondealer_regret_se']:6.4f}"
            f" {100 * r['nondealer_optimal']:6.1f}"
        )


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
#  Ablation study

# Each variant changes one thing about the base run.  Keys of the
# overrides are TrainConfig fields, plus `k` (Monte Carlo samples behind
# the training labels) and `hands` (number of training hands).
BASE_DATA = {"k": 20, "hands": 100000}


def ablation_variants() -> list[tuple[str, str, dict[str, Any]]]:
    """The (group, label, overrides) of every run in the study."""
    variants: list[tuple[str, str, dict[str, Any]]] = [("base", "base", {})]
    for depth in (1, 2, 3, 4, 5):
        for width in (8, 16, 32, 64, 128, 256):
            if (depth, width) != (BASE_CONFIG.depth, BASE_CONFIG.width):
                variants.append(
                    ("size", f"{depth}x{width}", {"depth": depth, "width": width})
                )
    variants.append(("heads", "scalar head", {"two_head": False}))
    for lr in (3e-4, 1e-3, 5e-3, 1e-2):
        variants.append(("learning rate", f"lr {lr:g}", {"lr": lr}))
    for lr in (2e-3, 5e-4):
        variants.append(
            ("schedule", f"constant lr {lr:g}", {"schedule": "constant", "lr": lr})
        )
    variants.append(("init", "He init", {"init": "he"}))
    variants += [
        ("scaling", "standardise inputs", {"standardize_inputs": True}),
        ("scaling", "standardise targets", {"standardize_targets": True}),
        (
            "scaling",
            "standardise both",
            {"standardize_inputs": True, "standardize_targets": True},
        ),
    ]
    variants += [
        ("features", "no nobs", {"drop": ("nobs",)}),
        ("features", "no keep-flush", {"drop": ("flush",)}),
        ("features", "no discard-suited", {"drop": ("suited",)}),
        ("features", "ranks only", {"drop": ("nobs", "flush", "suited")}),
    ]
    for k in (1, 5, 100):
        variants.append(("label noise", f"K={k}", {"k": k}))
    for hands in (3000, 10000, 30000, 300000, 1000000):
        variants.append(("data", f"{hands} hands", {"hands": hands}))
    return variants


@functools.lru_cache(maxsize=2)
def _load_tensors(path: str) -> Tensors:
    return to_tensors(Dataset.load(path))


def _ablation_task(task: tuple[str, str, dict[str, Any], int, int]) -> dict[str, Any]:
    """Trains and tests one (variant, seed); returns its record."""
    group, label, overrides, seed, steps = task
    fields = {k: v for k, v in overrides.items() if k not in ("k", "hands")}
    data = {**BASE_DATA, **{k: v for k, v in overrides.items() if k in BASE_DATA}}
    hands, h, c = _load_tensors(f"data/discard_train_k{data['k']}.npz")
    n = data["hands"]
    train = (hands[:n], h[:n], c[:n])
    steps_per_epoch = math.ceil(n / BASE_CONFIG.batch_size)
    config = dataclasses.replace(
        BASE_CONFIG, seed=seed, epochs=math.ceil(steps / steps_per_epoch), **fields
    )
    start = time.time()
    model, history = train_model(
        config, train, _load_tensors("data/discard_val_exact.npz"), verbose=False
    )
    seconds = time.time() - start
    best = min(history, key=lambda epoch: epoch["mse"])
    return {
        "group": group,
        "label": label,
        "seed": seed,
        "steps": steps,
        "config": vars(config),
        "data": data,
        "params": sum(p.numel() for p in model.parameters()),
        "best_epoch": int(best["epoch"]),
        "epochs": config.epochs,
        "seconds": seconds,
        "test": evaluate(model, _load_tensors("data/discard_test_exact.npz")),
    }


def _init_ablation_worker(threads: int) -> None:
    torch.set_num_threads(threads)


def run_ablation(
    path: str,
    steps: int,
    seeds: int,
    procs: int,
    threads: int,
    groups: Sequence[str] | None = None,
) -> None:
    """
    Runs every variant (of the chosen groups, plus the base) for each
    seed, appending one JSON record per run to `path`.  Runs that are
    already in the file are skipped, so an interrupted study can be
    resumed.
    """
    done = set()
    if os.path.exists(path):
        with open(path) as results_file:
            for line in results_file:
                record = json.loads(line)
                done.add((record["label"], record["seed"], record["steps"]))
    tasks = [
        (group, label, overrides, seed, steps)
        for group, label, overrides in ablation_variants()
        if groups is None or group in groups or group == "base"
        for seed in range(seeds)
        if (label, seed, steps) not in done
    ]
    print(f"{len(tasks)} runs to do ({len(done)} already done)", flush=True)
    start = time.time()
    with (
        multiprocessing.Pool(procs, _init_ablation_worker, (threads,)) as pool,
        open(path, "a") as results_file,
    ):
        for i, record in enumerate(pool.imap_unordered(_ablation_task, tasks), 1):
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()
            print(
                f"[{i}/{len(tasks)} {time.time() - start:.0f}s] {record['label']}"
                f" seed {record['seed']}: test rmse"
                f" {math.sqrt(record['test']['mse']):.4f}",
                flush=True,
            )


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    return mean, math.sqrt(var)


def ablation_report(path: str) -> str:
    """
    Summarises a results file as markdown tables, one per group.  For
    each variant, over the seeds: test RMSE in points (the square root of
    the mean of the two seats' MSE), regret (points per deal, averaged
    over the seats), percentage of optimal choices, and training time.
    `change` is the paired change in RMSE from the base run, by seed.
    """
    runs: dict[str, list[dict[str, Any]]] = {}
    groups: dict[str, str] = {}
    with open(path) as results_file:
        for line in results_file:
            record = json.loads(line)
            runs.setdefault(record["label"], []).append(record)
            groups[record["label"]] = record["group"]

    def rmse(record: dict[str, Any]) -> float:
        return math.sqrt(record["test"]["mse"])

    base_rmse = {r["seed"]: rmse(r) for r in runs.get("base", [])}

    def row(label: str) -> str:
        records = runs[label]
        mean, sd = _mean_sd([rmse(r) for r in records])
        regret = sum(
            (r["test"]["dealer_regret"] + r["test"]["nondealer_regret"]) / 2
            for r in records
        ) / len(records)
        optimal = sum(
            (r["test"]["dealer_optimal"] + r["test"]["nondealer_optimal"]) / 2
            for r in records
        ) / len(records)
        seconds = sum(r["seconds"] for r in records) / len(records)
        paired = [
            rmse(r) - base_rmse[r["seed"]] for r in records if r["seed"] in base_rmse
        ]
        change = (
            f"{sum(paired) / len(paired):+.4f}" if paired and label != "base" else ""
        )
        return (
            f"| {label} | {records[0]['params']:,} | {mean:.4f} ± {sd:.4f} |"
            f" {change} | {regret:.4f} | {100 * optimal:.1f} | {seconds:.0f} |"
        )

    header = (
        "| variant | params | test RMSE (pts) | change | regret | optimal % | train s |\n"
        "|---|---|---|---|---|---|---|"
    )
    lines = []
    for group in dict.fromkeys(g for g in groups.values() if g != "base"):
        lines += [f"### {group}", "", header]
        lines += [row("base")] if "base" in runs else []
        lines += [row(label) for label in runs if groups[label] == group]
        lines.append("")
    # the size study as a grid of RMSE
    size_records = [
        r for label in runs for r in runs[label] if r["group"] in ("size", "base")
    ]
    if size_records:
        widths = sorted({r["config"]["width"] for r in size_records})
        lines += [
            "### size grid: test RMSE (pts), rows are depth, columns are width",
            "",
        ]
        lines += ["| depth | " + " | ".join(map(str, widths)) + " |"]
        lines += ["|---" * (len(widths) + 1) + "|"]
        for depth in sorted({r["config"]["depth"] for r in size_records}):
            cells = []
            for width in widths:
                cell = [
                    rmse(r)
                    for r in size_records
                    if r["config"]["depth"] == depth and r["config"]["width"] == width
                ]
                cells.append(f"{_mean_sd(cell)[0]:.4f}" if cell else "")
            lines.append(f"| {depth} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------
#  Head-to-head


class LearnedDiscardPlayer(RandomCribbagePlayer):
    """Discards using a trained DiscardNet; plays cards at random."""

    def __init__(self, model: DiscardNet) -> None:
        super().__init__()
        self.model = model.eval()

    def discard(
        self,
        is_dealer: bool,
        hand: list[Card],
        player_score: int,
        opponent_score: int,
    ) -> list[int]:
        """Throws the two cards of the split the model values most."""
        with torch.no_grad():
            features = encode_hands(torch.tensor([hand]))
            best = int(predict_values(self.model, features, is_dealer).argmax())
        return DISCARD_IDX[best].tolist()


def make_player(spec: str) -> CribbagePlayer:
    """
    Builds a player from a name: `random`, `simple` (Simple, full),
    `simple-discard` (Simple's discard, random pegging), or
    `model:PATH` (a saved DiscardNet's discard, random pegging).
    """
    if spec == "random":
        return RandomCribbagePlayer()
    if spec == "simple":
        return SimpleCribbagePlayer()
    if spec == "simple-discard":
        return SimpleCribbagePlayer(estimate_playcard=False)
    if spec.startswith("model:"):
        return LearnedDiscardPlayer(load_model(spec[len("model:") :]))
    raise ValueError(f"unknown player {spec!r}")


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
    Plays `games` games of a against b, spread over `workers` processes
    in fixed-size chunks (so the result does not depend on `workers`).
    The dealer of each game is random.  Returns a's wins and the number of
    games.
    """
    chunk = 100
    sizes = [chunk] * (games // chunk) + ([games % chunk] if games % chunk else [])
    tasks = [(spec_a, spec_b, n, seed * 100000 + i) for i, n in enumerate(sizes)]
    with multiprocessing.Pool(workers) as pool:
        return sum(pool.imap_unordered(_play_games, tasks)), games


# ------------------------------------------------------------
#  Command line


def run_k_study(args: argparse.Namespace) -> None:
    """Runs the Monte Carlo sample-size study."""
    hands = deal_hands(args.hands, np.random.default_rng([args.seed, 0]))
    labels = compute_exact_labels(hands, args.workers, args.cache)
    ks = [int(k) for k in args.ks.split(",")]
    print_k_study(labels, k_study(hands, labels, ks, args.reps, args.seed))


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
    half_width = 1.96 * (rate * (1 - rate) / games) ** 0.5
    print(
        f"{args.a} beat {args.b} in {wins}/{games} games: "
        f"{100 * rate:.1f}% +/- {100 * half_width:.1f}% (95% CI),"
        f" {time.time() - start:.0f}s"
    )


def run_ablate(args: argparse.Namespace) -> None:
    """Runs the ablation study, then prints its report."""
    groups = args.groups.split(",") if args.groups else None
    run_ablation(args.out, args.steps, args.seeds, args.procs, args.threads, groups)
    print(ablation_report(args.out))


def run_ablate_report(args: argparse.Namespace) -> None:
    """Prints the report of an ablation results file."""
    print(ablation_report(args.results))


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

    study = commands.add_parser("k-study", help=run_k_study.__doc__)
    study.add_argument("--hands", type=int, default=300)
    study.add_argument("--ks", default="1,2,5,10,20,50,100,200,500")
    study.add_argument("--reps", type=int, default=5)
    study.add_argument("--cache", help=".npz file for the exact labels")
    study.set_defaults(func=run_k_study)

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

    ablate = commands.add_parser("ablate", help=run_ablate.__doc__)
    ablate.add_argument("--out", default="data/ablation.jsonl")
    ablate.add_argument(
        "--steps", type=int, default=1500, help="gradient steps per run"
    )
    ablate.add_argument("--seeds", type=int, default=3)
    ablate.add_argument("--procs", type=int, default=4, help="runs in parallel")
    ablate.add_argument("--threads", type=int, default=2, help="threads per run")
    ablate.add_argument("--groups", help="comma-separated groups (default: all)")
    ablate.set_defaults(func=run_ablate)

    report = commands.add_parser("ablate-report", help=run_ablate_report.__doc__)
    report.add_argument("--results", default="data/ablation.jsonl")
    report.set_defaults(func=run_ablate_report)

    match = commands.add_parser("head-to-head", help=run_head_to_head.__doc__)
    match.add_argument("--a", required=True, help="player spec, see make_player")
    match.add_argument("--b", required=True, help="player spec, see make_player")
    match.add_argument("--games", type=int, default=2000)
    match.set_defaults(func=run_head_to_head)

    for sub in (study, generate, train, match):
        sub.add_argument("--seed", type=int, default=0)
    for sub in (study, generate, match):
        sub.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
