#!/usr/bin/env python

"""
discard_pegging.py

Does the choice of discard depend on how the kept cards peg?

The discard network values a split as the expected hand score plus or
minus the expected crib score.  It ignores pegging, which is worth about
six points a hand to each player, and which depends on the four cards
kept and on whether the player is the dealer.  This module measures how
much that matters, before any model is built for it:

- `run` estimates, for each of 15 splits of a test hand and for each seat,
  the mean pegging margin of keeping that split, playing a pegging
  network on both sides against an opponent whose hand is what a discard
  network keeps from six random cards.  The opponent's cards and the
  starter are shared by all 15 splits of a sample, and the whole
  estimate is made twice, with independent samples.
- `report` compares the choice that maximises hand and crib with the
  choice that also counts pegging.  Choosing with one replicate and
  scoring with the other gives an estimate of the gain that is not
  inflated by picking the noisiest split.

- `labels` estimates the pegging margin of every split for many training
  hands (one replicate, few samples; the network averages the noise out).
- `train` fits a scalar `DiscardNet` (dealer flag as input) to those margins.
  `PegAwareDiscardPolicy` adds its output to the hand and crib values of
  the base discard network; the spec is `model:BASE@PEGNET`.
- `duel` plays duplicate hands, pegging and show, of a discard policy against
  a base policy, both against the same opponent.

Invoke as `python -m cribbage.discard_pegging --help`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import time
from collections.abc import Sequence

import numpy as np
import torch

from cribbage.cards import KEEP_COMBINATIONS, Card
from cribbage.discard_model import (
    DISCARD_IDX,
    KEEP_IDX,
    Dataset,
    DiscardNet,
    TrainConfig,
    encode_hands,
    load_model,
    predict_values,
    save_model,
)
from cribbage.pegging import PegState
from cribbage.pegging_model import NetPeggingPolicy
from cribbage.discard_policy import DiscardPolicy
from cribbage.player import DiscardView
from cribbage.scoring import score_hand

NUM_SPLITS = len(KEEP_COMBINATIONS)


def pegging_values(
    hand: Sequence[Card],
    is_dealer: bool,
    peg: NetPeggingPolicy,
    discard_model: DiscardNet,
    samples: int,
    seed: str,
) -> np.ndarray:
    """
    The mean pegging margin (mine less the opponent's) of keeping each of the
    15 splits of `hand`, over `samples` draws of the opponent's six cards and
    the starter.  Both sides peg with `peg`; the opponent keeps what
    `discard_model` keeps.
    """
    rng = random.Random(seed)
    gen = np.random.default_rng(rng.getrandbits(63))
    others = np.array([c for c in range(52) if c not in hand])
    picks = others[np.argsort(gen.random((samples, len(others))), axis=1)]
    dealt, starters = picks[:, :6], picks[:, 6]
    with torch.no_grad():
        values = predict_values(
            discard_model, encode_hands(torch.as_tensor(dealt)), not is_dealer
        )
    best = values.argmax(dim=1).numpy()
    rows = np.arange(samples)[:, None]
    opp_kept = dealt[rows, KEEP_IDX.numpy()[best]].tolist()
    opp_thrown = dealt[rows, DISCARD_IDX.numpy()[best]].tolist()
    dealer = 0 if is_dealer else 1
    totals = np.zeros(NUM_SPLITS)
    for s in range(samples):
        for i, keep in enumerate(KEEP_COMBINATIONS):
            mine = [hand[j] for j in keep]
            thrown = [hand[j] for j in range(6) if j not in keep]
            state = PegState.start([mine, opp_kept[s]], dealer)
            discards = (thrown, opp_thrown[s])
            peg.reseed(f"{seed}/{s}")
            while not state.done:
                seat = state.turn
                view = state.view(
                    seat, discards[seat], int(starters[s]), seat == dealer, 0, 0
                )
                state.play_card(peg.choose(view))
            totals[i] += state.margin()
    return totals / samples


def split_values(h: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Hand and crib value of each split, as (hands, 2 seats, 15): non-dealer, dealer."""
    return np.stack([h - c, h + c], axis=1)


def analyse(u: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """
    `u` is (N, 2, 15), the hand and crib value of each split by seat (0 non-dealer,
    1 dealer).  `p` is (N, 2, 2, 15), two independent estimates of the pegging
    margin of each split.  For each seat: how often counting pegging changes the
    choice, the gain in the full value (hand + crib + pegging) from the changed
    choice, scored on the replicate that did not choose it, and the spread of the
    pegging value across the splits.  Each is a mean over hands, with the
    standard error of that mean in `<name>_se`, as a 2-vector by seat.
    """
    n = len(u)
    base = u.argmax(axis=2)  # (N, 2)
    rows = np.arange(n)[:, None]
    seats = np.arange(2)[None, :]
    gains = []
    changed = []
    for chooser, scorer in ((0, 1), (1, 0)):
        pick = (u + p[:, :, chooser]).argmax(axis=2)
        full = u + p[:, :, scorer]
        gains.append(full[rows, seats, pick] - full[rows, seats, base])
        changed.append((pick != base).astype(float))
    mean_p = p.mean(axis=2)
    stats = {
        "gain": np.mean(gains, axis=0),
        "changed": np.mean(changed, axis=0),
        "spread": mean_p.max(axis=2) - mean_p.min(axis=2),
        "noise": np.sqrt(((p[:, :, 0] - p[:, :, 1]) ** 2 / 2).mean(axis=2)),
        "mean_peg": mean_p.mean(axis=2),
    }
    result = {}
    for name, values in stats.items():
        result[name] = values.mean(axis=0)
        result[name + "_se"] = values.std(axis=0, ddof=1) / np.sqrt(n)
    return result


def format_report(stats: dict[str, np.ndarray], hands: int, samples: int) -> str:
    lines = [
        f"{hands} test hands, {samples} samples per replicate, 2 replicates",
        f"{'':28}{'non-dealer':>20}{'dealer':>20}",
    ]
    rows = [
        ("choice changes (fraction)", "changed"),
        ("gain in full value, pts/hand", "gain"),
        ("pegging spread over splits", "spread"),
        ("pegging noise sd per split", "noise"),
        ("mean pegging margin", "mean_peg"),
    ]
    for label, name in rows:
        cells = "".join(
            f"{stats[name][s]:>12.3f} ± {1.96 * stats[name + '_se'][s]:.3f}"
            for s in range(2)
        )
        lines.append(f"{label:28}{cells}")
    return "\n".join(lines)


# ------------------------------------------------------------
#  Running the study

_WORKER: dict[str, object] = {}


def _init_worker(peg_path: str, discard_path: str) -> None:
    torch.set_num_threads(1)
    _WORKER["peg"] = NetPeggingPolicy.load(peg_path)
    _WORKER["discard"] = load_model(discard_path)


def _task(task: tuple[int, tuple[int, ...], int, int, int, int]) -> tuple[int, int, int, np.ndarray]:
    index, hand, role, rep, samples, seed = task
    peg = _WORKER["peg"]
    model = _WORKER["discard"]
    assert isinstance(peg, NetPeggingPolicy) and isinstance(model, DiscardNet)
    values = pegging_values(
        hand, bool(role), peg, model, samples, f"{seed}/{index}/{role}/{rep}"
    )
    return index, role, rep, values


def run_study(
    dataset: Dataset,
    hands: int,
    samples: int,
    peg_path: str,
    discard_path: str,
    seed: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the split values `u` (N, 2, 15) and pegging estimates `p` (N, 2, 2, 15)."""
    u = split_values(dataset.h[:hands], dataset.c[:hands])
    tasks = [
        (i, tuple(int(c) for c in dataset.hands[i]), role, rep, samples, seed)
        for i in range(hands)
        for role in (0, 1)
        for rep in (0, 1)
    ]
    p = np.zeros((hands, 2, 2, NUM_SPLITS))
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        workers, initializer=_init_worker, initargs=(peg_path, discard_path)
    ) as pool:
        for index, role, rep, values in pool.imap_unordered(_task, tasks, chunksize=4):
            p[index, role, rep] = values
    return u, p


def run_run(args: argparse.Namespace) -> None:
    """Estimates the pegging value of every split of test hands; saves it."""
    start = time.time()
    dataset = Dataset.load(args.data)
    u, p = run_study(
        dataset, args.hands, args.samples, args.peg, args.discard, args.seed, args.workers
    )
    np.savez_compressed(args.out, u=u, p=p, samples=args.samples)
    print(format_report(analyse(u, p), args.hands, args.samples))
    print(f"({time.time() - start:.0f}s; saved {args.out})")



# ------------------------------------------------------------
#  Labels, network and policy


def run_labels(args: argparse.Namespace) -> None:
    """Estimates the pegging margin of every split for training hands."""
    start = time.time()
    dataset = Dataset.load(args.data)
    hands = args.hands
    tasks = [
        (i, tuple(int(c) for c in dataset.hands[i]), role, 0, args.samples, args.seed)
        for i in range(hands)
        for role in (0, 1)
    ]
    p = np.zeros((hands, 2, NUM_SPLITS), dtype=np.float32)
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        args.workers, initializer=_init_worker, initargs=(args.peg, args.discard)
    ) as pool:
        for done, (index, role, _, values) in enumerate(
            pool.imap_unordered(_task, tasks, chunksize=8), 1
        ):
            p[index, role] = values
            if done % 2000 == 0:
                print(f"  {done}/{len(tasks)} ({time.time() - start:.0f}s)", flush=True)
    np.savez_compressed(
        args.out, hands=dataset.hands[:hands], p=p, samples=args.samples
    )
    print(f"wrote {hands} hands to {args.out} ({time.time() - start:.0f}s)")


def _peg_loss(
    model: DiscardNet, hands: torch.Tensor, p: torch.Tensor
) -> torch.Tensor:
    """Squared error of the predicted pegging margins, both seats; p is (B, 2, 15)."""
    features = encode_hands(hands)
    losses = [
        torch.nn.functional.mse_loss(
            predict_values(model, features, bool(seat)), p[:, seat]
        )
        for seat in (0, 1)
    ]
    return (losses[0] + losses[1]) / 2


def run_train(args: argparse.Namespace) -> None:
    """Fits a pegging-value network to the labels; reports test-set gain."""
    torch.set_num_threads(args.threads)
    saved = np.load(args.labels)
    hands = torch.from_numpy(saved["hands"]).long()
    p = torch.from_numpy(saved["p"]).float()
    cut = int(len(hands) * 0.95)
    config = TrainConfig(
        two_head=False,
        width=args.width,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        standardize_inputs=True,
        standardize_targets=True,
    )
    torch.manual_seed(args.seed)
    model = config.make_net()
    # the target statistics of the scalar head are those of the pegging margins
    with torch.no_grad():
        assert model.out_mean is not None and model.out_std is not None
        model.out_mean[:] = p[:cut].mean()
        model.out_std[:] = p[:cut].std()
        sample = encode_hands(hands[:20000]).reshape(-1, encode_hands(hands[:1]).shape[-1])
        assert model.in_mean is not None and model.in_std is not None
        cols = model.columns[model.columns < sample.shape[1]]
        x = sample[:, cols]
        std = x.std(dim=0)
        model.in_mean[: len(cols)] = x.mean(dim=0)
        model.in_std[: len(cols)] = torch.where(std < 1e-6, torch.ones_like(std), std)
    steps = (cut + args.batch_size - 1) // args.batch_size
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=args.epochs * steps
    )
    test = np.load(args.test)
    test_hands = torch.from_numpy(
        Dataset.load(args.test_data).hands[: len(test["u"])]
    ).long()
    best, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(cut)
        total = 0.0
        for first in range(0, cut, args.batch_size):
            idx = order[first : first + args.batch_size]
            loss = _peg_loss(model, hands[idx], p[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = float(_peg_loss(model, hands[cut:], p[cut:]))
        stats = test_gain(model, test_hands, test["u"], test["p"])
        print(
            f"epoch {epoch + 1:2d} train {total / cut:.4f} val {val:.4f}"
            f"  test gain {stats[0]:.4f}/{stats[1]:.4f} (non-dealer/dealer)",
            flush=True,
        )
        if val < best:
            best = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    stats = test_gain(model, test_hands, test["u"], test["p"])
    print(f"best: val {best:.4f}, test gain {stats[0]:.4f}/{stats[1]:.4f}")
    save_model(model, config, args.out)


@torch.no_grad()
def test_gain(
    model: DiscardNet, hands: torch.Tensor, u: np.ndarray, p: np.ndarray
) -> np.ndarray:
    """
    The gain, in full value, of choosing by u + the network's pegging value
    instead of by u, scored with the mean of the two measured replicates of
    the test hands, as (non-dealer, dealer).
    """
    model.eval()
    features = encode_hands(hands)
    truth = p.mean(axis=2)
    gains = []
    for seat in (0, 1):
        peg = predict_values(model, features, bool(seat)).numpy()
        pick = (u[:, seat] + peg).argmax(axis=1)
        base = u[:, seat].argmax(axis=1)
        rows = np.arange(len(u))
        full = u[:, seat] + truth[:, seat]
        gains.append((full[rows, pick] - full[rows, base]).mean())
    return np.array(gains)


class PegAwareDiscardPolicy(DiscardPolicy):
    """Discards by the base network's hand and crib values plus a pegging network's."""

    def __init__(self, base: DiscardNet, peg: DiscardNet) -> None:
        super().__init__()
        self.base = base.eval()
        self.peg = peg.eval()

    @classmethod
    def load(cls, spec: str) -> PegAwareDiscardPolicy:
        """From `BASEPATH@PEGPATH`."""
        base, _, peg = spec.partition("@")
        return cls(load_model(base), load_model(peg))

    def choose(self, view: DiscardView) -> tuple[Card, Card]:
        with torch.no_grad():
            features = encode_hands(torch.tensor([view.hand]))
            values = predict_values(
                self.base, features, view.is_dealer
            ) + predict_values(self.peg, features, view.is_dealer)
        first, second = (
            view.hand[i] for i in DISCARD_IDX[int(values.argmax())].tolist()
        )
        return first, second


# ------------------------------------------------------------
#  Duplicate hands

_DUEL: dict[str, object] = {}


def _init_duel(peg_path: str, specs: tuple[str, ...]) -> None:
    from cribbage.specs import make_discard_policy

    torch.set_num_threads(1)
    _DUEL["peg"] = NetPeggingPolicy.load(peg_path)
    _DUEL["policies"] = [make_discard_policy(s) for s in specs]


def play_hand(
    deal: np.ndarray, dealer: int, discarders: Sequence[DiscardPolicy], peg: NetPeggingPolicy
) -> int:
    """
    Points of seat 0 less seat 1 in one hand: pegging, both hands and the
    crib.  `deal` is (13,): six cards each and the starter; seat `s` throws
    with `discarders[s]` and both peg with `peg`.
    """
    dealt = [tuple(int(c) for c in deal[6 * s : 6 * s + 6]) for s in (0, 1)]
    starter = int(deal[12])
    kept, thrown = [], []
    for seat in (0, 1):
        out = discarders[seat].choose(DiscardView(dealt[seat], seat == dealer, 0, 0))
        thrown.append(tuple(out))
        kept.append([c for c in dealt[seat] if c not in out])
    state = PegState.start(kept, dealer)
    while not state.done:
        seat = state.turn
        view = state.view(seat, thrown[seat], starter, seat == dealer, 0, 0)
        state.play_card(peg.choose(view))
    margin = state.margin()
    shows = [score_hand(kept[s], starter) for s in (0, 1)]
    crib = score_hand([*thrown[0], *thrown[1]], starter, crib=True)
    margin += shows[0] - shows[1] + (crib if dealer == 0 else -crib)
    return margin


def _duel_task(task: tuple[int, int, int]) -> list[list[int]]:
    """Hands [first, first+num): returns, per hand, the margin of A less that of B,
    for A and B each seated against the opponent, as a 4-list per hand:
    (A non-dealer, B non-dealer, A dealer, B dealer)."""
    first, num, seed = task
    peg = _DUEL["peg"]
    policies = _DUEL["policies"]
    assert isinstance(peg, NetPeggingPolicy) and isinstance(policies, list)
    a, b, opponent = policies
    out = []
    for i in range(first, first + num):
        rng = np.random.default_rng([seed, i])
        deal = rng.permutation(52)[:13]
        row = []
        for dealer_is_me in (False, True):
            for me in (a, b):
                dealer = 0 if dealer_is_me else 1
                peg.reseed(f"{seed}/{i}")
                discarders = [me, opponent]
                row.append(play_hand(deal, dealer, discarders, peg))
        out.append(row)
    return out


def run_duel(args: argparse.Namespace) -> None:
    """Plays duplicate hands: policy A and policy B each against the opponent."""
    start = time.time()
    specs = (args.a, args.b, args.opponent)
    chunk = 50
    tasks = [(i, min(chunk, args.hands - i), args.seed) for i in range(0, args.hands, chunk)]
    rows: list[list[int]] = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        args.workers, initializer=_init_duel, initargs=(args.peg, specs)
    ) as pool:
        for part in pool.imap(_duel_task, tasks):
            rows.extend(part)
    m = np.array(rows, dtype=float)  # (hands, 4)
    n = len(m)
    print(f"A = {args.a}\nB = {args.b}\nopponent = {args.opponent}; {n} deals, each hand played "
          f"in both seats")
    for label, cols in (("non-dealer", (0, 1)), ("dealer", (2, 3))):
        diff = m[:, cols[0]] - m[:, cols[1]]
        print(f"  {label}: A {m[:, cols[0]].mean():+.3f}  B {m[:, cols[1]].mean():+.3f}"
              f"  A-B {diff.mean():+.3f} ± {1.96 * diff.std(ddof=1) / np.sqrt(n):.3f}")
    diff = (m[:, 0] + m[:, 2]) / 2 - (m[:, 1] + m[:, 3]) / 2
    print(f"  both seats: A-B {diff.mean():+.3f} ± {1.96 * diff.std(ddof=1) / np.sqrt(n):.3f} pts/hand"
          f"   ({time.time() - start:.0f}s)")


def run_report(args: argparse.Namespace) -> None:
    """Prints the report for a saved study."""
    saved = np.load(args.inp)
    print(format_report(analyse(saved["u"], saved["p"]), len(saved["u"]), int(saved["samples"])))


def main() -> None:
    """Does pegging change which cards to discard?"""
    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help=run_run.__doc__)
    run.add_argument("--data", default="data/discard_test_exact.npz")
    run.add_argument("--hands", type=int, default=500)
    run.add_argument("--samples", type=int, default=200, help="per replicate")
    run.add_argument("--peg", default="data/pegging/rl/net_w100_post.pt")
    run.add_argument("--discard", default="data/runs/two_k100.pt")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    run.add_argument("--out", default="data/discard_pegging.npz")
    run.set_defaults(func=run_run)

    report = commands.add_parser("report", help=run_report.__doc__)
    report.add_argument("--in", dest="inp", default="data/discard_pegging.npz")
    report.set_defaults(func=run_report)

    labels = commands.add_parser("labels", help=run_labels.__doc__)
    labels.add_argument("--data", default="data/discard_train_k100.npz")
    labels.add_argument("--hands", type=int, default=12000)
    labels.add_argument("--samples", type=int, default=50)
    labels.add_argument("--peg", default="data/pegging/rl/net_w100_post.pt")
    labels.add_argument("--discard", default="data/runs/two_k100.pt")
    labels.add_argument("--seed", type=int, default=1)
    labels.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    labels.add_argument("--out", default="data/discard_pegging_labels.npz")
    labels.set_defaults(func=run_labels)

    train = commands.add_parser("train", help=run_train.__doc__)
    train.add_argument("--labels", default="data/discard_pegging_labels.npz")
    train.add_argument("--test", default="data/discard_pegging.npz")
    train.add_argument("--test-data", default="data/discard_test_exact.npz")
    train.add_argument("--width", type=int, default=256)
    train.add_argument("--depth", type=int, default=3)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=2e-3)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--out", default="data/runs/peg_value.pt")
    train.set_defaults(func=run_train)

    duel = commands.add_parser("duel", help=run_duel.__doc__)
    duel.add_argument("--a", required=True, help="discard spec, e.g. model:X@Y")
    duel.add_argument("--b", default="model:data/runs/two_k100.pt")
    duel.add_argument("--opponent", default="model:data/runs/two_k100.pt")
    duel.add_argument("--hands", type=int, default=20000)
    duel.add_argument("--peg", default="data/pegging/rl/net_w100_post.pt")
    duel.add_argument("--seed", type=int, default=0)
    duel.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    duel.set_defaults(func=run_duel)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
