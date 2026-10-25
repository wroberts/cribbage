#!/usr/bin/env python

"""
pegging_rl_expt_2026.py

Can a small network learn to peg as well as a search?

Step 1 of the plan, which is supervised regression on search values, in
the manner of `discard_expt_2026`:

- `generate` plays out pegging on fixed deals.  At each decision with
  more than one distinct legal card, the teacher (`pimc-roll`, W sampled
  worlds with greedy rollouts) values every candidate card, and the
  decision is recorded.  The game continues with the teacher's choice,
  or a random legal card with probability `--explore`, for both players.
- `train` fits a network that scores one candidate card at a time, given
  the mover's information (their hand, the unseen cards by face, the
  sequence so far) and the candidate.  Only differences between the
  candidates of a decision matter, so the loss is on mean-centred values.
- `evaluate` plays the network (`net:PATH` in `pegging_expt_2026` specs)
  against the ladder with duplicate deals.

All features are by card face, so they are invariant to suits.  Inference
is a few small numpy matrix products, with no torch call.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from cribbage.cards import WORTH, Card, distinct_by_face, face
from cribbage.pegging import PegState, PegView
from cribbage.pegging_expt_2026 import (
    Deals,
    PegPolicy,
    PimcRollout,
    WorldSampler,
    distinct_moves,
    make_deals,
    rollout,
    run_pegging,
)
from cribbage.scoring import score_play

MAX_CANDIDATES = 4
# state: hand faces, unseen faces, count, sequence length, last three
# faces, go, dealer, opponent cards left, own cards left
STATE_DIM = 13 + 13 + 1 + 1 + 3 * 13 + 1 + 1 + 5 + 4
# candidate: its face, immediate score, count after, makes 31
CAND_DIM = 13 + 1 + 1 + 1


# ------------------------------------------------------------
#  Features


def featurise(
    hand: Sequence[Card],
    legal: Sequence[Card],
    seq: Sequence[Card],
    count: int,
    unseen_faces: np.ndarray,
    opp_said_go: bool,
    is_dealer: bool,
    opp_cards_left: int,
) -> tuple[np.ndarray, np.ndarray, list[Card]]:
    """
    Features of a decision: the state (STATE_DIM,), one row per candidate
    (k, CAND_DIM), and the card of each candidate.  The candidates are the
    legal cards, one per face (equal faces play alike).
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    for card in hand:
        state[face(card)] += 1
    state[13:26] = unseen_faces * 0.25
    state[26] = count / 31
    state[27] = len(seq) / 8
    for back, card in enumerate(seq[-3:][::-1]):
        state[28 + 13 * back + face(card)] = 1
    state[67] = opp_said_go
    state[68] = is_dealer
    state[69 + opp_cards_left] = 1
    state[74 + len(hand) - 1] = 1
    cards = distinct_by_face(legal)
    cands = np.zeros((len(cards), CAND_DIM), dtype=np.float32)
    for row, card in enumerate(cards):
        cands[row, face(card)] = 1
        cands[row, 13] = score_play([*seq, card]) / 6
        cands[row, 14] = (count + WORTH[card]) / 31
        cands[row, 15] = count + WORTH[card] == 31
    return state, cands, cards


def encode(view: PegView) -> tuple[np.ndarray, np.ndarray, list[Card]]:
    """Features of the decision a player sees (see `featurise`)."""
    unseen_faces = np.bincount([face(c) for c in view.unseen()], minlength=13)
    return featurise(
        view.hand,
        view.legal,
        view.seq,
        view.count,
        unseen_faces,
        view.opp_said_go,
        view.is_dealer,
        view.opp_cards_left,
    )


# ------------------------------------------------------------
#  Labels from the teacher


@dataclass
class Labels:
    """Decisions with their teacher values; unused candidate slots are masked."""

    state: np.ndarray  # (N, STATE_DIM)
    cand: np.ndarray  # (N, MAX_CANDIDATES, CAND_DIM)
    value: np.ndarray  # (N, MAX_CANDIDATES) teacher margin of each candidate
    mask: np.ndarray  # (N, MAX_CANDIDATES) bool

    def __len__(self) -> int:
        return len(self.state)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path, state=self.state, cand=self.cand, value=self.value, mask=self.mask
        )

    @classmethod
    def load(cls, path: str) -> Labels:
        with np.load(path) as data:
            return cls(data["state"], data["cand"], data["value"], data["mask"])

    @classmethod
    def concat(cls, parts: Sequence[Labels]) -> Labels:
        return cls(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                     ("state", "cand", "value", "mask")))


def label_deals(
    deals: Deals, worlds: int, explore: float, seed: str
) -> Labels:
    """Plays every deal out with the teacher on both sides, recording decisions."""
    teacher = PimcRollout(worlds)
    teacher.reseed(seed)
    rng = random.Random(seed)
    states, cands, values, masks = [], [], [], []
    for i in range(len(deals)):
        hands = [[int(c) for c in deals.kept[i, s]] for s in (0, 1)]
        thrown = [tuple(int(c) for c in deals.thrown[i, s]) for s in (0, 1)]
        starter = int(deals.starter[i])
        game = PegState.start(hands, 1)
        while not game.done:
            seat = game.turn
            view = game.view(seat, thrown[seat], starter, seat == 1, 0, 0)
            if len(view.legal) == 1:
                card = view.legal[0]
            else:
                teacher_values = teacher.values(view)
                if len(teacher_values) > 1:
                    state, cand, cards = encode(view)
                    k = len(cards)
                    pad = MAX_CANDIDATES - k
                    states.append(state)
                    cands.append(np.pad(cand, ((0, pad), (0, 0))))
                    values.append(np.pad([teacher_values[c] for c in cards], (0, pad)))
                    masks.append(np.arange(MAX_CANDIDATES) < k)
                if rng.random() < explore:
                    card = rng.choice(view.legal)
                else:
                    card = teacher.argmax(teacher_values)
            game.play_card(card)
    return Labels(
        np.array(states, dtype=np.float32).reshape(-1, STATE_DIM),
        np.array(cands, dtype=np.float32).reshape(-1, MAX_CANDIDATES, CAND_DIM),
        np.array(values, dtype=np.float32).reshape(-1, MAX_CANDIDATES),
        np.array(masks, dtype=bool).reshape(-1, MAX_CANDIDATES),
    )


def _label_chunk(task: tuple[Deals, int, float, str]) -> Labels:
    deals, worlds, explore, seed = task
    return label_deals(deals, worlds, explore, seed)


def generate_labels(
    deals: Deals, worlds: int, explore: float, seed: int, workers: int
) -> Labels:
    """Labels for `deals`, spread over `workers` processes in fixed chunks."""
    size = 250
    tasks = [
        (deals.slice(lo, lo + size), worlds, explore, f"{seed}/{lo}")
        for lo in range(0, len(deals), size)
    ]
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            parts = pool.map(_label_chunk, tasks)
    else:
        parts = [_label_chunk(t) for t in tasks]
    return Labels.concat(parts)


# ------------------------------------------------------------
#  The network


class PegNet(nn.Module):
    """Scores each candidate card: an MLP on (state, candidate) features."""

    def __init__(self, width: int = 256, depth: int = 3) -> None:
        super().__init__()
        self.width, self.depth = width, depth
        layers: list[nn.Module] = []
        size = STATE_DIM + CAND_DIM
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.ReLU()]
            size = width
        layers.append(nn.Linear(size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Scores (B, K) from state (B, S) and candidates (B, K, C)."""
        expanded = state[:, None, :].expand(-1, cand.shape[1], -1)
        return self.layers(torch.cat([expanded, cand], dim=-1))[..., 0]


def centred(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Subtracts each row's mean over its unmasked entries; masked ones become 0."""
    m = mask.float()
    mean = (x * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True)
    return (x - mean) * m


def regrets(pred: torch.Tensor, labels: dict[str, torch.Tensor]) -> dict[str, float]:
    """
    Mean margin lost, relative to the teacher's best candidate, by choosing
    the network's favourite; also for greedy (ties split) and for a random
    candidate; and how often the network picks a best candidate.
    """
    value, mask = labels["value"], labels["mask"]
    best = value.masked_fill(~mask, -1e9).max(dim=1).values
    pick = pred.masked_fill(~mask, -1e9).argmax(dim=1)
    chosen = value.gather(1, pick[:, None])[:, 0]
    immediate = labels["cand"][..., 13].masked_fill(~mask, -1e9)
    ties = (immediate == immediate.max(dim=1, keepdim=True).values) & mask
    greedy = (value * ties).sum(dim=1) / ties.sum(dim=1)
    uniform = (value * mask).sum(dim=1) / mask.sum(dim=1)
    return {
        "net": float((best - chosen).mean()),
        "greedy": float((best - greedy).mean()),
        "random": float((best - uniform).mean()),
        "top": float((chosen >= best - 1e-6).float().mean()),
    }


def to_tensors(labels: Labels) -> dict[str, torch.Tensor]:
    return {
        "state": torch.from_numpy(labels.state),
        "cand": torch.from_numpy(labels.cand),
        "value": torch.from_numpy(labels.value),
        "mask": torch.from_numpy(labels.mask),
    }


def split(labels: Labels, val_fraction: float) -> tuple[Labels, Labels]:
    """Train/validation split by position; rows are grouped by deal, so it is a split of deals."""
    n = int(len(labels) * (1 - val_fraction))
    cut = lambda lo, hi: Labels(  # noqa: E731
        labels.state[lo:hi], labels.cand[lo:hi], labels.value[lo:hi], labels.mask[lo:hi]
    )
    return cut(0, n), cut(n, len(labels))


def train_net(
    labels: Labels,
    width: int = 256,
    depth: int = 3,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 2e-3,
    val_fraction: float = 0.05,
    seed: int = 0,
    log: bool = True,
) -> tuple[PegNet, dict[str, float]]:
    """Fits a PegNet to centred teacher values; returns it and its validation regrets."""
    torch.manual_seed(seed)
    train, val = split(labels, val_fraction)
    data, val_data = to_tensors(train), to_tensors(val)
    model = PegNet(width, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * math.ceil(len(train) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    gen = torch.Generator().manual_seed(seed)
    stats: dict[str, float] = {}
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train), generator=gen)
        total = 0.0
        for lo in range(0, len(train), batch_size):
            b = order[lo : lo + batch_size]
            mask = data["mask"][b]
            pred = centred(model(data["state"][b], data["cand"][b]), mask)
            target = centred(data["value"][b], mask)
            loss = ((pred - target) ** 2).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            pred = model(val_data["state"], val_data["cand"])
            stats = regrets(pred, val_data)
            val_mask = val_data["mask"]
            stats["mse"] = float(
                (
                    (centred(pred, val_mask) - centred(val_data["value"], val_mask)) ** 2
                ).sum()
                / val_mask.sum()
            )
        if log:
            print(
                f"epoch {epoch + 1:2d}  train mse {total / len(train):.4f}  "
                f"val mse {stats['mse']:.4f}  regret net {stats['net']:.4f} "
                f"(greedy {stats['greedy']:.4f}, random {stats['random']:.4f})  "
                f"top {stats['top']:.3f}",
                flush=True,
            )
    return model, stats


def save_net(model: PegNet, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"width": model.width, "depth": model.depth, "state_dict": model.state_dict()}, path
    )


def load_net(path: str) -> PegNet:
    blob = torch.load(path, weights_only=True)
    model = PegNet(blob["width"], blob["depth"])
    model.load_state_dict(blob["state_dict"])
    return model.eval()


# ------------------------------------------------------------
#  The policy


class NetPeg(PegPolicy):
    """Plays the candidate the network scores highest; inference is plain numpy."""

    def __init__(self, model: PegNet) -> None:
        super().__init__()
        linears = [m for m in model.layers if isinstance(m, nn.Linear)]
        self.weights = [
            (l.weight.detach().numpy().T.copy(), l.bias.detach().numpy().copy())
            for l in linears
        ]

    @classmethod
    def load(cls, path: str) -> NetPeg:
        return cls(load_net(path))

    def scores(self, state: np.ndarray, cands: np.ndarray) -> np.ndarray:
        x = np.concatenate([np.broadcast_to(state, (len(cands), len(state))), cands], axis=1)
        for i, (w, b) in enumerate(self.weights):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)
        return x[:, 0]

    def choose(self, view: PegView) -> Card:
        if len(view.legal) == 1:
            return view.legal[0]
        state, cands, cards = encode(view)
        scores = self.scores(state, cands)
        return self.argmax({card: float(s) for card, s in zip(cards, scores)})


# ------------------------------------------------------------
#  Search with the network as the rollout policy


class NetRollout:
    """
    Plays out a sampled world with a network, each side seeing only what
    its own player would.  The world's cards are fixed at its root, so what
    a mover has not seen is its faces not in the root hands or played
    cards (`unplayed`), plus the other player's current hand, less the
    mover's own discards.
    """

    def __init__(
        self,
        net: NetPeg,
        root_hands: Sequence[Sequence[Card]],
        played: set[Card],
        discards: Sequence[Sequence[Card]],
        dealer_seat: int,
    ) -> None:
        self.net = net
        faces = np.full(13, 4.0, dtype=np.float32)
        for card in [*played, *root_hands[0], *root_hands[1]]:
            faces[face(card)] -= 1
        self.unplayed = faces
        self.discard_faces = np.zeros((2, 13), dtype=np.float32)
        for seat in (0, 1):
            for card in discards[seat]:
                self.discard_faces[seat, face(card)] += 1
        self.dealer_seat = dealer_seat

    def features(
        self, state: PegState, legal: list[int]
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """What `encode` gives for the player to move, with hand indices."""
        seat = state.turn
        hand = state.hands[seat]
        other = state.hands[1 - seat]
        unseen = self.unplayed - self.discard_faces[seat]
        for card in other:
            unseen[face(card)] += 1
        vec, cands, cards = featurise(
            hand,
            [hand[i] for i in legal],
            state.seq,
            state.count,
            unseen,
            state.is_go,
            seat == self.dealer_seat,
            len(other),
        )
        return vec, cands, [hand.index(card) for card in cards]

    def move(self, state: PegState, rng: random.Random) -> int:
        """The hand index of the network's favourite legal card."""
        legal = state.legal()
        if len(legal) == 1:
            return legal[0]
        vec, cands, idxs = self.features(state, legal)
        scores = self.net.scores(vec, cands)
        top = scores.max()
        best = [idxs[r] for r in range(len(idxs)) if scores[r] >= top - 1e-6]
        return best[0] if len(best) == 1 else rng.choice(best)


class PimcNet(PegPolicy):
    """
    PIMC like `pimc-roll`, but rollouts are played out by a trained
    network, which also plays the opponent.  Spec: `pimc-net:W:PATH`.
    """

    def __init__(self, worlds: int, net: NetPeg) -> None:
        super().__init__()
        self.worlds = worlds
        self.net = net

    def values(self, view: PegView) -> dict[Card, float]:
        """Mean margin of each distinct legal card."""
        sampler = WorldSampler(view)
        unseen = view.unseen()
        totals: dict[int, float] = {}
        dealer_seat = 0 if view.is_dealer else 1
        for _ in range(self.worlds):
            world = sampler.sample(self.rng)
            # the opponent's discards are among the cards it does not hold
            others = [c for c in unseen if c not in world.hands[1]]
            ctx = NetRollout(
                self.net,
                world.hands,
                set(view.played) | {view.starter},
                [view.discards, self.rng.sample(others, 2)],
                dealer_seat,
            )
            for idx in distinct_moves(world).values():
                child = world.copy()
                child.play(idx)
                rollout(child, ctx.move, self.rng)
                totals[idx] = totals.get(idx, 0.0) + child.margin()
        return {view.hand[idx]: total / self.worlds for idx, total in totals.items()}

    def choose(self, view: PegView) -> Card:
        if len(view.legal) == 1:
            return view.legal[0]
        return self.argmax(self.values(view))


# ------------------------------------------------------------
#  Command line


def run_generate(args: argparse.Namespace) -> None:
    """Labels pegging decisions with the pimc-roll teacher."""
    start = time.time()
    deals = make_deals(args.hands, args.seed, args.discard)
    labels = generate_labels(deals, args.worlds, args.explore, args.seed, args.workers)
    labels.save(args.out)
    print(
        f"wrote {len(labels)} decisions from {args.hands} hands to {args.out}"
        f" in {time.time() - start:.0f}s"
    )


def run_train(args: argparse.Namespace) -> None:
    """Trains a PegNet on a label file."""
    torch.set_num_threads(args.threads)
    labels = Labels.load(args.labels)
    model, stats = train_net(
        labels, args.width, args.depth, args.epochs, args.batch_size, args.lr, seed=args.seed
    )
    save_net(model, args.out)
    print(f"saved {args.out}; final validation: {stats}")


def run_evaluate(args: argparse.Namespace) -> None:
    """Plays a trained net against pegging policies on duplicate deals."""
    deals = make_deals(args.deals, args.seed, args.discard)
    spec = f"net:{args.net}"
    print(f"{spec}, {args.deals} deals x2 seats (95% half-widths)")
    for opp in args.opponents.split(","):
        result = run_pegging(spec, opp, deals, args.seed, args.workers)
        margin, half = result.per_hand()
        ms_net, ms_opp = result.ms_per_hand()
        print(
            f"  vs {opp:<14} {margin:+.3f} +/- {half:.3f} pts/hand"
            f"   (net {ms_net:.3f} ms/hand, {opp} {ms_opp:.3f} ms/hand)",
            flush=True,
        )


def main() -> None:
    """Learning to peg from search values."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    discard = "model:data/runs/two_k100.pt"

    gen = commands.add_parser("generate", help=run_generate.__doc__)
    gen.add_argument("--hands", type=int, default=20000)
    gen.add_argument("--worlds", type=int, default=100, help="teacher worlds per decision")
    gen.add_argument("--explore", type=float, default=0.25)
    gen.add_argument("--discard", default=discard)
    # evaluation uses seed 0 deals, so training deals must differ
    gen.add_argument("--seed", type=int, default=1000)
    gen.add_argument("--out", default="data/pegging/rl/labels.npz")
    gen.set_defaults(func=run_generate)

    train = commands.add_parser("train", help=run_train.__doc__)
    train.add_argument("--labels", default="data/pegging/rl/labels.npz")
    train.add_argument("--width", type=int, default=256)
    train.add_argument("--depth", type=int, default=3)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--lr", type=float, default=2e-3)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--out", default="data/pegging/rl/net.pt")
    train.set_defaults(func=run_train)

    ev = commands.add_parser("evaluate", help=run_evaluate.__doc__)
    ev.add_argument("--net", default="data/pegging/rl/net.pt")
    ev.add_argument("--opponents", default="greedy,greedy2,pimc-roll:30")
    ev.add_argument("--deals", type=int, default=20000)
    ev.add_argument("--discard", default=discard)
    ev.add_argument("--seed", type=int, default=0)
    ev.set_defaults(func=run_evaluate)

    for sub in (gen, ev):
        sub.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
