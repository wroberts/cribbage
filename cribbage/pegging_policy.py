"""
pegging_policy.py

Policies for choosing a card to play in the pegging.  A policy sees a
`PegView` and returns one of `view.legal`.  These are the simple ones; the
policies that search are in `pegging_search.py`.

- `first`: the first legal card (deterministic; used for testing).
- `random`: any legal card.
- `greedy` (`Greedy1PeggingPolicy`): the card that scores most right now
  (Simple's rule).
- `greedy2`: greedy, less the opponent's expected best reply (analytic,
  under a uniform guess at the opponent's hand).
"""

from __future__ import annotations

import math

from cribbage.cards import WORTH, Card
from cribbage.pegging import PegView
from cribbage.player import Policy
from cribbage.scoring import score_play


class PeggingPolicy(Policy):
    """Chooses which card to play; `choose` returns a card from `view.legal`."""

    needs_oracle = False

    def choose(self, view: PegView) -> Card:
        raise NotImplementedError()

    def argmax(self, values: dict[Card, float]) -> Card:
        """The key with the highest value; ties broken at random."""
        top = max(values.values())
        best = [k for k, v in values.items() if v == top]
        return best[0] if len(best) == 1 else self.rng.choice(best)


class FirstPeggingPolicy(PeggingPolicy):
    def choose(self, view: PegView) -> Card:
        return view.legal[0]


class RandomPeggingPolicy(PeggingPolicy):
    def choose(self, view: PegView) -> Card:
        return self.rng.choice(view.legal)


class Greedy1PeggingPolicy(PeggingPolicy):
    """Simple's rule: the card with the best immediate score."""

    def choose(self, view: PegView) -> Card:
        values = {c: float(score_play([*view.seq, c])) for c in view.legal}
        return self.argmax(values)


class Greedy2PeggingPolicy(PeggingPolicy):
    """
    Greedy, less the opponent's best immediate reply.  Their hand is
    n cards from the U unseen; the best playable reply of rank k (0 is
    best) is theirs to play with probability C(U-1-k, n-1) / C(U, n).
    """

    def choose(self, view: PegView) -> Card:
        if len(view.legal) == 1:
            return view.legal[0]
        pool = view.unseen()
        n = view.opp_cards_left
        total = math.comb(len(pool), n) if n else 1
        values: dict[Card, float] = {}
        for card in view.legal:
            immediate = score_play([*view.seq, card])
            reply = 0.0
            if n and not view.opp_said_go:
                room = 31 - view.count - WORTH[card]
                scores = sorted(
                    (score_play([*view.seq, card, x]) for x in pool if WORTH[x] <= room),
                    reverse=True,
                )
                reply = (
                    sum(
                        s * math.comb(len(pool) - 1 - k, n - 1)
                        for k, s in enumerate(scores)
                    )
                    / total
                )
            values[card] = immediate - reply
        return self.argmax(values)
