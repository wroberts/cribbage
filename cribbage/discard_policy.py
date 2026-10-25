"""
discard_policy.py

Policies for choosing two cards to discard to the crib.  A policy sees a
`DiscardView` and returns two cards of `view.hand`.

- `random`: any two cards.
- `expected` (`ExpectedHandDiscardPolicy`): keeps the four cards whose hand
  scores best on average over random starter cards (Simple's rule).
"""

from __future__ import annotations

from cribbage.cards import KEEP_COMBINATIONS, Card, unseen_cards
from cribbage.player import DiscardView, Policy
from cribbage.scoring import score_hand


class DiscardPolicy(Policy):
    """Chooses which two cards to discard; `choose` returns them."""

    def choose(self, view: DiscardView) -> tuple[Card, Card]:
        raise NotImplementedError()


class RandomDiscardPolicy(DiscardPolicy):
    def choose(self, view: DiscardView) -> tuple[Card, Card]:
        first, second = self.rng.sample(view.hand, 2)
        return first, second


class ExpectedHandDiscardPolicy(DiscardPolicy):
    """
    Keeps the four cards with the best mean score over `num_samples` random
    starter cards (the same ones for every candidate hand).
    """

    def __init__(self, num_samples: int = 20) -> None:
        super().__init__()
        self.num_samples = num_samples

    def choose(self, view: DiscardView) -> tuple[Card, Card]:
        hand = list(view.hand)
        starters = self.rng.choices(unseen_cards(hand), k=self.num_samples)
        results = {}
        for keep_idxs in KEEP_COMBINATIONS:
            keep = [hand[i] for i in keep_idxs]
            total = sum(score_hand(keep, draw=starter) for starter in starters)
            results[tuple(sorted(keep))] = total / self.num_samples
        best_hand = max((v, k) for (k, v) in results.items())[1]
        first, second = (card for card in hand if card not in best_hand)
        return first, second
