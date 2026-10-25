#!/usr/bin/env python

"""
simpleplayer.py
(c) Will Roberts  13 January, 2017

Simple AI cribbage player.  This player uses Monte Carlo simulation to
try to pick the best cards to discard and play.
"""

import random

import numpy as np

from cribbage.cards import KEEP_COMBINATIONS, Card, make_deck
from cribbage.pegging import PegView
from cribbage.player import CribbagePlayer, DiscardView
from cribbage.scoring import score_hand, score_play


class SimpleCribbagePlayer(CribbagePlayer):
    """
    Cribbage player with simple AI!!!
    """

    def __init__(
        self,
        estimate_discard: bool = True,
        num_discard_samples: int = 20,
        estimate_playcard: bool = True,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.estimate_discard = estimate_discard
        self.num_discard_samples = num_discard_samples
        self.estimate_playcard = estimate_playcard

    def discard(self, view: DiscardView) -> tuple[Card, Card]:
        """
        Discards the two cards that leave the hand with the best average
        score over random starter cards (or at random, if not estimating).
        """
        hand = list(view.hand)
        if not self.estimate_discard:
            first, second = random.sample(hand, 2)
            return first, second
        deck = sorted(set(make_deck()) - set(hand))
        npdeck = np.array(deck)
        nphand = np.array(hand)
        results = {}
        for keep_idxs in KEEP_COMBINATIONS:
            keep = list(nphand[list(keep_idxs)])
            num_samples = self.num_discard_samples
            # pre-sample draw cards
            draw_card_idxs = np.random.randint(low=0, high=len(deck), size=num_samples)
            draw_cards = npdeck[draw_card_idxs]
            samples = [score_hand(keep, draw=draw_cards[i]) for i in range(num_samples)]
            results[tuple(sorted(keep))] = sum(samples) / len(samples)
        # get the best hand
        best_hand = max((v, k) for (k, v) in results.items())[1]
        first, second = (card for card in hand if card not in best_hand)
        return first, second

    def play_card(self, view: PegView) -> Card:
        """Plays the card that scores most right now (at random, if not estimating)."""
        if not self.estimate_playcard:
            return random.choice(view.legal)
        results = {card: score_play([*view.seq, card]) for card in view.legal}
        max_value = max(results.values())
        return random.choice([k for (k, v) in results.items() if v == max_value])
