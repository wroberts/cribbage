#!/usr/bin/env python

"""
randomplayer.py
(c) Will Roberts  25 December, 2016

A CribbagePlayer that always makes a random move.
"""

import random
from cribbage.cards import Card
from cribbage.pegging import PegView
from cribbage.player import CribbagePlayer, DiscardView


class RandomCribbagePlayer(CribbagePlayer):
    """A CribbagePlayer that always makes a random move."""

    def __init__(
        self,
    ) -> None:
        """Constructor."""
        super().__init__()

    def discard(self, view: DiscardView) -> tuple[Card, Card]:
        """Discards two random cards."""
        first, second = random.sample(view.hand, 2)
        return first, second

    def play_card(self, view: PegView) -> Card:
        """Plays a random legal card."""
        return random.choice(view.legal)
