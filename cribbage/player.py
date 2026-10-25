#!/usr/bin/env python

"""
player.py
(c) Will Roberts  25 December, 2016

Abstract base class for a cribbage player.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cribbage.cards import Card
from cribbage.pegging import PegView

if TYPE_CHECKING:
    from cribbage.discard_policy import DiscardPolicy
    from cribbage.pegging_policy import PeggingPolicy


@dataclass(frozen=True)
class DiscardView:
    """
    What a player sees when asked to discard.

    - `hand`: the six cards dealt to the player
    - `is_dealer`: whether the player is the dealer (the crib is theirs)
    - `score`, `opp_score`: the game scores of the player and their opponent
    """

    hand: tuple[Card, ...]
    is_dealer: bool
    score: int
    opp_score: int


class CribbagePlayer:
    """
    Abstract base class for an object that plays a game of cribbage.
    """

    def __init__(self) -> None:
        """Constructor."""
        pass

    def discard(self, view: DiscardView) -> tuple[Card, Card]:
        """
        Asks the player to select two cards from `view.hand` for discarding
        to the crib.  Returns the two cards.
        """
        raise NotImplementedError()

    def play_card(self, view: PegView) -> Card:
        """
        Asks the player to select one card from `view.legal` to play during
        the pegging.  Returns the card.
        """
        raise NotImplementedError()

    def round_over(self) -> None:
        """
        Notification that the current round is over.
        """
        pass

    def game_over(self, has_won: bool) -> None:
        """
        Notification that the current game is over.  The argument to the
        function is a flag indicating if this player won or not.

        Arguments:
        - `has_won`:
        """
        pass


class Policy:
    """A rule for making one kind of choice, with its own random generator."""

    def __init__(self) -> None:
        # seeded from the global generator, so `random.seed` makes it reproducible
        self.rng = random.Random(random.getrandbits(64))

    def reseed(self, seed: int | str) -> None:
        self.rng = random.Random(seed)


class PolicyPlayer(CribbagePlayer):
    """A player that discards with a `DiscardPolicy` and pegs with a `PeggingPolicy`."""

    def __init__(
        self, discard_policy: DiscardPolicy, pegging_policy: PeggingPolicy
    ) -> None:
        super().__init__()
        self.discard_policy = discard_policy
        self.pegging_policy = pegging_policy

    def reseed(self, seed: int | str) -> None:
        """Reseeds both policies, for reproducible games."""
        self.discard_policy.reseed(f"{seed}/discard")
        self.pegging_policy.reseed(f"{seed}/pegging")

    def discard(self, view: DiscardView) -> tuple[Card, Card]:
        return self.discard_policy.choose(view)

    def play_card(self, view: PegView) -> Card:
        return self.pegging_policy.choose(view)
