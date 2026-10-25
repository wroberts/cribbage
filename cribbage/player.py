#!/usr/bin/env python

"""
player.py
(c) Will Roberts  25 December, 2016

Abstract base class for a cribbage player.
"""
from __future__ import annotations

from dataclasses import dataclass

from cribbage.cards import Card
from cribbage.pegging import PegView


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
