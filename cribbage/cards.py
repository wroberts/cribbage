#!/usr/bin/env python

"""
cards.py
(c) Will Roberts  25 December, 2016

Define cards and card behaviour for cribbage.

Cards are integer values between 0 and 51, inclusive.  This
representation can be broken down into a "face value" (0 representing
Ace to 12 representing King), and a "suit" value (0 representing
Spades to 3 representing Clubs).
"""

import itertools
import random
from collections.abc import Iterable, Sequence

# ------------------------------------------------------------
# Cards

CARD_FACES = "A234567890JQK"
CARD_SUITS = "SHDC"
# a card is an integer in [0, 51]; a hand is a list of cards
Card = int
Hand = list[Card]

CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
# the value of each card towards the count, by card (not face)
WORTH = [CARD_VALUES[c % 13] for c in range(52)]

# the 15 ways of keeping 4 of the 6 dealt cards, as indices into the hand
KEEP_COMBINATIONS = list(itertools.combinations(range(6), 4))


def make_card(face: int, suit: int) -> Card:
    """Creates a card value with the given `face` and `suit`."""
    return face + suit * 13


def split_card(vint: Card) -> tuple[int, int]:
    """Splits a card value into its `face` and `suit` values."""
    return vint % 13, vint // 13


def face(card: Card) -> int:
    """The face value of a card (0 for Ace to 12 for King)."""
    return card % 13


def card_tostring(vint: Card) -> str:
    """Returns a string representation of the given card value."""
    value, suit = split_card(vint)
    return f"{CARD_FACES[value]}{CARD_SUITS[suit]}"


def string_tocard(sval: str) -> Card:
    """Converts a string representation into a card value."""
    return make_card(CARD_FACES.index(sval[0]), CARD_SUITS.index(sval[1]))


def make_deck() -> list[Card]:
    """Creates a list containing all 52 card values."""
    return list(range(52))


# ------------------------------------------------------------
# Utility Functions


def unseen_cards(*known: Iterable[Card]) -> list[Card]:
    """The cards of the deck that are in none of the `known` groups."""
    seen = set().union(*known)
    return [card for card in range(52) if card not in seen]


def splits(hand: Sequence[Card]) -> list[tuple[list[Card], list[Card]]]:
    """
    Returns the 15 (keep, discard) splits of a 6-card hand, in the order
    of `KEEP_COMBINATIONS`.
    """
    return [
        (
            [hand[i] for i in keep_idxs],
            [hand[i] for i in range(6) if i not in keep_idxs],
        )
        for keep_idxs in KEEP_COMBINATIONS
    ]


def make_random_hand() -> Hand:
    """Create a random 4-card cribbage hand."""
    return random.sample(make_deck(), 4)


def make_random_hand_and_draw() -> tuple[Hand, Card]:
    """Create a random 4-card cribbage hand, and a random card as the starter."""
    hand_draw = random.sample(make_deck(), 5)
    return hand_draw[:4], hand_draw[4]


# ------------------------------------------------------------
# Determining cards' worth (for making 15s and 31s)


def card_worth(card: Card) -> int:
    """
    Returns the number of points the given card value is worth.

    Arguments:
    - `card`:
    """
    return WORTH[card]


def cards_worth(cards: Iterable[Card]) -> int:
    """
    Calls `card_worth` on every value in `cards` and returns the sum.

    Arguments:
    - `cards`:
    """
    return sum(card_worth(card) for card in cards)


# ------------------------------------------------------------
# Notes

# def n_choose_k(n,k):
#     '''Returns the number of ways of choosing `k` items from a set of `n`.'''
#     return math.factorial(n) // (math.factorial(k) * math.factorial(n-k))

# the number of ways of making 15s in a given hand with starter card
# n_choose_k(5,2) + n_choose_k(5,3) + n_choose_k(5,4) + n_choose_k(5,5)
# 26
# the number of ways of choosing two cards to discard into the crib
# n_choose_k(6,2)
# 15
