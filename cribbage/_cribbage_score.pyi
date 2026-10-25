"""
Type stub for the Cython scoring extension (`_cribbage_score.pyx`).

The C versions mirror `cribbage_score.py`, with these differences: `draw`
in `score_hand` is required (it cannot be None), and `cards_worth` needs
a sized sequence rather than any iterable.
"""

from collections.abc import Sequence
from cribbage.cards import Card, Hand

def score_hand(
    hand: Hand,
    draw: Card,
    crib: bool = False,
    verbose: bool = False,
) -> int: ...
def score_play(linear_play: Sequence[Card], verbose: bool = False) -> int: ...
def card_worth(card: Card) -> int: ...
def cards_worth(cards: Sequence[Card]) -> int: ...
