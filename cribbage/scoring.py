"""
scoring.py

The scoring functions, from the C extension when it is built and from
the pure Python version otherwise.  Import them from here.
"""

from cribbage.cards import card_worth

try:
    from cribbage._cribbage_score import cards_worth, score_hand, score_play
except ImportError:
    from cribbage.cards import cards_worth
    from cribbage.cribbage_score import score_hand, score_play

__all__ = [
    "card_worth",
    "cards_worth",
    "score_hand",
    "score_play",
]
