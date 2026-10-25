#!/usr/bin/env python

"""
c_cribbage_score.py
(c) Will Roberts  25 December, 2016

A module to deal with scoring cribbage hands and plays.
"""

from collections import Counter
import itertools
import math
from collections.abc import Sequence
from itertools import pairwise
from .cards import (
    CARD_FACES,
    CARD_VALUES,
    Card,
    Hand,
    cards_worth,
    split_card,
)

# the longest possible play: a count of 31 cannot be exceeded
MAX_PLAY_LENGTH = 13


def _check_cards(cards: Sequence[Card]) -> None:
    """
    Raises ValueError unless `cards` are distinct card values in [0, 51].
    """
    for card in cards:
        if not 0 <= card <= 51:
            raise ValueError(f"card values must be in [0, 51]: {card!r}")
    if len(set(cards)) != len(cards):
        raise ValueError(f"duplicate cards: {list(cards)!r}")


def score_hand(
    hand: Hand,
    draw: Card | None = None,
    crib: bool = False,
    verbose: bool = False,
) -> int:
    """
    Scores a Cribbage hand.

    Arguments:
    - `hand`: a list of four card values
    - `draw`: a card value, or None
    - `crib`: a flag indicating if the given hand is a crib or not

    Raises ValueError if the hand does not have four cards, or if any
    card is outside [0, 51] or appears twice.
    """
    if len(hand) != 4:
        raise ValueError("hand must have four cards")
    _check_cards(list(hand) + ([] if draw is None else [draw]))
    score = 0
    # split card values
    split_values_hand = [split_card(c) for c in hand]
    split_values = split_values_hand[:]
    if draw is not None:
        draw_split = split_card(draw)
        _draw_face, draw_suit = draw_split
        split_values.append(draw_split)
    face_values = [f for (f, s) in split_values]
    # score runs
    # sorted_faces = sorted(face_values)
    face_counts = Counter(face_values)
    sorted_uniq_faces: list[int | None] = [*sorted(face_counts.keys()), None]
    # a run is a contiguous
    run_begin: int | None = None
    for last_v, this_v in pairwise(sorted_uniq_faces):
        # only the second item of a pair can be the None sentinel
        assert last_v is not None
        if this_v == last_v + 1:
            if run_begin is None:
                run_begin = last_v
            else:
                pass
        else:
            if run_begin is None:
                pass
            else:
                # finish a run
                run_length = last_v - run_begin + 1
                if run_length >= 3:
                    # interesting
                    run_score = (
                        math.prod(
                            face_counts[x]
                            for x in range(run_begin, run_begin + run_length)
                        )
                        * run_length
                    )
                    if verbose:
                        print("run", run_begin, last_v, run_score)
                    score += run_score
                run_begin = None
    # score pairs/triples/quads/etc.
    for _f, count in face_counts.items():
        # count = 2 -> 2
        # count = 3 -> 6
        # count = n -> 2 * (n CHOOSE 2)
        if count > 1:
            pair_score = math.factorial(count) // math.factorial(count - 2)
            if verbose:
                print("pair", _f, count, pair_score)
            score += pair_score
    # score 15s
    card_values = [CARD_VALUES[x] for x in face_values]
    for comb_len in [2, 3, 4, 5]:
        for vlist in itertools.combinations(card_values, comb_len):
            if sum(vlist) == 15:
                if verbose:
                    print("fifteen")
                score += 2
    # score flush
    suit_values = set([s for (f, s) in split_values_hand])
    if len(suit_values) == 1:
        if draw is not None and list(suit_values)[0] == draw_suit:
            # draw 5
            if verbose:
                print("flush 5")
            score += len(hand) + 1
        else:
            if not crib:
                if verbose:
                    print("flush 4")
                score += len(hand)
    # score special jack
    if draw is not None and [
        f for (f, s) in split_values_hand if s == draw_suit and f == 10
    ]:
        if verbose:
            print("special jack")
        score += 1
    if verbose:
        print("score", score)
    return score


def score_play(linear_play: Sequence[Card], verbose: bool = False) -> int:
    """
    Scores the last play in a game.

    Arguments:
    - `linear_play`: a list containing the card values played, in
      order, during the round

    Raises ValueError if linear_play is empty or longer than 13
    cards, or if any card is outside [0, 51] or appears twice.
    """
    if not 1 <= len(linear_play) <= MAX_PLAY_LENGTH:
        raise ValueError("linear_play must have between 1 and 13 cards")
    _check_cards(linear_play)
    # the value we will return
    play_score = 0
    # we only need to know the face values thare played
    face_values = [split_card(c)[0] for c in linear_play]
    if verbose:
        print("Score", " ".join(CARD_FACES[f] for f in face_values))
    if cards_worth(linear_play) == 15:
        if verbose:
            print("fifteen")
        play_score += 2
    # look for pairs and tuples
    for backwards in range(len(linear_play), 1, -1):
        back_list = sorted(face_values[-backwards:])
        if len(set(back_list)) == 1:
            # pairlike
            pair_score = [2, 6, 12][len(back_list) - 2]
            if verbose:
                print(f"pair {pair_score} points")
            play_score += pair_score
            break
        elif back_list == list(range(min(back_list), max(back_list) + 1)):
            # run
            run_score = len(back_list)
            if run_score >= 3:
                if verbose:
                    print(f"run {run_score} points")
                play_score += run_score
                break
    return play_score
