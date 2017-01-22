#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
c_cribbage_score.py
(c) Will Roberts  25 December, 2016

A module to deal with scoring cribbage hands and plays.
'''

from __future__ import absolute_import, print_function
from collections import Counter
import itertools
import math
import numpy as np
from .cards import CARD_FACES, CARD_VALUES, card_worth, cards_worth, split_card
from .utils import pairwise

def score_hand(hand, draw=None, crib=False, verbose=False):
    '''
    Scores a Cribbage hand.

    Arguments:
    - `hand`: a list of four card values
    - `draw`: a card value, or None
    - `crib`: a flag indicating if the given hand is a crib or not
    '''
    assert len(hand) == 4
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
    #sorted_faces = sorted(face_values)
    face_counts = Counter(face_values)
    sorted_uniq_faces = sorted(face_counts.keys()) + [None]
    # a run is a contiguous
    run_begin = None
    for last_v, this_v in pairwise(sorted_uniq_faces):
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
                    run_score = np.product([face_counts[x] for x in
                                            range(run_begin, run_begin + run_length)]) * run_length
                    if verbose:
                        print('run', run_begin, last_v, run_score)
                    score += run_score
                run_begin = None
    # score pairs/triples/quads/etc.
    for _f, count in face_counts.items():
        # count = 2 -> 2
        # count = 3 -> 6
        # count = n -> 2 * (n CHOOSE 2)
        if count > 1:
            pair_score = math.factorial(count) / math.factorial(count - 2)
            if verbose:
                print('pair', _f, count, pair_score)
            score += pair_score
    # score 15s
    card_values = [CARD_VALUES[x] for x in face_values]
    for comb_len in [2, 3, 4, 5]:
        for vlist in itertools.combinations(card_values, comb_len):
            if sum(vlist) == 15:
                if verbose:
                    print('fifteen')
                score += 2
    # score flush
    suit_values = set([s for (f, s) in split_values_hand])
    if len(suit_values) == 1:
        if draw is not None and list(suit_values)[0] == draw_suit:
            # draw 5
            if verbose:
                print('flush 5')
            score += len(hand) + 1
        else:
            if not crib:
                if verbose:
                    print('flush 4')
                score += len(hand)
    # score special jack
    if draw is not None and [f for (f, s) in split_values_hand if s == draw_suit and f == 10]:
        if verbose:
            print('special jack')
        score += 1
    if verbose:
        print('score', score)
    return score

def is_legal_play(card, linear_play):
    '''
    Tests whether the given card value is a legal play in a game of
    cribbage, assuming that the cards in `linear_play` have previously
    been played during the same round.

    Arguments:
    - `card`:
    - `linear_play`:
    '''
    lp_score = cards_worth(linear_play)
    c_score = card_worth(card)
    return lp_score + c_score <= 31

def get_legal_play_idxs(hand, linear_play):
    '''
    Faster version of is_legal_play for finding legal plays in a hand.

    Arguments:
    - `hand`:
    - `linear_play`:
    '''
    lp_score = cards_worth(linear_play)
    return [idx for idx, card in enumerate(hand)
            if card_worth(card) + lp_score <= 31]

def score_play(linear_play, verbose=False):
    '''
    Scores the last play in a game.

    Arguments:
    - `linear_play`: a list containing the card values played, in
      order, during the round
    '''
    # the value we will return
    play_score = 0
    # we only need to know the face values thare played
    face_values = [split_card(c)[0] for c in linear_play]
    if verbose:
        print('Score', ' '.join(CARD_FACES[f] for f in face_values))
    if cards_worth(linear_play) == 15:
        if verbose:
            print('fifteen')
        play_score += 2
    # look for pairs and tuples
    for backwards in range(len(linear_play), 1, -1):
        back_list = sorted(face_values[-backwards:])
        if len(set(back_list)) == 1:
            # pairlike
            pair_score = [2, 6, 12][len(back_list)-2]
            if verbose:
                print('pair {} points'.format(pair_score))
            play_score += pair_score
            break
        elif back_list == range(min(back_list), max(back_list) + 1):
            # run
            run_score = len(back_list)
            if run_score >= 3:
                if verbose:
                    print('run {} points'.format(run_score))
                play_score += run_score
                break
    return play_score
