#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
c_cribbage_score.py
(c) Will Roberts  25 December, 2016

A module to deal with scoring cribbage hands and plays.
'''

from __future__ import print_function
from collections import Counter
import itertools
import math
import numpy as np
from .cards import CARD_VALUES, pairwise, split_card

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
    face_values = [f for (f,s) in split_values]
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
    for comb_len in [2,3,4,5]:
        for vlist in itertools.combinations(card_values, comb_len):
            if sum(vlist) == 15:
                if verbose:
                    print('fifteen')
                score += 2
    # score flush
    suit_values = set([s for (f,s) in split_values_hand])
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
    if draw is not None and [f for (f,s) in split_values_hand if s == draw_suit and f == 10]:
        if verbose:
            print('special jack')
        score += 1
    if verbose:
        print('score', score)
    return score
