#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import itertools
import math
import numpy as np
import random

# Cards are integers:

CARD_FACES = 'A234567890JQK'
CARD_SUITS = 'SHDC'
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def make_card(face, suit):
    return face + suit * 13

def split_card(vint):
    return vint % 13, vint // 13

def print_card(vint):
    v, s = split_card(vint)
    return '{}{}'.format(CARD_FACES[v], CARD_SUITS[s])

def make_deck():
    return range(52)

# ------------------------------------------------------------
# Cribbage

def make_random_hand(n=5):
    return random.sample(make_deck(), n)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def score_hand(hand, verbose=False):
    '''
    Scores a Cribbage hand.

    Arguments:
    - `hand`: a list of four or five card values
    '''
    score = 0
    # split card values
    split_values = [split_card(c) for c in hand]
    face_values = [f for (f,s) in split_values]
    # score runs
    #sorted_faces = sorted(face_values)
    face_counts = Counter(face_values)
    sorted_uniq_faces = sorted(face_counts.keys())
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
                        print 'run', run_begin, last_v, run_score
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
                print 'pair', _f, count, pair_score
            score += pair_score
    # score 15s
    card_values = [CARD_VALUES[x] for x in face_values]
    for comb_len in [2,3,4,5]:
        for vlist in itertools.combinations(card_values, comb_len):
            if sum(vlist) == 15:
                if verbose:
                    print 'fifteen'
                score += 2
    # score flush
    suit_values = [s for (f,s) in split_values]
    if len(set(suit_values)) == 1:
        if verbose:
            print 'flush'
        score += len(hand)
    # TODO: score special jack
    if verbose:
        print 'score', score
    return score

def test_score():
    global hand
    hand = make_random_hand()
    print hand
    print ', '.join([print_card(v) for v in hand])
    score_hand(hand, verbose=True)
