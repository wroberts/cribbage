#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Cards are integers: 

CARD_VALUES   = 'A234567890JQK'
CARD_SUITS = 'SHDC'

def make_card(value, suit):
    return value + suit * 13

def split_card(vint):
    return vint % 13, vint // 13

def print_card(vint):
    v, s = split_card(vint)
    return '{}{}'.format(CARD_VALUES[v], CARD_SUITS[s])

def make_deck():
    return range(52)

# ------------------------------------------------------------
# Cribbage

import random
def make_random_hand(n=5):
    return random.sample(make_deck(), n)

def score_hand(hand):
    '''
    Scores a Cribbage hand.
    
    Arguments:
    - `hand`: a list of four or five card values
    '''
    pass
