#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
test_scoring.py
(c) Will Roberts  25 December, 2016

Test the cribbage_scoring module.
'''

from ..cards import card_tostring, make_random_hand_and_draw
from ..cribbage_score import score_hand

def test_score():
    '''Utility function to test the score_hand method.'''
    #global hand, draw
    hand, draw = make_random_hand_and_draw()
    print(hand)
    print(draw)
    print('hand', ', '.join([card_tostring(v) for v in hand]))
    print('draw', card_tostring(draw))
    score_hand(hand, draw, verbose=True)
