#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
neural.py
(c) Will Roberts   6 January, 2017

Notes on how to reason about cribbage games using neural networks.

See also
http://r6.ca/cs486/
for inspiration.
'''

from cribbage.cards import split_card
import numpy as np

def one_hot(vec, start_offset, value):
    '''
    One hot encodes `value` into `vec`, using the range of vector
    positions between `start_offset` and `start_offset` + `max_value`,
    inclusive.

    Arguments:
    - `vec`:
    - `start_offset`:
    - `value`:
    '''
    vec[start_offset + value] = 1

def encode_categories(vec, start_offset, values):
    '''
    One-hot encoding of multiple values (in `values`) into `vec`.

    Arguments:
    - `vec`:
    - `start_offset`:
    - `values`:
    '''
    for value in values:
        vec[start_offset + value] = 1

# ------------------------------------------------------------
# Discard representation

# 1 unit: is it my crib?
# 52 on/off with 6 on for own hand
# own score: 121 units
# other player's score: 121 units

def discard_repr(current_round, player_idx):
    rv = np.zeros(1+51+121+121, dtype=int)
    is_dealer = current_round.dealer_idx == player_idx
    rv[0] = int(is_dealer)
    hand = current_round.dealt_hands[player_idx]
    encode_categories(rv, 1, hand)
    own_score = current_round.game.scores[player_idx]
    one_hot(rv, 53, own_score)
    other_score = current_round.game.scores[int(not player_idx)]
    one_hot(rv, 174, other_score)
    return rv

# ------------------------------------------------------------
# Play card representation

# 1 unit: is it my crib?
# 52 on/off with up to 4 on for own hand
# 52 on/off indicating cards seen so far (counting) including starter card
# 1 unit: is it go?
# state of play:
#     The last 104 inputs represent the state of play. The first 13
#     represent the rank of the first card played, the second 13 cards
#     represent the rank of the second card played, and so forth.
# own score: 121 units
# other player's score: 121 units

def play_repr(current_round, player_idx):
    rv = np.zeros(1+52+52+1+104+121+121, dtype=int)
    is_dealer = current_round.dealer_idx == player_idx
    rv[0] = int(is_dealer)
    hand = current_round.dealt_hands[player_idx]
    encode_categories(rv, 1, hand)
    played_cards = current_round.played_cards
    encode_categories(rv, 53, played_cards)
    is_go = current_round.is_go
    rv[105] = int(is_go)
    play_state = [split_card(card)[0] for card in current_round.linear_play] # TODO: speedup
    for bank_idx, card_value in enumerate(play_state[-8:]):
        one_hot(rv, 106 + 13 * bank_idx, card_value)
    own_score = current_round.game.scores[player_idx]
    one_hot(rv, 211, own_score)
    other_score = current_round.game.scores[int(not player_idx)]
    one_hot(rv, 332, other_score)
    return rv
