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

# ------------------------------------------------------------
# Discard representation

# 1 unit: is it my crib?
# 52 on/off with 6 on for own hand
# own score: 121 units
# other player's score: 121 units

def discard_repr(current_round, player_idx):
    is_dealer = current_round.dealer_idx == player_idx
    hand = current_round.dealt_hands[player_idx]
    own_score = current_round.game.scores[player_idx]
    other_score = current_round.game.scores[int(not player_idx)]

# ------------------------------------------------------------
# Play card representation

# 1 unit: is it my crib?
# 52 on/off with up to 4 on for own hand
# 52 on/off indicating cards seen so far (counting) including starter card
# state of play:
#     The last 104 inputs represent the state of play. The first 13
#     represent the rank of the first card played, the second 13 cards
#     represent the rank of the second card played, and so forth.
# own score: 121 units
# other player's score: 121 units

def play_repr(current_round, player_idx):
    is_dealer = current_round.dealer_idx == player_idx
    hand = current_round.dealt_hands[player_idx]
    played_cards = current_round.played_cards
    play_state = [split_card(card)[0] for card in current_round.linear_play] # TODO: speedup
    own_score = current_round.game.scores[player_idx]
    other_score = current_round.game.scores[int(not player_idx)]
