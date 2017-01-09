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
from cribbage.player import CribbagePlayer
import numpy as np

# ------------------------------------------------------------
# Utility functions

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

def discard_repr(is_dealer,
                 hand,
                 player_score,
                 opponent_score):
    rv = np.zeros(1+51+121+121, dtype=int)
    rv[0] = int(is_dealer)
    encode_categories(rv, 1, hand)
    one_hot(rv, 53, player_score)
    one_hot(rv, 174, opponent_score)
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

def play_repr(is_dealer,
              hand,
              played_cards,
              is_go,
              linear_play,
              player_score,
              opponent_score):
    rv = np.zeros(1+52+52+1+104+121+121, dtype=int)
    rv[0] = int(is_dealer)
    encode_categories(rv, 1, hand)
    encode_categories(rv, 53, played_cards)
    rv[105] = int(is_go)
    play_state = [split_card(card)[0] for card in linear_play] # TODO: speedup
    for bank_idx, card_value in enumerate(play_state[-8:]):
        one_hot(rv, 106 + 13 * bank_idx, card_value)
    one_hot(rv, 211, player_score)
    one_hot(rv, 332, opponent_score)
    return rv

# ------------------------------------------------------------
# Recording cribbage game state

class NeuralRecordingCribbagePlayer(CribbagePlayer):
    '''
    A CribbagePlayer wrapper that records the game states it observes.
    '''

    def __init__(self, player):
        '''Constructor.'''
        super(NeuralRecordingCribbagePlayer, self).__init__()
        self.player = player
        self.discard_states = []
        self.play_card_states = []

    def reset(self):
        '''
        Resets this player's internal state record lists.
        '''
        self.discard_states = []
        self.play_card_states = []

    def discard(self,
                is_dealer,
                hand,
                player_score,
                opponent_score):
        '''
        Asks the player to select two cards from `hand` for discarding to
        the crib.

        Return is a list of two indices into the hand array.

        Arguments:
        - `is_dealer`: a flag to indicate whether the given player is
          currently the dealer or not
        - `hand`: an array of 6 card values
        - `player_score`: the score of the current player
        - `opponent_score`: the score of the current player's opponent
        '''
        state = discard_repr(is_dealer,
                             hand,
                             player_score,
                             opponent_score)
        self.discard_states.append(state)
        return self.player.discard(is_dealer, hand, player_score, opponent_score)

    def play_card(self,
                  is_dealer,
                  hand,
                  played_cards,
                  is_go,
                  linear_play,
                  player_score,
                  opponent_score,
                  legal_moves):
        '''
        Asks the player to select one card from `hand` to play during a
        cribbage round.

        Return an index into the hand array.

        Arguments:
        - `is_dealer`: a flag to indicate whether the given player is
          currently the dealer or not
        - `hand`: an array of 1 to 4 card values
        - `played_cards`: a set of card values, containing all cards
          seen so far in this round (including the starter card)
        - `is_go`: a flag to indicate if the play is currently in go or not
        - `linear_play`: the array of card values that have been
          played in this round by both players, zippered into a single
          list
        - `player_score`: the score of the current player
        - `opponent_score`: the score of the current player's opponent
        - `legal_moves`: a list of indices into `hand` indicating
          which cards from the hand may be played legally at this
          point in the game
        '''
        state = play_repr(is_dealer,
                          hand,
                          played_cards,
                          is_go,
                          linear_play,
                          player_score,
                          opponent_score)
        self.play_card_states.append(state)
        return self.player.play_card(is_dealer, hand, played_cards, is_go,
                                     linear_play, player_score, opponent_score,
                                     legal_moves)
