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
from cribbage.cribbage_score import is_legal_play
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
# Discard state and action representation

# 1 unit: is it my crib?
# 52 on/off with 6 on for own hand
# own score: 121 units
# other player's score: 121 units

def discard_state_repr(is_dealer,
                       hand,
                       player_score,
                       opponent_score):
    rv = np.zeros(1+52+121+121, dtype=int)
    rv[0] = int(is_dealer)
    encode_categories(rv, 1, hand)
    one_hot(rv, 53, player_score)
    one_hot(rv, 174, opponent_score)
    return rv

def discard_action_repr(discards):
    '''
    Constructs a vector representation of a discard action.

    Arguments:
    - `discards`: an iterable of length 2 containing two card values
      (0-51 incl.)
    '''
    rv = np.zeros(52, dtype=int)
    encode_categories(rv, 0, discards)
    return rv

# ------------------------------------------------------------
# Play card state and action representation

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

def play_state_repr(is_dealer,
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
    one_hot(rv, 210, player_score)
    one_hot(rv, 331, opponent_score)
    return rv

def play_action_repr(play_card):
    '''
    Constructs a vector representation of a play_card action.

    Arguments:
    - `play_card`: a card value (0-51 incl.)
    '''
    rv = np.zeros(52, dtype=int)
    one_hot(rv, 0, play_card)
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
        self.last_discard_state = None
        self.last_discard_action = None
        self.discard_states = []
        self.last_play_card_state = None
        self.last_play_card_action = None
        self.play_card_states = []

    def reset(self):
        '''
        Resets this player's internal state record lists.
        '''
        self.last_discard_state = None
        self.last_discard_action = None
        self.discard_states = []
        self.last_play_card_state = None
        self.last_play_card_action = None
        self.play_card_states = []

    def record_discard_state(self, reward, state, action):
        if self.last_discard_state is not None:
            self.discard_states.append((self.last_discard_state,
                                        self.last_discard_action,
                                        reward,
                                        state))
        self.last_discard_state = state
        self.last_discard_action = action

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
        state = discard_state_repr(is_dealer,
                                   hand,
                                   player_score,
                                   opponent_score)
        discard_idxs = self.player.discard(is_dealer, hand, player_score, opponent_score)
        # sanity checking
        assert len(set(discard_idxs)) == 2
        assert all(0 <= i < 6 for i in discard_idxs)
        discard_idxs = set(discard_idxs)
        discards = [c for i, c in enumerate(hand) if i in discard_idxs]
        # construct a vector representation of discards
        action = discard_action_repr(discards)
        # reward is zero since game is not over
        self.record_discard_state(0, state, action)
        return discard_idxs

    def record_play_card_state(self, reward, state, action):
        if self.last_play_card_state is not None:
            self.play_card_states.append((self.last_play_card_state,
                                          self.last_play_card_action,
                                          reward,
                                          state))
        self.last_play_card_state = state
        self.last_play_card_action = action

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
        state = play_state_repr(is_dealer,
                                hand,
                                played_cards,
                                is_go,
                                linear_play,
                                player_score,
                                opponent_score)
        play_idx = self.player.play_card(
            is_dealer, hand, played_cards, is_go,
            linear_play, player_score, opponent_score,
            legal_moves)
        # sanity checking
        assert 0 <= play_idx < len(hand)
        play_card = hand[play_idx]
        assert is_legal_play(play_card, linear_play)
        # construct a vector representation of play_card
        action = play_action_repr(play_card)
        # reward is zero since game is not over
        self.record_play_card_state(0, state, action)
        return play_idx

    def round_over(self):
        '''
        Notification that the current round is over.
        '''
        self.player.round_over()

    def game_over(self, has_won):
        '''
        Notification that the current game is over.  The argument to the
        function is a flag indicating if this player won or not.

        Arguments:
        - `has_won`:
        '''
        self.record_discard_state(int(has_won), None, None)
        self.record_play_card_state(int(has_won), None, None)
        self.player.game_over(has_won)
