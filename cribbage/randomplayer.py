#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
randomplayer.py
(c) Will Roberts  25 December, 2016

A CribbagePlayer that always makes a random move.
'''

from __future__ import absolute_import, print_function
import random
from cribbage.player import CribbagePlayer

class RandomCribbagePlayer(CribbagePlayer):
    '''A CribbagePlayer that always makes a random move.'''

    def __init__(self, ):
        '''Constructor.'''
        super(RandomCribbagePlayer, self).__init__()

    def discard(self, hand, is_dealer):
        '''
        Asks the player to select two cards from `hand` for discarding to
        the crib.

        Return is a list of two indices into the hand array.

        Arguments:
        - `hand`: an array of 6 card values
        - `is_dealer`: a flag to indicate whether the given player is
          currently the dealer or not
        '''
        return random.sample(range(6), 2)

    def play_card(self, hand, is_dealer, own_pile, other_pile, linear_play, legal_moves):
        '''
        Asks the player to select one card from `hand` to play during a
        cribbage round.

        Return an index into the hand array.

        Arguments:
        - `hand`: an array of 1 to 4 card values
        - `is_dealer`: a flag to indicate whether the given player is
          currently the dealer or not
        - `own_pile`: an array of 0 to 3 card values that the given
          player has already played in this round
        - `other_pile`: an array of 0 to 4 card values that the
          player's opponent has played in this round
        - `linear_play`: the array of card values that have been
          played in this round by both players, zippered into a single
          list
        - `legal_moves`: a list of indices into `hand` indicating
          which cards from the hand may be played legally at this
          point in the game
        '''
        return random.choice(legal_moves)
