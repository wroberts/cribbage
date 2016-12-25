#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import itertools
import random
from cribbage.cards import make_deck
from cribbage.player import CribbagePlayer
try:
    from cribbage._cribbage_score import score_hand
except ImportError:
    from cribbage.cribbage_score import score_hand

class SimpleCribbagePlayer(CribbagePlayer):
    '''
    Cribbage player with simple AI!!!
    '''

    def __init__(self):
        '''Constructor.'''
        super(SimpleCribbagePlayer, self).__init__()

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
        deck = sorted(set(make_deck()) - set(hand))
        results = {}
        for keep in itertools.combinations(hand, 4):
            samples = [score_hand(keep, draw=random.choice(deck))
                       for i in range(1000)]
            results[tuple(sorted(keep))] = sum(samples) / float(len(samples))
        # get the best hand
        best_hand = max((v,k) for (k,v) in results.items())[1]
        # convert back into indices into hand
        discard_values = set(hand) - set(best_hand)
        # return indices to discard
        return [idx for idx, card in enumerate(hand) if card in discard_values]

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
        # TODO
        return random.choice(legal_moves)
