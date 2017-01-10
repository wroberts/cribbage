#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
player.py
(c) Will Roberts  25 December, 2016

Abstract base class for a cribbage player.
'''

class CribbagePlayer(object):
    '''
    Abstract base class for an object that plays a game of cribbage.
    '''

    def __init__(self):
        '''Constructor.'''
        pass

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def round_over(self):
        '''
        Notification that the current round is over.
        '''
        pass

    def game_over(self, has_won):
        '''
        Notification that the current game is over.  The argument to the
        function is a flag indicating if this player won or not.

        Arguments:
        - `has_won`:
        '''
        pass
