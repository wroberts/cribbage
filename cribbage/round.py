#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
round.py
(c) Will Roberts  25 December, 2016

An object representing a single round in a game of cribbage between
two CribbagePlayers.
'''

from __future__ import absolute_import, print_function
import random
from cribbage.cards import card_tostring, cards_worth, make_deck, split_card
from cribbage.cribbage_score import is_legal_play, score_play
try:
    from cribbage._cribbage_score import score_hand
except ImportError:
    from cribbage.cribbage_score import score_hand

class Round(object):
    '''
    A single round of cribbage in a game between two CribbagePlayers.
    '''

    def __init__(self, game, players, dealer_idx):
        '''
        Constructor.

        Arguments:
        - `players`:
        - `dealer_idx`:
        '''
        self.game = game
        self.players = players
        self.dealer_idx = dealer_idx
        # the deck of the current game round
        self.deck = None
        # the hands of the two players during the current game round
        self.hands = None
        # copy of self.hands directly after the deal
        self.dealt_hands = None
        # copy of self.hands directly after discarding, before play
        self.kept_hands = None
        # the crib of the current game round
        self.crib = None
        # the card value of the "starter" or "cut" card for the
        # current game round
        self.starter_card = None
        # the player whose turn it currently is to play a card
        self.turn_idx = None
        # the array of cards that have been played already during the
        # current game round by the two players, linearised into
        # a single list, based on the time that the cards were played
        self.linear_play = None
        # during a round, we keep track of the last_player
        self.last_player = None
        # the set of all cards that have been "seen" by both players
        # during this round, used for counting cards
        self.played_cards = set()
        # a flag that indicates whether the current sequence of the
        # current game round has gone to "Go" or not
        self.is_go = False
        # a flag that indicates whether the last player hit 31 or not
        # during the current sequence of the current game round
        self.flag_31 = False

    def deal(self, verbose=False):
        '''
        Shuffles and deals the cards for a single round of cribbage.

            The dealer shuffles and deals six cards to each player.
            Once the cards have been dealt, each player chooses four
            cards to retain, then discards the other two face-down to
            form the "crib", which will be used later by the dealer.
            At this point, each player's hand and the crib will
            contain exactly four cards. The player on the dealer's
            left cuts the deck and the dealer reveals the top card,
            called the "starter" or the "cut". If this card is a Jack,
            the dealer scores two points for "his heels."

        Returns True if the game is not over after the deal; False if
        the game is over (or was already over before the deal).

        Arguments:
        - `verbose`:
        '''
        if self.game.over:
            return False
        # create a new shuffled deck
        self.deck = make_deck()
        random.shuffle(self.deck)
        # deal out 6 cards to each player
        nondealer_hand = self.deck[:12][::2]
        dealer_hand = self.deck[:12][1::2]
        self.deck = self.deck[12:]
        # self.dealer_idx indicates the player who is dealer
        self.hands = ([nondealer_hand, dealer_hand] if self.dealer_idx
                      else [dealer_hand, nondealer_hand])
        if verbose:
            print('Dealing cards')
            self.print_state()
        # copy self.hands after dealing
        self.dealt_hands = [x[:] for x in self.hands]
        # ask players to select cards to discard to crib
        self.crib = []
        for player_idx, player in enumerate(self.players):
            discard_idxs = player.discard(
                is_dealer=self.dealer_idx == player_idx,
                hand=self.dealt_hands[player_idx],
                player_score=self.game.scores[player_idx],
                opponent_score=self.game.scores[int(not player_idx)])
            # sanity checking
            assert len(set(discard_idxs)) == 2
            assert all(0 <= i < 6 for i in discard_idxs)
            discard_idxs = set(discard_idxs)
            discards = [c for i, c in enumerate(self.hands[player_idx])
                        if i in discard_idxs]
            if verbose:
                print('Player {} discards:'.format(player_idx+1),
                      ' '.join(card_tostring(c) for c in sorted(discards)))
            self.crib.extend(discards)
            self.hands[player_idx] = [c for i, c in enumerate(self.hands[player_idx])
                                      if i not in discard_idxs]
        # copy self.hands after discarding
        self.kept_hands = [x[:] for x in self.hands]
        if verbose:
            self.print_state()
        # randomly cut a card from the deck to serve as the "starter"
        self.starter_card = random.choice(self.deck)
        # check for "his nibs"
        starter_value = split_card(self.starter_card)[0]
        if verbose:
            print('Starter card is ', card_tostring(self.starter_card))
        if starter_value == 10:
            if verbose:
                print('Dealer gets two points for his nibs')
            if not self.game.award_points(self.dealer_idx, 2, verbose=verbose):
                return False
        # add the starter to the set of played cards
        self.played_cards.add(self.starter_card)
        return True

    def play(self, verbose=False):
        '''
        Plays out cards (pegging) for a single round of cribbage.

            Starting with the non-dealer player, each player in turn
            lays one card face up on the table in front of him or her,
            stating the count--that is, the cumulative value of the
            cards that have been laid (for example, the first player
            lays a five and says "five", the next lays a six and says
            "eleven", and so on)--without the count going above
            31. The cards are not laid in the centre of the table as,
            at the end of the "play," each player needs to pick up the
            cards they have laid. Players score points during this
            process for causing the count to reach exactly fifteen,
            for runs (consecutively played, but not necessarily in
            order) and for pairs. Three or four of a kind are counted
            as multiple pairs: three of a kind is the same as three
            different pairs, or 6 points, and four of a kind is 6
            different kinds of pairs, or 12 points.

            If a player cannot play without causing the count exceed
            31, he calls "Go." Continuing with the player on his left,
            the other player(s) continue(s) the play until no one can
            play without the count exceeding 31. A player is obligated
            to play a card unless there is no card in their hand that
            can be played without the count exceeding 31 (one cannot
            voluntarily pass). Once 31 is reached or no one is able to
            play, the player who played the last card scores one point
            if the count is still under 31 and two if it is exactly
            31. The count is then reset to zero and those players with
            cards remaining repeat the process starting with the
            player to the left of the player who played the last
            card. When no player has any cards the game proceeds to
            the "show."

            Players choose the order in which to lay their cards in
            order to maximize their scores; experienced players refer
            to this as either good or poor "pegging" or
            "pegsmanship". If one player reaches the target (usually
            61 or 121), the game ends immediately and that player
            wins. When the scores are level during a game, the
            players' pegs will be side by side, and it is thought that
            this gave rise to the phrase "level pegging".[5]

        Returns True if the game is not over after the play; False if
        the game is over (or was already over before the play).

        Arguments:
        - `verbose`:
        '''
        # start with non-dealer player
        self.turn_idx = int(not self.dealer_idx)
        # a game is composed of sequences
        # ------
        # set up a new sequence
        # ------
        # players take turns laying one card face up on the table
        # without the count going over 31
        self.linear_play = []
        # a flag that indicates whether this sequence has gone to "Go" or
        # not
        self.is_go = False
        # a flag that indicates whether the last player hit 31 or not
        self.flag_31 = False
        # keep track of the last player in this sequence
        self.last_player = None

        # loop forever
        while True:
            # determine which plays the player can legally make
            legal_moves = [idx for idx, card in
                           enumerate(self.hands[self.turn_idx]) if
                           is_legal_play(card, self.linear_play)]

            # a sequence is over if the game is in "Go" and the player
            # has no legal moves, or if the player has previously hit
            # 31
            if (self.is_go and not legal_moves) or self.flag_31:
                # then last_player gets awarded 1 or 2 points, and we restart the sequence
                if self.flag_31:
                    if not self.game.award_points(self.last_player, 2, verbose=verbose):
                        return False
                else:
                    if not self.game.award_points(self.last_player, 1, verbose=verbose):
                        return False
                # restart the sequence
                if verbose:
                    print('Sequence is over')
                self.linear_play = []
                self.is_go = False
                self.flag_31 = False

                # the round is over if no player has cards left
                if sum(map(len, self.hands)) == 0:
                    if verbose:
                        print('Round is over')
                        self.print_state()
                    # notify the players that the round is over
                    for player in self.players:
                        player.round_over()
                    # exit the loop
                    break

                # restart with the opponent of the player who played the last
                # card
                self.turn_idx = int(not self.last_player)
                self.last_player = None

                continue

            # if there are no legal moves that this player can play,
            # and the game is not yet in "Go"
            if not legal_moves:
                # then we call 'Go' and make it the other player's
                # turn
                if verbose:
                    print('Player {} says go'.format(self.turn_idx+1))
                self.turn_idx = int(not self.turn_idx)
                self.is_go = True
                continue

            # ask the player to choose
            play_idx = self.players[self.turn_idx].play_card(
                is_dealer=self.dealer_idx == self.turn_idx,
                hand=self.hands[self.turn_idx],
                played_cards=self.played_cards,
                is_go=self.is_go,
                linear_play=self.linear_play,
                player_score=self.game.scores[self.turn_idx],
                opponent_score=self.game.scores[int(not self.turn_idx)],
                legal_moves=legal_moves)

            # sanity checking
            assert 0 <= play_idx < len(self.hands[self.turn_idx])
            play_card = self.hands[self.turn_idx][play_idx]
            assert is_legal_play(play_card, self.linear_play)

            # add the played card to the set of played cards
            self.played_cards.add(play_card)

            # record the last player to make a move
            self.last_player = self.turn_idx

            # make the move
            self.hands[self.turn_idx] = [c for i, c in enumerate(self.hands[self.turn_idx])
                                         if i != play_idx]
            self.linear_play.append(play_card)
            if verbose:
                print('Player {} plays:'.format(self.turn_idx+1),
                      card_tostring(play_card), end='')
                print('  Count:', cards_worth(self.linear_play))
                self.print_state()

            # if the move makes the count hit 31, set the 31 flag
            if cards_worth(self.linear_play) == 31:
                if verbose:
                    print('Player {} hit 31'.format(self.turn_idx+1))
                self.flag_31 = True

            # if the opponent player has no more cards left, toggle
            # the "Go" flag
            if not self.hands[int(not self.turn_idx)]:
                if verbose:
                    print('Player {} says go'.format(int(not self.turn_idx) + 1))
                self.is_go = True

            # score the move for player_1 if it's a 15, pair, or run
            play_score = score_play(self.linear_play, verbose=verbose)
            if play_score:
                if verbose:
                    print('Player {} scores:'.format(self.turn_idx+1), play_score)
                if not self.game.award_points(self.turn_idx, play_score, verbose=verbose):
                    return False

            # players take turns (except when one player is in "Go",
            # or this player has hit 31)
            if not self.is_go and not self.flag_31:
                self.turn_idx = int(not self.turn_idx)

        return True

    def show(self, verbose=False):
        '''
        Scores all hands after a round of cribbage.

            Once the play is complete, each player in turn, starting
            with the player on the left of the dealer, displays his
            hand on the table and scores points based on its content
            in conjunction with the starter card. Points are scored
            for combinations of cards totalling fifteen, runs, pairs
            (multiple pairs are scored pair by pair, but may be
            referred to as three or four of a kind), a flush and
            having a Jack of the same suit as the starter card ("one
            for his nob [or nobs or nibs]"). A four-card flush scores
            four and cannot include the cut or starter; a five-card
            flush scores five.

            The dealer scores his hand last and then turns the cards
            in the crib face up. These cards are then scored by the
            dealer as an additional hand, also in conjunction with the
            starter card. Unlike the dealer's own hand, the crib
            cannot score a four-card flush, but it can score a
            five-card flush with the starter.

            All scores from 0 to 29 are possible, with the exception
            of 19, 25, 26 and 27.[6] Players may refer colloquially to
            a hand scoring zero points as having a score of
            nineteen.[7]

        Returns True if the game is not over after the show; False if
        the game is over (or was already over before the show).

        Arguments:
        - `verbose`:
        '''
        # score the non-dealer player's hand
        nondealer_idx = int(not self.dealer_idx)
        nondealer_hand = self.kept_hands[nondealer_idx]
        if verbose:
            print('Scoring player {}:'.format(nondealer_idx + 1),
                  ' '.join([card_tostring(c) for c in sorted(nondealer_hand)]))
            print('Starter card is ', card_tostring(self.starter_card))
        hand_score = score_hand(nondealer_hand, self.starter_card, verbose=verbose)
        if verbose:
            print('Player {}\'s hand scores'.format(nondealer_idx + 1), hand_score)
        if not self.game.award_points(nondealer_idx, hand_score, verbose=verbose):
            return False

        # score the dealer's hand
        dealer_hand = self.kept_hands[self.dealer_idx]
        if verbose:
            print('Scoring player {}:'.format(self.dealer_idx + 1),
                  ' '.join([card_tostring(c) for c in sorted(dealer_hand)]))
            print('Starter card is ', card_tostring(self.starter_card))
        hand_score = score_hand(dealer_hand, self.starter_card, verbose=verbose)
        if verbose:
            print('Player {}\'s hand scores'.format(self.dealer_idx + 1), hand_score)
        if not self.game.award_points(self.dealer_idx, hand_score, verbose=verbose):
            return False

        # score the dealer's crib
        if verbose:
            print('Scoring crib:',
                  ' '.join([card_tostring(c) for c in sorted(self.crib)]))
            print('Starter card is ', card_tostring(self.starter_card))
        hand_score = score_hand(self.crib, self.starter_card, crib=True, verbose=verbose)
        if verbose:
            print('Player {}\'s crib scores,'.format(self.dealer_idx + 1), hand_score)
        if not self.game.award_points(self.dealer_idx, hand_score, verbose=verbose):
            return False

        return True

    def print_state(self):
        '''Print a representation of the current round state to stdout.'''
        for idx in range(2):
            print('Player {}{}  '.format(idx+1, '(D)' if idx == self.dealer_idx else '   '), end='')
            if self.hands:
                print('{:17}'.format(' '.join([card_tostring(c) for c in sorted(self.hands[idx])])),
                      end='')
            print('  {:3} Points'.format(self.game.scores[idx]), end='')
            if self.crib and idx == self.dealer_idx:
                print('  Crib', ' '.join([card_tostring(c) for c in sorted(self.crib)]), end='')
            print()
