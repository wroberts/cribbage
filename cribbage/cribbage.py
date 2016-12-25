#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import Counter
import itertools
import math
import random
import numpy as np

# ------------------------------------------------------------
# Cards

# Cards are integers:

CARD_FACES = 'A234567890JQK'
CARD_SUITS = 'SHDC'
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def make_card(face, suit):
    '''Creates a card value with the given `face` and `suit`.'''
    return face + suit * 13

def split_card(vint):
    '''Splits a card value into its `face` and `suit` values.'''
    return vint % 13, vint // 13

def card_tostring(vint):
    '''Returns a string representation of the given card value.'''
    v, s = split_card(vint)
    return '{}{}'.format(CARD_FACES[v], CARD_SUITS[s])

def make_deck():
    '''Creates a list containing all 52 card values.'''
    return range(52)

# ------------------------------------------------------------
# Cribbage scoring

def make_random_hand():
    '''Create a random 4-card cribbage hand.'''
    return random.sample(make_deck(), 4)

def make_random_hand_and_draw():
    '''Create a random 4-card cribbage hand, and a random card as the starter.'''
    hd = random.sample(make_deck(), 5)
    return hd[:4], hd[4]

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def score_hand(hand, draw=None, crib=False, verbose=False):
    '''
    Scores a Cribbage hand.

    Arguments:
    - `hand`: a list of four card values
    - `draw`: a card value, or None
    - `crib`: a flag indicating if the given hand is a crib or not
    '''
    assert len(hand) == 4
    score = 0
    # split card values
    split_values_hand = [split_card(c) for c in hand]
    split_values = split_values_hand[:]
    if draw is not None:
        draw_split = split_card(draw)
        _draw_face, draw_suit = draw_split
        split_values.append(draw_split)
    face_values = [f for (f,s) in split_values]
    # score runs
    #sorted_faces = sorted(face_values)
    face_counts = Counter(face_values)
    sorted_uniq_faces = sorted(face_counts.keys()) + [None]
    # a run is a contiguous
    run_begin = None
    for last_v, this_v in pairwise(sorted_uniq_faces):
        if this_v == last_v + 1:
            if run_begin is None:
                run_begin = last_v
            else:
                pass
        else:
            if run_begin is None:
                pass
            else:
                # finish a run
                run_length = last_v - run_begin + 1
                if run_length >= 3:
                    # interesting
                    run_score = np.product([face_counts[x] for x in
                                            range(run_begin, run_begin + run_length)]) * run_length
                    if verbose:
                        print('run', run_begin, last_v, run_score)
                    score += run_score
                run_begin = None
    # score pairs/triples/quads/etc.
    for _f, count in face_counts.items():
        # count = 2 -> 2
        # count = 3 -> 6
        # count = n -> 2 * (n CHOOSE 2)
        if count > 1:
            pair_score = math.factorial(count) / math.factorial(count - 2)
            if verbose:
                print('pair', _f, count, pair_score)
            score += pair_score
    # score 15s
    card_values = [CARD_VALUES[x] for x in face_values]
    for comb_len in [2,3,4,5]:
        for vlist in itertools.combinations(card_values, comb_len):
            if sum(vlist) == 15:
                if verbose:
                    print('fifteen')
                score += 2
    # score flush
    suit_values = set([s for (f,s) in split_values_hand])
    if len(suit_values) == 1:
        if draw is not None and list(suit_values)[0] == draw_suit:
            # draw 5
            if verbose:
                print('flush 5')
            score += len(hand) + 1
        else:
            if not crib:
                if verbose:
                    print('flush 4')
                score += len(hand)
    # score special jack
    if draw is not None and [f for (f,s) in split_values_hand if s == draw_suit and f == 10]:
        if verbose:
            print('special jack')
        score += 1
    if verbose:
        print('score', score)
    return score

def test_score():
    '''Utility function to test the score_hand method.'''
    global hand, draw
    hand, draw = make_random_hand_and_draw()
    print(hand)
    print(draw)
    print('hand', ', '.join([card_tostring(v) for v in hand]))
    print('draw', card_tostring(draw))
    score_hand(hand, draw, verbose=True)

def n_choose_k(n,k):
    '''Returns the number of ways of choosing `k` items from a set of `n`.'''
    return math.factorial(n) // (math.factorial(k) * math.factorial(n-k))

# the number of ways of making 15s in a given hand with starter card
# n_choose_k(5,2) + n_choose_k(5,3) + n_choose_k(5,4) + n_choose_k(5,5)
# 26
# the number of ways of choosing two cards to discard into the crib
# n_choose_k(6,2)
# 15


# ------------------------------------------------------------
# Cribbage Game

class CribbagePlayer(object):
    '''
    Abstract base class for an object that plays a game of cribbage.
    '''

    def __init__(self):
        '''Constructor.'''
        pass

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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

def card_score(card):
    '''
    Returns the number of points the given card value is worth.

    Arguments:
    - `card`:
    '''
    return CARD_VALUES[split_card(card)[0]]

def cards_score(cards):
    '''
    Calls `card_score` on every value in `cards` and returns the sum.

    Arguments:
    - `cards`:
    '''
    return sum(card_score(card) for card in cards)

def is_legal_play(card, linear_play):
    '''
    Tests whether the given card value is a legal play in a game of
    cribbage, assuming that the cards in `linear_play` have previously
    been played during the same round.

    Arguments:
    - `card`:
    - `linear_play`:
    '''
    lp_score = cards_score(linear_play)
    c_score = card_score(card)
    return lp_score + c_score <= 31

def score_play(linear_play, verbose=False):
    '''
    Scores the last play in a game.

    Arguments:
    - `linear_play`: a list containing the card values played, in
      order, during the round
    '''
    # the value we will return
    play_score = 0
    # we only need to know the face values thare played
    face_values = [split_card(c)[0] for c in linear_play]
    if verbose:
        print('Score', ' '.join(CARD_FACES[f] for f in face_values))
    if cards_score(linear_play) == 15:
        if verbose:
            print('fifteen')
        play_score +=2
    # look for pairs and tuples
    for backwards in range(len(linear_play), 1, -1):
        back_list = sorted(face_values[-backwards:])
        if len(set(back_list)) == 1:
            # pairlike
            pair_score = [2,6,12][len(back_list)-2]
            if verbose:
                print('pair {} points'.format(pair_score))
            play_score += pair_score
            break
        elif back_list == range(min(back_list), max(back_list) + 1):
            # run
            run_score = len(back_list)
            if run_score >= 3:
                if verbose:
                    print('run {} points'.format(run_score))
                play_score += run_score
                break
    return play_score

class Round(object):
    '''
    A single round of cribbage in a gaame between two CribbagePlayers.
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
        # copy of self.hands
        self.orig_hands = None
        # the crib of the current game round
        self.crib = None
        # the card value of the "starter" or "cut" card for the
        # current game round
        self.starter_card = None
        # the player whose turn it currently is to play a card
        self.turn_idx = None
        # the array of cards that have been played already during the
        # current game round by the two players
        self.faceups = None
        # the same cards recorded in self.faceups, but linearised into
        # a single list, based on the time that the cards were played
        self.linear_play = None
        # during a round, we keep track of the last_player
        self.last_player = None
        # the cards played by both players during a round
        self.round_played = []
        # a flag that indicates whether the current sequence of the
        # current game round has gone to "Go" or not
        self.is_go = False
        # a flag that indicates whether the last player hit 31 or not
        # during the current sequence of the current game round
        self.flag_31 = False

    def deal_round(self, verbose=False):
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
        # ask players to select cards to discard to crib
        self.crib = []
        for idx, player in enumerate(self.players):
            discard_idxs = player.discard(self.hands[idx], idx == self.dealer_idx)
            # sanity checking
            assert len(set(discard_idxs)) == 2
            assert all(0 <= i < 6 for i in discard_idxs)
            discard_idxs = set(discard_idxs)
            discards = [c for i,c in enumerate(self.hands[idx])
                        if i in discard_idxs]
            if verbose:
                print('Player {} discards:'.format(idx+1),
                      ' '.join(card_tostring(c) for c in sorted(discards)))
            self.crib.extend(discards)
            self.hands[idx] = [c for i,c in enumerate(self.hands[idx])
                               if i not in discard_idxs]
        # copy self.hands after discarding
        self.orig_hands = [x[:] for x in self.hands]
        if verbose:
            self.print_state()
        # randomly cut a card from the deck to serve as the "starter"
        self.starter_card = random.choice(self.deck)
        # check for "his nibs"
        starter_value, _starter_suit = split_card(self.starter_card)
        if verbose:
            print('Starter card is ', card_tostring(self.starter_card))
        if starter_value == 10:
            if verbose:
                print('Dealer gets two points for his nibs')
            if not self.game.award_points(self.dealer_idx, 2, verbose=verbose):
                return False
        return True

    def play_round(self, verbose=False):
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
        # we keep track of all the cards that each player has played
        # during a hand
        self.round_played = []
        # a game is composed of sequences
        # ------
        # set up a new sequence
        # ------
        # players take turns laying one card face up on the table
        # without the count going over 31
        self.faceups = [[], []]
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
                self.round_played.extend(self.faceups)
                self.faceups = [[], []]
                self.linear_play = []
                self.is_go = False
                self.flag_31 = False

                # the round is over if no player has cards left
                if sum(map(len, self.hands)) == 0:
                    if verbose:
                        print('Round is over')
                        self.print_state()
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
                self.hands[self.turn_idx],
                self.turn_idx == self.dealer_idx,
                self.faceups[self.turn_idx],
                self.faceups[int(not self.turn_idx)],
                self.linear_play,
                legal_moves)
            # record the last player to make a move
            self.last_player = self.turn_idx
            # sanity checking
            assert 0 <= play_idx < len(self.hands[self.turn_idx])
            play_card = self.hands[self.turn_idx][play_idx]
            assert is_legal_play(play_card, self.linear_play)

            # make the move
            self.hands[self.turn_idx] = [c for i,c in enumerate(self.hands[self.turn_idx])
                                         if i != play_idx]
            self.faceups[self.turn_idx].append(play_card)
            self.linear_play.append(play_card)
            if verbose:
                print('Player {} plays:'.format(self.turn_idx+1),
                      card_tostring(play_card), end='')
                print('  Count:', cards_score(self.linear_play))
                self.print_state()

            # if the move makes the count hit 31, set the 31 flag
            if cards_score(self.linear_play) == 31:
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

    def show_round(self, verbose=False):
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
        nondealer_hand = self.orig_hands[nondealer_idx]
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
        dealer_hand = self.orig_hands[self.dealer_idx]
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

class Game(object):
    '''
    An object representing a game of cribbage between two
    CribbagePlayers.
    '''

    def __init__(self, players):
        '''
        Constructor.

        Arguments:
        - `players`: a list of two CribbagePlayer objects
        '''
        self.players = players
        # scores start at zero
        self.scores = [0, 0]
        # The players cut for first deal, and the person who cuts the
        # lowest card deals.
        # Randomly pick one player to start.
        self.dealer_idx = random.randrange(2)
        self.target_score = 121
        # a flag to cache the game state
        self.over = False
        # a game consists of a series of Round objects
        self.rounds = []
        # the last one in the series is called the current_round
        self.current_round = None

    def play_round(self, verbose=False):
        '''
        Executes a single round of cribbage.

            Play proceeds through a succession of "hands", each hand
            consisting of a "deal", "the play" and "the show." At any
            time during any of these stages, if a player reaches the
            target score (usually 121), play ends immediately with
            that player being the winner of the game. This can even
            happen during the deal, since the dealer can score if a
            Jack is cut as the starter.

        Returns True if the game is not over after the round; False if
        the game is over (or was already over before the round).

        Arguments:
        - `verbose`:
        '''
        if verbose:
            print('Starting new round')
        self.current_round = Round(self, self.players, self.dealer_idx)
        self.rounds.append(self.current_round)
        if self.current_round.deal_round(verbose=verbose):
            if self.current_round.play_round(verbose=verbose):
                if self.current_round.show_round(verbose=verbose):
                    # swap the dealer
                    self.dealer_idx = int(not self.dealer_idx)
                    return True
        return False

    def award_points(self, player_idx, num_points, verbose=False):
        '''
        Awards `num_points` to player `player_idx`.

        Returns True if neither player has yet reached `target_score`,
        or False if one or more players have already finished the
        game.

        Arguments:
        - `player_idx`:
        - `num_points`:
        '''
        if self.over:
            return False
        # check that neither player is over target_score
        if any(score >= self.target_score for score in self.scores):
            self.over = True
            return False
        # add the points to the given player's score
        self.scores[player_idx] += num_points
        # check if that player is now over target_score
        if self.scores[player_idx] >= self.target_score:
            if verbose:
                print('Player {} wins with {} points'.format(
                    player_idx + 1, self.scores[player_idx]))
            self.over = True
            return False
        return True

# testing
player1 = RandomCribbagePlayer()
player2 = RandomCribbagePlayer()
game = Game([player1, player2])
#game.do_round(verbose=True)
while game.play_round(verbose=True):
    print('New round')