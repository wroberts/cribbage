#!/usr/bin/env python

'''
test_scoring.py
(c) Will Roberts  25 December, 2016

Test the cribbage_scoring module.
'''

import itertools
import random
import pytest
from cribbage.cards import card_tostring, make_random_hand_and_draw, card_worth, cards_worth, string_tocard
from cribbage.cribbage_score import score_hand, score_play
from cribbage._cribbage_score import score_play as c_score_play
from cribbage._cribbage_score import score_hand as c_score_hand
from cribbage._cribbage_score import card_worth as c_card_worth
from cribbage._cribbage_score import cards_worth as c_cards_worth

def explore_score_hand():
    '''Utility function to test the score_hand method.'''
    global hand, draw
    hand, draw = make_random_hand_and_draw()
    print(hand)
    print(draw)
    print('hand', ', '.join([card_tostring(v) for v in hand]))
    print('draw', card_tostring(draw))
    score_hand(hand, draw, verbose=True)

def make_testcases():
    '''Produce random test cases.'''
    for i in range(100):
        hand, draw = make_random_hand_and_draw()
        print('assert score_hand({!r}, {!r}) == {}'.format(hand, draw, score_hand(hand, draw)))

def score_hand_tester(fn):
    '''Randomly built test cases to peg scoring behaviour.'''
    assert fn([22, 48, 25, 47], 0) == 2
    assert fn([12, 16, 34, 35], 46) == 3
    assert fn([49, 39, 48, 13], 5) == 2
    assert fn([45, 3, 40, 15], 43) == 6
    assert fn([8, 50, 29, 9], 7) == 3
    assert fn([28, 47, 26, 23], 6) == 0
    assert fn([37, 12, 30, 49], 44) == 10
    assert fn([31, 19, 4, 21], 10) == 7
    assert fn([16, 39, 19, 11], 14) == 2
    assert fn([42, 20, 48, 47], 31) == 5
    assert fn([5, 6, 38, 47], 29) == 2
    assert fn([50, 32, 4, 37], 36) == 8
    assert fn([13, 19, 1, 26], 5) == 6
    assert fn([29, 13, 27, 46], 51) == 4
    assert fn([34, 32, 21, 38], 41) == 2
    assert fn([37, 47, 19, 20], 28) == 5
    assert fn([2, 33, 21, 1], 14) == 4
    assert fn([8, 47, 44, 11], 18) == 12
    assert fn([20, 48, 18, 35], 46) == 4
    assert fn([18, 31, 15, 12], 45) == 4
    assert fn([45, 9, 32, 16], 34) == 2
    assert fn([11, 42, 17, 12], 35) == 6
    assert fn([20, 27, 17, 41], 21) == 2
    assert fn([41, 26, 42, 10], 17) == 7
    assert fn([43, 13, 28, 33], 18) == 4
    assert fn([26, 2, 31, 8], 11) == 2
    assert fn([34, 2, 25, 33], 49) == 0
    assert fn([27, 44, 11, 42], 35) == 0
    assert fn([3, 50, 9, 43], 24) == 8
    assert fn([20, 38, 44, 37], 47) == 2
    assert fn([40, 32, 24, 18], 37) == 4
    assert fn([0, 12, 4, 51], 14) == 6
    assert fn([45, 33, 44, 15], 51) == 5
    assert fn([24, 35, 28, 19], 41) == 2
    assert fn([9, 10, 25, 12], 43) == 10
    assert fn([33, 42, 3, 38], 24) == 2
    assert fn([25, 47, 19, 35], 7) == 6
    assert fn([25, 33, 1, 39], 15) == 5
    assert fn([18, 26, 45, 46], 47) == 10
    assert fn([16, 15, 44, 19], 33) == 7
    assert fn([4, 6, 14, 28], 33) == 6
    assert fn([39, 24, 41, 42], 43) == 7
    assert fn([38, 14, 10, 12], 13) == 2
    assert fn([42, 3, 32, 30], 47) == 4
    assert fn([18, 7, 51, 35], 20) == 2
    assert fn([17, 19, 11, 7], 34) == 7
    assert fn([17, 18, 23, 45], 28) == 7
    assert fn([49, 7, 46, 30], 34) == 4
    assert fn([28, 43, 33, 46], 39) == 2
    assert fn([25, 37, 7, 27], 33) == 2
    assert fn([35, 18, 38, 17], 24) == 6
    assert fn([11, 35, 22, 48], 29) == 6
    assert fn([43, 22, 28, 5], 27) == 4
    assert fn([29, 43, 23, 49], 39) == 11
    assert fn([39, 34, 13, 2], 33) == 2
    assert fn([8, 36, 42, 10], 41) == 2
    assert fn([49, 51, 45, 12], 41) == 3
    assert fn([39, 17, 4, 23], 9) == 10
    assert fn([38, 0, 27, 17], 50) == 4
    assert fn([31, 42, 22, 24], 9) == 2
    assert fn([1, 7, 21, 22], 30) == 7
    assert fn([33, 3, 2, 18], 31) == 6
    assert fn([12, 27, 21, 23], 24) == 4
    assert fn([46, 23, 20, 49], 22) == 5
    assert fn([26, 31, 32, 39], 5) == 8
    assert fn([34, 9, 12, 8], 26) == 2
    assert fn([18, 4, 34, 8], 26) == 10
    assert fn([7, 12, 5, 20], 25) == 4
    assert fn([11, 33, 42, 50], 49) == 2
    assert fn([5, 19, 6, 4], 28) == 12
    assert fn([19, 38, 26, 1], 21) == 0
    assert fn([7, 45, 14, 26], 8) == 5
    assert fn([36, 17, 7, 46], 32) == 9
    assert fn([10, 32, 36, 6], 47) == 4
    assert fn([25, 29, 17, 35], 44) == 9
    assert fn([7, 36, 15, 44], 48) == 0
    assert fn([0, 3, 10, 6], 9) == 10
    assert fn([12, 46, 45, 43], 36) == 6
    assert fn([37, 36, 15, 43], 28) == 7
    assert fn([40, 35, 42, 30], 31) == 7
    assert fn([26, 30, 50, 21], 11) == 8
    assert fn([21, 25, 39, 49], 36) == 2
    assert fn([33, 24, 0, 43], 34) == 4
    assert fn([47, 3, 10, 13], 12) == 5
    assert fn([38, 7, 16, 31], 20) == 2
    assert fn([27, 10, 13, 30], 49) == 6
    assert fn([16, 32, 20, 6], 39) == 8
    assert fn([3, 41, 45, 11], 24) == 2
    assert fn([46, 32, 51, 25], 31) == 7
    assert fn([33, 6, 7, 35], 44) == 12
    assert fn([15, 33, 19, 39], 28) == 6
    assert fn([34, 1, 50, 15], 25) == 4
    assert fn([29, 9, 7, 10], 33) == 2
    assert fn([38, 9, 4, 50], 49) == 12
    assert fn([41, 7, 12, 25], 8) == 2
    assert fn([37, 16, 21, 5], 41) == 2
    assert fn([49, 35, 25, 11], 41) == 5
    assert fn([17, 45, 25, 46], 44) == 8
    assert fn([20, 31, 40, 6], 50) == 7
    assert fn([31, 36, 18, 30], 10) == 8

def score_hand_tester_2(fn):
    '''
    Test cases taken from here:

    http://codegolf.stackexchange.com/q/5515
    '''
    # 5S 5H 5D JS | KS  gives  21
    assert fn([4, 17, 30, 10], 12, False) == 21
    # AS 2D 3H JS | 4S !  gives  9
    assert fn([0, 27, 15, 10], 3, True) == 9
    # JD 3C 4H 5H | 5S  gives  12
    assert fn([36, 41, 16, 17], 4, False) == 12
    # 9S 8S 7S 6S | 5H !  gives  9
    assert fn([8, 7, 6, 5], 17, True) == 9
    # 9S 8S 7S 6S | 5H  gives  13
    assert fn([8, 7, 6, 5], 17, False) == 13
    # 8D 7D 6D 5D | 4D !  gives  14
    assert fn([33, 32, 31, 30], 29, True) == 14
    # 8D 7D 6D 5D | 4D  gives  14
    assert fn([33, 32, 31, 30], 29, False) == 14
    # AD KD 3C QD | 6D  gives  0
    assert fn([26, 38, 41, 37], 31, False) == 0

def test_score_hand():
    score_hand_tester(score_hand)

def test_c_score_hand():
    score_hand_tester(c_score_hand)

def test_score_hand_2():
    score_hand_tester_2(score_hand)

def test_c_score_hand_2():
    score_hand_tester_2(c_score_hand)

def test_card_worth():
    '''
    Test card_worth against its C implementation.
    '''
    for card in range(52):
        assert c_card_worth(card) == card_worth(card)

def test_cards_worth():
    '''
    Test cards_worth against its C implementation.
    '''
    for i in range(1000):
        cards = random.sample(range(52), 5)
        assert c_cards_worth(cards) == cards_worth(cards)

def test_score_play():
    '''
    Test score_play and its C implementation on a set of fixed card
    play patterns.
    '''
    # https://en.wikipedia.org/wiki/Rules_of_cribbage#Example_plays
    EXAMPLE_PLAYS = [
        ('0H', 0),
        ('0H 5S', 2),
        ('0H 5S 7C', 0),
        ('0H 5S 7C 6H', 3),

        ('6D', 0),
        ('6D 4S', 0),
        ('6D 4S 4H', 2),

        ('7C', 0),
        ('7C 7D', 2),
        ('7C 7D 7S', 6),
        ('7C 7D 7S 5S', 0),
        ('7C 7D 7S 5S 5C', 2),

        ('8H', 0),
        ('8H 0S', 0),
        ('8H 0S 0H', 2),

        ('AS 2S 3S', 3),
        ('AS 2S 3S 4S', 4),
        ('AS 2S 3S 4S 5S', 7), # fifteen
        ('2S 3S 4S 5S 6S', 5),
        ('AS 2S 3S 4S 5S 6S', 6),
        ('AS 2S 3S 4S 5S 6S 7S', 7),
    ]
    for (cards, score) in EXAMPLE_PLAYS:
        cards = [string_tocard(card) for card in cards.split()]
        assert score_play(cards) == score
        assert c_score_play(cards) == score

def test_score_play_2():
    '''
    Tess score_play against the C implementation of score_play.
    '''
    num_tests = 0
    while num_tests < 1000:
        num_cards = random.randint(1,6)
        cards = random.sample(range(52), num_cards)
        if cards_worth(cards) <= 31:
            assert score_play(cards) == c_score_play(cards)
            num_tests += 1

def ref_score_hand(hand, draw, is_crib=False):
    '''Brute-force reference implementation of score_hand.'''
    cards = list(hand) + [draw]
    faces = [c % 13 for c in cards]
    values = [min(f + 1, 10) for f in faces]
    score = 0
    for n in range(2, 6):
        for combo in itertools.combinations(range(5), n):
            if sum(values[i] for i in combo) == 15:
                score += 2
    counts = [faces.count(r) for r in range(13)]
    score += sum(c * (c - 1) for c in counts)
    for length in (5, 4, 3):
        total = 0
        for lo in range(13 - length + 1):
            prod = 1
            for r in range(lo, lo + length):
                prod *= counts[r]
            total += length * prod
        if total:
            score += total
            break
    suits = [c // 13 for c in cards]
    if len(set(suits[:4])) == 1:
        if suits[4] == suits[0]:
            score += 5
        elif not is_crib:
            score += 4
    score += any(c % 13 == 10 and c // 13 == suits[4] for c in hand)
    return score

def score_hand_special_cases(fn):
    # his nobs plus two fifteens: JS 2H 4D 6C | 9S
    assert fn([10, 14, 29, 44], 8, False) == 5
    # perfect 29: 5S 5H 5D JC | 5C
    assert fn([4, 17, 30, 49], 43, False) == 29
    # double run: 3S 4H 4D 5C | KS (6 + 2 + 2)
    assert fn([2, 16, 29, 43], 12, False) == 10
    # triple run: 3S 4H 4D 4C | 5S (9 + 6 + 2)
    assert fn([2, 16, 29, 42], 4, False) == 17
    # double-double run: 3S 3H 4D 4C | 5S (12 + 4 + 4)
    assert fn([2, 15, 29, 42], 4, False) == 20
    # five-card flush in the crib still scores
    assert fn([0, 1, 2, 3], 4, True) == 12

def test_score_hand_special_cases():
    score_hand_special_cases(score_hand)

def test_c_score_hand_special_cases():
    score_hand_special_cases(c_score_hand)

@pytest.mark.parametrize('fn', [score_hand, c_score_hand])
def test_score_hand_invalid(fn):
    '''Both scorers reject bad cards and duplicates.'''
    for hand, draw in [([52, 1, 2, 3], 4),
                       ([0, 1, 2, 3], 52),
                       ([0, 1, 2, 255], 4),
                       ([-1, 1, 2, 3], 4),
                       ([0, 0, 2, 3], 4),
                       ([0, 1, 2, 3], 3),
                       ([4, 4, 4, 4], 4),
                       ([0, 1, 2], 4),
                       ([0, 1, 2, 3, 5], 4)]:
        with pytest.raises(ValueError):
            fn(hand, draw)
        with pytest.raises(ValueError):
            fn(hand, draw, True)

def test_c_score_hand_exhaustive_ranks():
    '''Every rank combination, with and without flush/nobs suits.'''
    for faces in itertools.combinations_with_replacement(range(13), 5):
        for suits in ((0, 1, 2, 3, 0), (0, 0, 0, 0, 1), (0, 0, 0, 0, 0), (1, 2, 3, 0, 1)):
            cards = [f + 13 * s for f, s in zip(faces, suits)]
            if len(set(cards)) < 5:
                continue
            hand, draw = cards[:4], cards[4]
            assert c_score_hand(hand, draw, False) == ref_score_hand(hand, draw, False)

def ref_score_play(cards):
    '''Brute-force reference implementation of score_play.'''
    faces = [c % 13 for c in cards]
    score = 0
    if sum(min(f + 1, 10) for f in faces) == 15:
        score += 2
    n = 1
    while n < len(faces) and faces[-1 - n] == faces[-1]:
        n += 1
    score += {1: 0, 2: 2, 3: 6, 4: 12}[n]
    for length in range(len(faces), 2, -1):
        tail = faces[-length:]
        if len(set(tail)) == length and max(tail) - min(tail) + 1 == length:
            score += length
            break
    return score

def test_score_play_special_cases():
    CASES = [
        ('3S 5S 4S', 3),            # unordered run
        ('KS 3S 5S 4S', 3),         # run preceded by a non-run card
        ('3S 4S 4H 5S', 0),         # duplicate breaks the run
        ('QS KS AS', 0),            # no ace-high wrap
        ('JS QS KS', 3),
        ('AS AH AD 2S 2H 2D 3S 8S', 0),  # 8-card span with duplicates
        ('AS AH AD AC 2S 2H 2D 2C 3S 3H 3D 3C 4S', None),
    ]
    for cards, score in CASES:
        cards = [string_tocard(card) for card in cards.split()]
        expected = ref_score_play(cards) if score is None else score
        assert ref_score_play(cards) == expected
        assert score_play(cards) == expected
        assert c_score_play(cards) == expected

def test_score_play_random_vs_reference():
    num_tests = 0
    while num_tests < 5000:
        cards = random.sample(range(52), random.randint(1, 13))
        if cards_worth(cards) > 31:
            continue
        expected = ref_score_play(cards)
        assert score_play(cards) == expected
        assert c_score_play(cards) == expected
        num_tests += 1

@pytest.mark.parametrize('fn', [score_play, c_score_play])
def test_score_play_invalid(fn):
    '''Both scorers reject empty, overlong, out-of-range and repeated plays.'''
    for cards in ([],
                  list(range(14)),
                  [0, 1, 52],
                  [0, 1, -1],
                  [5, 5],
                  [3, 4, 5, 3]):
        with pytest.raises(ValueError):
            fn(cards)
    # 13 cards is the longest legal play
    fn([0, 13, 26, 39, 1, 14, 27, 40, 2, 15, 28, 41, 3])

@pytest.mark.parametrize('fn', [score_hand, c_score_hand])
def test_score_hand_returns_int(fn):
    # pairs/runs must not leak floats (or numpy ints) into the score
    for _ in range(500):
        hand, draw = make_random_hand_and_draw()
        assert type(fn(hand, draw)) is int
        assert type(fn(hand, draw, True)) is int

@pytest.mark.parametrize('fn', [score_play, c_score_play])
def test_score_play_returns_int(fn):
    for _ in range(500):
        play = random.sample(range(52), random.randint(1, 8))
        assert type(fn(play)) is int
