# cython: boundscheck=False, wraparound=False, cdivision=True
cdef extern from "cribbage_score.h":
    int c_score_hand "score_hand" (unsigned char card1,
                                   unsigned char card2,
                                   unsigned char card3,
                                   unsigned char card4,
                                   unsigned char draw_card,
                                   unsigned char is_crib)
    int c_score_play "score_play" (unsigned char cards[13])

def score_hand(hand, int draw, crib=False, verbose=False):
    '''
    Scores a hand of four cards with the given draw card.

    Raises ValueError if the hand does not have four cards, or if any
    card is outside [0, 51] or appears twice.
    '''
    if len(hand) != 4:
        raise ValueError('hand must have four cards')
    cdef int c1 = hand[0]
    cdef int c2 = hand[1]
    cdef int c3 = hand[2]
    cdef int c4 = hand[3]
    cdef int score
    if not (0 <= c1 <= 51 and 0 <= c2 <= 51 and 0 <= c3 <= 51 and
            0 <= c4 <= 51 and 0 <= draw <= 51):
        raise ValueError('card values must be in [0, 51]: {!r} {!r}'.format(hand, draw))
    score = c_score_hand(c1, c2, c3, c4, draw, 1 if crib else 0)
    if score < 0:
        raise ValueError('invalid hand: duplicate cards')
    return score

def score_play(linear_play, verbose=False):
    '''
    Scores the last card played in linear_play.

    Raises ValueError if linear_play is empty or longer than 13 cards,
    or if any card is outside [0, 51] or appears twice.
    '''
    cdef unsigned char cards[13]
    cdef int num_cards = len(linear_play)
    cdef int idx
    cdef int score
    if num_cards == 0 or num_cards > 13:
        raise ValueError('linear_play must have between 1 and 13 cards')
    for idx in range(num_cards):
        cards[idx] = _check_card(linear_play[idx])
    if num_cards < 13:
        cards[num_cards] = 0xFF
    score = c_score_play(cards)
    if score < 0:
        raise ValueError('invalid play: duplicate cards')
    return score

cdef int _check_card(card) except -1:
    cdef int c = card
    if c < 0 or c > 51:
        raise ValueError('card values must be in [0, 51]: {!r}'.format(card))
    return c

cpdef card_worth(card):
    cdef int face = card % 13 + 1
    if face > 10:
        return 10
    else:
        return face

cpdef cards_worth(cards):
    cdef int idx
    cdef int n = len(cards)
    cdef int rv = 0
    cdef int card
    cdef int face
    for idx in range(n):
        card = cards[idx]
        face = card % 13 + 1
        if face > 10:
            rv += 10
        else:
            rv += face
    return rv

def is_legal_play(card, linear_play):
    cdef int lp_score = cards_worth(linear_play)
    cdef int c_score = card_worth(card)
    return lp_score + c_score <= 31

def get_legal_play_idxs(hand, linear_play):
    '''
    Faster version of is_legal_play for finding legal plays in a hand.

    Arguments:
    - `hand`:
    - `linear_play`:
    '''
    cdef int lp_score = cards_worth(linear_play)
    return [idx for idx, card in enumerate(hand)
            if card_worth(card) + lp_score <= 31]
