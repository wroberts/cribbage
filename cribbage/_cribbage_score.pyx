cdef extern from "cribbage_score.h":
    int c_score_hand "score_hand" (unsigned char card1,
                                   unsigned char card2,
                                   unsigned char card3,
                                   unsigned char card4,
                                   unsigned char draw_card,
                                   unsigned char is_crib)

def score_hand(hand, draw, crib=False, verbose=False):
    return c_score_hand(hand[0], hand[1], hand[2], hand[3], draw, crib)

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
