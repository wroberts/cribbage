cdef extern from "cribbage_score.h":
    int c_score_hand "score_hand" (unsigned char card1,
                                   unsigned char card2,
                                   unsigned char card3,
                                   unsigned char card4,
                                   unsigned char draw_card,
                                   unsigned char is_crib)

def score_hand(hand, draw, crib=False, verbose=False):
    return c_score_hand(hand[0], hand[1], hand[2], hand[3], draw, crib)
