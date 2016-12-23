cdef extern from "cribbage_score.h":
    int c_score_hand "score_hand" (int card1, int card2, int card3, int card4, int draw_card)

def score_hand(hand, draw):
    return c_score_hand(hand[0], hand[1], hand[2], hand[3], draw)
