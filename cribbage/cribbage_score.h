#ifndef _CRIBBAGE_SCORE_H_
#define _CRIBBAGE_SCORE_H_

// a type for a card value [0-51 incl.], suit value [0-4 incl.], or face value [0-12 incl.]
typedef unsigned char card_t;
// a type for a count value
typedef char count_t;
// a type for an unsigned count value
typedef unsigned char ucount_t;
// a type for a score
typedef int score_t;
// a type for a flag
typedef unsigned char flag_t;

score_t score_hand(card_t card1,
                   card_t card2,
                   card_t card3,
                   card_t card4,
                   card_t draw_card,
                   flag_t is_crib);

// a cribbage game play can only last a maximum of 13 cards before a
// score of 31 must be achieved
#define MAX_PLAY_LENGTH 13

typedef card_t play_list_t[MAX_PLAY_LENGTH];

score_t score_play(play_list_t linear_play);

#endif /* _CRIBBAGE_SCORE_H_ */
