#include <stdio.h>
#include "cribbage_score.h"

#define CARD_FACE(c) (c % 13)
#define CARD_SUIT(c) (c / 13)
#define COMP_SWAP(a,b) if ((b) < (a)) { count_t t = a; a = b; b = t; }
//#define DEBUG 1

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

/*
 * The values of having a set of card face_values of the given size:
 * 0 of a kind - 0 points
 * 1 of a kind - 0 points
 * 2 of a kind - 2 points
 * 3 of a kind - 6 points (2 x 3)
 * 4 of a kind - 12 points (2 x 6)
 */
static const score_t PAIR_SCORES[5] = { 0, 0, 2, 6, 12 };

/*
 * The values of each face_value, for adding up to 15.
 */
static const score_t CARD_VALUES[13] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10 };

/*
 * The ways to combine two or more cards from a set of five cards,
 * specified in indices into a 5-element array.
 * 0xFF indicates a blank.
 */
#define NUM_COMBINATIONS 26
static const unsigned char COMBINATIONS[NUM_COMBINATIONS][5] = {
    { 0, 1, 0xFF, 0xFF, 0xFF },
    { 0, 2, 0xFF, 0xFF, 0xFF },
    { 0, 3, 0xFF, 0xFF, 0xFF },
    { 0, 4, 0xFF, 0xFF, 0xFF },
    { 1, 2, 0xFF, 0xFF, 0xFF },
    { 1, 3, 0xFF, 0xFF, 0xFF },
    { 1, 4, 0xFF, 0xFF, 0xFF },
    { 2, 3, 0xFF, 0xFF, 0xFF },
    { 2, 4, 0xFF, 0xFF, 0xFF },
    { 3, 4, 0xFF, 0xFF, 0xFF },
    { 0, 1, 2, 0xFF, 0xFF },
    { 0, 1, 3, 0xFF, 0xFF },
    { 0, 1, 4, 0xFF, 0xFF },
    { 0, 2, 3, 0xFF, 0xFF },
    { 0, 2, 4, 0xFF, 0xFF },
    { 0, 3, 4, 0xFF, 0xFF },
    { 1, 2, 3, 0xFF, 0xFF },
    { 1, 2, 4, 0xFF, 0xFF },
    { 1, 3, 4, 0xFF, 0xFF },
    { 2, 3, 4, 0xFF, 0xFF },
    { 0, 1, 2, 3, 0xFF },
    { 0, 1, 2, 4, 0xFF },
    { 0, 1, 3, 4, 0xFF },
    { 0, 2, 3, 4, 0xFF },
    { 1, 2, 3, 4, 0xFF },
    { 0, 1, 2, 3, 4 },
};

#ifdef DEBUG
void print_face_counts(count_t face_counts[5][2])
{
    printf("%d:%d, %d:%d, %d:%d, %d:%d, %d:%d\n",
           face_counts[0][0],face_counts[0][1],
           face_counts[1][0],face_counts[1][1],
           face_counts[2][0],face_counts[2][1],
           face_counts[3][0],face_counts[3][1],
           face_counts[4][0],face_counts[4][1]);
}
#endif // DEBUG

/*
 * Scores a cribbage hand.
 *
 * Hand is specified with card values card1 through card4; the draw
 * card is given in draw_card.
 *
 * Returns the number of points to score for the given hand and draw.
 */
score_t score_hand(card_t card1, card_t card2, card_t card3, card_t card4, card_t draw_card, flag_t is_crib)
{
    // sanity check
    if (card1 > 51 ||
        card2 > 51 ||
        card3 > 51 ||
        card4 > 51 ||
        draw_card > 51)
    {
        return -1;
    }

    // the value we will return
    score_t score = 0;

    // split card values into face and suit
    card_t f1 = CARD_FACE(card1), s1 = CARD_SUIT(card1);
    card_t f2 = CARD_FACE(card2), s2 = CARD_SUIT(card2);
    card_t f3 = CARD_FACE(card3), s3 = CARD_SUIT(card3);
    card_t f4 = CARD_FACE(card4), s4 = CARD_SUIT(card4);
    card_t fd = CARD_FACE(draw_card), sd = CARD_SUIT(draw_card);

    // we need the counts of the face values of cards
    // 5 x 2 array:
    // first number is the face value; these are listed in increasing order
    // second number is the count
    // value of -1 means unused
    count_t face_counts[5][2] = { { f1, 1 },
                                  { f2, 1 },
                                  { f3, 1 },
                                  { f4, 1 },
                                  { fd, 1 } };
    // http://stackoverflow.com/a/3903172/1062499
    // sorting network: [[1 2][3 4][1 3][2 5][1 2][3 4][2 3][4 5][3 4]]
    COMP_SWAP(face_counts[0][0],face_counts[1][0]);
    COMP_SWAP(face_counts[2][0],face_counts[3][0]);
    COMP_SWAP(face_counts[0][0],face_counts[2][0]);
    COMP_SWAP(face_counts[1][0],face_counts[4][0]);
    COMP_SWAP(face_counts[0][0],face_counts[1][0]);
    COMP_SWAP(face_counts[2][0],face_counts[3][0]);
    COMP_SWAP(face_counts[1][0],face_counts[2][0]);
    COMP_SWAP(face_counts[3][0],face_counts[4][0]);
    COMP_SWAP(face_counts[2][0],face_counts[3][0]);
#ifdef DEBUG
    printf("sorted "); print_face_counts(face_counts);
#endif // DEBUG
    // uniquify face_counts
    for (int i = 0; i<4; ++i)
    {
        while (face_counts[i][0] != -1 && face_counts[i][0] == face_counts[i+1][0])
        {
            // increment the current count
            face_counts[i][1] += 1;
            // shift everybody down
            for (int j = i+1; j < 4; ++j)
            {
                face_counts[j][0] = face_counts[j+1][0];
            }
            face_counts[4][0] = -1;
        }
    }
#ifdef DEBUG
    printf("sorted unique "); print_face_counts(face_counts);
#endif // DEBUG

    // now we locate runs
    int run_begin = -1;
    for (int i = 1; i <= 5; ++i)
    {
        if (i != 5 && face_counts[i-1][0] + 1 == face_counts[i][0])
        {
            if (run_begin == -1)
            {
                // start a run
                run_begin = i - 1;
            }
        }
        else {
            if (run_begin != -1)
            {
                // finish a run
                score_t run_length = i - run_begin;
                if (run_length >= 3)
                {
                    // this is an interesting run
                    score_t run_score = run_length;
                    for (int j = run_begin; j < i; j++)
                    {
                        run_score *= face_counts[j][1];
                    }
#ifdef DEBUG
                    printf("run %d %d %d\n", run_begin, i - 1, run_score);
#endif // DEBUG
                    score += run_score;
                }
                run_begin = -1;
            }
        }
    }

    // score pairs/triples/quads
    for (int i = 0; i < 5; i++)
    {
        if (face_counts[i][0] >= 0)
        {
            score_t pair_score = PAIR_SCORES[(ucount_t)face_counts[i][1]];
#ifdef DEBUG
            if (pair_score)
                printf("pair %d %d %d\n", face_counts[i][0], face_counts[i][1], pair_score);
#endif // DEBUG
            score += pair_score;
        }
    }

    // score 15s
    score_t face_values[5] = { CARD_VALUES[f1],
                               CARD_VALUES[f2],
                               CARD_VALUES[f3],
                               CARD_VALUES[f4],
                               CARD_VALUES[fd] };
    for (int i = 0; i < NUM_COMBINATIONS; i++)
    {
        score_t acc = 0;
        for (int j = 0; j < 5; j++)
        {
            if (COMBINATIONS[i][j] == 0xFF) break;
            acc += face_values[COMBINATIONS[i][j]];
        }
        if (acc == 15)
        {
#ifdef DEBUG
            printf("fifteen\n");
#endif // DEBUG
            score += 2;
        }
    }

    // score flush
    if (s1 == s2 && s1 == s3 && s1 == s4)
    {
        if (s1 == sd)
        {
#ifdef DEBUG
            printf("flush 5\n");
#endif // DEBUG
            score += 5;
        }
        else
        {
            if (!is_crib)
            {
#ifdef DEBUG
                printf("flush 4\n");
#endif // DEBUG
                score += 4;
            }
        }
    }

    // score special jack
    if ((s1 == sd && f1 == 10) ||
        (s2 == sd && f2 == 10) ||
        (s3 == sd && f3 == 10) ||
        (s4 == sd && f4 == 10))
    {
#ifdef DEBUG
        printf("special jack\n");
#endif // DEBUG
        score += 1;
    }

    // return the score
    return score;
}

/**
 * Scores a cribbage play.
 *
 * Play is specified with card values stored in an array with maximum
 * length MAX_PLAY_LENGTH.  Card values after the last valid value are
 * set to 0xFF.
 *
 * Returns the number of points to score for playing the last card in
 * linear_play.
 */
score_t score_play(play_list_t linear_play)
{
    // compute the face values of the cards in linear_play
    card_t face_values[MAX_PLAY_LENGTH];
    // we also use this opportunity to count the length of the
    // linear_play array
    int len_play = 0;
    for (; len_play < MAX_PLAY_LENGTH && linear_play[len_play] <= 51; len_play++)
    {
        face_values[len_play] = CARD_FACE(linear_play[len_play]);
    }
    if (len_play == 0) return -1;

    // printf("----------\n");
    // printf("len_play is %d\n", len_play);
    // printf("face values are: ");
    // for (int i = 0; i < len_play; i++) printf("%d ", face_values[i]);
    // printf("\n");

    // the value we will return
    score_t score = 0;

    // check if the card values summed up are worth 15 points
    score_t acc = 0;
    for (int i = 0; i < len_play; i++)
    {
        acc += CARD_VALUES[face_values[i]];
    }
    if (acc == 15)
    {
#ifdef DEBUG
        printf("fifteen\n");
#endif // DEBUG
        score += 2;
    }

    // look for pairs and triples
    card_t last_value = face_values[len_play - 1];
    int tuple_len = len_play - 2;
    for (; tuple_len >= 0 && face_values[tuple_len] == last_value; tuple_len--);
    // for a pair: last_value represents [len_play - 1] and
    // tuple_len has the value of [len_play - 3]
    // in general, the length of the tuple is (len_play - 1) - tuple_len
    tuple_len = (len_play - 1) - tuple_len;
    score_t pair_score = PAIR_SCORES[(ucount_t)tuple_len];
#ifdef DEBUG
    if (pair_score)
        printf("pair %d points\n", pair_score);
#endif // DEBUG
    score += pair_score;

    // look for runs: O(n^2)
    //
    // these are subsequences of the face_values array which contain
    // consequetive values
    //
    // we're only interested in subsequences of at least length 3
    //
    // score the first (longest) one we find
    for (int low_index = 0; low_index <= len_play - 3; low_index++)
    {
        // we're now considering the subsequence starting at low_index
        // and going up to (not including) len_play
        //
        // keep track of the largest and smallest values we've seen
        card_t run_lo = face_values[low_index], run_hi = face_values[low_index];
        for (int i = low_index + 1; i < len_play; i++)
        {
            run_lo = MIN(face_values[i], run_lo);
            run_hi = MAX(face_values[i], run_hi);
        }
        // now we check to see if run_hi - run_lo + 1 is the length of
        // the subsequence
        // printf("low_index is %d, run_lo is %d, run_hi is %d\n", low_index, run_lo, run_hi);
        if ((run_hi - run_lo + 1) == (len_play - low_index))
        {
            // now we need to check that there are no duplicate values
            // in the subsequence
            //
            // the longest possible run is of length 7
            int value_seen[7] = {0, 0, 0, 0, 0, 0, 0};
            int duplicate = 0;
            for (int i = low_index; i < len_play; i++)
            {
                if (value_seen[face_values[i] - run_lo])
                {
                    // duplicate!
                    duplicate = 1;
                    break;
                }
                value_seen[face_values[i] - run_lo] = 1;
            }

            if (!duplicate)
            {
                // we found a run
                score_t run_score = (len_play - low_index);
#ifdef DEBUG
                printf("run %d points\n", run_score);
#endif // DEBUG
                score += run_score;
                break;
            }
        }
    }

    return score;
}
