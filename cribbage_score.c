#include <stdio.h>
#include "cribbage_score.h"

#define CARD_FACE(c) (c % 13)
#define CARD_SUIT(c) (c / 13)
#define COMP_SWAP(a,b) if ((b) < (a)) { count_t t = a; a = b; b = t; }
//#define DEBUG 1

void print_face_counts(count_t face_counts[5][2])
{
    printf("%d:%d, %d:%d, %d:%d, %d:%d, %d:%d\n",
           face_counts[0][0],face_counts[0][1],
           face_counts[1][0],face_counts[1][1],
           face_counts[2][0],face_counts[2][1],
           face_counts[3][0],face_counts[3][1],
           face_counts[4][0],face_counts[4][1]);
}

/*
 * The values of having a set of card face_values of the given size:
 * 0 of a kind - 0 points
 * 1 of a kind - 0 points
 * 2 of a kind - 2 points
 * 3 of a kind - 6 points (2 x 3)
 * 4 of a kind - 12 points (3 x 4)
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
static const int NUM_COMBINATIONS = 26;
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

/*
 * Scores a cribbage hand.
 *
 * Hand is specified with card values card1 through card4; the draw
 * card is given in draw_card.
 *
 * Returns the number of points to score for the given hand and draw.
 */
score_t score_hand(card_t card1, card_t card2, card_t card3, card_t card4, card_t draw_card)
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
#ifdef DEBUG
            printf("flush 4\n");
#endif // DEBUG
            score += 4;
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
