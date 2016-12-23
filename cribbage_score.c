#include <stdio.h>

#define CARD_FACE(c) (c % 13)
#define CARD_SUIT(c) (c / 13)
#define COMP_SWAP(a,b) if ((b) < (a)) { int t = a; a = b; b = t; }
#define DEBUG 1

void print_face_counts(int face_counts[5][2])
{
    printf("%d:%d, %d:%d, %d:%d, %d:%d, %d:%d\n",
           face_counts[0][0],face_counts[0][1],
           face_counts[1][0],face_counts[1][1],
           face_counts[2][0],face_counts[2][1],
           face_counts[3][0],face_counts[3][1],
           face_counts[4][0],face_counts[4][1]);
}

int score_hand(int card1, int card2, int card3, int card4, int draw_card)
{
    // the value we will return
    int score = 0;

    // split card values into face and suit
    int f1 = CARD_FACE(card1), s1 = CARD_SUIT(card1);
    int f2 = CARD_FACE(card2), s2 = CARD_SUIT(card2);
    int f3 = CARD_FACE(card3), s3 = CARD_SUIT(card3);
    int f4 = CARD_FACE(card4), s4 = CARD_SUIT(card4);
    int fd = CARD_FACE(draw_card), sd = CARD_SUIT(draw_card);

    // we need the counts of the face values of cards
    // 5 x 2 array:
    // first number is the face value; these are listed in increasing order
    // second number is the count
    // value of -1 means unused
    int face_counts[5][2] = { { -1, 1 }, { -1, 1 }, { -1, 1 }, { -1, 1 }, { -1, 1 } };
    face_counts[0][0] = f1;
    face_counts[1][0] = f2;
    face_counts[2][0] = f3;
    face_counts[3][0] = f4;
    face_counts[4][0] = fd;
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
                int run_length = i - run_begin;
                if (run_length >= 3)
                {
                    // this is an interesting run
                    int run_score = run_length;
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
    int PAIR_SCORES[5] = {0, 0, 2, 6, 12};
    for (int i = 0; i < 5; i++)
    {
        if (face_counts[i][0] != -1)
        {
            int pair_score = PAIR_SCORES[face_counts[i][1]];
            if (pair_score)
#ifdef DEBUG
                printf("pair %d %d %d\n", face_counts[i][0], face_counts[i][1], pair_score);
#endif // DEBUG
            score += pair_score;
        }
    }

    // score 15s
    int CARD_VALUES[13] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10 };
    int face_values[5] = { CARD_VALUES[f1],
                           CARD_VALUES[f2],
                           CARD_VALUES[f3],
                           CARD_VALUES[f4],
                           CARD_VALUES[fd] };
    int FIFTEEN_COMBS[26][5] = {
        { 0, 1, -1, -1, -1 },
        { 0, 2, -1, -1, -1 },
        { 0, 3, -1, -1, -1 },
        { 0, 4, -1, -1, -1 },
        { 1, 2, -1, -1, -1 },
        { 1, 3, -1, -1, -1 },
        { 1, 4, -1, -1, -1 },
        { 2, 3, -1, -1, -1 },
        { 2, 4, -1, -1, -1 },
        { 3, 4, -1, -1, -1 },
        { 0, 1, 2, -1, -1 },
        { 0, 1, 3, -1, -1 },
        { 0, 1, 4, -1, -1 },
        { 0, 2, 3, -1, -1 },
        { 0, 2, 4, -1, -1 },
        { 0, 3, 4, -1, -1 },
        { 1, 2, 3, -1, -1 },
        { 1, 2, 4, -1, -1 },
        { 1, 3, 4, -1, -1 },
        { 2, 3, 4, -1, -1 },
        { 0, 1, 2, 3, -1 },
        { 0, 1, 2, 4, -1 },
        { 0, 1, 3, 4, -1 },
        { 0, 2, 3, 4, -1 },
        { 1, 2, 3, 4, -1 },
        { 0, 1, 2, 3, 4 },
    };
    for (int i = 0; i < 26; i++)
    {
        int acc = 0;
        for (int j = 0; j < 5; j++)
        {
            if (FIFTEEN_COMBS[i][j] == -1) break;
            acc += face_values[FIFTEEN_COMBS[i][j]];
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
