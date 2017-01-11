#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
from cribbage.game import Game
from cribbage.randomplayer import RandomCribbagePlayer
from cribbage.simpleplayer import SimpleCribbagePlayer

def compare_players(players, num_games=1000):
    stats = [0, 0]
    for i in range(num_games):
        g = Game(players)
        g.play()
        stats[g.winner] += 1
    return stats

# ------------------------------------------------------------
# Cribbage Game

stats = compare_players([RandomCribbagePlayer(), RandomCribbagePlayer()])
print('Random vs. random:', stats)

# stats
# [487, 513]

stats = compare_players([RandomCribbagePlayer(), SimpleCribbagePlayer()], 500)
print('Random vs. simple:', stats)

# with discard()
# stats
# [16, 484]

stats = compare_players([RandomCribbagePlayer(),
                         SimpleCribbagePlayer(estimate_playcard=False)],
                        500)
print('Random vs. simple (only discard):', stats)

# with play_card()
# stats
# [12, 488]
# 0.976 success against random player

# http://www.socscistatistics.com/tests/chisquare/Default2.aspx
# The chi-square statistic is 0.5879. The p-value is .443236.

stats = compare_players([RandomCribbagePlayer(),
                         SimpleCribbagePlayer(estimate_discard=False)],
                        500)
print('Random vs. simple (only play_card):', stats)

# stats
# [161, 339]

stats = compare_players([SimpleCribbagePlayer(),
                         SimpleCribbagePlayer(estimate_playcard=False)],
                        500)
print('Simple vs. simple (only discard):', stats)

# stats
# [326, 174]

# stats (after optimizing code)
# [298, 202]
# [325, 175]

def myfunc():
    stats = compare_players([SimpleCribbagePlayer(),
                             SimpleCribbagePlayer(estimate_playcard=False)],
                            100)

import cProfile
cProfile.run('myfunc()', sort='time')

# deck=make_deck()
# random.shuffle(deck)
# p=SimpleCribbagePlayer()
# hand=deck[:6]
# def wrap_discard():
#     for i in range(1000):
#         p.discard(hand,False)
# import hotshot
# prof = hotshot.Profile("stones.prof")
# prof.runcall(wrap_discard)
# prof.close()

# import hotshot.stats
# stats = hotshot.stats.load("stones.prof")
# stats.sort_stats('time', 'calls')
# stats.print_stats(20)

stats = compare_players([SimpleCribbagePlayer(estimate_discard=False),
                         SimpleCribbagePlayer(estimate_playcard=False)],
                        500)
print('Simple (only play_card) vs. simple (only discard):', stats)

# stats
# [48, 452]

# estimate the number of discard() samples we need for good
# performance in simpleplayer
for num_samples in [5, 10, 20, 50, 100, 200, 500, 1000]:
    stats = compare_players([SimpleCribbagePlayer(),
                             SimpleCribbagePlayer(num_discard_samples=num_samples)],
                            200)
    print('Simple vs simple (num_discard_samples = {}):'.format(num_samples), stats)

# Simple vs simple (num_discard_samples = 5): [116, 84]
# Simple vs simple (num_discard_samples = 10): [99, 101]
# Simple vs simple (num_discard_samples = 20): [96, 104]
# Simple vs simple (num_discard_samples = 50): [107, 93]
# Simple vs simple (num_discard_samples = 100): [104, 96]
# Simple vs simple (num_discard_samples = 200): [102, 98]
# Simple vs simple (num_discard_samples = 500): [98, 102]
# Simple vs simple (num_discard_samples = 1000): [102, 98]

from cribbage.neural import record_both_player_states

#states = record_both_player_states(RandomCribbagePlayer(), SimpleCribbagePlayer())

import random
def random_discard_sars_gen(random_seed = None):
    '''
    Infinite generator over discard (state, action, reward,
    next_state) tuples, using a random player.  Produces about 2700
    states per second on samarkand.

    Arguments:
    - `random_seed`:
    '''
    random.seed(random_seed)
    player = RandomCribbagePlayer()
    while True:
        discard_states1, _pcs1, discard_states2, _pcs2 = record_both_player_states(player, player)
        for state in discard_states1:
            yield state
        for state in discard_states2:
            yield state

def random_discard_state_gen(random_seed = None):
    '''
    Infinite generator over discard states, using a random player.
    Produces about 2700 states per second on samarkand.

    Arguments:
    - `random_seed`:
    '''
    for (state, _a, _r, _s) in random_discard_sars_gen(random_seed):
        yield state
