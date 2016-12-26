#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
from cribbage.game import Game
from cribbage.randomplayer import RandomCribbagePlayer
from cribbage.simpleplayer import SimpleCribbagePlayer

# ------------------------------------------------------------
# Cribbage Game

stats = [0,0]
for i in range(1000):
    g = Game([RandomCribbagePlayer(), RandomCribbagePlayer()])
    g.play()
    stats[g.winner] += 1

# stats
# [487, 513]

stats = [0,0]
for i in range(500):
    g = Game([RandomCribbagePlayer(), SimpleCribbagePlayer()])
    g.play()
    stats[g.winner] += 1

# with discard()
# stats
# [16, 484]

# with play_card()
# stats
# [12, 488]
# 0.976 success against random player

# http://www.socscistatistics.com/tests/chisquare/Default2.aspx
# The chi-square statistic is 0.5879. The p-value is .443236.

stats = [0,0]
for i in range(500):
    g = Game([RandomCribbagePlayer(), SimpleCribbagePlayer(estimate_discard=False)])
    g.play()
    stats[g.winner] += 1

# stats
# [161, 339]

stats = [0,0]
for i in range(500):
    g = Game([SimpleCribbagePlayer(), SimpleCribbagePlayer(estimate_playcard=False)])
    g.play()
    stats[g.winner] += 1

# stats
# [326, 174]

# stats (after optimizing code)
# [298, 202]
# [325, 175]

def myfunc():
    stats = [0,0]
    for i in range(100):
        g = Game([SimpleCribbagePlayer(), SimpleCribbagePlayer(estimate_playcard=False)])
        g.play()
        stats[g.winner] += 1

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

stats = [0,0]
for i in range(500):
    g = Game([SimpleCribbagePlayer(estimate_discard=False), SimpleCribbagePlayer(estimate_playcard=False)])
    g.play()
    stats[g.winner] += 1

# stats
# [48, 452]
