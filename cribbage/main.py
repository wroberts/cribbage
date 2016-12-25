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

# stats
# [16, 484]
