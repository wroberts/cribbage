#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
from cribbage.game import Game
from cribbage.randomplayer import RandomCribbagePlayer

# ------------------------------------------------------------
# Cribbage Game

# testing
player1 = RandomCribbagePlayer()
player2 = RandomCribbagePlayer()
game = Game([player1, player2])
#game.do_round(verbose=True)
while game.play_round(verbose=True):
    print('New round')
