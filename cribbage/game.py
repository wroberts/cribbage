#!/usr/bin/env python

"""
game.py
(c) Will Roberts  25 December, 2016

An object representing a game of cribbage between two CribbagePlayers.
"""

import random
from cribbage.player import CribbagePlayer
from cribbage.round import Round


class Game:
    """
    An object representing a game of cribbage between two
    CribbagePlayers.
    """

    def __init__(self, players: list[CribbagePlayer]) -> None:
        """
        Constructor.

        Arguments:
        - `players`: a list of two CribbagePlayer objects
        """
        self.players = players
        # scores start at zero
        self.scores = [0, 0]
        # The players cut for first deal, and the person who cuts the
        # lowest card deals.
        # Randomly pick one player to start.
        self.dealer_idx = random.randrange(2)
        self.target_score = 121
        # a flag to cache the game state
        self.over = False
        # indicates the index of the player that won
        self.winner: int | None = None
        # a game consists of a series of Round objects
        self.rounds: list[Round] = []
        # the last one in the series is called the current_round
        self.current_round: Round | None = None

    def play_round(self, verbose: bool = False) -> bool:
        """
        Executes a single round of cribbage.

            Play proceeds through a succession of "hands", each hand
            consisting of a "deal", "the play" and "the show." At any
            time during any of these stages, if a player reaches the
            target score (usually 121), play ends immediately with
            that player being the winner of the game. This can even
            happen during the deal, since the dealer can score if a
            Jack is cut as the starter.

        Returns True if the game is not over after the round; False if
        the game is over (or was already over before the round).

        Arguments:
        - `verbose`:
        """
        if verbose:
            print("Starting new round")
        self.current_round = Round(self, self.players, self.dealer_idx)
        self.rounds.append(self.current_round)
        if self.current_round.deal(verbose=verbose):
            if self.current_round.play(verbose=verbose):
                if self.current_round.show(verbose=verbose):
                    # swap the dealer
                    self.dealer_idx = int(not self.dealer_idx)
                    return True
        # notify the players of the winner
        for player_idx, player in enumerate(self.players):
            player.game_over(player_idx == self.winner)
        return False

    def play(self, verbose: bool = False) -> None:
        """
        Plays through the whole game of cribbage until it is over.
        """
        while self.play_round(verbose=verbose):
            if verbose:
                print("New round")

    def award_points(
        self, player_idx: int, num_points: int, verbose: bool = False
    ) -> bool:
        """
        Awards `num_points` to player `player_idx`.

        Returns True if neither player has yet reached `target_score`,
        or False if one or more players have already finished the
        game.

        Arguments:
        - `player_idx`:
        - `num_points`:
        """
        if self.over:
            return False
        # check that neither player is over target_score
        if any(score >= self.target_score for score in self.scores):
            self.over = True
            return False
        # add the points to the given player's score
        self.scores[player_idx] += num_points
        # check if that player is now over target_score
        if self.scores[player_idx] >= self.target_score:
            if verbose:
                print(
                    f"Player {player_idx + 1} wins with "
                    f"{self.scores[player_idx]} points"
                )
            self.over = True
            self.winner = player_idx
            return False
        return True


def compare_players(players: list[CribbagePlayer], num_games: int = 1000) -> list[int]:
    """
    Utility function to compare two player objects.

    This function plays the two players against each other for the
    specified number of games, and returns a two-item list indicating
    how many time each player won.

    Arguments:
    - `players`: a list of two CribbagePlayer objects
    - `num_games`: the number of games to play
    """
    stats = [0, 0]
    for _idx in range(num_games):
        game = Game(players)
        game.play()
        # a finished game always has a winner
        assert game.winner is not None
        stats[game.winner] += 1
    return stats
