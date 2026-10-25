#!/usr/bin/env python

"""
test_round.py

Test the rules of play in Round and Game, using scripted players.
"""

import random

import pytest
from cribbage.cards import string_tocard
from cribbage.game import Game
from cribbage.pegging import Go, PegState, Play
from cribbage.player import CribbagePlayer
from cribbage.specs import make_player
from cribbage.round import Deal, Round
from cribbage.scoring import score_hand
from cribbage.test.helpers import make_round, random_deal


class ScriptedPlayer(CribbagePlayer):
    """Plays the cards it is given, in order, and keeps what it was shown."""

    def __init__(self, plays=()):
        super().__init__()
        self.plays = list(plays)
        self.discard_views = []
        self.views = []
        self.rounds_over = 0

    def discard(self, view):
        self.discard_views.append(view)
        return view.hand[0], view.hand[1]

    def play_card(self, view):
        self.views.append(view)
        return self.plays.pop(0)

    def round_over(self):
        self.rounds_over += 1


# player 0 holds KS KH KD 4S and player 1 QS QH QD 3S, and player 1 deals.
# The first sequence is KS, QS, KH: the count is 30, player 1 says go and
# player 0 (holding nothing that fits) scores one for the last card.  Player
# 1 leads the second: QH, KD, QD; player 0 says go (4S does not fit), and
# player 1 scores one.  Player 0 leads the third, 4S then 3S, and player 1
# scores one for the last card.
KEPT = [[12, 25, 38, 3], [11, 24, 37, 2]]
SCRIPT = [[12, 25, 38, 3], [11, 24, 37, 2]]


def scripted_round(plays=SCRIPT, target=10**9):
    players = [ScriptedPlayer(plays[0]), ScriptedPlayer(plays[1])]
    game = Game(players)
    game.target_score = target
    return make_round(players, KEPT, 51, 1, game), players


def test_play_scores_go_and_last_card():
    rnd, players = scripted_round()
    assert rnd.play()
    assert rnd.game.scores == [1, 2]
    assert rnd.peg.done and rnd.peg.points == [1, 2]
    assert rnd.peg is not None
    seqs = rnd.peg.sequences
    assert [s.cards for s in seqs] == [[12, 11, 25], [24, 38, 37], [3, 2]]
    assert [(s.closed_by, s.closing_points) for s in seqs] == [(0, 1), (1, 1), (1, 1)]
    assert [type(e) for e in seqs[1].events] == [Play, Play, Play, Go]
    assert seqs[1].events[-1] == Go(player=0, count=30, cards_left=1)
    assert all(p.rounds_over == 1 for p in players)


def test_players_are_asked_only_when_they_can_play():
    rnd, players = scripted_round()
    rnd.play()
    # player 0 plays 12, 25 | 38 | 3 and player 1 plays 11 | 24, 37 | 2
    assert [len(p.views) for p in players] == [4, 4]
    assert players[0].plays == [] and players[1].plays == []


def test_views_are_from_the_players_own_seat():
    rnd, players = scripted_round()
    rnd.play()
    first0, first1 = players[0].views[0], players[1].views[0]
    assert first0.hand == (12, 25, 38, 3) and first0.legal == first0.hand
    assert first0.count == 0 and not first0.opp_said_go
    assert not first0.is_dealer and first1.is_dealer
    assert first0.starter == first1.starter == 51
    assert first0.opp_hand is None and first1.opp_hand is None
    # the discards are each player's own
    assert first0.discards != first1.discards
    assert not set(first0.discards) & set(first1.discards)
    # player 1's first decision: one card on the table, played by the
    # opponent, who is player 1 from player 0's side but 0 from its own
    assert first1.seq == [12] and first1.sequence.events == [Play(1, 10, 12, 0)]
    assert first1.opp_cards_left == 3 and first1.opp_plays == [((), 12)]
    assert first1.played == {12}
    # player 0 then plays KH on a count of 20, and sees the sequences so far
    third = players[0].views[1]
    assert third.seq == [12, 11] and third.legal == (25, 38, 3)
    assert [e.player for e in third.sequence.events] == [0, 1]


def test_a_player_sees_that_the_opponent_said_go():
    # player 0 holds 0S 9S AS 2S and player 1 JS QH KH JH.  0S, JS, 9S makes
    # a run of three (3) and a count of 29, where player 1 says go.  Player 0
    # plays AS (30) and scores the go (1).  In the next sequence (QH, 2S, KH)
    # player 0 runs out of cards and says go; player 1 cannot play JH on 22,
    # and scores the last card.  Player 1 then leads JH, with player 0 saying
    # go before it starts, and scores the last card again.
    players = [ScriptedPlayer([9, 8, 0, 1]), ScriptedPlayer([10, 24, 25, 23])]
    rnd = make_round(players, [[9, 8, 0, 1], [10, 24, 25, 23]], 51, 1)
    assert rnd.play()
    p0, p1 = players
    third = p0.views[2]
    assert third.seq == [9, 10, 8] and third.opp_said_go and third.legal == (0, 1)
    # player 0 had no cards to lead with, so player 1's lead was in go
    assert [v.opp_said_go for v in p1.views] == [False, False, False, True]
    assert [v.opp_said_go for v in p0.views] == [False, False, True, False]
    assert rnd.peg is not None
    seqs = rnd.peg.sequences
    assert Go(player=1, count=29, cards_left=3) in seqs[0].events
    assert seqs[1].events[-1] == Go(player=0, count=22, cards_left=0)
    assert seqs[2].events[0] == Go(player=0, count=0, cards_left=0)
    assert [(s.closed_by, s.closing_points) for s in seqs] == [(0, 1), (1, 1), (1, 1)]
    assert rnd.game.scores == [4, 2]


def test_an_illegal_card_is_refused():
    rnd, players = scripted_round(plays=[[12], [40]])
    with pytest.raises(AssertionError):
        rnd.play()


def test_the_game_ends_the_moment_a_player_reaches_the_target():
    # one point is enough: player 0 scores the go and the game is over, so
    # player 1 is never asked to lead the second sequence
    rnd, players = scripted_round(target=1)
    assert not rnd.play()
    assert rnd.game.over and rnd.game.winner == 0
    assert rnd.game.scores == [1, 0]
    assert len(players[1].views) == 1
    assert all(p.rounds_over == 0 for p in players)


def test_points_for_a_play_come_before_those_for_the_last_card():
    # player 1 reaches the target with a fifteen, so the go point that
    # would follow is never awarded
    # player 0 holds 5S, player 1 holds KS; 5, K makes fifteen for player 1
    players = [ScriptedPlayer([4]), ScriptedPlayer([12])]
    game = Game(players)
    game.target_score = 2
    rnd = make_round(players, [[4], [12]], 51, 1, game)
    assert not rnd.play()
    assert game.scores == [0, 2] and game.winner == 1


def test_nibs():
    jack = string_tocard("JH")
    players = [ScriptedPlayer(), ScriptedPlayer()]
    rnd = make_round(players, KEPT, jack, dealer=1)
    assert rnd.nibs()
    assert rnd.game.scores == [0, 2]
    rnd = make_round(players, KEPT, string_tocard("5H"), dealer=0)
    assert rnd.nibs()
    assert rnd.game.scores == [0, 0]


def test_nibs_can_win_the_game():
    players = [ScriptedPlayer(), ScriptedPlayer()]
    game = Game(players)
    game.target_score = 2
    rnd = make_round(players, KEPT, string_tocard("JH"), 0, game)
    assert not rnd.nibs()
    assert game.winner == 0


def test_show_scores_hands_and_crib():
    players = [ScriptedPlayer(), ScriptedPlayer()]
    # the dealer's (player 1's) hand is the 29 hand: three fives and the jack
    # of the starter's suit, with the starter a five
    starter = string_tocard("5C")
    kept = [
        [string_tocard(c) for c in ("AS", "3H", "7D", "KC")],
        [string_tocard(c) for c in ("5S", "5H", "5D", "JC")],
    ]
    rnd = make_round(players, kept, starter, 1)
    assert rnd.show()
    assert rnd.cards is not None
    crib = rnd.cards.crib
    assert score_hand(kept[1], starter) == 29
    assert rnd.game.scores == [
        score_hand(kept[0], starter),
        29 + score_hand(crib, starter, crib=True),
    ]


def test_deal_asks_each_player_to_discard():
    players = [ScriptedPlayer(), ScriptedPlayer()]
    game = Game(players, rng=random.Random(5))
    rnd = Round(game, players, dealer_idx=0)
    assert rnd.deal()
    cards = rnd.cards
    assert isinstance(cards, Deal) and cards.dealer == 0
    views = [p.discard_views[0] for p in players]
    assert [v.is_dealer for v in views] == [True, False]
    assert [(v.score, v.opp_score) for v in views] == [(0, 0), (0, 0)]
    for seat in (0, 1):
        assert views[seat].hand == cards.dealt[seat] and len(cards.dealt[seat]) == 6
        # the scripted player throws its first two cards
        assert cards.discards[seat] == cards.dealt[seat][:2]
        assert cards.kept[seat] == cards.dealt[seat][2:]
    everything = [*cards.dealt[0], *cards.dealt[1], cards.starter]
    assert len(set(everything)) == 13
    assert sorted(cards.crib) == sorted([*cards.discards[0], *cards.discards[1]])


def test_discarding_a_card_not_held_is_refused():
    class Cheat(ScriptedPlayer):
        def discard(self, view):
            return 99, view.hand[0]

    players = [Cheat(), ScriptedPlayer()]
    rnd = Round(Game(players), players, dealer_idx=0)
    with pytest.raises(AssertionError):
        rnd.deal()


def test_the_same_rng_gives_the_same_game():
    def play(seed):
        random.seed(0)
        players: list[CribbagePlayer] = [make_player("random"), make_player("random")]
        game = Game(players, rng=random.Random(seed))
        rnd = Round(game, players, game.dealer_idx)
        rnd.deal()
        return game.dealer_idx, rnd.cards

    assert play(1) == play(1)
    assert play(1) != play(2)


def test_kept_hands_keep_the_order_dealt():
    cards = Deal(
        dealer=0,
        dealt=((1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)),
        discards=((5, 2), (12, 7)),
        starter=0,
    )
    assert cards.kept == ((1, 3, 4, 6), (8, 9, 10, 11))
    assert cards.crib == [5, 2, 12, 7]


def test_random_games_always_finish():
    for seed in range(20):
        players: list[CribbagePlayer] = [make_player("random"), make_player("random")]
        game = Game(players, rng=random.Random(seed))
        random.seed(seed)
        game.play()
        assert game.over and game.winner in (0, 1)
        assert game.scores[game.winner] >= 121


def test_state_matches_round():
    """The points Round awards for the play are those PegState adds up."""
    rng = random.Random(1)
    for trial in range(300):
        hands, starter = random_deal(rng)
        dealer = trial % 2
        players: list[CribbagePlayer] = [make_player("random"), make_player("random")]
        rnd = make_round(players, hands, starter, dealer)
        # record who played what, by wrapping the players' choices
        log = []
        for seat, player in enumerate(players):
            original = player.play_card

            def spy(view, original=original, seat=seat):
                card = original(view)
                log.append((seat, card))
                return card

            player.play_card = spy  # type: ignore[method-assign]
        random.seed(trial)
        rnd.play()
        state = PegState.start(hands, dealer)
        for seat, card in log:
            assert not state.done
            assert state.turn == seat
            state.play_card(card)
        assert state.done
        assert state.points == rnd.game.scores
        assert all(type(p) is int for p in state.points)
        assert rnd.peg is not None and rnd.peg.history() == state.history()
