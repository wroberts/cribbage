#!/usr/bin/env python

"""
test_pegging.py

Test the pegging module: the rules of the play, and its event history.
"""

import json
import random

from cribbage.pegging import Go, PegState, Play, Sequence, SequenceEvent

KING, QUEEN, ACE, FIVE = 12, 11, 0, 4


def random_hands(rng):
    n = rng.choice([4, 4, 4, 3, 2, 1])
    m = rng.choice([n, n, 4, 1])
    cards = rng.sample(range(52), n + m)
    return [cards[:n], cards[n:]]


def play_out(state, rng):
    while not state.done:
        state.play(rng.choice(state.legal()))
    return state


def test_event_types():
    play = Play(player=0, count=7, card=3, points=2)
    go = Go(player=1, count=7, cards_left=0)
    assert play.type == "play" and go.type == "go"
    assert isinstance(play, SequenceEvent) and isinstance(go, SequenceEvent)
    assert play.to_dict() == {
        "type": "play",
        "player": 0,
        "count": 7,
        "card": 3,
        "points": 2,
    }
    for event in (play, go):
        assert SequenceEvent.from_dict(json.loads(json.dumps(event.to_dict()))) == event


def test_go_with_no_card_that_fits():
    # three kings against three queens: the count reaches 30 and the
    # queens' player cannot continue, so the kings' player scores the go
    # player 0 holds three kings (KS KH KD), player 1 three queens (QS QH QD);
    # player 0 leads
    state = PegState.start([[12, 25, 38], [11, 24, 37]], 1)
    # KS, QS, KH
    for card in (12, 11, 25):
        state.play_card(card)
    first = state.sequences[0]
    assert [e.type for e in first.events] == ["play", "play", "play", "go"]
    assert first.events[-1] == Go(player=1, count=30, cards_left=2)
    assert first.said_go(1) and not first.said_go(0)
    assert first.closed and (first.closed_by, first.closing_points) == (0, 1)
    assert first.cards == [12, 11, 25]  # KS QS KH
    # the next sequence starts with the player after the last one
    assert len(state.sequences) == 2
    assert state.seq == [] and state.turn == 1 and not state.is_go


def test_go_when_out_of_cards():
    # the opponent has no cards: they say go at once, with none left
    # player 0 holds a king (KS), player 1 has no cards
    state = PegState.start([[KING], []], 1)
    assert state.sequences[0].events == []
    state.play_card(KING)
    assert state.done
    events = state.sequences[0].events
    assert events == [Play(0, 10, KING, 0), Go(1, 10, 0)]
    assert state.points == [1, 0]


def test_go_from_the_start_of_a_sequence():
    # the player who must lead has no cards
    # player 0 has no cards, player 1 holds a queen (QS); player 0 would lead
    state = PegState.start([[], [QUEEN]], 1)
    assert state.sequences[0].events == [Go(0, 0, 0)]
    assert state.is_go and state.turn == 1
    state.play_card(QUEEN)
    assert state.done and state.points == [0, 1]


def test_thirty_one():
    # player 0 holds KS KD, player 1 holds KH and AS: K, K, K, A makes 31
    state = PegState.start([[12, 38], [25, ACE]], 1)
    # KS KH KD AS
    for card in (12, 25, 38, ACE):
        state.play_card(card)
    assert state.done
    only = state.sequences[0]
    assert only.count == 31
    assert (only.closed_by, only.closing_points) == (1, 2)
    assert not any(e.type == "go" for e in only.events)


def test_is_go_is_the_opponent_having_said_go():
    # player 0 holds three kings (KS KH KD), player 1 three queens (QS QH QD);
    # player 0 leads
    state = PegState.start([[12, 25, 38], [11, 24, 37]], 1)
    state.play_card(12)
    state.play_card(11)
    assert not state.is_go
    state.place(state.hands[0].index(25))
    state.advance()
    assert state.sequences[0].said_go(1)


def test_replaying_the_events_gives_the_same_state():
    rng = random.Random(3)
    for trial in range(300):
        hands = random_hands(rng)
        original = play_out(PegState.start(hands, trial % 2), rng)
        replay = PegState.start(hands, trial % 2)
        for event in original.events:
            if isinstance(event, Play):
                assert replay.turn == event.player
                replay.play_card(event.card)
        assert replay.done
        assert replay.history() == original.history()
        assert replay.points == original.points


def test_events_account_for_the_points():
    rng = random.Random(4)
    for trial in range(300):
        state = play_out(PegState.start(random_hands(rng), trial % 2), rng)
        total = [0, 0]
        for seq in state.sequences:
            assert seq.closed and seq.count <= 31
            count = 0
            for event in seq.events:
                assert event.count >= count
                count = event.count
                if isinstance(event, Play):
                    total[event.player] += event.points
            assert seq.count == count
            assert seq.closed_by == seq.last
            total[seq.closed_by] += seq.closing_points
        assert total == state.points
        assert all(type(p) is int for p in state.points)


def test_history_survives_json():
    rng = random.Random(5)
    for trial in range(50):
        state = play_out(PegState.start(random_hands(rng), trial % 2), rng)
        data = json.loads(json.dumps(state.history()))
        assert [Sequence.from_dict(d).to_dict() for d in data] == state.history()
        assert [e for s in data for e in s["events"]] == [
            e.to_dict() for e in state.events
        ]


def test_copies_do_not_share_the_open_sequence():
    # player 0 holds KS QS, player 1 holds 5S QH
    state = PegState.start([[KING, QUEEN], [FIVE, 24]], 1)
    state.play_card(KING)
    other = state.copy()
    other.play_card(FIVE)
    assert state.seq == [KING] and other.seq == [KING, FIVE]
    assert len(state.events) == 1 and len(other.events) == 2
    # closed sequences are shared, and playing a copy does not change them
    rng = random.Random(6)
    # player 0 holds KS KH KD 4S, player 1 holds QS QH QD 3S: after K, Q, K the count is 30 and
    # neither 3S nor 4S fits
    state = PegState.start([[12, 25, 38, 3], [11, 24, 37, 2]], 1)
    # KS QS KH
    for card in (12, 11, 25):
        state.play_card(card)
    assert len(state.sequences) == 2
    snapshot = state.history()
    other = state.copy()
    assert other.sequences[0] is state.sequences[0]
    play_out(other, rng)
    assert state.history() == snapshot
    assert other.history()[0] == snapshot[0]


def test_unrecorded_states_play_identically():
    rng = random.Random(7)
    for trial in range(200):
        hands = random_hands(rng)
        a = PegState.start(hands, trial % 2)
        b = PegState.start(hands, trial % 2)
        b.record = False
        while not a.done:
            assert (a.turn, a.seq, a.count, a.is_go, a.last, a.points) == (
                b.turn,
                b.seq,
                b.count,
                b.is_go,
                b.last,
                b.points,
            )
            idx = rng.choice(a.legal())
            a.play(idx)
            b.play(idx)
        assert b.done and b.points == a.points and b.hands == a.hands
        assert b.events == []


def go_state():
    """
    Player 0 holds 0S 9S AS 2S and player 1 JS QH KH JH; player 0 leads.
    After 0S, JS, 9S the count is 29, player 1 has said go and player 0 may
    still play AS or 2S.
    """
    state = PegState.start([[9, 8, 0, 1], [10, 24, 25, 23]], 1)
    for card in (9, 10, 8):
        state.play_card(card)
    return state


def test_view_shows_the_player_their_own_seat():
    # player 0 holds KS KH KD 4S, player 1 QS QH QD 3S
    state = PegState.start([[12, 25, 38, 3], [11, 24, 37, 2]], 1)
    # KS, QS, KH close the first sequence (player 0 scores the go)
    for card in (12, 11, 25):
        state.play_card(card)
    assert state.turn == 1
    v0 = state.view(0, [40, 41], 42, True, 7, 5)
    v1 = state.view(1, [43, 44], 42, False, 5, 7)
    # each sees the sequences with themselves as player 0
    assert [e.player for e in v0.sequences[0].events] == [0, 1, 0, 1]
    assert [e.player for e in v1.sequences[0].events] == [1, 0, 1, 0]
    assert v0.sequences[0].closed_by == 0 and v1.sequences[0].closed_by == 1
    assert v1.sequences[0].events[0] == Play(1, 10, 12, 0)
    # the engine's own record is untouched
    assert state.sequences[0].events[0] == Play(0, 10, 12, 0)
    assert state.sequences[0].closed_by == 0
    assert v0.hand == (38, 3) and v1.hand == (24, 37, 2)
    assert (v0.is_dealer, v0.score, v0.opp_score) == (True, 7, 5)
    assert (v1.is_dealer, v1.score, v1.opp_score) == (False, 5, 7)
    assert v0.discards == (40, 41) and v1.starter == 42


def test_view_derives_what_a_player_can_work_out():
    state = go_state()
    view = state.view(0, [40, 41], 42, False, 0, 0)
    assert view.hand == (0, 1) and view.legal == (0, 1)
    assert view.seq == [9, 10, 8] and view.count == 29
    assert view.opp_said_go and state.is_go
    assert view.played == {9, 10, 8}
    assert view.opp_cards_left == 3
    assert view.opp_plays == [((9,), 10)]
    assert view.opp_hand is None
    # what may be in the opponent's hand or the crib: everything not seen
    assert sorted(view.unseen()) == sorted(
        set(range(52)) - {9, 10, 8, 0, 1, 40, 41, 42}
    )
    assert set(state.hands[1]) <= set(view.unseen())


def test_view_is_a_snapshot():
    state = go_state()
    view = state.view(0, [40, 41], 42, False, 0, 0)
    state.play_card(0)
    assert view.seq == [9, 10, 8] and view.count == 29
    assert state.sequences[0].cards == [9, 10, 8, 0]


def test_view_legal_cards_follow_the_count():
    state = PegState.start([[12, 25, 38, 3], [11, 24, 37, 2]], 1)
    for card in (12, 11):
        state.play_card(card)
    view = state.view(0, [40, 41], 42, True, 0, 0)
    assert view.count == 20 and view.legal == (25, 38, 3)
    # the count is 30 and neither 3S nor 4S fits
    state = go_state()
    state.play_card(0)
    assert state.sequences[0].count == 30


def test_from_view():
    state = go_state()
    world = PegState.from_view(state.view(0, [40, 41], 42, False, 0, 0), [24, 25, 23])
    assert world.hands == [[0, 1], [24, 25, 23]]
    assert world.turn == 0 and world.seq == [9, 10, 8] and world.count == 29
    assert world.is_go and world.last == 0
    assert not world.record and world.events == []
    assert world.legal() == [0, 1]
    # it plays on like the real state
    other = state.copy()
    before = other.points[:]
    world.play(0)
    other.play(0)
    assert (world.turn, world.seq) == (other.turn, other.seq)
    assert world.points == [other.points[0] - before[0], other.points[1] - before[1]]
