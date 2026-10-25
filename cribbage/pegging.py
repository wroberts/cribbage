"""
pegging.py

The pegging phase ("the play") of one round, as a game state with its
history.

A round's play is a stack of `Sequence`s, each a run of `Play` and `Go`
events from a count of zero up to 31 or until neither player can play.
`PegState` holds both hands and applies the rules of `Round.play`
(go, 31, last card) without the `Game` and `CribbagePlayer` machinery, so
it is cheap to copy, replay and search.  Players are numbered 0 and 1.
Cards are 0..51, so a hand never repeats a card.

What a player may see of the play is a `PegView`: the same history, but
from the player's own seat (they are player 0 and their opponent is
player 1) and without the opponent's hand.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceType
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from typing import Any, ClassVar

from cribbage.cards import WORTH, Card, unseen_cards
from cribbage.scoring import score_play


@dataclass(frozen=True, slots=True)
class SequenceEvent:
    """
    Something that happened in a sequence.  Subclasses set `type`, which
    names them in the dict (and JSON) form.

    - `player`: who did it
    - `count`: the count of the sequence after the event
    """

    type: ClassVar[str]
    player: int
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **asdict(self)}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SequenceEvent:
        fields = {k: v for k, v in data.items() if k != "type"}
        return EVENT_TYPES[data["type"]](**fields)


@dataclass(frozen=True, slots=True)
class Play(SequenceEvent):
    """`player` lays `card`, scoring `points` for 15s, pairs and runs."""

    type: ClassVar[str] = "play"
    card: Card
    points: int


@dataclass(frozen=True, slots=True)
class Go(SequenceEvent):
    """
    `player` says go: they cannot play at this count, either because every
    card they hold is too big, or because they have no cards left
    (`cards_left` is 0).
    """

    type: ClassVar[str] = "go"
    cards_left: int


EVENT_TYPES: dict[str, type[SequenceEvent]] = {"play": Play, "go": Go}


class Sequence:
    """
    One sequence of the play: its events, and the points for the last card
    (1) or for 31 (2) once it is closed.  The other attributes are derived
    from the events, and kept up to date for speed.

    - `cards`: the cards played so far, in order
    - `count`: their total value
    - `last`: the last player to play a card, if any
    - `go`: a bitmask of the players who have said go
    - `closed_by`, `closing_points`: who scored for the last card or 31,
      and how much; set when the sequence is closed
    """

    __slots__ = (
        "events",
        "cards",
        "count",
        "last",
        "go",
        "closed_by",
        "closing_points",
    )

    def __init__(self) -> None:
        self.events: list[SequenceEvent]
        self.cards: list[Card]
        self.count: int
        self.last: int | None
        self.go: int
        self.closed_by: int | None
        self.closing_points: int
        self.reset()

    def reset(self) -> None:
        """Empties the sequence, ready for the next one."""
        self.events = []
        self.cards = []
        self.count = 0
        self.last = None
        self.go = 0
        self.closed_by = None
        self.closing_points = 0

    @classmethod
    def from_events(
        cls,
        events: SequenceType[SequenceEvent],
        closed_by: int | None = None,
        closing_points: int = 0,
    ) -> Sequence:
        seq = cls()
        for event in events:
            seq.events.append(event)
            seq.count = event.count
            if isinstance(event, Play):
                seq.cards.append(event.card)
                seq.last = event.player
            else:
                seq.go |= 1 << event.player
        seq.closed_by = closed_by
        seq.closing_points = closing_points
        return seq

    def copy(self) -> Sequence:
        other = Sequence.__new__(Sequence)
        other.events = self.events[:]
        other.cards = self.cards[:]
        other.count = self.count
        other.last = self.last
        other.go = self.go
        other.closed_by = self.closed_by
        other.closing_points = self.closing_points
        return other

    @property
    def closed(self) -> bool:
        return self.closed_by is not None

    def said_go(self, player: int) -> bool:
        return bool(self.go >> player & 1)

    def flipped(self) -> Sequence:
        """A copy with the two players' places swapped."""
        events = [replace(event, player=1 - event.player) for event in self.events]
        closed_by = None if self.closed_by is None else 1 - self.closed_by
        return Sequence.from_events(events, closed_by, self.closing_points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.events],
            "closed_by": self.closed_by,
            "closing_points": self.closing_points,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sequence:
        events = [SequenceEvent.from_dict(e) for e in data["events"]]
        return cls.from_events(events, data["closed_by"], data["closing_points"])


@dataclass(frozen=True)
class PegView:
    """
    What a player sees when asked to play a card, from their own seat:
    they are player 0 in `sequences` and their opponent is player 1.
    The sequences are shared with the engine's own record, so they must
    not be changed.

    - `hand`: the cards the player holds
    - `legal`: those of `hand` that may be played now
    - `sequences`: the sequences so far, public to both players; the last
      one is open
    - `discards`: the two cards the player threw into the crib
    - `starter`: the cut card
    - `is_dealer`: whether the player is the dealer (the crib is theirs)
    - `score`, `opp_score`: the game scores of the player and their opponent
    - `opp_hand`: the opponent's hand; set only by a harness, for an oracle
    """

    hand: tuple[Card, ...]
    legal: tuple[Card, ...]
    sequences: tuple[Sequence, ...]
    discards: tuple[Card, ...]
    starter: Card
    is_dealer: bool
    score: int
    opp_score: int
    opp_hand: tuple[Card, ...] | None = None

    @property
    def sequence(self) -> Sequence:
        """The current sequence."""
        return self.sequences[-1]

    @property
    def seq(self) -> list[Card]:
        """The cards played in the current sequence."""
        return self.sequences[-1].cards

    @property
    def count(self) -> int:
        """The count of the current sequence."""
        return self.sequences[-1].count

    @property
    def opp_said_go(self) -> bool:
        """Whether the opponent has said go in the current sequence."""
        return self.sequences[-1].said_go(1)

    @cached_property
    def played(self) -> frozenset[Card]:
        """The cards played so far in the round (the starter is not one)."""
        return frozenset(card for seq in self.sequences for card in seq.cards)

    @cached_property
    def opp_cards_left(self) -> int:
        """How many cards the opponent still holds (each keeps four)."""
        plays = sum(
            1
            for seq in self.sequences
            for event in seq.events
            if isinstance(event, Play) and event.player == 1
        )
        return 4 - plays

    def unseen(self) -> list[Card]:
        """Cards that may be in the opponent's hand or in the crib."""
        return unseen_cards(self.played, self.hand, self.discards, (self.starter,))

    @cached_property
    def opp_plays(self) -> list[tuple[tuple[Card, ...], Card]]:
        """(the sequence so far, the card) for each of the opponent's plays."""
        plays = []
        for seq in self.sequences:
            for i, event in enumerate(seq.events):
                if isinstance(event, Play) and event.player == 1:
                    before = tuple(
                        e.card for e in seq.events[:i] if isinstance(e, Play)
                    )
                    plays.append((before, event.card))
        return plays


class PegState:
    """
    The pegging phase of one round, as a mutable game state.  A state is
    always either terminal (`done`) or waiting for `turn` to choose a card.

    - `hands`: the cards each player still holds
    - `sequences`: the sequences so far; unless the state is `done`, the
      last one is open, and the others are closed and never change
    - `points`: pegging points scored so far by each player (a cache of
      what the events add up to)
    - `record`: whether to keep the events of each sequence.  Search
      switches this off for its copies, which saves about a third of the
      time; the cards, count and go flags of the sequences are kept
      regardless.
    """

    __slots__ = ("hands", "turn", "sequences", "points", "done", "record")

    def __init__(
        self,
        hands: list[list[Card]],
        turn: int,
        sequences: list[Sequence] | None = None,
        record: bool = True,
    ) -> None:
        self.record = record
        self.hands = hands
        self.turn = turn
        self.sequences = [Sequence()] if sequences is None else sequences
        self.points = [0, 0]
        self.done = False

    @classmethod
    def start(
        cls, hands: SequenceType[SequenceType[Card]], dealer_idx: int
    ) -> PegState:
        """The state at the start of a round's play: the non-dealer leads."""
        state = cls([list(hands[0]), list(hands[1])], 1 - dealer_idx)
        state.advance()
        return state

    @classmethod
    def from_view(cls, view: PegView, opp_hand: SequenceType[Card]) -> PegState:
        """
        The state a player sees in `view`, if the opponent holds `opp_hand`.
        The player to move is player 0.  The state keeps the cards, count
        and go flags of the current sequence, but not its events, and does
        not record further ones.
        """
        current = view.sequences[-1].copy()
        current.events = []
        return cls([list(view.hand), list(opp_hand)], 0, [current], record=False)

    def copy(self) -> PegState:
        other = PegState.__new__(PegState)
        other.hands = [self.hands[0][:], self.hands[1][:]]
        # closed sequences never change, so they are shared
        other.sequences = self.sequences[:-1]
        other.sequences.append(self.sequences[-1].copy())
        other.turn = self.turn
        other.points = self.points[:]
        other.done = self.done
        other.record = self.record
        return other

    @property
    def seq(self) -> list[Card]:
        """The cards played in the current sequence."""
        return self.sequences[-1].cards

    @property
    def count(self) -> int:
        """The count of the current sequence."""
        return self.sequences[-1].count

    @property
    def last(self) -> int | None:
        """The last player to play a card in the current sequence."""
        return self.sequences[-1].last

    @property
    def is_go(self) -> bool:
        """Whether the opponent of the player to move has said go."""
        return bool(self.sequences[-1].go >> (1 - self.turn) & 1)

    @property
    def events(self) -> list[SequenceEvent]:
        """Every event of the play, in order."""
        return [event for seq in self.sequences for event in seq.events]

    def history(self) -> list[dict[str, Any]]:
        """The sequences in a form that can be JSON encoded."""
        return [seq.to_dict() for seq in self.sequences]

    def view(
        self,
        seat: int,
        discards: SequenceType[Card],
        starter: Card,
        is_dealer: bool,
        score: int,
        opp_score: int,
    ) -> PegView:
        """
        What the player in `seat` sees now.  The state must be recording.
        The sequences are relabelled so that `seat` is player 0.
        """
        assert self.record
        hand = self.hands[seat]
        room = 31 - self.sequences[-1].count
        legal = tuple(card for card in hand if WORTH[card] <= room)
        closed, current = self.sequences[:-1], self.sequences[-1]
        if seat == 0:
            sequences = (*closed, current.copy())
        else:
            sequences = (*(s.flipped() for s in closed), current.flipped())
        return PegView(
            tuple(hand),
            legal,
            sequences,
            tuple(discards),
            starter,
            is_dealer,
            score,
            opp_score,
        )

    def margin(self) -> int:
        """Points scored so far by player 0, less those of player 1."""
        return self.points[0] - self.points[1]

    def legal(self) -> list[int]:
        """Indices into the hand of `turn` of the cards that may be played."""
        room = 31 - self.sequences[-1].count
        return [i for i, c in enumerate(self.hands[self.turn]) if WORTH[c] <= room]

    def advance(self) -> tuple[int, int] | None:
        """
        Resolves go calls and the ends of sequences, up to the next choice.
        Returns (player, points) if a sequence was closed, for the last
        card or for 31.
        """
        closed = None
        while True:
            seq = self.sequences[-1]
            turn = self.turn
            room = 31 - seq.count
            for card in self.hands[turn]:
                if WORTH[card] <= room:
                    can_play = True
                    break
            else:
                can_play = False
            if seq.count == 31 or (not can_play and seq.go >> (1 - turn) & 1):
                # the sequence is over: the last player scores 2 for 31, else 1
                assert seq.last is not None
                points = 2 if seq.count == 31 else 1
                self.points[seq.last] += points
                seq.closed_by = seq.last
                seq.closing_points = points
                closed = (seq.last, points)
                if not self.hands[0] and not self.hands[1]:
                    self.done = True
                    return closed
                self.turn = 1 - seq.last
                if self.record:
                    self.sequences.append(Sequence())
                else:
                    # no history is kept, so the sequence is reused
                    seq.reset()
            elif not can_play:
                # "go": the other player continues
                if self.record:
                    seq.events.append(Go(turn, seq.count, len(self.hands[turn])))
                seq.go |= 1 << turn
                self.turn = 1 - turn
            else:
                return closed

    def place(self, idx: int) -> int:
        """
        The player to move plays `hand[idx]` (which must be legal), without
        resolving what follows.  Returns the points the play scores.
        """
        turn = self.turn
        seq = self.sequences[-1]
        card = self.hands[turn].pop(idx)
        seq.cards.append(card)
        seq.count += WORTH[card]
        seq.last = turn
        points = score_play(seq.cards)
        if self.record:
            seq.events.append(Play(turn, seq.count, card, points))
        self.points[turn] += points
        # players take turns, except once the other has said go or at 31
        if seq.count != 31 and not seq.go >> (1 - turn) & 1:
            self.turn = 1 - turn
        return points

    def play(self, idx: int) -> None:
        """The player to move plays `hand[idx]` (which must be legal)."""
        self.place(idx)
        self.advance()

    def play_card(self, card: Card) -> None:
        """The player to move plays `card` (which must be legal)."""
        self.play(self.hands[self.turn].index(card))
