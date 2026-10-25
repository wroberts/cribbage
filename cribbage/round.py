#!/usr/bin/env python

"""
round.py
(c) Will Roberts  25 December, 2016

An object representing a single round in a game of cribbage between
two CribbagePlayers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from cribbage.cards import Card, card_tostring, make_deck, split_card
from cribbage.pegging import Go, PegState, Play
from cribbage.player import CribbagePlayer, DiscardView
from cribbage.scoring import score_hand

if TYPE_CHECKING:
    from cribbage.game import Game


@dataclass(frozen=True)
class Deal:
    """
    The cards of one round, and where they went.  Players are indexed by
    seat, 0 and 1.

    - `dealer`: the seat of the dealer
    - `dealt`: the six cards dealt to each player
    - `discards`: the two cards each player threw into the crib
    - `starter`: the cut card
    """

    dealer: int
    dealt: tuple[tuple[Card, ...], tuple[Card, ...]]
    discards: tuple[tuple[Card, ...], tuple[Card, ...]]
    starter: Card

    @classmethod
    def from_kept(
        cls,
        dealer: int,
        kept: Sequence[Sequence[Card]],
        discards: Sequence[Sequence[Card]],
        starter: Card,
    ) -> Deal:
        """A deal from the cards each player kept and threw (kept ones first)."""
        keep = [tuple(int(c) for c in kept[seat]) for seat in (0, 1)]
        throw = [tuple(int(c) for c in discards[seat]) for seat in (0, 1)]
        return cls(
            dealer,
            (keep[0] + throw[0], keep[1] + throw[1]),
            (throw[0], throw[1]),
            int(starter),
        )

    @property
    def kept(self) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
        """The four cards each player kept, in the order dealt."""
        keep0, keep1 = (
            tuple(c for c in self.dealt[seat] if c not in self.discards[seat])
            for seat in (0, 1)
        )
        return keep0, keep1

    @property
    def crib(self) -> list[Card]:
        """The four cards of the crib."""
        return [*self.discards[0], *self.discards[1]]


class Round:
    """
    A single round of cribbage in a game between two CribbagePlayers.
    """

    def __init__(
        self, game: Game, players: Sequence[CribbagePlayer], dealer_idx: int
    ) -> None:
        """
        Constructor.

        Arguments:
        - `game`:
        - `players`:
        - `dealer_idx`:
        """
        self.game = game
        self.players = list(players)
        self.dealer_idx = dealer_idx
        # the cards of the round, once they are dealt and discarded
        self.cards: Deal | None = None
        # the state of the pegging, once it has started
        self.peg: PegState | None = None

    @classmethod
    def from_deal(
        cls, game: Game, players: Sequence[CribbagePlayer], cards: Deal
    ) -> Round:
        """
        A round with the given cards, already dealt and discarded: `play`
        and `show` may be called next (`nibs` for his nibs).
        """
        rnd = cls(game, players, cards.dealer)
        rnd.cards = cards
        return rnd

    def deal(self, verbose: bool = False) -> bool:
        """
        Shuffles and deals the cards for a single round of cribbage.

            The dealer shuffles and deals six cards to each player.
            Once the cards have been dealt, each player chooses four
            cards to retain, then discards the other two face-down to
            form the "crib", which will be used later by the dealer.
            At this point, each player's hand and the crib will
            contain exactly four cards. The player on the dealer's
            left cuts the deck and the dealer reveals the top card,
            called the "starter" or the "cut". If this card is a Jack,
            the dealer scores two points for "his heels."

        Returns True if the game is not over after the deal; False if
        the game is over (or was already over before the deal).

        Arguments:
        - `verbose`:
        """
        if self.game.over:
            return False
        rng = self.game.rng
        # create a new shuffled deck
        deck = make_deck()
        rng.shuffle(deck)
        # deal out 6 cards to each player
        nondealer_hand = tuple(deck[:12][::2])
        dealer_hand = tuple(deck[:12][1::2])
        deck = deck[12:]
        # self.dealer_idx indicates the player who is dealer
        dealt = (
            (nondealer_hand, dealer_hand)
            if self.dealer_idx
            else (dealer_hand, nondealer_hand)
        )
        if verbose:
            print("Dealing cards")
            self.print_state(dealt)
        # ask players to select cards to discard to crib
        discards = []
        for player_idx, player in enumerate(self.players):
            chosen = tuple(
                player.discard(
                    DiscardView(
                        hand=dealt[player_idx],
                        is_dealer=self.dealer_idx == player_idx,
                        score=self.game.scores[player_idx],
                        opp_score=self.game.scores[int(not player_idx)],
                    )
                )
            )
            # sanity checking
            assert len(set(chosen)) == 2
            assert all(c in dealt[player_idx] for c in chosen)
            if verbose:
                print(
                    f"Player {player_idx + 1} discards:",
                    " ".join(card_tostring(c) for c in sorted(chosen)),
                )
            discards.append(chosen)
        # randomly cut a card from the deck to serve as the "starter"
        self.cards = Deal(
            self.dealer_idx, dealt, (discards[0], discards[1]), rng.choice(deck)
        )
        if verbose:
            self.print_state(self.cards.kept)
        return self.nibs(verbose=verbose)

    def nibs(self, verbose: bool = False) -> bool:
        """
        Reveals the starter; if it is a Jack, the dealer scores two points
        for "his heels".

        Returns True if the game is not over afterwards.
        """
        assert self.cards is not None
        starter_value = split_card(self.cards.starter)[0]
        if verbose:
            print("Starter card is ", card_tostring(self.cards.starter))
        if starter_value == 10:
            if verbose:
                print("Dealer gets two points for his nibs")
            if not self.game.award_points(self.dealer_idx, 2, verbose=verbose):
                return False
        return True

    def play(self, verbose: bool = False) -> bool:
        """
        Plays out cards (pegging) for a single round of cribbage.

            Starting with the non-dealer player, each player in turn
            lays one card face up on the table in front of him or her,
            stating the count--that is, the cumulative value of the
            cards that have been laid (for example, the first player
            lays a five and says "five", the next lays a six and says
            "eleven", and so on)--without the count going above
            31. The cards are not laid in the centre of the table as,
            at the end of the "play," each player needs to pick up the
            cards they have laid. Players score points during this
            process for causing the count to reach exactly fifteen,
            for runs (consecutively played, but not necessarily in
            order) and for pairs. Three or four of a kind are counted
            as multiple pairs: three of a kind is the same as three
            different pairs, or 6 points, and four of a kind is 6
            different kinds of pairs, or 12 points.

            If a player cannot play without causing the count exceed
            31, he calls "Go." Continuing with the player on his left,
            the other player(s) continue(s) the play until no one can
            play without the count exceeding 31. A player is obligated
            to play a card unless there is no card in their hand that
            can be played without the count exceeding 31 (one cannot
            voluntarily pass). Once 31 is reached or no one is able to
            play, the player who played the last card scores one point
            if the count is still under 31 and two if it is exactly
            31. The count is then reset to zero and those players with
            cards remaining repeat the process starting with the
            player to the left of the player who played the last
            card. When no player has any cards the game proceeds to
            the "show."

            Players choose the order in which to lay their cards in
            order to maximize their scores; experienced players refer
            to this as either good or poor "pegging" or
            "pegsmanship". If one player reaches the target (usually
            61 or 121), the game ends immediately and that player
            wins. When the scores are level during a game, the
            players' pegs will be side by side, and it is thought that
            this gave rise to the phrase "level pegging".[5]

        Returns True if the game is not over after the play; False if
        the game is over (or was already over before the play).

        Arguments:
        - `verbose`:
        """
        assert self.cards is not None
        cards = self.cards
        # the non-dealer leads
        peg = self.peg = PegState.start(cards.kept, self.dealer_idx)
        printed = 0

        def report() -> None:
            nonlocal printed
            for event in peg.events[printed:]:
                if isinstance(event, Go):
                    print(f"Player {event.player + 1} says go")
                else:
                    assert isinstance(event, Play)
                    print(
                        f"Player {event.player + 1} plays:",
                        card_tostring(event.card),
                        "  Count:",
                        event.count,
                    )
            printed = len(peg.events)

        while not peg.done:
            seat = peg.turn
            view = peg.view(
                seat,
                cards.discards[seat],
                cards.starter,
                self.dealer_idx == seat,
                self.game.scores[seat],
                self.game.scores[int(not seat)],
            )
            card = self.players[seat].play_card(view)
            # sanity checking
            assert card in view.legal
            idx = peg.hands[seat].index(card)

            # make the move, and score it
            play_points = peg.place(idx)
            if verbose:
                report()
                if play_points:
                    print(f"Player {seat + 1} scores:", play_points)
            if not self.game.award_points(seat, play_points, verbose=verbose):
                return False

            # resolve go calls and the end of the sequence, scoring the
            # last card or 31
            closed = peg.advance()
            if verbose:
                report()
            if closed is not None:
                if verbose:
                    print("Sequence is over")
                if not self.game.award_points(*closed, verbose=verbose):
                    return False

        if verbose:
            print("Round is over")
            self.print_state()
        # notify the players that the round is over
        for player in self.players:
            player.round_over()
        return True

    def show(self, verbose: bool = False) -> bool:
        """
        Scores all hands after a round of cribbage.

            Once the play is complete, each player in turn, starting
            with the player on the left of the dealer, displays his
            hand on the table and scores points based on its content
            in conjunction with the starter card. Points are scored
            for combinations of cards totalling fifteen, runs, pairs
            (multiple pairs are scored pair by pair, but may be
            referred to as three or four of a kind), a flush and
            having a Jack of the same suit as the starter card ("one
            for his nob [or nobs or nibs]"). A four-card flush scores
            four and cannot include the cut or starter; a five-card
            flush scores five.

            The dealer scores his hand last and then turns the cards
            in the crib face up. These cards are then scored by the
            dealer as an additional hand, also in conjunction with the
            starter card. Unlike the dealer's own hand, the crib
            cannot score a four-card flush, but it can score a
            five-card flush with the starter.

            All scores from 0 to 29 are possible, with the exception
            of 19, 25, 26 and 27.[6] Players may refer colloquially to
            a hand scoring zero points as having a score of
            nineteen.[7]

        Returns True if the game is not over after the show; False if
        the game is over (or was already over before the show).

        Arguments:
        - `verbose`:
        """
        assert self.cards is not None
        starter_card = self.cards.starter
        kept_hands = self.cards.kept
        # the non-dealer's hand, then the dealer's hand and the dealer's crib
        nondealer_idx = int(not self.dealer_idx)
        for player_idx, cards, is_crib in (
            (nondealer_idx, kept_hands[nondealer_idx], False),
            (self.dealer_idx, kept_hands[self.dealer_idx], False),
            (self.dealer_idx, self.cards.crib, True),
        ):
            if verbose:
                print(
                    "Scoring crib:" if is_crib else f"Scoring player {player_idx + 1}:",
                    " ".join([card_tostring(c) for c in sorted(cards)]),
                )
                print("Starter card is ", card_tostring(starter_card))
            points = score_hand(list(cards), starter_card, crib=is_crib, verbose=verbose)
            if verbose:
                print(
                    f"Player {player_idx + 1}'s {'crib' if is_crib else 'hand'} scores",
                    points,
                )
            if not self.game.award_points(player_idx, points, verbose=verbose):
                return False
        return True

    def print_state(self, hands: Sequence[Sequence[Card]] | None = None) -> None:
        """
        Print a representation of the current round state to stdout: the
        given hands, or else those still held in the pegging, or else those
        kept.
        """
        if hands is None:
            if self.peg is not None:
                hands = self.peg.hands
            elif self.cards is not None:
                hands = self.cards.kept
        crib = self.cards.crib if self.cards is not None else []
        for idx in range(2):
            print(
                f"Player {idx + 1}{'(D)' if idx == self.dealer_idx else '   '}  ",
                end="",
            )
            if hands:
                print(
                    f"{' '.join([card_tostring(c) for c in sorted(hands[idx])]):17}",
                    end="",
                )
            print(f"  {self.game.scores[idx]:3} Points", end="")
            if crib and idx == self.dealer_idx:
                print(
                    "  Crib",
                    " ".join([card_tostring(c) for c in sorted(crib)]),
                    end="",
                )
            print()
