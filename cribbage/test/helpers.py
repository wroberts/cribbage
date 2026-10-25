"""
helpers.py

Builders for the tests: views and rounds from a few cards.
"""

from collections.abc import Sequence

from cribbage.cards import WORTH, Card
from cribbage.game import Game
from cribbage.pegging import Go, PegView, Play
from cribbage.pegging import Sequence as PegSequence
from cribbage.player import CribbagePlayer
from cribbage.round import Deal, Round


def make_view(
    hand: Sequence[Card],
    seq: Sequence[Card] = (),
    opp_said_go: bool = False,
    is_dealer: bool = False,
    discards: Sequence[Card] = (),
    starter: Card | None = None,
) -> PegView:
    """
    A view of the first sequence of a round: the cards in `seq` have been
    played, alternately, the last by the opponent (or, if `opp_said_go`,
    by the player, after which the opponent said go).
    """
    events: list[Play | Go] = []
    count = 0
    last = 0 if opp_said_go else 1
    for i, card in enumerate(seq):
        count += WORTH[card]
        player = last if (len(seq) - 1 - i) % 2 == 0 else 1 - last
        events.append(Play(player, count, card, 0))
    if opp_said_go:
        events.append(Go(1, count, 4))
    if starter is None:
        used = {*hand, *seq, *discards}
        starter = next(c for c in range(51, -1, -1) if c not in used)
    room = 31 - count
    return PegView(
        tuple(hand),
        tuple(c for c in hand if WORTH[c] <= room),
        (PegSequence.from_events(events),),
        tuple(discards),
        starter,
        is_dealer,
        0,
        0,
    )


def make_round(
    players: Sequence[CribbagePlayer],
    kept: Sequence[Sequence[Card]],
    starter: Card,
    dealer: int,
    game: Game | None = None,
) -> Round:
    """
    A round for the two players with the given kept hands.  The discards
    are other cards, and the game has no target score.
    """
    spare = [c for c in range(52) if c not in {*kept[0], *kept[1], starter}]
    if game is None:
        game = Game(list(players))
        game.target_score = 10**9
    return Round.from_deal(
        game,
        list(players),
        Deal.from_kept(dealer, kept, [spare[:2], spare[2:4]], starter),
    )


def random_deal(rng):
    """Two four-card hands and a starter, all different."""
    cards = rng.sample(range(52), 9)
    return [cards[:4], cards[4:8]], cards[8]
