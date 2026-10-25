"""
specs.py

Build policies and players from short spec strings, so that experiments
can name them on the command line and pass them between processes.

A discard policy spec is one of:

- `random`
- `expected[:N]`: the best expected hand over N random starters (20)
- `model:PATH`: a trained `DiscardNet` (see `discard_model`)

A pegging policy spec is one of:

- `first`, `random`, `greedy`, `greedy2`, `oracle`
- `pimc-roll:N[:R]`, `pimc-solve:N`, `ismcts:I`, each with an optional
  `/opp` suffix, e.g. `pimc-solve:50/opp`
- `net:PATH`, `pimc-net:WORLDS:PATH`: a trained `PegNet` (see
  `pegging_model`)

A player spec is `DISCARD+PEGGING`, e.g. `expected+greedy`, or one of the
shorthands `random` (`random+random`), `simple` (`expected+greedy`),
`simple-discard` (`expected+random`) and `model:PATH` (`model:PATH+random`).
"""

from __future__ import annotations

from cribbage.discard_policy import (
    DiscardPolicy,
    ExpectedHandDiscardPolicy,
    RandomDiscardPolicy,
)
from cribbage.pegging_policy import (
    FirstPeggingPolicy,
    Greedy1PeggingPolicy,
    Greedy2PeggingPolicy,
    PeggingPolicy,
    RandomPeggingPolicy,
)
from cribbage.pegging_search import (
    IsmctsPeggingPolicy,
    OraclePeggingPolicy,
    PimcRolloutPeggingPolicy,
    PimcSolvePeggingPolicy,
)
from cribbage.player import PolicyPlayer

PLAYER_SHORTHANDS = {
    "random": "random+random",
    "simple": "expected+greedy",
    "simple-discard": "expected+random",
}


def make_discard_policy(spec: str) -> DiscardPolicy:
    """Builds a discard policy from its name, e.g. `random` or `expected:50`."""
    if spec.startswith("model:"):
        # imported here, as it needs torch
        from cribbage.discard_model import NetDiscardPolicy

        return NetDiscardPolicy.load(spec[len("model:") :])
    name, *args = spec.split(":")
    nums = [int(a) for a in args]
    if name == "random" and not nums:
        return RandomDiscardPolicy()
    if name == "expected" and len(nums) <= 1:
        return ExpectedHandDiscardPolicy(*nums)
    raise ValueError(f"unknown discard policy {spec!r}")


def make_pegging_policy(spec: str) -> PeggingPolicy:
    """
    Builds a pegging policy from its name, e.g. `greedy`, `pimc-roll:20:2`,
    `pimc-solve:50/opp`, `ismcts:400`, `oracle`.
    """
    if spec.startswith("net:"):
        # a network trained by pegging_model (the path may contain "/")
        from cribbage.pegging_model import NetPeggingPolicy

        return NetPeggingPolicy.load(spec[len("net:") :])
    if spec.startswith("pimc-net:"):
        # pimc-net:WORLDS:PATH, rollouts played by a network
        from cribbage.pegging_model import NetPeggingPolicy, PimcNetPeggingPolicy

        worlds, _, path = spec[len("pimc-net:") :].partition(":")
        return PimcNetPeggingPolicy(int(worlds), NetPeggingPolicy.load(path))
    base, *flags = spec.split("/")
    if set(flags) - {"opp"}:
        raise ValueError(f"unknown flag in {spec!r}")
    opp = "opp" in flags
    name, *args = base.split(":")
    nums = [int(a) for a in args]
    if name in ("first", "random", "greedy", "greedy2", "oracle") and (nums or opp):
        raise ValueError(f"{name} takes no arguments: {spec!r}")
    if name == "first":
        return FirstPeggingPolicy()
    if name == "random":
        return RandomPeggingPolicy()
    if name == "greedy":
        return Greedy1PeggingPolicy()
    if name == "greedy2":
        return Greedy2PeggingPolicy()
    if name == "oracle":
        return OraclePeggingPolicy()
    if name == "pimc-roll" and 1 <= len(nums) <= 2:
        return PimcRolloutPeggingPolicy(
            nums[0], nums[1] if len(nums) > 1 else 1, opp_aware=opp
        )
    if name == "pimc-solve" and len(nums) == 1:
        return PimcSolvePeggingPolicy(nums[0], opp_aware=opp)
    if name == "ismcts" and len(nums) == 1:
        return IsmctsPeggingPolicy(nums[0], opp_aware=opp)
    raise ValueError(f"unknown pegging policy {spec!r}")


def make_player(spec: str) -> PolicyPlayer:
    """A player from `DISCARD+PEGGING` or a shorthand (see the module docstring)."""
    if spec in PLAYER_SHORTHANDS:
        spec = PLAYER_SHORTHANDS[spec]
    elif "+" not in spec and spec.startswith("model:"):
        spec += "+random"
    discard, plus, pegging = spec.partition("+")
    if not plus:
        raise ValueError(f"expected DISCARD+PEGGING: {spec!r}")
    return PolicyPlayer(make_discard_policy(discard), make_pegging_policy(pegging))
