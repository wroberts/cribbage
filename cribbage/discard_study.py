#!/usr/bin/env python

"""
discard_study.py

Studies of the discard model's inputs and training, built on
`discard_model`:

- `k-study` measures how many Monte Carlo samples K are needed for a good
  crib estimate.  The measure is regret: the exact expected points lost by
  choosing the discard that the K-sample estimate prefers, relative to the
  exact best discard.
- `ablate` and `ablate-report` run and report an ablation study of the
  network and training choices: size, heads, learning rate and schedule,
  initialisation, standardisation, features, label noise and amount of data.

Run `python -m cribbage.discard_study --help` for the commands.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import multiprocessing
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from cribbage.cards import Card
from cribbage.discard_model import (
    BASE_CONFIG,
    NUM_KEEPS,
    Dataset,
    ExactLabels,
    OpponentThrowModel,
    Tensors,
    UniformThrows,
    candidate_values,
    crib_values_mc,
    deal_hands,
    evaluate,
    exact_labels,
    to_tensors,
    train_model,
)

# ------------------------------------------------------------
#  How many Monte Carlo samples do we need?


def argmax_random_tiebreak(values: np.ndarray, rng: np.random.Generator) -> int:
    """argmax, breaking ties at random (matters when K is small)."""
    return int(rng.choice(np.flatnonzero(values >= values.max() - 1e-9)))


def _exact_labels_task(hand: Sequence[Card]) -> ExactLabels:
    return exact_labels(hand)


def compute_exact_labels(
    hands: np.ndarray, workers: int, cache: str | None = None
) -> list[ExactLabels]:
    """
    Exact labels for each hand, in parallel.  If `cache` names an .npz
    file for the same hands, labels are read from it; otherwise they are
    computed and written to it.
    """
    if cache is not None and os.path.exists(cache):
        data = np.load(cache)
        if np.array_equal(data["hands"], hands):
            return [
                ExactLabels(h, c, s)
                for h, c, s in zip(data["h"], data["c"], data["c_std"])
            ]
    with multiprocessing.Pool(workers) as pool:
        labels = pool.map(_exact_labels_task, [list(h) for h in hands], chunksize=1)
    if cache is not None:
        np.savez(
            cache,
            hands=hands,
            h=np.array([x.h for x in labels]),
            c=np.array([x.c for x in labels]),
            c_std=np.array([x.c_std for x in labels]),
        )
    return labels


def k_study(
    hands: np.ndarray,
    labels: list[ExactLabels],
    ks: Sequence[int],
    reps: int,
    seed: int,
    model: OpponentThrowModel = UniformThrows(),
) -> dict[str, dict[str, float]]:
    """
    For each K, and for both seats, measures how well the discard chosen
    by a K-sample crib estimate does against the exact best discard.

    - `crib_rmse`: root mean squared error of the C estimate
    - `optimal`: fraction of choices whose exact value ties the best
    - `regret`: mean exact expected points lost against the best discard
    - `regret_se`: standard error of that mean, across hands

    Also reports two baselines, "H only" (ignore the crib entirely) and
    "random".  Each hand is estimated `reps` times at each K.
    """
    rng = np.random.default_rng([seed, 1])
    results: dict[str, dict[str, float]] = {}

    # regrets[name][is_dealer] -> per-hand mean regret; optimal likewise
    regrets: dict[str, dict[bool, list[float]]] = {}
    optimal: dict[str, dict[bool, list[float]]] = {}
    rmse_sq: dict[str, list[float]] = {}

    def record(
        name: str, is_dealer: bool, h: np.ndarray, c: np.ndarray, idxs: list[int]
    ) -> None:
        """Scores the chosen split indices against the exact values."""
        true = candidate_values(h, c, is_dealer)
        best = true.max()
        regrets.setdefault(name, {}).setdefault(is_dealer, []).append(
            float(np.mean([best - true[i] for i in idxs]))
        )
        optimal.setdefault(name, {}).setdefault(is_dealer, []).append(
            float(np.mean([true[i] >= best - 1e-9 for i in idxs]))
        )

    for hand, exact in zip(hands, labels):
        h, c = exact.h, exact.c
        for is_dealer in (True, False):
            record(
                "H only",
                is_dealer,
                h,
                c,
                [argmax_random_tiebreak(h, rng) for _ in range(reps)],
            )
            record(
                "random",
                is_dealer,
                h,
                c,
                [int(rng.integers(NUM_KEEPS)) for _ in range(reps)],
            )
        for k in ks:
            name = f"K={k}"
            estimates = [crib_values_mc(list(hand), k, rng, model) for _ in range(reps)]
            rmse_sq.setdefault(name, []).append(
                float(np.mean([np.mean((e - c) ** 2) for e in estimates]))
            )
            for is_dealer in (True, False):
                record(
                    name,
                    is_dealer,
                    h,
                    c,
                    [
                        argmax_random_tiebreak(candidate_values(h, e, is_dealer), rng)
                        for e in estimates
                    ],
                )

    for name in regrets:
        row: dict[str, float] = {}
        if name in rmse_sq:
            row["crib_rmse"] = float(np.sqrt(np.mean(rmse_sq[name])))
        for is_dealer, label in ((True, "dealer"), (False, "nondealer")):
            r = np.array(regrets[name][is_dealer])
            row[f"{label}_regret"] = float(r.mean())
            row[f"{label}_regret_se"] = float(r.std(ddof=1) / np.sqrt(len(r)))
            row[f"{label}_optimal"] = float(np.mean(optimal[name][is_dealer]))
        results[name] = row
    return results


def print_k_study(
    labels: list[ExactLabels], results: dict[str, dict[str, float]]
) -> None:
    """Prints the label spread and the K-study results as a table."""
    h = np.array([x.h for x in labels])
    c = np.array([x.c for x in labels])
    c_std = np.array([x.c_std for x in labels])
    print(f"hands: {len(labels)}")
    print(
        "mean spread across the 15 splits (max - min):"
        f" H {np.mean(h.max(axis=1) - h.min(axis=1)):.2f},"
        f" C {np.mean(c.max(axis=1) - c.min(axis=1)):.2f}"
    )
    print(f"mean std of a single crib score: {c_std.mean():.2f}")
    print()
    print(
        f"{'':9} {'crib':>6} | {'dealer':^24} | {'non-dealer':^24}\n"
        f"{'':9} {'rmse':>6} | {'regret':>10} {'se':>6} {'opt%':>6} |"
        f" {'regret':>10} {'se':>6} {'opt%':>6}"
    )
    for name, r in results.items():
        rmse = f"{r['crib_rmse']:6.3f}" if "crib_rmse" in r else f"{'':6}"
        print(
            f"{name:9} {rmse} | "
            f"{r['dealer_regret']:10.4f} {r['dealer_regret_se']:6.4f}"
            f" {100 * r['dealer_optimal']:6.1f} | "
            f"{r['nondealer_regret']:10.4f} {r['nondealer_regret_se']:6.4f}"
            f" {100 * r['nondealer_optimal']:6.1f}"
        )


# ------------------------------------------------------------
#  Ablation study

# Each variant changes one thing about the base run.  Keys of the
# overrides are TrainConfig fields, plus `k` (Monte Carlo samples behind
# the training labels) and `hands` (number of training hands).
BASE_DATA = {"k": 20, "hands": 100000}


def ablation_variants() -> list[tuple[str, str, dict[str, Any]]]:
    """The (group, label, overrides) of every run in the study."""
    variants: list[tuple[str, str, dict[str, Any]]] = [("base", "base", {})]
    for depth in (1, 2, 3, 4, 5):
        for width in (8, 16, 32, 64, 128, 256, 512):
            if (depth, width) != (BASE_CONFIG.depth, BASE_CONFIG.width):
                variants.append(
                    ("size", f"{depth}x{width}", {"depth": depth, "width": width})
                )
    variants.append(("heads", "scalar head", {"two_head": False}))
    for lr in (3e-4, 1e-3, 5e-3, 1e-2):
        variants.append(("learning rate", f"lr {lr:g}", {"lr": lr}))
    for lr in (2e-3, 5e-4):
        variants.append(
            ("schedule", f"constant lr {lr:g}", {"schedule": "constant", "lr": lr})
        )
    variants.append(("init", "He init", {"init": "he"}))
    variants += [
        ("scaling", "standardise inputs", {"standardize_inputs": True}),
        ("scaling", "standardise targets", {"standardize_targets": True}),
        (
            "scaling",
            "standardise both",
            {"standardize_inputs": True, "standardize_targets": True},
        ),
    ]
    variants += [
        ("features", "no nobs", {"drop": ("nobs",)}),
        ("features", "no keep-flush", {"drop": ("flush",)}),
        ("features", "no discard-suited", {"drop": ("suited",)}),
        ("features", "ranks only", {"drop": ("nobs", "flush", "suited")}),
    ]
    for k in (1, 5, 100):
        variants.append(("label noise", f"K={k}", {"k": k}))
    for hands in (3000, 10000, 30000, 300000, 1000000):
        variants.append(("data", f"{hands} hands", {"hands": hands}))
    return variants


@functools.lru_cache(maxsize=2)
def _load_tensors(path: str) -> Tensors:
    return to_tensors(Dataset.load(path))


def _ablation_task(task: tuple[str, str, dict[str, Any], int, int]) -> dict[str, Any]:
    """Trains and tests one (variant, seed); returns its record."""
    group, label, overrides, seed, steps = task
    fields = {k: v for k, v in overrides.items() if k not in ("k", "hands")}
    data = {**BASE_DATA, **{k: v for k, v in overrides.items() if k in BASE_DATA}}
    hands, h, c = _load_tensors(f"data/discard_train_k{data['k']}.npz")
    n = data["hands"]
    train = (hands[:n], h[:n], c[:n])
    steps_per_epoch = math.ceil(n / BASE_CONFIG.batch_size)
    config = dataclasses.replace(
        BASE_CONFIG, seed=seed, epochs=math.ceil(steps / steps_per_epoch), **fields
    )
    start = time.time()
    model, history = train_model(
        config, train, _load_tensors("data/discard_val_exact.npz"), verbose=False
    )
    seconds = time.time() - start
    best = min(history, key=lambda epoch: epoch["mse"])
    return {
        "group": group,
        "label": label,
        "seed": seed,
        "steps": steps,
        "config": vars(config),
        "data": data,
        "params": sum(p.numel() for p in model.parameters()),
        "best_epoch": int(best["epoch"]),
        "epochs": config.epochs,
        "seconds": seconds,
        "test": evaluate(model, _load_tensors("data/discard_test_exact.npz")),
    }


def _init_ablation_worker(threads: int) -> None:
    torch.set_num_threads(threads)


def run_ablation(
    path: str,
    steps: int,
    seeds: int,
    procs: int,
    threads: int,
    groups: Sequence[str] | None = None,
) -> None:
    """
    Runs every variant (of the chosen groups, plus the base) for each
    seed, appending one JSON record per run to `path`.  Runs that are
    already in the file are skipped, so an interrupted study can be
    resumed.
    """
    done = set()
    if os.path.exists(path):
        with open(path) as results_file:
            for line in results_file:
                record = json.loads(line)
                done.add((record["label"], record["seed"], record["steps"]))
    tasks = [
        (group, label, overrides, seed, steps)
        for group, label, overrides in ablation_variants()
        if groups is None or group in groups or group == "base"
        for seed in range(seeds)
        if (label, seed, steps) not in done
    ]
    print(f"{len(tasks)} runs to do ({len(done)} already done)", flush=True)
    start = time.time()
    with (
        multiprocessing.Pool(procs, _init_ablation_worker, (threads,)) as pool,
        open(path, "a") as results_file,
    ):
        for i, record in enumerate(pool.imap_unordered(_ablation_task, tasks), 1):
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()
            print(
                f"[{i}/{len(tasks)} {time.time() - start:.0f}s] {record['label']}"
                f" seed {record['seed']}: test rmse"
                f" {math.sqrt(record['test']['mse']):.4f}",
                flush=True,
            )


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    return mean, math.sqrt(var)


def ablation_report(path: str) -> str:
    """
    Summarises a results file as markdown tables, one per group.  For
    each variant, over the seeds: test RMSE in points (the square root of
    the mean of the two seats' MSE), regret (points per deal, averaged
    over the seats), percentage of optimal choices, and training time.
    `change` is the paired change in RMSE from the base run, by seed.
    """
    runs: dict[str, list[dict[str, Any]]] = {}
    groups: dict[str, str] = {}
    with open(path) as results_file:
        for line in results_file:
            record = json.loads(line)
            runs.setdefault(record["label"], []).append(record)
            groups[record["label"]] = record["group"]

    def rmse(record: dict[str, Any]) -> float:
        return math.sqrt(record["test"]["mse"])

    base_rmse = {r["seed"]: rmse(r) for r in runs.get("base", [])}

    def row(label: str) -> str:
        records = runs[label]
        mean, sd = _mean_sd([rmse(r) for r in records])
        regret = sum(
            (r["test"]["dealer_regret"] + r["test"]["nondealer_regret"]) / 2
            for r in records
        ) / len(records)
        optimal = sum(
            (r["test"]["dealer_optimal"] + r["test"]["nondealer_optimal"]) / 2
            for r in records
        ) / len(records)
        seconds = sum(r["seconds"] for r in records) / len(records)
        paired = [
            rmse(r) - base_rmse[r["seed"]] for r in records if r["seed"] in base_rmse
        ]
        change = (
            f"{sum(paired) / len(paired):+.4f}" if paired and label != "base" else ""
        )
        return (
            f"| {label} | {records[0]['params']:,} | {mean:.4f} ± {sd:.4f} |"
            f" {change} | {regret:.4f} | {100 * optimal:.1f} | {seconds:.0f} |"
        )

    header = (
        "| variant | params | test RMSE (pts) | change | regret | optimal % | train s |\n"
        "|---|---|---|---|---|---|---|"
    )
    lines = []
    for group in dict.fromkeys(g for g in groups.values() if g != "base"):
        lines += [f"### {group}", "", header]
        lines += [row("base")] if "base" in runs else []
        lines += [row(label) for label in runs if groups[label] == group]
        lines.append("")
    # the size study as a grid of RMSE
    size_records = [
        r for label in runs for r in runs[label] if r["group"] in ("size", "base")
    ]
    if size_records:
        widths = sorted({r["config"]["width"] for r in size_records})
        lines += [
            "### size grid: test RMSE (pts), rows are depth, columns are width",
            "",
        ]
        lines += ["| depth | " + " | ".join(map(str, widths)) + " |"]
        lines += ["|---" * (len(widths) + 1) + "|"]
        for depth in sorted({r["config"]["depth"] for r in size_records}):
            cells = []
            for width in widths:
                cell = [
                    rmse(r)
                    for r in size_records
                    if r["config"]["depth"] == depth and r["config"]["width"] == width
                ]
                cells.append(f"{_mean_sd(cell)[0]:.4f}" if cell else "")
            lines.append(f"| {depth} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------
#  Command line


def run_k_study(args: argparse.Namespace) -> None:
    """Runs the Monte Carlo sample-size study."""
    hands = deal_hands(args.hands, np.random.default_rng([args.seed, 0]))
    labels = compute_exact_labels(hands, args.workers, args.cache)
    ks = [int(k) for k in args.ks.split(",")]
    print_k_study(labels, k_study(hands, labels, ks, args.reps, args.seed))


def run_ablate(args: argparse.Namespace) -> None:
    """Runs the ablation study, then prints its report."""
    groups = args.groups.split(",") if args.groups else None
    run_ablation(args.out, args.steps, args.seeds, args.procs, args.threads, groups)
    print(ablation_report(args.out))


def run_ablate_report(args: argparse.Namespace) -> None:
    """Prints the report of an ablation results file."""
    print(ablation_report(args.results))


def main() -> None:
    """Studies of the discard model."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    study = commands.add_parser("k-study", help=run_k_study.__doc__)
    study.add_argument("--hands", type=int, default=300)
    study.add_argument("--ks", default="1,2,5,10,20,50,100,200,500")
    study.add_argument("--reps", type=int, default=5)
    study.add_argument("--cache", help=".npz file for the exact labels")
    study.set_defaults(func=run_k_study)

    ablate = commands.add_parser("ablate", help=run_ablate.__doc__)
    ablate.add_argument("--out", default="data/ablation.jsonl")
    ablate.add_argument(
        "--steps", type=int, default=1500, help="gradient steps per run"
    )
    ablate.add_argument("--seeds", type=int, default=3)
    ablate.add_argument("--procs", type=int, default=4, help="runs in parallel")
    ablate.add_argument("--threads", type=int, default=2, help="threads per run")
    ablate.add_argument("--groups", help="comma-separated groups (default: all)")
    ablate.set_defaults(func=run_ablate)

    report = commands.add_parser("ablate-report", help=run_ablate_report.__doc__)
    report.add_argument("--results", default="data/ablation.jsonl")
    report.set_defaults(func=run_ablate_report)

    study.add_argument("--seed", type=int, default=0)
    study.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
