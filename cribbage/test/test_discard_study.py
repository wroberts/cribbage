#!/usr/bin/env python

"""
test_discard_study.py

Test the discard_study module.
"""

import json
from typing import Any

import numpy as np
import torch
from cribbage.discard_model import (
    BASE_CONFIG,
    TrainConfig,
    deal_hands,
    encode_hands,
    exact_labels,
    generate_dataset,
    load_model,
    predict_values,
    save_model,
    to_tensors,
    train_model,
)
from cribbage.discard_study import ablation_report, ablation_variants, k_study


def test_k_study_smoke():
    hands = deal_hands(3, np.random.default_rng(0))
    labels = [exact_labels(list(hand)) for hand in hands]
    results = k_study(hands, labels, ks=[1, 5], reps=2, seed=0)
    assert set(results) == {"H only", "random", "K=1", "K=5"}
    for row in results.values():
        assert row["dealer_regret"] >= 0 and row["nondealer_regret"] >= 0


def test_ablation_options_train_and_round_trip(tmp_path):
    train = to_tensors(generate_dataset("train", num=300, k=5, seed=1, workers=1))
    val = to_tensors(generate_dataset("val", num=100, k=5, seed=1, workers=1))
    variants: list[dict[str, Any]] = [
        dict(init="he"),
        dict(schedule="constant"),
        dict(standardize_inputs=True),
        dict(standardize_targets=True),
        dict(standardize_inputs=True, standardize_targets=True, two_head=False),
        dict(drop=("nobs", "flush", "suited")),
    ]
    for i, extra in enumerate(variants):
        config = TrainConfig(width=8, depth=2, epochs=1, batch_size=64, **extra)
        model, _ = train_model(config, train, val, False)
        path = str(tmp_path / f"model{i}.pt")
        save_model(model, config, path)
        features = encode_hands(val[0][:5])
        with torch.no_grad():
            assert torch.allclose(
                predict_values(model, features, False),
                predict_values(load_model(path), features, False),
                atol=1e-6,
            ), extra


def test_ablation_variants_are_well_formed():
    variants = ablation_variants()
    labels = [label for _, label, _ in variants]
    assert len(set(labels)) == len(labels) and labels[0] == "base"
    fields = set(vars(BASE_CONFIG)) | {"k", "hands"}
    for _, _, overrides in variants:
        assert set(overrides) <= fields


def test_ablation_report(tmp_path):
    def record(label, group, seed, mse, width=256):
        test = dict(
            mse=mse,
            dealer_regret=0.01,
            nondealer_regret=0.03,
            dealer_optimal=0.9,
            nondealer_optimal=0.8,
        )
        return dict(
            label=label,
            group=group,
            seed=seed,
            params=1000,
            seconds=2.0,
            test=test,
            config=dict(width=width, depth=3),
        )

    rows = [record("base", "base", s, 0.04) for s in range(2)]
    rows += [record("32x3", "size", s, 0.09, width=32) for s in range(2)]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    report = ablation_report(str(path))
    assert "### size" in report and "| 32x3 |" in report
    # rmse goes from 0.2 to 0.3
    assert "+0.1000" in report and "0.2000 ± 0.0000" in report
    assert "size grid" in report
