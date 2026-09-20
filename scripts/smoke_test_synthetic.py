#!/usr/bin/env python
"""Synthetic smoke test for entropy_calib's pure-logic pieces (no GR00T
checkpoint, no LIBERO, no GPU needed) -- entropy estimation, labeling, and
sample selection, exercised on hand-built data with a known structure.

Run: python scripts/smoke_test_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entropy_calib import KDE, entropy_guided_sample_indices, label_precision_segments


def test_entropy_utils():
    torch.manual_seed(0)
    # Two step "families": tight cluster of samples (low entropy / precise)
    # vs. spread-out samples (high entropy / imprecise).
    tight = torch.randn(1, 20, 4) * 0.01
    wide = torch.randn(1, 20, 4) * 5.0

    kde = KDE()
    e_tight = kde.kde_entropy(tight).item()
    e_wide = kde.kde_entropy(wide).item()
    print(f"[entropy_utils] tight-cluster entropy={e_tight:.4f} wide-cluster entropy={e_wide:.4f}")
    assert e_wide > e_tight, "wide (imprecise) cluster should have higher entropy than tight one"
    print("[entropy_utils] OK")


def test_labeling_and_selection():
    rng = np.random.default_rng(0)
    t = 300
    # Build an entropy trace with two clear low-entropy ("precision") bands
    # and everything else high-entropy ("redundant"), so we know what the
    # correct label split should roughly look like.
    entropy = rng.normal(loc=5.0, scale=0.5, size=t)
    entropy[60:110] = rng.normal(loc=0.2, scale=0.05, size=50)   # precision band 1
    entropy[180:230] = rng.normal(loc=0.2, scale=0.05, size=50)  # precision band 2

    labels = label_precision_segments(entropy, min_cluster_size=5, warmup_steps=50)
    assert labels.shape == (t,)
    assert set(np.unique(labels)).issubset({0, 1})

    frac_precision_in_bands = np.mean(labels[60:110] == 0) + np.mean(labels[180:230] == 0)
    frac_precision_elsewhere = np.mean(
        np.concatenate([labels[110:180], labels[230:]]) == 0
    )
    print(
        f"[labeling] precision-label rate inside known-low-entropy bands="
        f"{frac_precision_in_bands / 2:.2f}, elsewhere={frac_precision_elsewhere:.2f}"
    )
    assert frac_precision_in_bands / 2 > frac_precision_elsewhere, (
        "labeling should mark the known low-entropy bands as precision far more "
        "often than the rest of the trace"
    )
    print("[labeling] OK")

    candidate_indices = np.arange(t)
    selected = entropy_guided_sample_indices(
        candidate_indices=candidate_indices,
        labels=labels,
        entropy=entropy,
        num_samples=10,
        redundant_fraction=0.2,
        seed=0,
    )
    assert len(selected) == 10
    assert len(set(selected)) == 10  # no duplicates
    selected_labels = labels[np.array(selected)]
    n_precision_selected = int((selected_labels == 0).sum())
    print(f"[sample_selection] selected {len(selected)} indices, {n_precision_selected} precision-critical")
    assert n_precision_selected >= 6, "expected the majority of the budget to land on precision-critical steps"
    print("[sample_selection] OK")


if __name__ == "__main__":
    test_entropy_utils()
    test_labeling_and_selection()
    print("ALL SMOKE TESTS PASSED")
