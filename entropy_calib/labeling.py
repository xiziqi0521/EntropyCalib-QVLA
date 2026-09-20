"""Entropy -> precision/redundant binary labeling.

Adapted from DemoSpeedup's `hdbscan_with_custom_merge`
(aloha/act/imitate_episodes.py, github.com/lingxiao-guo/DemoSpeedup). The
clustering logic is unchanged; file I/O and plotting side effects (the
original wrote PNGs to a checkpoint dir keyed by rollout id) were stripped
since this is used as a library function here, not a script step.

Label convention (kept identical to the source): 0 = precision-critical
(low predictive entropy -> the policy is confident, quantization error here
matters most), 1 = redundant / compressible (high entropy or noise).
"""

from __future__ import annotations

import numpy as np


def label_precision_segments(entropy_trace: np.ndarray, min_cluster_size: int = 5, warmup_steps: int = 50) -> np.ndarray:
    """Cluster a per-timestep entropy trace into precision (0) / redundant (1) labels.

    Args:
        entropy_trace: 1D array of per-timestep entropy values, length T.
        min_cluster_size: HDBSCAN's `min_cluster_size`.
        warmup_steps: number of leading timesteps forced to "noise" before
            clustering, matching the source's `initial_labels[:50] = -1`
            (avoids labeling the policy's warmup/settling period as precision).

    Returns:
        1D int array of length T, values in {0, 1}.
    """
    import hdbscan

    entropy_trace = np.asarray(entropy_trace, dtype=np.float64)
    if entropy_trace.ndim != 1:
        raise ValueError(f"entropy_trace must be 1D, got shape {entropy_trace.shape}")

    t = len(entropy_trace)
    std = np.std(entropy_trace)
    mean = np.mean(entropy_trace)
    if std == 0:
        # Degenerate: no variation at all, nothing is distinguishably
        # "precision-critical" -> treat everything as redundant.
        return np.ones(t, dtype=np.int64)

    entropy_z = (entropy_trace - mean) / std
    indices = np.arange(t)
    indices_z = (indices - np.mean(indices)) / (np.std(indices) if np.std(indices) > 0 else 1.0)
    data = np.stack((indices_z, entropy_z), axis=-1)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(data)
    initial_labels = clusterer.labels_.copy()

    warmup = min(warmup_steps, t)
    initial_labels[:warmup] = -1

    unique_labels = np.unique(initial_labels[initial_labels >= 0])
    refined_labels = np.full_like(initial_labels, -1)
    for label in unique_labels:
        cluster_points = data[initial_labels == label]
        if np.mean(cluster_points[:, 1] < 0):
            refined_labels[initial_labels == label] = 0
        else:
            refined_labels[initial_labels == label] = -1

    return np.abs(refined_labels).astype(np.int64)
