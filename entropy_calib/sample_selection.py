"""Entropy-guided calibration sample selection.

This is new code (no DemoSpeedup/Omega-QVLA equivalent existed before). It
replaces the role of Omega-QVLA's `choose_sample_indices`
(tools/analyze_layerwise_quant_drift.py:378-397), which picks calibration
timesteps by blind uniform stride/linspace. Here selection is weighted
toward "precision-critical" (label == 0, low predictive entropy) timesteps,
since GPTQ/DuQuant calibration quality depends on which activations the
Gram-matrix statistics are built from, and quantization error on
precision-critical action-chunk predictions is the one that actually hurts
task success.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _rank_by_entropy(
    pool_indices: np.ndarray,
    pool_entropy: Optional[np.ndarray],
    rng: np.random.Generator,
    ascending: bool,
) -> np.ndarray:
    if pool_entropy is None:
        order = rng.permutation(len(pool_indices))
    else:
        order = np.argsort(pool_entropy) if ascending else np.argsort(-pool_entropy)
    return pool_indices[order]


def _select_precision_redundant(
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    entropy: Optional[np.ndarray],
    num_samples: int,
    redundant_fraction: float,
    rng: np.random.Generator,
) -> list[int]:
    """The original (task-agnostic) global top-K-by-entropy selection."""
    precision_mask = labels == 0
    redundant_mask = ~precision_mask

    precision_pool = candidate_indices[precision_mask]
    redundant_pool = candidate_indices[redundant_mask]

    n_redundant_target = int(round(num_samples * redundant_fraction))
    n_precision_target = num_samples - n_redundant_target

    precision_ranked = _rank_by_entropy(
        precision_pool, None if entropy is None else entropy[precision_mask], rng, ascending=True
    )
    redundant_ranked = _rank_by_entropy(
        redundant_pool, None if entropy is None else entropy[redundant_mask], rng, ascending=False
    )

    n_precision = min(n_precision_target, len(precision_ranked))
    n_redundant = min(n_redundant_target, len(redundant_ranked))

    # Backfill budget shortfall from the other pool when one pool is too small.
    shortfall = num_samples - (n_precision + n_redundant)
    if shortfall > 0:
        spare_precision = len(precision_ranked) - n_precision
        spare_redundant = len(redundant_ranked) - n_redundant
        take_from_precision = min(shortfall, spare_precision)
        n_precision += take_from_precision
        shortfall -= take_from_precision
        take_from_redundant = min(shortfall, spare_redundant)
        n_redundant += take_from_redundant

    selected = np.concatenate([precision_ranked[:n_precision], redundant_ranked[:n_redundant]])
    return [int(i) for i in selected]


def entropy_guided_sample_indices(
    candidate_indices: Sequence[int],
    labels: Sequence[int],
    entropy: Optional[Sequence[float]] = None,
    num_samples: int = 10,
    redundant_fraction: float = 0.2,
    seed: int = 0,
    task_ids: Optional[Sequence] = None,
) -> list[int]:
    """Pick up to `num_samples` calibration indices, weighted toward label==0.

    Args:
        candidate_indices: dataset indices (e.g. into a LeRobot dataset's
            flat step list) eligible for calibration.
        labels: same length as `candidate_indices`; 0 = precision-critical,
            1 = redundant (see `entropy_calib.labeling.label_precision_segments`).
        entropy: optional, same length as `candidate_indices`. When given,
            precision-pool candidates are ranked by *lowest* entropy first
            (most confident / most precision-critical) and redundant-pool
            candidates by *highest* entropy first (most clearly redundant).
            When omitted, falls back to seeded random sampling within each pool.
        num_samples: total calibration budget.
        redundant_fraction: fraction of the budget reserved for label==1
            samples, kept for calibration diversity rather than spending the
            whole budget on precision segments alone.
        seed: RNG seed used only when `entropy` is not provided.
        task_ids: optional, same length as `candidate_indices` (e.g. a LIBERO
            task prompt string, or any hashable per-task label). When given,
            selection first RESERVES one slot per unique task -- that task's
            single most precision-critical (lowest-entropy) candidate --
            guaranteeing every task contributes at least one calibration
            sample regardless of how its frames rank globally by entropy.
            The remaining budget (num_samples - num_unique_tasks) is then
            filled by the same precision/redundant global-ranking logic
            among the not-yet-reserved candidates. If there are more unique
            tasks than `num_samples`, only the tasks whose best candidate is
            most precision-critical get a reserved slot (graceful
            degradation toward the task-agnostic behavior as the budget
            shrinks below the number of tasks).

            Without this, plain global top-K-by-entropy selection can (and
            in practice did -- see EntropyCalib-QVLA/MECHANISM_CN.md) skip
            several tasks entirely at typical small calibration budgets,
            since a handful of tasks' frames can dominate the low-entropy
            end of the global ranking, leaving GPTQ with zero activation
            statistics for the skipped tasks' scenes.

    Returns:
        Sorted list of selected dataset indices, length <= num_samples
        (fewer only if the candidate pools are smaller than the budget).
    """
    candidate_indices = np.asarray(candidate_indices)
    labels = np.asarray(labels)
    if candidate_indices.shape != labels.shape:
        raise ValueError(
            f"candidate_indices and labels must have the same shape, "
            f"got {candidate_indices.shape} vs {labels.shape}"
        )
    if entropy is not None:
        entropy = np.asarray(entropy)
        if entropy.shape != candidate_indices.shape:
            raise ValueError(
                f"entropy must match candidate_indices shape, "
                f"got {entropy.shape} vs {candidate_indices.shape}"
            )

    rng = np.random.default_rng(seed)

    if task_ids is None:
        selected = _select_precision_redundant(
            candidate_indices, labels, entropy, num_samples, redundant_fraction, rng
        )
        return sorted(selected)

    task_ids = np.asarray(task_ids, dtype=object)
    if task_ids.shape != candidate_indices.shape:
        raise ValueError(
            f"task_ids must match candidate_indices shape, "
            f"got {task_ids.shape} vs {candidate_indices.shape}"
        )

    # Step 1: for each unique task, find its single most precision-critical
    # (lowest-entropy) candidate -- the one reserved for that task if budget
    # allows. Randomly pick within the task when entropy is unavailable.
    unique_tasks = np.unique(task_ids)
    per_task_best_idx = []
    per_task_best_entropy = []
    for task in unique_tasks:
        task_mask = task_ids == task
        task_pool = candidate_indices[task_mask]
        if entropy is None:
            best = task_pool[rng.integers(len(task_pool))]
            best_entropy = 0.0  # arbitrary; only used for reserved-task priority ordering
        else:
            task_entropy = entropy[task_mask]
            best_local = np.argmin(task_entropy)
            best = task_pool[best_local]
            best_entropy = task_entropy[best_local]
        per_task_best_idx.append(int(best))
        per_task_best_entropy.append(best_entropy)

    # Step 2: if budget < number of tasks, prioritize reserving the tasks
    # whose best candidate is most precision-critical first.
    priority_order = np.argsort(per_task_best_entropy)
    n_reserved = min(num_samples, len(unique_tasks))
    reserved_indices = [per_task_best_idx[i] for i in priority_order[:n_reserved]]

    # Step 3: fill the remaining budget from the leftover pool using the
    # original global precision/redundant ranking.
    remaining_budget = num_samples - len(reserved_indices)
    if remaining_budget <= 0:
        return sorted(reserved_indices)

    reserved_set = set(reserved_indices)
    keep_mask = np.array([int(i) not in reserved_set for i in candidate_indices])
    remaining_selected = _select_precision_redundant(
        candidate_indices[keep_mask],
        labels[keep_mask],
        None if entropy is None else entropy[keep_mask],
        remaining_budget,
        redundant_fraction,
        rng,
    )
    return sorted(reserved_indices + remaining_selected)
