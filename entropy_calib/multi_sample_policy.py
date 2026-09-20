"""Multi-sample stochastic query adapter for GR00T / pi0.5 policies.

Plays the role of DemoSpeedup's `label_entropy` sampling loop
(aloha/act/imitate_episodes.py:288-340, github.com/lingxiao-guo/DemoSpeedup),
which repeatedly queries a policy for stochastic action-chunk predictions to
estimate per-timestep predictive entropy. There, the stochasticity comes from
ACT's CVAE latent or DP's diffusion sampling. Here it comes from GR00T/pi0.5's
DiT action head, which is diffusion-based (same "sampleable stochastic head"
category as DemoSpeedup's DP path): re-running denoising from a fresh random
seed on the same observation yields a fresh action-chunk sample.

This module only *reads* Omega-QVLA-official-baseline as a library (via
sys.path, same pattern its own tools/ scripts use internally) -- it does not
modify that repo.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch


def compute_entropy_trace(
    policy: Any,
    samples: Sequence[dict],
    action_keys: Sequence[str],
    normalized_input_fn: Callable[[Any, dict], Any],
    seed_everything_fn: Callable[[int], None],
    flatten_action_dict_fn: Callable[[dict, Sequence[str]], np.ndarray],
    num_noise_samples: int = 10,
    reduce: str = "mean",
) -> np.ndarray:
    """Compute one predictive-entropy scalar per calibration sample.

    Args:
        policy: a loaded `Gr00tPolicy` (or compatible), as returned by
            Omega-QVLA's `tools.analyze_layerwise_quant_drift.load_policy`.
        samples: list of sample dicts with at least an "obs" key and a
            "seed" key, i.e. the same structure `load_libero_samples` /
            `load_dataset_samples` in Omega-QVLA already produce.
        action_keys: `data_config.action_keys`, passed straight to
            `flatten_action_dict_fn`.
        normalized_input_fn: pass Omega-QVLA's
            `tools.analyze_layerwise_quant_drift.normalized_input_no_inference`.
        seed_everything_fn: pass Omega-QVLA's
            `tools.analyze_layerwise_quant_drift.seed_everything`.
        flatten_action_dict_fn: pass Omega-QVLA's
            `tools.analyze_layerwise_quant_drift.flatten_action_dict`.
        num_noise_samples: how many stochastic denoising passes per
            observation (DemoSpeedup's default multi-sample count is 10).
        reduce: "mean" averages the per-chunk-step std-based entropy over the
            whole predicted action horizon; "first" uses only the immediate
            next-action step (index 0), which is the step calibration
            actually executes on.

    Returns:
        1D float array, one entropy value per entry in `samples`, in the
        same order.
    """
    if reduce not in ("mean", "first"):
        raise ValueError(f"reduce must be 'mean' or 'first', got {reduce!r}")

    entropies = np.zeros(len(samples), dtype=np.float64)
    with torch.no_grad():
        for i, sample in enumerate(samples):
            normalized = normalized_input_fn(policy, sample["obs"])
            base_seed = int(sample.get("seed", 0))

            chunk_samples = []
            for k in range(num_noise_samples):
                seed_everything_fn(base_seed * 1000 + k)
                action_dict = policy.model.get_action(normalized)
                chunk_samples.append(flatten_action_dict_fn(action_dict, action_keys))

            # (num_noise_samples, horizon, dim)
            stacked = np.stack(chunk_samples, axis=0)
            # std across noise samples, then mean across action dims ->
            # one entropy value per predicted horizon step, mirrors
            # DemoSpeedup's `torch.mean(torch.std(action_samples, dim=1), dim=-1)`.
            per_step_entropy = stacked.std(axis=0).mean(axis=-1)  # (horizon,)

            entropies[i] = per_step_entropy.mean() if reduce == "mean" else per_step_entropy[0]

    return entropies
