"""Multi-sample stochastic query adapter for pi0.5 (openpi `Policy`).

Pi0.5's PyTorch policy (`openpi.policies.policy.Policy.infer`) draws fresh
flow-matching/diffusion noise internally whenever `noise=None` is passed
(the default), so repeated calls on the same observation are already
stochastic -- no special hook is needed, unlike DemoSpeedup's ACT/DP path
which had to call a dedicated `get_samples()` method. This plays the same
role as `entropy_calib.multi_sample_policy.compute_entropy_trace` (the
GR00T/Gr00tPolicy variant), but against `policy.infer(obs)` instead of
`policy.model.get_action(normalized)`.

This module only imports the `openpi` package (from `/private/xzq/openpi`,
already on `sys.path` when run inside its own `.venv`). It does not import
or modify Omega-QVLA.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_entropy_trace_pi05(
    policy: Any,
    samples: Sequence[dict],
    num_noise_samples: int = 10,
    reduce: str = "mean",
    base_seed: int = 0,
) -> np.ndarray:
    """Compute one predictive-entropy scalar per pi0.5 calibration observation.

    Args:
        policy: an openpi `Policy` in PyTorch mode, as returned by
            `load_pi05_policy` (Omega-QVLA's `tools/build_pi05_a2lite_gptq_perstep.py`).
        samples: list of obs dicts in the format `record_libero_obs_for_pi05.py`
            produces: {"observation/image", "observation/wrist_image",
            "observation/state", "prompt"}.
        num_noise_samples: stochastic `policy.infer()` calls per observation.
        reduce: "mean" averages the per-chunk-step std-based entropy over the
            whole predicted action horizon; "first" uses only the immediate
            next-action step.
        base_seed: seed offset; sample `i`'s noise draws use
            `seed_everything(base_seed + i * 1000 + k)`.

    Returns:
        1D float array, one entropy value per entry in `samples`.
    """
    if reduce not in ("mean", "first"):
        raise ValueError(f"reduce must be 'mean' or 'first', got {reduce!r}")

    entropies = np.zeros(len(samples), dtype=np.float64)
    for i, obs in enumerate(samples):
        chunk_samples = []
        for k in range(num_noise_samples):
            seed_everything(base_seed + i * 1000 + k)
            result = policy.infer(obs)
            chunk_samples.append(np.asarray(result["actions"], dtype=np.float32))  # (horizon, dim)

        stacked = np.stack(chunk_samples, axis=0)  # (num_noise_samples, horizon, dim)
        per_step_entropy = stacked.std(axis=0).mean(axis=-1)  # (horizon,)
        entropies[i] = per_step_entropy.mean() if reduce == "mean" else per_step_entropy[0]

    return entropies
