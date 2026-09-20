#!/usr/bin/env python
"""Filter a pi0.5 LIBERO obs pickle down to an entropy-guided calibration subset.

This is the pi0.5 integration point, and it is deliberately a *data*
transform, not a code fork of Omega-QVLA's calibration script: it reads a
larger obs pool + the entropy cache from `compute_entropy_labels_pi05.py`,
and writes out a smaller `.pt` obs pickle in the exact same format
(list[dict] with "observation/image", "observation/wrist_image",
"observation/state", "prompt"), containing just the samples
`entropy_guided_sample_indices` selected.

Point Omega-QVLA-official-baseline's own, completely unmodified
`tools/build_pi05_a2lite_gptq_perstep.py --obs-path` at the output of this
script instead of at a blindly-truncated pool -- no fork of that ~350-line
GPTQ/A2-lite rotation script is needed or maintained here.

Example:
    python scripts/select_entropy_calibration_obs.py \\
        --obs-path /path/to/larger_pool_obs.pt \\
        --entropy-cache cache/pi05_libero_object_entropy.npz \\
        --num-samples 10 \\
        --output-obs-path cache/pi05_libero_object_obs_entropy_selected.pt

    # Then, from Omega-QVLA-official-baseline, unmodified:
    $OPENPI_ROOT/.venv/bin/python -m tools.build_pi05_a2lite_gptq_perstep \\
        --checkpoint ... \\
        --obs-path /private/xzq/EntropyCalib-QVLA/cache/pi05_libero_object_obs_entropy_selected.pt \\
        --max-samples 10 --output ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entropy_calib import entropy_guided_sample_indices  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--obs-path", required=True, help="Full obs pool (must match what --entropy-cache scored).")
    p.add_argument("--entropy-cache", required=True, help=".npz produced by compute_entropy_labels_pi05.py")
    p.add_argument("--num-samples", type=int, default=10, help="Calibration budget.")
    p.add_argument("--redundant-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-obs-path", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pool = torch.load(args.obs_path, weights_only=False)
    if isinstance(pool, dict) and "samples" in pool:
        pool = pool["samples"]

    cache = np.load(args.entropy_cache, allow_pickle=True)
    if str(cache["obs_path"]) != str(args.obs_path):
        print(
            f"[select-entropy-obs] WARNING: entropy cache was computed on "
            f"'{cache['obs_path']}', selecting from '{args.obs_path}'"
        )
    if len(pool) != len(cache["dataset_index"]):
        raise ValueError(
            f"obs pool has {len(pool)} samples but entropy cache covers "
            f"{len(cache['dataset_index'])} -- they must match 1:1"
        )

    # Group by each observation's LIBERO task prompt so selection can
    # guarantee per-task calibration coverage (see sample_selection.py's
    # task_ids docstring / MECHANISM_CN.md for why plain global top-K-by
    # -entropy selection can silently skip whole tasks otherwise).
    task_ids = [pool[i]["prompt"] for i in cache["dataset_index"]]

    selected = entropy_guided_sample_indices(
        candidate_indices=cache["dataset_index"],
        labels=cache["label"],
        entropy=cache["entropy"],
        num_samples=args.num_samples,
        redundant_fraction=args.redundant_fraction,
        seed=args.seed,
        task_ids=task_ids,
    )
    n_precision = int((cache["label"][np.isin(cache["dataset_index"], selected)] == 0).sum())
    n_tasks_covered = len({pool[i]["prompt"] for i in selected})
    print(
        f"[select-entropy-obs] selected {len(selected)}/{len(pool)} obs "
        f"({n_precision} precision-critical, target budget={args.num_samples}, "
        f"{n_tasks_covered} unique tasks covered)"
    )

    filtered = [pool[i] for i in selected]
    out_path = Path(args.output_obs_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered, out_path)
    print(f"[select-entropy-obs] wrote {out_path} ({len(filtered)} samples)")


if __name__ == "__main__":
    main()
