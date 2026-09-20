#!/usr/bin/env python
"""Compute per-timestep predictive entropy + precision/redundant labels over
a LeRobot-format dataset, using a GR00T/pi0.5 policy loaded the same way
Omega-QVLA-official-baseline's own calibration tools load it.

This does not modify Omega-QVLA-official-baseline or DemoSpeedup; it only
imports read-only helpers from the former (`tools.analyze_layerwise_quant_drift`)
via sys.path, the same way Omega-QVLA's own `tools/build_gptq_weights.py`
imports from that module internally.

Output: an .npz cache with, per dataset step:
    dataset_index, trajectory_id, base_index, entropy, label
so `build_gptq_weights_entropy.py` can turn this into a calibration sample
set without recomputing entropy every run.

Example:
    python scripts/compute_entropy_labels.py \\
        --checkpoint /path/to/gr00t-n1.5-checkpoint \\
        --dataset-path /private/xzq/Omega-QVLA-official-baseline/demo_data/robot_sim.PickNPlace \\
        --data-config examples.Libero.custom_data_config:LiberoDataConfig \\
        --output /private/xzq/EntropyCalib-QVLA/cache/robot_sim_picknplace_entropy.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OMEGA_QVLA_ROOT = Path(
    __import__("os").environ.get(
        "OMEGA_QVLA_ROOT", "/private/xzq/Omega-QVLA-official-baseline"
    )
)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(OMEGA_QVLA_ROOT))

from entropy_calib import compute_entropy_trace, label_precision_segments  # noqa: E402
from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    flatten_action_dict,
    load_policy,
    normalized_input_no_inference,
    seed_everything,
)
from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-noise-samples", type=int, default=10,
                    help="Stochastic denoising passes per step used to estimate entropy.")
    p.add_argument("--entropy-reduce", default="mean", choices=["mean", "first"])
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=0,
                    help="Cap on how many dataset steps to score (0 = all). Entropy scoring "
                         "requires --num-noise-samples policy forward passes per step, so this "
                         "is the main cost knob.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, help="Output .npz cache path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    policy.model.eval()

    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=data_config.modality_config(),
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        video_backend=args.video_backend,
    )

    total_steps = len(dataset)
    n_steps = total_steps if args.max_steps <= 0 else min(args.max_steps, total_steps)
    print(f"[entropy-calib] scoring {n_steps}/{total_steps} dataset steps "
          f"({args.num_noise_samples} noise samples/step)")

    samples = []
    for dataset_index in range(n_steps):
        trajectory_id, base_index = dataset.all_steps[dataset_index]
        obs = dataset.get_step_data(trajectory_id, base_index)
        samples.append(
            {
                "dataset_index": dataset_index,
                "trajectory_id": trajectory_id,
                "base_index": base_index,
                "seed": args.seed + dataset_index,
                "obs": obs,
            }
        )

    entropy = compute_entropy_trace(
        policy=policy,
        samples=samples,
        action_keys=data_config.action_keys,
        normalized_input_fn=normalized_input_no_inference,
        seed_everything_fn=seed_everything,
        flatten_action_dict_fn=flatten_action_dict,
        num_noise_samples=args.num_noise_samples,
        reduce=args.entropy_reduce,
    )
    labels = label_precision_segments(
        entropy, min_cluster_size=args.min_cluster_size, warmup_steps=args.warmup_steps
    )

    n_precision = int((labels == 0).sum())
    print(f"[entropy-calib] {n_precision}/{n_steps} steps labeled precision-critical")

    np.savez(
        out_path,
        dataset_index=np.array([s["dataset_index"] for s in samples], dtype=np.int64),
        trajectory_id=np.array([s["trajectory_id"] for s in samples], dtype=np.int64),
        base_index=np.array([s["base_index"] for s in samples], dtype=np.int64),
        entropy=entropy,
        label=labels,
        dataset_path=str(args.dataset_path),
    )
    print(f"[entropy-calib] wrote {out_path}")


if __name__ == "__main__":
    main()
