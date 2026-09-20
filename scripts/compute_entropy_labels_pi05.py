#!/usr/bin/env python
"""Score a pi0.5 LIBERO obs pickle for predictive entropy + precision labels.

Run with the **openpi venv**
(`/private/xzq/openpi/.venv/bin/python`), not the `omega_qvla` conda env --
this needs `openpi` + a real pi0.5 PyTorch checkpoint, and does not need
`gr00t` at all (entropy scoring only calls `policy.infer()`).

Consumes obs pickles in the format `record_libero_obs_for_pi05.py` produces
(Omega-QVLA's `tools/record_libero_obs_for_pi05.py`; see that script or the
existing `duquant_act_stats/*.pt` files under `/private/xzq/Omega-QVLA` for
ready-made examples). To get real benefit from entropy-guided *selection*
downstream, record more observations than your final calibration budget
(e.g. 50) so there's an actual pool to choose the most precision-critical
ones from -- the existing 10-sample pickles are already at the typical
final budget size, so selection over them is closer to a no-op.

Neither Omega-QVLA-official-baseline, Omega-QVLA, nor openpi is modified;
`load_pi05_policy` is imported read-only from Omega-QVLA-official-baseline's
`tools/build_pi05_a2lite_gptq_perstep.py` (that import only needs the three
lightweight `gr00t.quantization.*` submodules plus `openpi`, not the rest
of `gr00t`).

Example:
    /private/xzq/openpi/.venv/bin/python scripts/compute_entropy_labels_pi05.py \\
        --checkpoint /private/xzq/openpi/checkpoints/pi05_libero/pi05_independent_pytorch_l20_seed42/30000 \\
        --obs-path /private/xzq/Omega-QVLA/duquant_act_stats/pi05_libero_object_obs.pt \\
        --output cache/pi05_libero_object_entropy.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
OMEGA_QVLA_ROOT = Path(os.environ.get("OMEGA_QVLA_ROOT", "/private/xzq/Omega-QVLA-official-baseline"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(OMEGA_QVLA_ROOT))

from entropy_calib import compute_entropy_trace_pi05, label_precision_segments  # noqa: E402
from tools.build_pi05_a2lite_gptq_perstep import load_pi05_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-config", default="pi05_libero")
    p.add_argument("--obs-path", required=True, help="Obs pickle from record_libero_obs_for_pi05.py")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-noise-samples", type=int, default=10,
                    help="Stochastic policy.infer() calls per observation used to estimate entropy.")
    p.add_argument("--entropy-reduce", default="mean", choices=["mean", "first"])
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=0,
                    help="DemoSpeedup skips the first 50 rollout timesteps (policy warmup during a "
                         "closed-loop trajectory). These obs pickles are independent single-step "
                         "observations, not a rollout, so there's no warmup period -- default 0.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, help="Output .npz cache path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[entropy-calib-pi05] loading policy ... {args.checkpoint}")
    policy, _model = load_pi05_policy(args.checkpoint, args.data_config, args.device)

    samples = torch.load(args.obs_path, weights_only=False)
    if isinstance(samples, dict) and "samples" in samples:
        samples = samples["samples"]
    n = len(samples)
    print(f"[entropy-calib-pi05] scoring {n} obs samples "
          f"({args.num_noise_samples} noise samples/obs)")

    entropy = compute_entropy_trace_pi05(
        policy=policy,
        samples=samples,
        num_noise_samples=args.num_noise_samples,
        reduce=args.entropy_reduce,
        base_seed=args.seed,
    )
    labels = label_precision_segments(
        entropy, min_cluster_size=args.min_cluster_size, warmup_steps=args.warmup_steps
    )

    n_precision = int((labels == 0).sum())
    print(f"[entropy-calib-pi05] {n_precision}/{n} obs labeled precision-critical")
    print(f"[entropy-calib-pi05] entropy: min={entropy.min():.4f} max={entropy.max():.4f} "
          f"mean={entropy.mean():.4f}")

    np.savez(
        out_path,
        dataset_index=np.arange(n, dtype=np.int64),
        prompt=np.array([s.get("prompt", "") for s in samples], dtype=object),
        entropy=entropy,
        label=labels,
        obs_path=str(args.obs_path),
    )
    print(f"[entropy-calib-pi05] wrote {out_path}")


if __name__ == "__main__":
    main()
