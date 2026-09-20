#!/usr/bin/env python
"""Entropy-guided GPTQ calibration for GR00T W4A4 packs.

Mirrors Omega-QVLA-official-baseline's `tools/build_gptq_weights.py` --
same GPTQ math, same output pack format -- but replaces its calibration
sample source. The original picks samples by blind LIBERO
one-per-task/stride sampling (`load_libero_samples` /
`load_dataset_samples` + `choose_sample_indices`); this script instead
selects from an entropy/label cache produced by `compute_entropy_labels.py`,
weighted toward "precision-critical" (low predictive-entropy) timesteps via
`entropy_calib.sample_selection.entropy_guided_sample_indices`.

Everything else -- the Gram-matrix accumulation (`collect_layer_gram`),
per-layer GPTQ solve (`gptq_quantize_weight`), optional DuQuant rotation,
and the saved `.pt` pack format -- is imported unchanged from Omega-QVLA, so
the output drops into its existing `merge_packs.py` / eval scripts as-is.

Neither Omega-QVLA-official-baseline nor DemoSpeedup is modified by this
file; both are only imported read-only via sys.path.

v1 scope: the plain GPTQ recipe over a LeRobot dataset path (matching
`compute_entropy_labels.py`'s data source), not live LIBERO rollouts and
not the DiT per-step/SVD recipe (`build_dit_a2lite_svd_gptq_perstep.py`).
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

from entropy_calib import entropy_guided_sample_indices  # noqa: E402
from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    flatten_action_dict,
    normalized_input_no_inference,
    seed_everything,
)
from tools.build_gptq_weights import (  # noqa: E402
    _autocast_context,
    _build_duquant_rotation,
    _clear_quant_env,
    _resolve_per_kind,
    collect_layer_gram,
    resolve_target_layers,
)
from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import COMPUTE_DTYPE  # noqa: E402,F401 (kept for parity with build_gptq_weights.py)
from gr00t.quantization.gptq_layers import gptq_quantize_weight  # noqa: E402
from tools.analyze_layerwise_quant_drift import get_named_module, load_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--weight-bits", type=int, default=4)
    p.add_argument("--gptq-block-size", type=int, default=128)
    p.add_argument("--gptq-damp-percent", type=float, default=0.01)
    p.add_argument("--gptq-err-comp-gamma", type=float, default=1.0)
    p.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")

    p.add_argument("--dataset-path", required=True,
                    help="LeRobot dataset the entropy cache was computed over.")
    p.add_argument("--entropy-cache", required=True,
                    help=".npz produced by scripts/compute_entropy_labels.py")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)

    p.add_argument("--num-samples", type=int, default=10,
                    help="Calibration budget (same semantics as build_gptq_weights.py).")
    p.add_argument("--redundant-fraction", type=float, default=0.2,
                    help="Fraction of --num-samples reserved for label==1 (redundant) steps.")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--token-cap", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--include-regex", default=None)
    p.add_argument("--exclude-regex", default=None)
    p.add_argument("--scope", default="")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--max-layers", type=int, default=0)

    p.add_argument("--duquant-rotation", action="store_true")
    p.add_argument("--duquant-block-size", type=int, default=64)
    p.add_argument("--duquant-rot-mode", default="svd",
                    choices=["svd", "hadamard", "svd_hadamard", "random_hadamard"])
    p.add_argument("--duquant-permute", action="store_true")
    p.add_argument("--duquant-output-rotation", action="store_true")
    p.add_argument("--duquant-block-out", type=int, default=0)
    p.add_argument("--attn-block-size", type=int, default=-1)
    p.add_argument("--attn-block-out", type=int, default=-1)
    p.add_argument("--attn-rot-mode", default="",
                    choices=["", "svd", "hadamard", "svd_hadamard", "random_hadamard"])
    p.add_argument("--attn-gptq-damp", type=float, default=-1.0)
    p.add_argument("--attn-group-size", type=int, default=0)
    p.add_argument("--mlp-block-size", type=int, default=-1)
    p.add_argument("--mlp-block-out", type=int, default=-1)
    p.add_argument("--mlp-rot-mode", default="",
                    choices=["", "svd", "hadamard", "svd_hadamard", "random_hadamard"])
    p.add_argument("--mlp-gptq-damp", type=float, default=-1.0)
    p.add_argument("--mlp-group-size", type=int, default=0)

    args = p.parse_args()
    if args.include_regex is None:
        from tools.analyze_layerwise_quant_drift import DEFAULT_INCLUDE_REGEX
        args.include_regex = DEFAULT_INCLUDE_REGEX
    if args.exclude_regex is None:
        from tools.analyze_layerwise_quant_drift import DEFAULT_EXCLUDE_REGEX
        args.exclude_regex = DEFAULT_EXCLUDE_REGEX
    return args


def load_entropy_guided_samples(args, dataset: LeRobotSingleDataset) -> list[dict]:
    cache = np.load(args.entropy_cache, allow_pickle=False)
    if str(cache["dataset_path"]) != str(args.dataset_path):
        print(
            f"[GPTQ-entropy] WARNING: entropy cache was computed on "
            f"'{cache['dataset_path']}', calibrating against '{args.dataset_path}'"
        )

    candidate_indices = cache["dataset_index"]
    labels = cache["label"]
    entropy = cache["entropy"]

    selected = entropy_guided_sample_indices(
        candidate_indices=candidate_indices,
        labels=labels,
        entropy=entropy,
        num_samples=args.num_samples,
        redundant_fraction=args.redundant_fraction,
        seed=args.sample_seed,
    )
    print(
        f"[GPTQ-entropy] selected {len(selected)}/{args.num_samples} calibration samples "
        f"({int((labels[np.isin(candidate_indices, selected)] == 0).sum())} precision-critical)"
    )

    samples = []
    for dataset_index in selected:
        trajectory_id, base_index = dataset.all_steps[dataset_index]
        obs = dataset.get_step_data(trajectory_id, base_index)
        samples.append({"dataset_index": dataset_index, "seed": args.seed + dataset_index, "obs": obs})
    return samples


def main() -> None:
    args = parse_args()
    _clear_quant_env()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    print(f"[GPTQ-entropy] loading FP policy ... {args.checkpoint}")
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    policy.model.eval()

    layer_names = resolve_target_layers(policy, args)
    print(f"[GPTQ-entropy] target layers: {len(layer_names)}")

    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=data_config.modality_config(),
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        video_backend=args.video_backend,
    )
    samples = load_entropy_guided_samples(args, dataset)
    normalized_samples = [
        {"seed": s["seed"], "normalized": normalized_input_no_inference(policy, s["obs"])}
        for s in samples
    ]
    print(f"[GPTQ-entropy] prepared {len(normalized_samples)} normalized observations")

    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    records: dict[str, dict] = {}

    for idx, layer_name in enumerate(layer_names, start=1):
        module = get_named_module(policy.model, layer_name)
        if not isinstance(module, torch.nn.Linear):
            continue
        print(f"[GPTQ-entropy] layer {idx}/{len(layer_names)}: {layer_name}")
        solve_device = module.weight.device
        H, n_tokens = collect_layer_gram(
            policy, normalized_samples, layer_name, args.token_cap, rng, str(args.device),
        )
        W = module.weight.detach().to(dtype=torch.float32, device=solve_device)
        H_dev = H.to(dtype=torch.float32, device=solve_device)

        per_kind = _resolve_per_kind(args, layer_name)
        rotation_R = None
        rotation_R_out = None
        if args.duquant_rotation:
            rotation_R, rotation_R_out = _build_duquant_rotation(W, args, per_kind, solve_device)
            W = W @ rotation_R
            if rotation_R_out is not None:
                W = rotation_R_out @ W
            H_dev = rotation_R.t() @ H_dev @ rotation_R

        baseline_q = gptq_quantize_weight(
            W,
            H_dev,
            bits=args.weight_bits,
            block_size=args.gptq_block_size,
            damp_percent=per_kind["gptq_damp"],
            group_size=int(per_kind["group_size"]),
            err_comp_gamma=float(args.gptq_err_comp_gamma),
        )

        rec_dict = {
            "baseline_q": baseline_q.to(dtype=save_dtype).cpu().contiguous(),
            "weight_bits": int(args.weight_bits),
            "n_calib_tokens": int(n_tokens),
            "gptq_block_size": int(args.gptq_block_size),
            "gptq_damp_percent": float(args.gptq_damp_percent),
        }
        if rotation_R is not None:
            rec_dict["duquant_rotation"] = rotation_R.to(dtype=torch.float16).cpu().contiguous()
        if rotation_R_out is not None:
            rec_dict["duquant_rotation_out"] = rotation_R_out.to(dtype=torch.float16).cpu().contiguous()
        records[layer_name] = rec_dict
        print(f"[GPTQ-entropy] saved {layer_name} (shape={tuple(baseline_q.shape)} calib_tokens={n_tokens})")

    payload = {
        "__meta__": {
            "checkpoint": args.checkpoint,
            "weight_bits": int(args.weight_bits),
            "num_samples": int(args.num_samples),
            "token_cap": int(args.token_cap),
            "gptq_block_size": int(args.gptq_block_size),
            "gptq_damp_percent": float(args.gptq_damp_percent),
            "gptq_err_comp_gamma": float(args.gptq_err_comp_gamma),
            "duquant_rotation": bool(args.duquant_rotation),
            "duquant_rot_mode": args.duquant_rot_mode if args.duquant_rotation else None,
            "save_dtype": args.save_dtype,
            "sample_selection": "entropy_guided",
            "entropy_cache": str(args.entropy_cache),
            "redundant_fraction": float(args.redundant_fraction),
        }
    }
    payload.update(records)
    tmp_path = str(out_path) + f".tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    print(f"[GPTQ-entropy] wrote {out_path}")


if __name__ == "__main__":
    main()
