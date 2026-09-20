#!/usr/bin/env python
"""Reconstructed pi0.5 ATM/OHB calibration script, driven by entropy-guided samples.

Omega-QVLA-official-baseline's `gr00t/atm/pi05_atm.py` documents a
calibration API ("used by tools/calibrate_atm_pi05.py") -- but that tool is
missing from both `Omega-QVLA-official-baseline` and `Omega-QVLA` on disk.
The existing `atm_alpha_beta_pi05/*.json` calibration files (36 attention
blocks x 7 sublayers = 252 layers, matching the user's reported "252 layers,
good accuracy" pi0.5 result) were almost certainly produced by that missing
script, driving DuQuant's runtime ATM/OHB per-head scaling
(`enable_pi05_atm_if_configured`), NOT the offline GPTQ-pack path
(`build_pi05_a2lite_gptq_perstep.py`) -- that path has an unrelated bug: it
depends on `gr00t.quantization.dit_step_context`, which is only ever entered
by GR00T's own model code, never by openpi's pi0.5 denoising loop, so it
always captures zero activations for pi0.5 regardless of calibration data.

This script reconstructs the missing calibration step using pi05_atm.py's
still-intact, documented capture API:
  register_pi05_atm_capture(model, cb)  -- per-head logits std (post-RoPE, pre-softmax)
  register_pi05_ohb_capture(model, cb)  -- per-head output RMS

**Aggregation formula is a reconstruction, not a byte-exact reproduction**
of whatever the lost script did (no aggregation logic survives anywhere in
either repo -- GR00T's own `dit_atm.py` has the same gap). Given the
existing JSON's value ranges (alpha ~0.87-1.17, beta ~1.0-1.06, both
centered near 1.0), this implements the natural reading of "ATM/OHB
per-head scaling": normalize each head's captured statistic to the
per-layer mean across heads, so per-head quantization difficulty is
equalized before a single shared scale is applied:
    alpha_h = mean_h(std_h) / std_h
    beta_h  = mean_h(rms_h)  / rms_h
Verify against your own held-out eval before trusting this over the
original JSON files for anything but this project's own A/B comparison.

Neither Omega-QVLA-official-baseline, Omega-QVLA, nor openpi is modified;
`load_pi05_policy` is imported read-only from Omega-QVLA-official-baseline,
`register_pi05_atm_capture` / `register_pi05_ohb_capture` /
`clear_pi05_atm_capture` / `uninstall_pi05_capture` read-only from its
`gr00t.atm.pi05_atm`.

Example:
    /private/xzq/openpi/.venv/bin/python scripts/calibrate_atm_pi05_entropy.py \\
        --checkpoint /private/xzq/openpi/checkpoints/pi05_libero/pi05_independent_pytorch_l20_seed42/30000 \\
        --obs-path cache/pi05_object_entropy_selected10.pt \\
        --output cache/atm/object_w4a4_entropy10.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
OMEGA_QVLA_ROOT = Path(os.environ.get("OMEGA_QVLA_ROOT", "/private/xzq/Omega-QVLA-official-baseline"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(OMEGA_QVLA_ROOT))

from tools.build_pi05_a2lite_gptq_perstep import load_pi05_policy  # noqa: E402
from gr00t.atm.pi05_atm import (  # noqa: E402
    clear_pi05_atm_capture,
    register_pi05_atm_capture,
    register_pi05_ohb_capture,
    uninstall_pi05_capture,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-config", default="pi05_libero")
    p.add_argument("--obs-path", required=True,
                    help="Calibration obs pickle, e.g. from select_entropy_calibration_obs.py")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True, help="Output alpha/beta JSON, same schema as atm_alpha_beta_pi05/*.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[calibrate-atm-pi05-entropy] loading policy ... {args.checkpoint}")
    policy, model = load_pi05_policy(args.checkpoint, args.data_config, args.device)

    samples = torch.load(args.obs_path, weights_only=False)
    if isinstance(samples, dict) and "samples" in samples:
        samples = samples["samples"]
    print(f"[calibrate-atm-pi05-entropy] calibrating on {len(samples)} obs samples")

    # layer_name -> list of per-head tensors, one per calibration sample forward.
    std_captures: dict[str, list] = defaultdict(list)
    rms_captures: dict[str, list] = defaultdict(list)

    def atm_cb(layer_name: str, std: torch.Tensor) -> None:
        std_captures[layer_name].append(std.detach().cpu())

    def ohb_cb(layer_name: str, rms: torch.Tensor) -> None:
        rms_captures[layer_name].append(rms.detach().cpu())

    n_atm_layers = register_pi05_atm_capture(model, atm_cb)
    n_ohb_layers = register_pi05_ohb_capture(model, ohb_cb)
    print(f"[calibrate-atm-pi05-entropy] wired {n_atm_layers} ATM layers, {n_ohb_layers} OHB layers")

    with torch.no_grad():
        for i, obs in enumerate(samples, 1):
            _ = policy.infer(obs)
            if i == 1 or i % 5 == 0 or i == len(samples):
                print(f"[calibrate-atm-pi05-entropy] sample {i}/{len(samples)}", flush=True)

    clear_pi05_atm_capture(model)
    uninstall_pi05_capture(model)

    alpha_data: dict[str, dict] = {}
    all_layers = sorted(set(std_captures) | set(rms_captures))
    for layer_name in all_layers:
        entry: dict = {}
        if layer_name in std_captures:
            std_stack = torch.stack(std_captures[layer_name], dim=0)  # (n_samples, num_heads)
            std_per_head = std_stack.mean(dim=0)  # (num_heads,)
            alpha = (std_per_head.mean() / std_per_head.clamp_min(1e-8)).tolist()
            entry["all"] = alpha
        if layer_name in rms_captures:
            rms_stack = torch.stack(rms_captures[layer_name], dim=0)
            rms_per_head = rms_stack.mean(dim=0)
            beta = (rms_per_head.mean() / rms_per_head.clamp_min(1e-8)).tolist()
            entry["beta_perhead"] = beta
        alpha_data[layer_name] = entry

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(alpha_data, f, indent=2)
    print(f"[calibrate-atm-pi05-entropy] wrote {out_path} ({len(alpha_data)} layers)")


if __name__ == "__main__":
    main()
