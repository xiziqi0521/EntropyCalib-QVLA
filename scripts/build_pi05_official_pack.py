#!/usr/bin/env python
"""Build one side (PaliGemma LLM or Expert) of the "official flow" pi0.5
W4A4 pack, using entropy-guided (or any other) calibration obs, by calling
Omega-QVLA-official-baseline's own, completely unmodified
`tools/build_pi05_a2lite_gptq_perstep.py::main()` -- not a fork of it.

This is the recipe documented in
`/private/xzq/Omega-QVLA/results/README_all_evaluations_20260910.md` as
"官方量化流程 W4A4" (94.45% success, -1.30pp vs FP16 on the independent
step30000 checkpoint): PaliGemma LLM = 126 layers via plain GPTQ
(prefix-pass activations, one bucket), Expert = 126 layers via per-step RTN
(err_comp_gamma=0). The two runs merge into one 252-layer pack via
Omega-QVLA-official-baseline's own `tools/merge_packs.py` (also unmodified).

The one thing "官方量化流程" needed beyond Omega-QVLA-official-baseline's own
code was an external adapter to make openpi's pi0.5 denoising loop enter
`gr00t.quantization.dit_step_context.set_dit_quant_step(t)` (that context is
otherwise only entered by GR00T's own model code -- see
`entropy_calib/pi05_dit_step_patch.py`, vendored from
`/private/xzq/Omega-QVLA/tools/pi05_dit_step_patch.py`). This script applies
that patch before calling the official build script's main().

Usage:
    # Expert side (per-step RTN, 126 layers)
    python scripts/build_pi05_official_pack.py --component expert \\
        --checkpoint .../30000 --obs-path cache/pi05_object_entropy_selected10.pt \\
        --output cache/packs/expert_entropy/quantized.pt --max-samples 10

    # PaliGemma LLM side (plain GPTQ, prefix-only, 126 layers)
    python scripts/build_pi05_official_pack.py --component paligemma \\
        --checkpoint .../30000 --obs-path cache/pi05_object_entropy_selected10.pt \\
        --output cache/packs/paligemma_entropy/quantized.pt --max-samples 10

    # Then merge (Omega-QVLA-official-baseline's own tool, unmodified):
    python -m tools.merge_packs --out merged/quantized.pt \\
        cache/packs/paligemma_entropy/quantized.pt cache/packs/expert_entropy/quantized.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OMEGA_QVLA_ROOT = Path(os.environ.get("OMEGA_QVLA_ROOT", "/private/xzq/Omega-QVLA-official-baseline"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(OMEGA_QVLA_ROOT))

PALIGEMMA_INCLUDE = (
    r".*paligemma_with_expert\.paligemma\.model\.language_model\.layers\.[0-9]+\."
    r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*"
)
EXCLUDE = r"(?:^|\.)(vision_tower|vision_model|embeddings|embed_tokens|norm|layernorm|lm_head)(?:\.|$)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--component", required=True, choices=["expert", "paligemma"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-config", default="pi05_libero")
    p.add_argument("--obs-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--token-cap", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gptq-damp-percent", type=float, default=0.05)
    p.add_argument("--duquant-block-size", type=int, default=64)
    p.add_argument("--duquant-block-out", type=int, default=64)
    p.add_argument("--act-percentile", type=float, default=99.9)
    p.add_argument("--save-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--w-bits", type=int, default=4,
                    help="Weight quantization bit-width, passed through to the official "
                         "build script's own --w-bits (default 4, matching the validated "
                         "'official flow' recipe). Values other than 4 skip the deployed "
                         "packed-kernel path at eval time (GR00T-W4A4's can_run() only "
                         "accepts 4/4) and fall back to its fake-quantize simulation path "
                         "-- useful for a kernel-free accuracy probe at other bit-widths.")
    p.add_argument("--a-bits", type=int, default=4,
                    help="Activation quantization bit-width, passed through to the official "
                         "build script's own --a-bits. See --w-bits for the fallback-path note.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from entropy_calib.pi05_dit_step_patch import apply_patch
    apply_patch()

    from tools.build_pi05_a2lite_gptq_perstep import main as official_main

    argv = [
        "build_pi05_a2lite_gptq_perstep.py",
        "--checkpoint", args.checkpoint,
        "--data-config", args.data_config,
        "--obs-path", args.obs_path,
        "--output", args.output,
        "--max-samples", str(args.max_samples),
        "--token-cap", str(args.token_cap),
        "--num-steps", str(args.num_steps),
        "--device", args.device,
        "--gptq-damp-percent", str(args.gptq_damp_percent),
        "--duquant-block-size", str(args.duquant_block_size),
        "--duquant-block-out", str(args.duquant_block_out),
        "--act-percentile", str(args.act_percentile),
        "--save-dtype", args.save_dtype,
        "--w-bits", str(args.w_bits),
        "--a-bits", str(args.a_bits),
    ]
    if args.max_layers > 0:
        argv += ["--max-layers", str(args.max_layers)]

    if args.component == "expert":
        # Default --include-regex / --exclude-regex already target the
        # Expert (gemma_expert.model.layers...) -- matches "expert 126层
        # RTN/per-step" in the README. Leave --capture-prefix off: the
        # Expert only runs inside the (now-patched) denoising loop.
        argv += ["--use-rtn"]
    else:
        # PaliGemma LLM runs once in the prefix pass, never inside the
        # denoising loop -- matches "PaliGemma 126层 GPTQ" (plain GPTQ,
        # no --use-rtn, single activation bucket via --capture-prefix).
        argv += [
            "--include-regex", PALIGEMMA_INCLUDE,
            "--exclude-regex", EXCLUDE,
            "--capture-prefix",
        ]

    sys.argv = argv
    official_main()


if __name__ == "__main__":
    main()
