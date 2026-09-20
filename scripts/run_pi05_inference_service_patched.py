#!/usr/bin/env python
"""Launch Omega-QVLA-official-baseline's openpi_inference_service.py with
the dit_step_context patch applied first.

Discovered gap: `build_pi05_official_pack.py` (this project) already
patches PI0Pytorch.sample_actions to enter `set_dit_quant_step(t)` during
*calibration*, so the offline per-step act_scale_table gets built correctly.
But Omega-QVLA-official-baseline's own `scripts/openpi_inference_service.py`
was never patched, so at *eval/serving* time `get_current_dit_step()` is
always None -- `GptqLinear` then falls back to the mean of all per-step
scales (gr00t/quantization/gptq_layers.py, the `elif self._has_act_scale_table`
branch) instead of the exact per-step one. That is a real, silent
train/serve skew for any pack with a genuine per-step table (i.e. the
Expert side, built with per-step RTN) -- this script's whole point is
finding out how much that skew actually costs.

This launches the official, unmodified `openpi_inference_service.py`'s
`main()` after applying the same vendored patch used for calibration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OMEGA_QVLA_ROOT = Path(os.environ.get("OMEGA_QVLA_ROOT", "/private/xzq/Omega-QVLA-official-baseline"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(OMEGA_QVLA_ROOT))

from entropy_calib.pi05_dit_step_patch import apply_patch  # noqa: E402

apply_patch()

sys.path.insert(0, str(OMEGA_QVLA_ROOT / "scripts"))
import openpi_inference_service  # noqa: E402
import tyro  # noqa: E402

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    openpi_inference_service.main(tyro.cli(openpi_inference_service.ArgsConfig))
