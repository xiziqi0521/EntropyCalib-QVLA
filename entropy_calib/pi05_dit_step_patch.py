"""Monkey-patch openpi's Pi0Pytorch.sample_actions to enter gr00t's
set_dit_quant_step(t) context per denoising iteration.

Vendored verbatim from `/private/xzq/Omega-QVLA/tools/pi05_dit_step_patch.py`
(read-only source; neither that repo nor Omega-QVLA-official-baseline is
modified). Copied here rather than imported cross-repo so callers only ever
need Omega-QVLA-official-baseline on sys.path -- importing this file
alongside both repos' `gr00t` packages on sys.path at once risks resolving
`gr00t.quantization.dit_step_context` to whichever repo happened to load
first, which would silently desync from the `gr00t.quantization.gptq_layers`
the calling build script actually uses.

Why this exists
----------------
gr00t's calibration/quantization hooks (build_pi05_a2lite_gptq_perstep.py,
build_pi05_svdquant_weights.py, the GptqLinear runtime itself) all gate
activation capture on `gr00t.quantization.dit_step_context.get_current_dit_step()`.
That context is only ever entered by gr00t's OWN action head
(gr00t/model/action_head/flow_matching_action_head.py) -- openpi's pi0.5
implementation (openpi/src/openpi/models_pytorch/pi0_pytorch.py) has no
knowledge of it. Without this patch, every hook sees
`get_current_dit_step() -> None` for the entire denoising loop and skips
every layer ("no activations captured") -- this is exactly the failure mode
this project's build_gptq_weights_entropy.py first hit against pi0.5.

This module reimplements Pi0Pytorch.sample_actions's control flow (same
while-loop, same Euler step, same call to self.denoise_step) but brackets
each iteration with `set_dit_quant_step(t)`, then monkey-patches it onto
the class. It does NOT touch openpi's or Omega-QVLA-official-baseline's
source tree -- both repos stay completely unmodified.

(An earlier version of this patch used two small additive functions,
`enter_dit_quant_step`/`exit_dit_quant_step`, added directly to
Omega-QVLA-official-baseline's `gr00t/quantization/dit_step_context.py`,
to dodge a TorchDynamo graph-break on `ContextVar.set`/contextlib's
generator `__enter__`/`__exit__` that blocked `TORCHINDUCTOR_CUDAGRAPHS`
from capturing this loop. That graph-break only mattered for a cudagraphs
experiment on this *calibration* code path -- separately confirmed that
the actual deployed LIBERO eval servers never apply this patch at all
(openpi_inference_service.py runs vanilla, unpatched sample_actions; the
packed W4A4 kernel dynamically re-quantizes activations every call and
never reads the per-step scale table this patch's context feeds), so the
cudagraphs win didn't carry over to anything real. Reverted per explicit
instruction to keep both source repos untouched -- kernel/runtime
experiments now live on a separate copy, never in-place.)

Usage: import and call apply_patch() once, BEFORE constructing/loading the
policy, with Omega-QVLA-official-baseline (not Omega-QVLA) on sys.path:

    from entropy_calib.pi05_dit_step_patch import apply_patch
    apply_patch()
    ...
    policy = create_policy(...)

If the installed openpi version's sample_actions signature/body differs
from what this patch assumes, apply_patch() raises instead of silently
no-op'ing, so a mismatch is loud rather than a second round of empty packs.
"""
from __future__ import annotations

import inspect

import torch

from gr00t.quantization.dit_step_context import set_dit_quant_step

_PATCHED = False


def _patched_sample_actions(self, device, observation, noise=None, num_steps=10) -> torch.Tensor:
    """Same as openpi.models_pytorch.pi0_pytorch.PI0Pytorch.sample_actions,
    with set_dit_quant_step(t) entered around each denoise_step call.

    NOTE: this mirrors the `for _ in range(num_steps)` control flow of the
    currently-installed openpi (the original vendored patch assumed an
    older `while time >= -dt / 2` loop; openpi switched to a fixed-count
    for-loop for torch.compile/dynamo compatibility -- see the comment in
    openpi's own pi0_pytorch.py). apply_patch()'s sanity check will catch
    the next such drift.
    """
    bsize = observation.state.shape[0]
    if noise is None:
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        noise = self.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
        observation, train=False
    )

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    dt = torch.full((), -1.0 / num_steps, dtype=torch.float32, device=device)

    x_t = noise
    time = torch.ones((), dtype=torch.float32, device=device)
    for t in range(num_steps):
        expanded_time = time.expand(bsize)
        with set_dit_quant_step(t, total=num_steps):
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

        x_t = x_t + dt * v_t
        time = time + dt
    return x_t


def _make_att_2d_masks(pad_masks, att_masks):
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    return make_att_2d_masks(pad_masks, att_masks)


def apply_patch() -> None:
    """Monkey-patch Pi0Pytorch.sample_actions in-place. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    original_src = inspect.getsource(PI0Pytorch.sample_actions)
    required_markers = [
        "def sample_actions(self, device, observation, noise=None, num_steps=10)",
        "self.denoise_step(",
        "for _ in range(num_steps):",
    ]
    missing = [m for m in required_markers if m not in original_src]
    if missing:
        raise RuntimeError(
            "pi05_dit_step_patch: openpi's Pi0Pytorch.sample_actions no "
            f"longer matches the assumed implementation (missing: {missing}). "
            "Re-check pi0_pytorch.py and update _patched_sample_actions "
            "before re-running the build -- otherwise per-step activations "
            "will silently fail to be captured again."
        )

    PI0Pytorch.sample_actions = _patched_sample_actions
    _PATCHED = True
    print("[pi05_dit_step_patch] PI0Pytorch.sample_actions patched: "
          "set_dit_quant_step(t) now entered per denoising iteration.",
          flush=True)
